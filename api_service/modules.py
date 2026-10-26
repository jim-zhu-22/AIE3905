"""Trusted configuration adapters for teammate Python, CLI and HTTP programs."""

from dataclasses import dataclass
import base64
from io import BytesIO
import importlib
import json
import logging
from pathlib import Path
import subprocess
from typing import Annotated, Literal
from urllib.request import Request, urlopen

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

LOG = logging.getLogger("aie.api")
ID = r"^[a-z][a-z0-9_]{0,39}$"


class ModulePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ok", "unknown", "skipped", "error"]
    label: str | None = Field(default=None, max_length=200)
    score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    source: str = Field(default="unavailable", max_length=200)
    model_version: str | None = Field(default=None, max_length=200)
    reason: str | None = Field(default=None, max_length=1000)
    values: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def meaningful_result(self):
        if self.status == "ok" and self.label is None and not self.values:
            raise ValueError("Successful predictions need a label or values")
        if self.status != "ok" and not self.reason:
            raise ValueError("Non-successful predictions need a reason")
        return self


class ModuleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=ID)
    title: str = Field(min_length=1, max_length=100)
    enabled: bool = False
    adapter: Literal["material", "python", "command", "http"]
    factory: str | None = None
    endpoint: str | None = None
    command: list[str] = Field(default_factory=list)
    timeout_s: Annotated[float, Field(gt=0, le=60)] = 30
    options: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def configured(self):
        if self.id == "segmentation":
            raise ValueError("segmentation is reserved")
        if self.enabled:
            if self.adapter == "python" and (not self.factory or ":" not in self.factory):
                raise ValueError("Python adapters need module:factory")
            if self.adapter == "http" and (not self.endpoint or not self.endpoint.startswith(("http://", "https://"))):
                raise ValueError("HTTP adapters need an explicit http(s) endpoint")
            if self.adapter == "command" and not self.command:
                raise ValueError("Command adapters need an argument array")
        return self


class ModuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modules: list[ModuleSpec] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [item.id for item in self.modules]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate module id")
        return self


@dataclass
class InstanceContext:
    run_id: str
    instance_id: str
    rgb_path: Path
    mask_path: Path
    depth_path: Path | None
    bbox_xyxy: list[int]
    depth_meta: dict
    work_dir: Path

    def request_data(self):
        return {"schema_version": "module-1.0", "run_id": self.run_id, "instance_id": self.instance_id,
                "rgb_path": str(self.rgb_path), "mask_path": str(self.mask_path),
                "depth_path": str(self.depth_path) if self.depth_path else None,
                "bbox_xyxy": self.bbox_xyxy, "depth": self.depth_meta}


def normalize_prediction(raw, spec):
    if isinstance(raw, ModulePrediction):
        return raw
    raw = dict(raw)
    if "status" not in raw:
        raw["status"] = "ok" if raw.get("label") or raw.get("values") else "unknown"
    raw.setdefault("source", spec.id)
    if raw["status"] == "unknown":
        raw.setdefault("reason", "module_returned_no_decision")
    return ModulePrediction.model_validate(raw)


class MaterialAdapter:
    def __init__(self, spec):
        checkpoint = Path(str(spec.options.get("checkpoint", "runs/week1_resnet18/best.pt"))).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("材质权重未配置或不存在")
        from material_recognition.infer import Predictor
        self.predictor = Predictor(checkpoint)

    def analyze(self, context):
        # Existing classifier was trained on RGB crops; do not silently change preprocessing.
        with Image.open(context.rgb_path) as image:
            raw = self.predictor.predict(image.convert("RGB").crop(context.bbox_xyxy))
        return {"status": "ok", "label": raw["label"], "score": raw["score"],
                "model_version": raw.get("model_version"), "source": "material_recognition.Predictor",
                "values": {"region": "object_bbox", "scores": raw.get("scores", {})}}


class CommandAdapter:
    def __init__(self, spec):
        self.spec = spec

    def analyze(self, context):
        directory = context.work_dir / self.spec.id / context.instance_id
        directory.mkdir(parents=True)
        request_path, response_path = directory / "request.json", directory / "response.json"
        request_path.write_text(json.dumps(context.request_data(), ensure_ascii=False), encoding="utf-8")
        command = [part.replace("{request}", str(request_path)).replace("{response}", str(response_path))
                   for part in self.spec.command]
        # Separate environments are supported without shell interpolation.
        subprocess.run(command, shell=False, timeout=self.spec.timeout_s, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if response_path.stat().st_size > 1024 * 1024:
            raise ValueError("Module response exceeds 1 MiB")
        return json.loads(response_path.read_text(encoding="utf-8"))


class HttpAdapter:
    def __init__(self, spec):
        self.spec = spec

    def analyze(self, context):
        # A remote process cannot read local file paths: send explicit cropped images.
        payload = {"schema_version": "module-1.0", "run_id": context.run_id,
                   "instance_id": context.instance_id, "bbox_xyxy": context.bbox_xyxy,
                   "depth": context.depth_meta, "region": "bbox_crop"}
        for key, path in (("image_base64", context.rgb_path), ("mask_base64", context.mask_path)):
            with Image.open(path) as image:
                crop = image.crop(context.bbox_xyxy)
                output = BytesIO()
                crop.save(output, format="PNG")
                payload[key] = base64.b64encode(output.getvalue()).decode()
        # Raw depth is deliberately not transmitted without an agreed registration contract.
        request = Request(self.spec.endpoint, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=self.spec.timeout_s) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("Module response exceeds 1 MiB")
        return json.loads(raw)


class ModuleRegistry:
    def __init__(self, config_path=None, *, injected=None):
        self.entries = {}
        self.config_error = None
        try:
            path = config_path or Path(__file__).with_name("modules.default.json")
            config = ModuleConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))
        except (ValueError, OSError):
            LOG.exception("Module configuration failed")
            self.config_error = "模块配置读取失败，请检查配置文件和服务日志。"
            config = ModuleConfig()
        for spec in config.modules:
            self.entries[spec.id] = {"spec": spec, "adapter": None, "state": "disabled", "reason": "尚未启用"}
            if not spec.enabled:
                continue
            try:
                if injected and spec.id in injected:
                    adapter = injected[spec.id]
                elif spec.adapter == "material":
                    adapter = MaterialAdapter(spec)
                elif spec.adapter == "command":
                    adapter = CommandAdapter(spec)
                elif spec.adapter == "http":
                    adapter = HttpAdapter(spec)
                else:
                    module_name, factory_name = spec.factory.split(":", 1)
                    adapter = getattr(importlib.import_module(module_name), factory_name)(spec.options)
                if not callable(getattr(adapter, "analyze", None)):
                    raise ValueError("Adapters must implement analyze(context)")
                self.entries[spec.id].update(adapter=adapter, state="ready", reason=None)
            except Exception:
                LOG.exception("Module initialization failed id=%s", spec.id)
                self.entries[spec.id].update(state="unavailable", reason="模块加载失败，请检查依赖、配置与服务日志。")

    def describe(self):
        return [{"id": key, "title": item["spec"].title, "adapter": item["spec"].adapter,
                 "state": item["state"], "reason": item["reason"]} for key, item in self.entries.items()]

    def analyze(self, module_id, context):
        entry = self.entries[module_id]
        if entry["state"] != "ready":
            return ModulePrediction(status="skipped", reason=entry["reason"] or "module_unavailable")
        try:
            return normalize_prediction(entry["adapter"].analyze(context), entry["spec"])
        except Exception:
            LOG.exception("Module failed run_id=%s instance_id=%s module=%s",
                          context.run_id, context.instance_id, module_id)
            return ModulePrediction(status="error", source=module_id, reason="模块处理失败，其他模块结果已保留。")
