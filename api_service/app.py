"""HTTP adapters for archived segmentation bundles and the existing RGB model."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import importlib
import os
from pathlib import Path
import re
import sys
import sqlite3
import threading
from time import perf_counter
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from pydantic import ValidationError

from .errors import RequestGuard, fail, register_handlers
from .images import decode_bytes, decode_image
from .schemas import (MAX_IMAGE_BYTES, MAX_PIXELS, MAX_REQUEST_BYTES, DepthMetadata, ErrorResponse,
                      ExecutionInfo, Identifier, InstanceResult, SegmentOptions, SegmentRequest,
                      SegmentationResult)
from .storage import contained_file, load_bundle, staged_bundle
from .modules import ModuleRegistry
from .workspace import Workspace
from .workspace_routes import register_workspace_routes
from .events import register_event_route

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("aie.api")
VERSION = "0.4.0"


@dataclass
class Settings:
    mode: Literal["archive", "model"] = "archive"
    data_dir: Path = ROOT / "output" / "api"
    sam_weight: Path = ROOT / "week1_model_selection" / "sam2.1_t.pt"
    device: str = "cpu"
    max_request_bytes: int = MAX_REQUEST_BYTES
    workspace_enabled: bool = True
    module_config: Path | None = None

    @classmethod
    def from_env(cls):
        mode = os.getenv("AIE_API_MODE", "archive")
        device = os.getenv("AIE_DEVICE", "cpu")
        if mode not in {"archive", "model"} or device not in {"cpu", "cuda", "auto"}:
            raise ValueError("Invalid AIE_API_MODE or AIE_DEVICE")
        return cls(mode=mode, data_dir=Path(os.getenv("AIE_API_DATA_DIR", str(cls.data_dir))).resolve(),
                   sam_weight=Path(os.getenv("AIE_SAM_WEIGHT", str(cls.sam_weight))).resolve(), device=device,
                   workspace_enabled=os.getenv("AIE_WORKSPACE_ENABLED", "1") == "1",
                   module_config=Path(os.environ["AIE_MODULE_CONFIG"]).resolve() if os.getenv("AIE_MODULE_CONFIG") else None)


def default_model(settings):
    if os.getenv("AIE_SEGMENTATION_FACTORY"):
        module_name, factory = os.environ["AIE_SEGMENTATION_FACTORY"].split(":", 1)
        model = getattr(importlib.import_module(module_name), factory)(settings)
        if not callable(getattr(model, "predict", None)):
            raise ValueError("Segmentation factories must return a predict(image,prompt,mode) adapter")
        return model
    if not settings.sam_weight.is_file():
        raise FileNotFoundError("SAM 权重文件不存在。")
    sys.path.insert(0, str(ROOT / "week1_model_selection"))
    from segmentation_practice.pipeline import GroundedSegmenter
    model = GroundedSegmenter(sam_weight=str(settings.sam_weight), device=settings.device)
    model.model_info["thresholds"] = {"box": model.box_threshold, "text": model.text_threshold,
                                     "nms": model.nms_threshold}
    model.model_info["model_version"] = "seg_dino_tiny_sam21_v1"
    return model


def save_model_result(prediction, options, run_id, directory):
    """Export into a private staging directory; publication happens separately."""
    (directory / "masks").mkdir()
    instances = []
    dropped = 0
    width, height = prediction.image.size
    for candidate in prediction.instances:
        raw = candidate.mask
        if raw.shape != (height, width) or ((raw != 0) & (raw != 1)).any():
            raise ValueError("Model returned an invalid mask shape or nonbinary pixels")
        mask = Image.fromarray(raw.astype("uint8") * 255)
        bbox = mask.getbbox()
        if bbox is None:
            dropped += 1
            continue
        instance_id = f"i{len(instances):03d}"
        mask_path = f"masks/{instance_id}.png"
        item = InstanceResult(instance_id=instance_id, object_label=candidate.label,
            bbox_xyxy=list(bbox), mask_path=mask_path, det_score=candidate.det_score)
        mask.save(directory / mask_path)
        instances.append(item)
    canvas = prediction.image.convert("RGBA")
    tint = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    for item in instances:
        with Image.open(directory / item.mask_path) as mask:
            tint.paste((36, 220, 87, 100), (0, 0), mask.convert("L"))
    canvas = Image.alpha_composite(canvas, tint).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for item in instances:
        x1, y1, x2, y2 = item.bbox_xyxy
        draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline="red", width=2)
    canvas.save(directory / "overlay.jpg")
    depth = (options.depth or DepthMetadata()).model_copy(update={"depth_usable": False})
    warning_codes = ["depth_not_processed"] + (["empty_masks_discarded"] if dropped else [])
    return SegmentationResult(run_id=run_id, frame_id=options.frame_id,
        status="ok" if instances else "no_target", prompt=options.prompt,
        image_size=[width, height], model_version=prediction.model_info.get("model_version", "seg_dino_tiny_sam21_v1"),
        instance_count=len(instances), instances=instances, depth=depth, warnings=warning_codes,
        model_info=prediction.model_info, mode=options.mode, stage_latency_ms=prediction.latency_ms,
        execution=ExecutionInfo(mode="model", new_inference=True))


def with_refs(result, prefix):
    result = result.model_copy(deep=True)
    for item in result.instances:
        item.mask_ref = f"{prefix}/{item.mask_path}"
    result.overlay_ref = f"{prefix}/overlay.jpg"
    result.report_ref = f"{prefix}/report.md"
    result.segmentation_ref = f"{prefix}/segmentation.json"
    return result


def create_app(settings=None, *, model_factory=default_model, registry=None):
    settings = settings or Settings.from_env()
    model_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(api):
        # Archive deployments may mount an entirely read-only data directory.
        api.state.model = None
        api.state.model_error = None
        if settings.mode == "model":
            try:
                (settings.data_dir / "runs").mkdir(parents=True, exist_ok=True)
                api.state.model = model_factory(settings)
            except Exception:
                LOG.exception("Model initialization failed")
                api.state.model_error = "模型加载失败，请检查依赖、权重与服务日志。"
        api.state.registry = registry or ModuleRegistry(settings.module_config)
        api.state.workspace = None
        def segment_for_workspace(image, options, run_id, directory):
            if api.state.model is None:
                raise RuntimeError("Segmentation model unavailable")
            if not model_lock.acquire(blocking=False):
                raise RuntimeError("Segmentation model busy")
            try:
                prediction = api.state.model.predict(image, options.prompt, options.mode)
                if prediction.image.size != image.size:
                    raise ValueError("Model changed input coordinate system")
                return save_model_result(prediction, options, run_id, directory)
            finally:
                model_lock.release()
        if settings.workspace_enabled:
            try:
                api.state.workspace = Workspace(settings.data_dir, api.state.registry, segment_for_workspace)
            except (OSError, ValueError, sqlite3.Error):
                LOG.warning("Workspace unavailable (directory permissions or another owner)", exc_info=True)
        try:
            yield
        finally:
            if api.state.workspace is not None:
                api.state.workspace.close()

    responses = {code: {"model": ErrorResponse} for code in (400, 404, 409, 413, 415, 422, 429, 500, 503)}
    api = FastAPI(title="AIE 分割成果 API", version=VERSION, lifespan=lifespan, responses=responses,
        description="archive 发布已有成果；model 进行新图片推理。支持 Base64 和直接上传，深度仅记录元数据。")
    api.add_middleware(RequestGuard, max_bytes=settings.max_request_bytes)
    register_handlers(api)
    api.mount("/assets", StaticFiles(directory=Path(__file__).with_name("static")), name="assets")

    def bundle_directory(kind, bundle_id):
        parent = (settings.data_dir / kind).resolve()
        directory = (parent / bundle_id).resolve()
        if not directory.is_relative_to(parent):
            fail(404, "bundle_not_found", "结果包不存在。")
        return directory

    def sample_directories():
        parent = settings.data_dir / "samples"
        return [p for p in sorted(parent.glob("*")) if re.fullmatch(r"[A-Za-z0-9_-]{1,100}", p.name)
                and p.resolve().is_relative_to(parent.resolve()) and (p / "segmentation.json").is_file()]

    @api.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home():
        return Path(__file__).with_name("index.html").read_text(encoding="utf-8")

    @api.get("/health", tags=["运行状态"])
    def health():
        return {"status": "ok", "mode": settings.mode, "model_ready": api.state.model is not None,
                "depth_processing": False, "api_version": VERSION}

    @api.get("/ready", tags=["运行状态"])
    def ready():
        if settings.mode == "model":
            if api.state.model is None:
                fail(503, "model_not_ready", api.state.model_error or "模型尚未就绪。", retry_after=5)
        else:
            for directory in sample_directories():
                try:
                    load_bundle(directory)
                    break
                except (ValueError, OSError):
                    continue
            else:
                fail(503, "archive_not_ready", "尚无完整可读取的分割成果包。")
        return {"status": "ready", "mode": settings.mode, "model_ready": api.state.model is not None}

    @api.get("/v1/capabilities", tags=["运行状态"])
    def capabilities():
        return {"api_version": VERSION, "mode": settings.mode,
                "new_inference_available": api.state.model is not None,
                "input_formats": ["base64_json", "multipart_form"], "image_formats": ["PNG", "JPEG"],
                "max_image_bytes": MAX_IMAGE_BYTES, "max_image_pixels": MAX_PIXELS,
                "max_request_bytes": settings.max_request_bytes, "depth_pixels_processed": False,
                "max_concurrent_inferences": 1, "schema_version": "pipeline-1.0"}

    @api.get("/v1/examples", tags=["已有成果"])
    def examples():
        return {"execution_mode": "archived_sample", "new_inference": False,
                "examples": [{"sample_id": p.name, "result_ref": f"/v1/examples/{p.name}"} for p in sample_directories()]}

    @api.get("/v1/examples/{sample_id}", response_model=SegmentationResult, tags=["已有成果"])
    def example(sample_id: Identifier, request: Request):
        directory = bundle_directory("samples", sample_id)
        if not (directory / "segmentation.json").is_file():
            fail(404, "sample_not_found", "样例不存在，请先导入分割成果包。")
        try:
            result = load_bundle(directory)
        except (ValueError, OSError):
            LOG.exception("Invalid sample bundle request_id=%s", request.state.request_id)
            fail(500, "invalid_sample_bundle", "样例包不完整或不符合接口格式，请重新导入。")
        result.execution = ExecutionInfo(mode="archived_sample", new_inference=False,
            source_run_id=result.run_id, request_id=request.state.request_id)
        result.depth.depth_usable = False
        if "depth_file_not_packaged" not in result.warnings:
            result.warnings.append("depth_file_not_packaged")
        return with_refs(result, f"/artifacts/samples/{sample_id}")

    def run_inference(options, image, request, started):
        if api.state.model is None:
            fail(503, "model_not_ready", api.state.model_error or "当前为 archive 模式，未启用新图片推理。", retry_after=5)
        if not model_lock.acquire(blocking=False):
            fail(503, "model_busy", "模型正在处理另一张图片，请稍后重试。", retry_after=1)
        run_id = f"run_{uuid4().hex}"
        try:
            prediction = api.state.model.predict(image, options.prompt, options.mode)
            if prediction.image.size != image.size:
                raise ValueError("Model changed input coordinate system")
            with staged_bundle(settings.data_dir / "runs", run_id) as stage:
                result = save_model_result(prediction, options, run_id, stage)
                result = with_refs(result, f"/artifacts/runs/{run_id}")
                result.execution.request_id = request.state.request_id
                (stage / "report.md").write_text(
                    f"# 分割接口运行结果\n\n- run_id: {run_id}\n- frame_id: {options.frame_id}\n"
                    f"- request_id: {request.state.request_id}\n- 模式: model（新推理）\n"
                    f"- 实例数: {result.instance_count}\n- 深度: 未处理像素\n", encoding="utf-8")
                result.execution.api_elapsed_ms = round((perf_counter() - started) * 1000, 2)
                result.execution.timing_scope = "decode_inference_export_before_validation_and_publication"
                (stage / "segmentation.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
            LOG.info("Inference completed request_id=%s run_id=%s count=%s",
                     request.state.request_id, run_id, result.instance_count)
            return result
        except Exception:
            LOG.exception("Inference failed request_id=%s run_id=%s", request.state.request_id, run_id)
            fail(500, "inference_failed", f"推理失败；运行编号：{run_id}。请使用请求编号查询日志。")
        finally:
            model_lock.release()

    @api.post("/v1/segment", response_model=SegmentationResult, tags=["新图片推理"])
    def segment(payload: SegmentRequest, request: Request):
        started = perf_counter()
        return run_inference(payload, decode_image(payload.image_base64), request, started)

    @api.post("/v1/segment/upload", response_model=SegmentationResult, tags=["新图片推理"])
    def upload(request: Request, image: Annotated[UploadFile, File(description="PNG/JPEG 原始图片")],
               prompt: Annotated[str, Form()], frame_id: Annotated[str, Form()] = "uploaded_frame",
               mode: Annotated[str, Form()] = "full", depth_meta: Annotated[str | None, Form()] = None):
        started = perf_counter()
        try:
            depth = DepthMetadata.model_validate_json(depth_meta) if depth_meta is not None else None
            options = SegmentOptions(prompt=prompt, frame_id=frame_id, mode=mode, depth=depth)
            raw = image.file.read(MAX_IMAGE_BYTES + 1)
            return run_inference(options, decode_bytes(raw), request, started)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc
        finally:
            image.file.close()

    @api.get("/v1/runs/{run_id}", response_model=SegmentationResult, tags=["推理结果"])
    def run_result(run_id: Identifier):
        try:
            result = load_bundle(bundle_directory("runs", run_id))
        except (ValueError, OSError):
            fail(404, "run_not_found", "运行结果不存在或尚未完成。")
        return with_refs(result, f"/artifacts/runs/{run_id}")

    @api.get("/artifacts/{kind}/{bundle_id}/{relative:path}", tags=["结果文件"])
    def artifact(kind: Literal["samples", "runs"], bundle_id: Identifier, relative: str):
        directory = bundle_directory(kind, bundle_id)
        try:
            result = load_bundle(directory)
            allowed = {"segmentation.json", "report.md", "overlay.jpg"} | {i.mask_path for i in result.instances}
            if relative not in allowed:
                raise ValueError("Artifact not declared by bundle")
            path = contained_file(directory, relative)
        except (ValueError, OSError):
            fail(404, "artifact_not_found", "结果文件不存在。")
        media_type = {".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json",
                      ".md": "text/plain; charset=utf-8"}[path.suffix.lower()]
        return FileResponse(path, media_type=media_type)

    register_workspace_routes(api, settings)
    register_event_route(api)
    return api


app = create_app()
