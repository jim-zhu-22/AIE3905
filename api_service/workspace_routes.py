"""Local app routes: modules, persistent jobs, perception and teammate bundles."""

from io import BytesIO
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tempfile
from typing import Annotated
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

from fastapi import File, Form, Header, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import ValidationError
from starlette.background import BackgroundTask

from .errors import fail
from .images import decode_bytes
from .import_sample import import_sample
from .schemas import DepthMetadata, Identifier, MAX_IMAGE_BYTES, SegmentOptions, SegmentationResult
from .workspace import IdempotencyConflict, JobInfo, JobPage, PerceptionResult, SampleJob, submission_fingerprint


def register_workspace_routes(api, settings):
    @api.get("/v1/integration-guide", tags=["应用"])
    def integration_guide():
        return FileResponse(Path(__file__).resolve().parents[1] / "docs/app_integration.md", media_type="text/plain; charset=utf-8")

    @api.get("/v1/environment-guide", tags=["应用"])
    def environment_guide():
        return FileResponse(Path(__file__).resolve().parents[1] / "docs/local_inference.md", media_type="text/plain; charset=utf-8")

    @api.get("/v1/progress-report", tags=["应用"])
    def progress_report():
        return FileResponse(Path(__file__).resolve().parents[1] / "docs/app_progress_20261026.md", media_type="text/plain; charset=utf-8")

    @api.get("/v1/modules/template", tags=["应用"])
    def module_template():
        return FileResponse(Path(__file__).with_name("modules.default.json"), media_type="application/json", filename="modules.local.json")

    def workspace():
        if api.state.workspace is None:
            fail(503, "workspace_unavailable", "任务工作区不可用，请检查目录权限或是否已由其他进程使用。")
        return api.state.workspace

    def job_or_404(job_id):
        job = workspace().store.get(job_id)
        if job is None:
            fail(404, "job_not_found", "任务不存在。")
        return job

    @api.get("/v1/app/ready", tags=["应用"])
    def app_ready():
        workspace()
        return {"status":"ready", "workspace_available":True, "new_inference_available":api.state.model is not None}

    @api.get("/v1/modules", tags=["应用"])
    def modules():
        registry = api.state.registry
        return {"workspace_available": api.state.workspace is not None, "config_error": registry.config_error,
                "modules": [{"id":"segmentation","title":"目标检测与分割","adapter":"existing_pipeline",
                             "state":"ready" if api.state.model else "unavailable",
                             "reason":None if api.state.model else "模型尚未启用，可先查看已有样例"}] + registry.describe()}

    @api.get("/v1/jobs", response_model=JobPage, tags=["任务"])
    def jobs(limit: Annotated[int, Query(ge=1, le=100)] = 30, offset: Annotated[int, Query(ge=0)] = 0):
        return workspace().store.list(limit, offset)

    @api.post("/v1/jobs", response_model=JobInfo, status_code=202, tags=["任务"])
    def sample_job(payload: SampleJob, request: Request):
        parent = (settings.data_dir / "samples").resolve()
        directory = (parent / payload.sample_id).resolve()
        if not directory.is_relative_to(parent) or not (directory / "segmentation.json").is_file():
            fail(404, "sample_not_found", "找不到已有分割包。")
        try:
            from .storage import load_bundle
            result = load_bundle(directory)
            options = SegmentOptions(frame_id=result.frame_id, prompt=result.prompt, depth=result.depth)
            return workspace().submit(source="sample", options=options, modules=payload.modules,
                request_id=request.state.request_id, sample_id=payload.sample_id)
        except OverflowError:
            fail(429, "queue_full", "任务队列已满，请稍后重试。", retry_after=5)
        except (ValueError, OSError):
            fail(400, "invalid_sample_job", "样例不完整或模块选择不符合约定。")

    @api.post("/v1/perceive", response_model=JobInfo, status_code=202, tags=["统一感知"])
    def perceive(request: Request, rgb: Annotated[UploadFile, File()], prompt: Annotated[str, Form()],
                 frame_id: Annotated[str, Form()] = "uploaded_frame", mode: Annotated[str, Form()] = "full",
                 modules: Annotated[str, Form()] = "[]", meta: Annotated[str | None, Form()] = None,
                 depth: Annotated[UploadFile | None, File()] = None,
                 idempotency_key: Annotated[Identifier | None, Header(alias="Idempotency-Key")] = None):
        engine = workspace()
        try:
            raw_modules = json.loads(modules)
            selection = SampleJob(sample_id="upload", modules=raw_modules).modules
            depth_meta = DepthMetadata.model_validate_json(meta) if meta else DepthMetadata()
            options = SegmentOptions(frame_id=frame_id, prompt=prompt, mode=mode, depth=depth_meta)
            engine.validate_modules(selection)
            rgb_raw = rgb.file.read(MAX_IMAGE_BYTES + 1)
            image = decode_bytes(rgb_raw)
            depth_raw = None
            if depth is not None:
                depth_raw = depth.file.read(MAX_IMAGE_BYTES + 1)
                if len(depth_raw) > MAX_IMAGE_BYTES:
                    fail(413, "depth_too_large", "深度文件超过大小限制。")
                try:
                    with Image.open(BytesIO(depth_raw)) as depth_image:
                        if (depth_image.format != "PNG" or depth_image.mode not in {"I", "I;16", "I;16L", "I;16B"}
                                or depth_image.size != image.size):
                            fail(400, "invalid_depth", "深度图应为与 RGB 同尺寸的 16 位单通道 PNG。")
                        depth_image.verify()
                except (ValueError, OSError):
                    fail(400, "invalid_depth", "深度文件无法解码。")
                options.depth = depth_meta.model_copy(update={"depth_path":"depth.png", "depth_size":list(image.size),
                                                              "depth_dtype":"uint16", "depth_usable":False})
            fingerprint = submission_fingerprint(options,selection,rgb_raw,depth_raw)
            replay = engine.store.replay(idempotency_key,fingerprint)
            if replay is not None:
                return replay
            if api.state.model is None:
                fail(503, "model_not_ready", "模型尚未启用，当前只能处理已有样例。", retry_after=5)
            return engine.submit(source="upload", options=options, modules=selection, request_id=request.state.request_id,
                                 image=image, depth=depth_raw, client_key=idempotency_key, fingerprint=fingerprint)
        except IdempotencyConflict:
            fail(409,"idempotency_conflict","提交编号已用于另一张图片或另一组参数，请生成新编号。")
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc
        except json.JSONDecodeError:
            fail(422, "invalid_modules", "模块选择必须是 JSON 数组。")
        except OverflowError:
            fail(429, "queue_full", "任务队列已满，请稍后重试。", retry_after=5)
        except ValueError:
            fail(422, "invalid_modules", "请选择已注册且没有重复的模块。")
        finally:
            rgb.file.close()
            if depth is not None:
                depth.file.close()

    @api.get("/v1/jobs/{job_id}", response_model=JobInfo, tags=["任务"])
    def job(job_id: Identifier):
        return job_or_404(job_id)

    @api.post("/v1/jobs/{job_id}/cancel", response_model=JobInfo, tags=["任务"])
    def cancel(job_id: Identifier):
        job_or_404(job_id)
        if not workspace().store.cancel(job_id):
            fail(409, "job_not_queued", "只能取消尚未开始处理的任务。")
        return job_or_404(job_id)

    @api.get("/v1/jobs/{job_id}/result", response_model=PerceptionResult, tags=["任务"])
    def result(job_id: Identifier):
        current = job_or_404(job_id)
        if not current["result_ref"]:
            fail(409, "result_not_ready", "任务尚未生成可读取的结果。")
        return workspace().result(job_id)

    @api.get("/v1/jobs/{job_id}/artifacts/{relative:path}", tags=["任务"])
    def artifact(job_id: Identifier, relative: str):
        job_or_404(job_id)
        try:
            path = workspace().artifact(job_id, relative)
        except (ValueError, OSError):
            fail(404, "artifact_not_found", "结果文件不存在或任务尚未完成。")
        types = {".json":"application/json", ".md":"text/plain; charset=utf-8", ".png":"image/png", ".jpg":"image/jpeg"}
        return FileResponse(path, media_type=types[path.suffix])

    @api.get("/v1/jobs/{job_id}/download", tags=["任务"])
    def download(job_id: Identifier):
        current = job_or_404(job_id)
        if not current["result_ref"]:
            fail(409, "result_not_ready", "任务尚未生成可下载的结果。")
        engine = workspace()
        result = engine.result(job_id)
        names = {"perception.json", "segmentation.json", "report.md", "overlay.jpg"} | {i.mask_path for i in result.instances}
        if result.image_ref:
            names.add("rgb.png")
        if result.depth_ref:
            names.add("depth.png")
        temp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        path = Path(temp.name)
        temp.close()
        try:
            with ZipFile(path, "w", ZIP_DEFLATED) as output:
                for name in sorted(names):
                    output.write(engine.artifact(job_id, name), f"{job_id}/{name}")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return FileResponse(path, filename=f"{job_id}.zip", background=BackgroundTask(path.unlink, missing_ok=True))

    @api.post("/v1/bundles/import", status_code=201, tags=["同学成果接入"])
    def import_bundle(bundle: Annotated[UploadFile, File()], sample_id: Annotated[str, Form()]):
        engine = workspace()
        try:
            SampleJob(sample_id=sample_id)
            raw = bundle.file.read(MAX_IMAGE_BYTES + 1)
            if len(raw) > MAX_IMAGE_BYTES:
                fail(413, "bundle_too_large", "压缩包不能超过 15 MiB。")
            with ZipFile(BytesIO(raw)) as archive:
                entries = archive.infolist()
                if len(entries) > 1000 or sum(item.file_size for item in entries) > 200 * 1024 * 1024:
                    fail(413, "bundle_too_large", "压缩包解压后超过限制。")
                candidates = [item.filename for item in entries if PurePosixPath(item.filename).name == "segmentation.json"]
                if len(candidates) != 1:
                    raise ValueError("Exactly one segmentation.json is required")
                if archive.getinfo(candidates[0]).file_size > 1024 * 1024:
                    raise ValueError("Metadata too large")
                metadata = archive.read(candidates[0])
                if len(metadata) > 1024 * 1024:
                    raise ValueError("Metadata too large")
                segmentation = SegmentationResult.model_validate_json(metadata)
                prefix = str(PurePosixPath(candidates[0]).parent)
                prefix = "" if prefix == "." else prefix + "/"
                names = {"segmentation.json", "overlay.jpg", "report.md"} | {i.mask_path for i in segmentation.instances}
                if len({item.filename for item in entries}) != len(entries):
                    raise ValueError("Duplicate archive entries")
                with tempfile.TemporaryDirectory(dir=engine.root) as directory:
                    destination = Path(directory)
                    for relative in names:
                        rel = PurePosixPath(relative)
                        if rel.is_absolute() or PureWindowsPath(relative).drive or "\\" in relative or ".." in rel.parts:
                            raise ValueError("Unsafe archive path")
                        info = archive.getinfo(prefix + relative)
                        if info.file_size > 20 * 1024 * 1024:
                            raise ValueError("Artifact too large")
                        target = destination / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info) as source, target.open("wb") as output:
                            shutil.copyfileobj(source, output)
                    path, count = import_sample(destination, settings.data_dir, sample_id)
            return {"sample_id":sample_id, "instance_count":count, "result_ref":f"/v1/examples/{sample_id}"}
        except FileExistsError:
            fail(409, "sample_exists", "已有同名样例，请换一个名称。")
        except Exception as exc:
            from fastapi import HTTPException
            if isinstance(exc, HTTPException):
                raise
            fail(400, "invalid_bundle", "成果包不完整或不符合 pipeline-1.0 格式。")
        finally:
            bundle.file.close()
