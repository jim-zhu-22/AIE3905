"""Durable local task workspace and extensible per-instance perception pipeline."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sqlite3
import threading
from time import perf_counter
from typing import Annotated, Literal
from uuid import uuid4

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .modules import InstanceContext, ModulePrediction, ModuleRegistry
from .schemas import DepthMetadata, ExecutionInfo, Identifier, InstanceResult, SegmentOptions
from .storage import contained_file, load_bundle, staged_bundle

LOG = logging.getLogger("aie.api")
ACTIVE = {"queued", "running"}


class IdempotencyConflict(ValueError):
    """The caller reused a submission key for different input."""


def submission_fingerprint(options, modules, rgb, depth=None):
    digest = hashlib.sha256()
    digest.update(json.dumps({"options":options.model_dump(mode="json"), "modules":modules},
                             sort_keys=True, separators=(",", ":")).encode())
    for content in (rgb, depth or b""):
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


class SampleJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["sample"] = "sample"
    sample_id: Identifier
    modules: list[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")]] = Field(default_factory=list, max_length=20)


class PerceivedInstance(InstanceResult):
    attributes: dict[str, ModulePrediction] = Field(default_factory=dict)


class PerceptionResult(BaseModel):
    schema_version: Literal["perception-1.0"] = "perception-1.0"
    run_id: Identifier
    frame_id: str
    status: Literal["ok", "partial", "no_target"]
    prompt: str
    image_size: list[int]
    instance_count: int
    instances: list[PerceivedInstance]
    execution: ExecutionInfo
    depth: DepthMetadata
    stages: dict[str, dict]
    warnings: list[str]
    model_versions: dict[str, str] = Field(default_factory=dict)
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    overlay_ref: str
    image_ref: str | None = None
    depth_ref: str | None = None
    report_ref: str
    json_ref: str
    download_ref: str


class JobInfo(BaseModel):
    id: Identifier
    status: Literal["queued", "running", "completed", "partial", "failed", "cancelled"]
    created_at: str
    updated_at: str
    source: Literal["sample", "upload"]
    frame_id: str
    prompt: str
    modules: list[str]
    sample_id: str | None
    stage: str
    done: int = Field(ge=0)
    total: int = Field(ge=0)
    error: str | None
    request_id: str
    result_ref: str | None


class JobPage(BaseModel):
    total: int = Field(ge=0)
    jobs: list[JobInfo]


class JobStore:
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "jobs.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                source TEXT NOT NULL, frame_id TEXT NOT NULL, prompt TEXT NOT NULL, modules TEXT NOT NULL,
                sample_id TEXT, stage TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0, error TEXT, request_id TEXT NOT NULL)""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            for column in ("client_key", "fingerprint"):
                if column not in columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS jobs_client_key ON jobs(client_key) WHERE client_key IS NOT NULL")
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                data TEXT NOT NULL, created_at TEXT NOT NULL)""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def public(row):
        item = dict(row)
        item.pop("client_key", None)
        item.pop("fingerprint", None)
        item["modules"] = json.loads(item["modules"])
        item["result_ref"] = f"/v1/jobs/{item['id']}/result" if item["status"] in {"completed", "partial"} else None
        return item

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self.public(row) if row else None

    def list(self, limit=50, offset=0):
        with self.connect() as db:
            total = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return {"total": total, "jobs": [self.public(row) for row in rows]}

    def active_count(self):
        with self.connect() as db:
            return db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]

    def replay(self, client_key, fingerprint):
        if client_key is None:
            return None
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE client_key=?", (client_key,)).fetchone()
        if row is None:
            return None
        if row["fingerprint"] != fingerprint:
            raise IdempotencyConflict("同一个提交编号对应了不同图片或参数，请使用新编号。")
        return self.public(row)

    def _emit(self, db, job_id):
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return
        cursor = db.execute("INSERT INTO events(job_id,data,created_at) VALUES (?,?,?)",
                           (job_id,json.dumps(self.public(row),ensure_ascii=False),now()))
        if cursor.lastrowid % 100 == 0:
            db.execute("DELETE FROM events WHERE id<=?", (cursor.lastrowid - 10000,))

    def event_snapshot(self):
        with self.connect() as db:
            db.execute("BEGIN")
            cursor = db.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()[0]
            jobs = db.execute("SELECT * FROM jobs ORDER BY created_at DESC,id DESC LIMIT 100").fetchall()
        return {"cursor":cursor,"jobs":[self.public(row) for row in jobs]}

    def events_after(self, cursor, limit=100):
        with self.connect() as db:
            bounds = db.execute("SELECT MIN(id),MAX(id) FROM events").fetchone()
            reset = (bounds[0] is not None and cursor < bounds[0] - 1) or cursor > (bounds[1] or 0)
            rows = db.execute("SELECT id,data FROM events WHERE id>? ORDER BY id LIMIT ?", (cursor,limit)).fetchall()
        return reset,[{"id":row["id"],"job":json.loads(row["data"])} for row in rows]

    def create(self, job_id, source, options, modules, sample_id, request_id, client_key=None, fingerprint=None):
        stamp = now()
        with self.connect() as db:
            db.execute("""INSERT INTO jobs (id,status,created_at,updated_at,source,frame_id,prompt,modules,sample_id,stage,request_id,client_key,fingerprint)
                          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                       (job_id, "queued", stamp, stamp, source, options.frame_id, options.prompt,
                        json.dumps(modules), sample_id, "等待处理", request_id, client_key, fingerprint))
            self._emit(db, job_id)

    def update(self, job_id, **changes):
        allowed = {"status", "stage", "done", "total", "error"}
        if not set(changes).issubset(allowed):
            raise ValueError("Unsupported job update")
        changes["updated_at"] = now()
        with self.connect() as db:
            db.execute("UPDATE jobs SET " + ",".join(f"{key}=?" for key in changes) + " WHERE id=?",
                       (*changes.values(), job_id))
            self._emit(db, job_id)

    def cancel(self, job_id):
        with self.connect() as db:
            cursor = db.execute("UPDATE jobs SET status='cancelled',stage='已取消',updated_at=? WHERE id=? AND status='queued'",
                                (now(), job_id))
            changed = cursor.rowcount == 1
            if changed:
                self._emit(db, job_id)
            return changed

    def claim(self, job_id):
        with self.connect() as db:
            changed = db.execute("UPDATE jobs SET status='running',stage='读取输入',updated_at=? WHERE id=? AND status='queued'",
                                 (now(), job_id)).rowcount == 1
            if changed:
                self._emit(db, job_id)
            return changed


class Workspace:
    def __init__(self, data_dir, registry: ModuleRegistry, segment, *, max_pending=16):
        self.data_dir = data_dir.resolve()
        self.root = self.data_dir / "workspace"
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.segment = segment
        self.max_pending = max_pending
        self.submit_lock = threading.Lock()
        self._owner = (self.root / "owner.lock").open("a+b")
        try:
            self._owner.seek(0)
            if self._owner.read(1) == b"":
                self._owner.write(b"0")
                self._owner.flush()
            self._owner.seek(0)
            import os
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._owner.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.store = JobStore(self.root)
        except Exception:
            self._owner.close()
            raise
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aie-pipeline")
        try:
            self._recover()
        except Exception:
            self.close()
            raise

    def _recover(self):
        with self.store.connect() as db:
            rows = db.execute("SELECT * FROM jobs WHERE status IN ('queued','running')").fetchall()
        for row in rows:
            job = self.store.public(row)
            final = self.root / "results" / job["id"] / "perception.json"
            if final.is_file():
                try:
                    result = PerceptionResult.model_validate_json(final.read_text(encoding="utf-8"))
                    self.store.update(job["id"], status="partial" if result.status == "partial" else "completed", stage="已完成")
                except (ValueError, OSError):
                    LOG.exception("Damaged result during recovery job_id=%s", job["id"])
                    self.store.update(job["id"], status="failed", stage="恢复失败", error="结果文件不完整，请重新提交任务。")
            elif job["status"] == "running":
                self.store.update(job["id"], status="failed", stage="运行中断", error="服务在处理过程中重启，请重新提交。")
            else:
                self.pool.submit(self._run, job["id"])

    def close(self):
        self.pool.shutdown(wait=True)
        self._owner.close()

    def validate_modules(self, modules):
        if len(set(modules)) != len(modules) or any(key not in self.registry.entries for key in modules):
            raise ValueError("模块列表包含重复项或未注册模块。")

    def submit(self, *, source, options, modules, request_id, sample_id=None, image=None, depth=None,
               client_key=None, fingerprint=None):
        self.validate_modules(modules)
        with self.submit_lock:
            replay = self.store.replay(client_key, fingerprint)
            if replay is not None:
                return replay
            if self.store.active_count() >= self.max_pending:
                raise OverflowError("任务队列已满，请稍后重试。")
            job_id = f"job_{uuid4().hex}"
            # Input publication precedes the DB record, so queued jobs always have complete input.
            inputs = self.root / "inputs" / job_id
            inputs.mkdir(parents=True)
            try:
                (inputs / "options.json").write_text(options.model_dump_json(), encoding="utf-8")
                if image is not None:
                    image.save(inputs / "rgb.png")
                if depth is not None:
                    (inputs / "depth.png").write_bytes(depth)
                if source == "sample":
                    parent = (self.data_dir / "samples").resolve()
                    sample = (parent / sample_id).resolve()
                    if not sample.is_relative_to(parent):
                        raise ValueError("Invalid sample path")
                    result = load_bundle(sample, check_pixels=True)
                    snapshot = inputs / "sample"
                    snapshot.mkdir()
                    for relative in {"segmentation.json", "report.md", "overlay.jpg"} | {i.mask_path for i in result.instances}:
                        target = snapshot / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(contained_file(sample, relative), target)
                    load_bundle(snapshot, check_pixels=True)
                self.store.create(job_id, source, options, modules, sample_id, request_id, client_key, fingerprint)
            except Exception:
                if inputs.resolve().is_relative_to((self.root / "inputs").resolve()):
                    shutil.rmtree(inputs)
                raise
            self.pool.submit(self._run, job_id)
            return self.store.get(job_id)

    def _run(self, job_id):
        if not self.store.claim(job_id):
            return
        job = self.store.get(job_id)
        inputs = self.root / "inputs" / job_id
        try:
            options = SegmentOptions.model_validate_json((inputs / "options.json").read_text(encoding="utf-8"))
            with staged_bundle(self.root / "results", job_id) as stage:
                started = perf_counter()
                if job["source"] == "sample":
                    sample = inputs / "sample"
                    segmentation = load_bundle(sample)
                    original_id = segmentation.run_id
                    for relative in {"overlay.jpg"} | {i.mask_path for i in segmentation.instances}:
                        target = stage / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(contained_file(sample, relative), target)
                    segmentation = segmentation.model_copy(deep=True)
                    segmentation.run_id = job_id
                    segmentation.execution = ExecutionInfo(mode="archived_sample", new_inference=False, source_run_id=original_id,
                                                             request_id=job["request_id"])
                    segmentation.depth.depth_usable = False
                    image_path = None
                else:
                    self.store.update(job_id, stage="目标检测与分割")
                    with Image.open(inputs / "rgb.png") as rgb:
                        segmentation = self.segment(rgb.convert("RGB"), options, job_id, stage)
                    segmentation.execution.request_id = job["request_id"]
                    shutil.copyfile(inputs / "rgb.png", stage / "rgb.png")
                    image_path = stage / "rgb.png"
                depth_path = None
                if (inputs / "depth.png").is_file():
                    shutil.copyfile(inputs / "depth.png", stage / "depth.png")
                    depth_path = stage / "depth.png"
                prefix = f"/v1/jobs/{job_id}/artifacts"
                for item in segmentation.instances:
                    item.mask_ref = f"{prefix}/{item.mask_path}"
                segmentation.overlay_ref = f"{prefix}/overlay.jpg"
                segmentation.report_ref = f"{prefix}/report.md"
                segmentation.segmentation_ref = f"{prefix}/segmentation.json"
                stages = {"segmentation": {"status": "ok", "source": job["source"],
                           "new_inference": segmentation.execution.new_inference}}
                timings = {"segmentation": round((perf_counter() - started) * 1000, 2)}
                instances = [PerceivedInstance.model_validate(item.model_dump()) for item in segmentation.instances]
                versions = {"segmentation": segmentation.model_version}
                total = len(instances) * len(job["modules"])
                self.store.update(job_id, total=total)
                done = 0
                work_dir = stage / "module_work"
                for module_id in job["modules"]:
                    module_started = perf_counter()
                    outputs = []
                    for item in instances:
                        self.store.update(job_id, stage=f"{self.registry.entries[module_id]['spec'].title} · {item.instance_id}", done=done)
                        if image_path is None:
                            prediction = ModulePrediction(status="skipped", reason="样例包未包含原始 RGB，不能用叠加图代替模型输入。")
                        else:
                            context = InstanceContext(job_id, item.instance_id, image_path, stage / item.mask_path,
                                depth_path, item.bbox_xyxy, segmentation.depth.model_dump(), work_dir)
                            prediction = self.registry.analyze(module_id, context)
                        item.attributes[module_id] = prediction
                        outputs.append(prediction.status)
                        done += 1
                    state = "ok" if all(value == "ok" for value in outputs) else ("error" if outputs and all(value == "error" for value in outputs) else "partial")
                    stages[module_id] = {"status": state, "ok": outputs.count("ok"), "unknown": outputs.count("unknown"),
                                         "skipped": outputs.count("skipped"), "errors": outputs.count("error")}
                    module_versions = {item.attributes[module_id].model_version for item in instances if item.attributes[module_id].model_version}
                    if module_versions:
                        versions[module_id] = ",".join(sorted(module_versions))
                    timings[module_id] = round((perf_counter() - module_started) * 1000, 2)
                if work_dir.exists():
                    if not work_dir.resolve().is_relative_to(stage.resolve()):
                        raise ValueError("Unsafe module work directory")
                    shutil.rmtree(work_dir)
                partial = any(item["status"] != "ok" for key, item in stages.items() if key != "segmentation")
                status = "no_target" if not instances else ("partial" if partial else "ok")
                result = PerceptionResult(run_id=job_id, frame_id=segmentation.frame_id, status=status,
                    prompt=segmentation.prompt, image_size=segmentation.image_size, instance_count=len(instances),
                    instances=instances, execution=segmentation.execution, depth=segmentation.depth, stages=stages,
                    warnings=list(dict.fromkeys(segmentation.warnings + ["depth_pixels_not_processed"])),
                    model_versions=versions, stage_latency_ms=timings,
                    overlay_ref=f"{prefix}/overlay.jpg", image_ref=f"{prefix}/rgb.png" if image_path else None,
                    depth_ref=f"{prefix}/depth.png" if depth_path else None,
                    report_ref=f"{prefix}/report.md", json_ref=f"{prefix}/perception.json", download_ref=f"/v1/jobs/{job_id}/download")
                (stage / "segmentation.json").write_text(segmentation.model_dump_json(indent=2), encoding="utf-8")
                (stage / "perception.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
                lines = [f"# RGB-D 感知任务 {job_id}", "", f"- 帧：{result.frame_id}", f"- 状态：{result.status}",
                         f"- 实例数：{result.instance_count}", f"- 来源：{'新图片推理' if result.execution.new_inference else '已有分割样例'}",
                         "- 深度像素处理：未接入", "", "| 模块 | 状态 |", "| --- | --- |"]
                lines += [f"| {key} | {value['status']} |" for key, value in stages.items()]
                (stage / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.store.update(job_id, status="partial" if status == "partial" else "completed", stage="已完成", done=total, total=total)
        except Exception:
            LOG.exception("Workspace job failed job_id=%s request_id=%s", job_id, job["request_id"])
            self.store.update(job_id, status="failed", stage="处理失败", error="任务处理失败，请使用任务编号查询服务日志。")

    def result(self, job_id):
        path = contained_file(self.root / "results" / job_id, "perception.json")
        return PerceptionResult.model_validate_json(path.read_text(encoding="utf-8"))

    def artifact(self, job_id, relative):
        result = self.result(job_id)
        directory = self.root / "results" / job_id
        allowed = {"segmentation.json", "perception.json", "report.md", "overlay.jpg"} | {i.mask_path for i in result.instances}
        if result.image_ref:
            allowed.add("rgb.png")
        if (directory / "depth.png").is_file():
            allowed.add("depth.png")
        if relative not in allowed:
            raise ValueError("Undeclared artifact")
        return contained_file(directory, relative)
