"""Shared request/response contracts, independent of HTTP and model libraries."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 20_000_000
MAX_REQUEST_BYTES = 22 * 1024 * 1024
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")]
Pixel = Annotated[int, Field(strict=True, gt=0)]
FiniteScore = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Duration = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class DepthMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    depth_path: str | None = Field(default=None, max_length=2048)
    depth_size: list[Pixel] | None = Field(default=None, min_length=2, max_length=2)
    depth_dtype: str | None = Field(default=None, max_length=80)
    depth_unit: str = Field(default="unconfirmed", min_length=1, max_length=80)
    depth_scale_to_m: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    invalid_depth_values: list[Annotated[int, Field(strict=True)]] | None = Field(default=None, max_length=100)
    alignment: str = Field(default="unconfirmed", min_length=1, max_length=80)
    intrinsics: dict[str, Annotated[float, Field(allow_inf_nan=False)]] | None = None
    depth_usable: bool = False


class SegmentOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    frame_id: Identifier = "uploaded_frame"
    prompt: str = Field(min_length=1, max_length=300)
    mode: Literal["full", "tiled"] = "full"
    depth: DepthMetadata | None = None


class SegmentRequest(SegmentOptions):
    image_base64: str = Field(min_length=1, max_length=MAX_IMAGE_BYTES * 4 // 3 + 200,
                              description="RGB PNG/JPEG 图片的 Base64 或 data URL")


class InstanceResult(BaseModel):
    instance_id: Identifier
    object_label: str = Field(min_length=1, max_length=200)
    bbox_xyxy: list[Annotated[int, Field(strict=True)]] = Field(min_length=4, max_length=4)
    mask_path: str = Field(min_length=1, max_length=1024)
    mask_ref: str | None = None
    det_score: FiniteScore
    seg_score: FiniteScore | None = None


class ExecutionInfo(BaseModel):
    mode: Literal["archived_sample", "model"]
    new_inference: bool
    source_run_id: str | None = None
    request_id: str | None = None
    api_elapsed_ms: Duration | None = None
    timing_scope: str | None = None


class SegmentationResult(BaseModel):
    # Preserve additional upstream JSON metadata while validating shared fields.
    model_config = ConfigDict(extra="allow")
    schema_version: Literal["pipeline-1.0"] = "pipeline-1.0"
    run_id: Identifier
    frame_id: Identifier
    stage: Literal["segmentation"] = "segmentation"
    status: Literal["ok", "no_target"]
    prompt: str = Field(min_length=1, max_length=300)
    image_size: list[Pixel] = Field(min_length=2, max_length=2)
    model_version: str = Field(min_length=1, max_length=200)
    instance_count: int = Field(ge=0, strict=True)
    instances: list[InstanceResult]
    depth: DepthMetadata
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    execution: ExecutionInfo | None = None
    stage_latency_ms: dict[str, Duration] = Field(default_factory=dict)
    model_info: dict[str, JsonValue] = Field(default_factory=dict)
    overlay_ref: str | None = None
    report_ref: str | None = None
    segmentation_ref: str | None = None

    @model_validator(mode="after")
    def consistent_instances(self):
        if self.instance_count != len(self.instances):
            raise ValueError("instance_count does not match instances")
        if (self.status == "no_target") != (self.instance_count == 0):
            raise ValueError("status does not match instance_count")
        if len({item.instance_id for item in self.instances}) != self.instance_count:
            raise ValueError("Duplicate instance_id")
        if len({item.mask_path for item in self.instances}) != self.instance_count:
            raise ValueError("Duplicate mask_path")
        width, height = self.image_size
        if width * height > MAX_PIXELS:
            raise ValueError("Image exceeds pixel limit")
        for item in self.instances:
            x1, y1, x2, y2 = item.bbox_xyxy
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid bbox: {item.instance_id}")
        return self


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    fields: list[dict] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: ErrorDetail
