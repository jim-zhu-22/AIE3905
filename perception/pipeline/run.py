import argparse
from pathlib import Path
from perception.config import PipelineConfig, load_config
from perception.io.manifest import read_frames, write_json


def main():
    parser = argparse.ArgumentParser(description="RGB-D instance perception: single frame or CSV batch")
    parser.add_argument("--config", type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--rgb", type=Path)
    inputs.add_argument("--manifest", type=Path)
    parser.add_argument("--depth", type=Path)
    parser.add_argument("--segmentation", type=Path, help="Resume a pipeline-1.0 segmentation; requires --rgb")
    parser.add_argument("--depth-processing", choices=["reference", "statistics"])
    parser.add_argument("--prompt", default="box . bottle . bag . package .")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sam-weight")
    parser.add_argument("--material-weight")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--mode", choices=["full", "tiled"])
    parser.add_argument("--crop-mode", choices=["bbox", "masked"])
    parser.add_argument("--depth-scale", type=float)
    parser.add_argument("--aligned-depth", action="store_true", default=None)
    args = parser.parse_args()
    try:
        if args.segmentation and args.manifest:
            raise ValueError("--segmentation requires --rgb, not --manifest")
        overrides = {key: getattr(args, key) for key in ("sam_weight", "material_weight", "device", "mode", "crop_mode", "depth_scale", "aligned_depth", "depth_processing")}
        if args.segmentation and not args.config and not args.sam_weight:
            overrides["sam_weight"] = "unused_for_import"
        config = load_config(args.config, overrides, require_sam=not args.segmentation) if args.config else PipelineConfig(**{
            key: value for key, value in overrides.items() if value is not None}).validate(require_sam=not args.segmentation)
        if args.depth and args.manifest:
            raise ValueError("For batch input, put depth paths in the manifest")
        rows = read_frames(args.manifest, args.prompt) if args.manifest else [{
            "frame_id": "single", "rgb_path": args.rgb, "depth_path": args.depth, "prompt": args.prompt}]
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Choose a new or empty output directory")
        if config.depth_processing == "statistics" and any(row["depth_path"] for row in rows) and not config.aligned_depth:
            raise ValueError("Confirm depth registration in config before using depth")
        from perception.pipeline.service import PerceptionPipeline
        from perception.io.pipeline_contract import SavedSegmenter
        segmenter = SavedSegmenter(args.segmentation) if args.segmentation else None
        pipeline = PerceptionPipeline(config, segmenter=segmenter)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    summary = {"config": config.to_dict(), "model_load_ms": round(pipeline.load_ms, 2), "frames": []}
    from uuid import uuid4
    run_id = f"run_{uuid4().hex}"
    for row in rows:
        target = args.output / row["frame_id"] if args.manifest else args.output
        try:
            result = pipeline.process(row["rgb_path"], row["prompt"], target, row["depth_path"],
                                      frame_id=row["frame_id"] if args.manifest else Path(row["rgb_path"]).stem, run_id=run_id)
            import json
            contract = json.loads((target / "result.json").read_text(encoding="utf-8"))
            record = {"frame_id": row["frame_id"], "status": "ok", "instance_count": result["instance_count"],
                      "pipeline_status": contract["status"], "frame_processing_ms": result["frame_processing_ms"], "result": str(target / "result.json")}
        except Exception as error:
            record = {"frame_id": row["frame_id"], "status": "error", "error_type": type(error).__name__, "error": str(error)}
        summary["frames"].append(record)
        print(f"{record['frame_id']}: {record['status']}", flush=True)
        write_json(args.output / "batch_summary.json", summary)
    if any(row["status"] == "error" for row in summary["frames"]):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
