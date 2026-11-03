from typing import Any

def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def fixed_tiles(width: int, height: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    """For 1280x960: 704x528 tiles at x=0/576, y=0/432."""
    tw, th = min(704, width), min(528, height)
    xs = sorted({0, width - tw})
    ys = sorted({0, height - th})
    return [(f"x{x}_y{y}", (x, y, x + tw, y + th)) for y in ys for x in xs]


def deduplicate(detections: list[dict[str, Any]], threshold: float = 0.5) -> list[dict[str, Any]]:
    """Score-ordered box NMS after restoring full-image coordinates."""
    kept: list[dict[str, Any]] = []
    for candidate in sorted(detections, key=lambda x: x["score"], reverse=True):
        duplicate = next(
            (old for old in kept if box_iou(candidate["box"], old["box"]) >= threshold),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
        else:
            duplicate["source_tiles"].extend(candidate["source_tiles"])
    return kept
