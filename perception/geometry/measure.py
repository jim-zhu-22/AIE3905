import numpy as np

def measure(mask, depth=None, depth_scale=None):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Mask must have two dimensions")
    if depth_scale is not None and (not np.isfinite(depth_scale) or depth_scale <= 0):
        raise ValueError("depth_scale must be positive and finite")
    result = {"mask_area_px": int(mask.sum()), "depth_median_m": None,
              "valid_depth_fraction": None, "depth_status": "not_provided"}
    if depth is not None:
        if depth.ndim != 2 or depth.shape != mask.shape:
            raise ValueError("Depth must be a single channel image aligned to RGB")
        values = depth[mask]
        valid = values[np.isfinite(values) & (values > 0)]
        result["valid_depth_fraction"] = float(len(valid) / len(values)) if len(values) else 0.0
        result["depth_status"] = "no_valid_depth" if not len(valid) else "unit_unconfirmed"
        if depth_scale is not None and len(valid):
            result["depth_median_m"] = float(np.median(valid) * depth_scale)
            result["depth_status"] = "measured"
    return result
