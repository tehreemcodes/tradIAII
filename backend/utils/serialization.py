"""
JSON Serialization Utility
===========================
Recursively converts numpy types, NaN, Inf, and other non-JSON-serializable
values to safe Python primitives before returning API responses.

Usage:
    from backend.utils.serialization import to_json_safe
    return to_json_safe(response_dict)
"""
import math
from typing import Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def to_json_safe(obj: Any) -> Any:
    """
    Recursively convert an object to JSON-safe Python primitives.

    Handles:
    - numpy int/float scalars -> int/float
    - numpy arrays           -> list (recursed)
    - NaN / Inf float values -> None
    - pandas Timestamp       -> ISO 8601 string
    - pandas NA / NaT        -> None
    - dict / list / tuple    -> recursed
    - everything else        -> unchanged
    """
    if obj is None:
        return None

    if _HAS_PANDAS:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if obj is pd.NaT:
            return None
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass

    if _HAS_NUMPY:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return [to_json_safe(v) for v in obj.tolist()]

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj

    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]

    return obj
