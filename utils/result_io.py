# -*- coding: utf-8 -*-
"""
result_io.py - Unified result.json writer for KAO v3.2
"""
import json
from pathlib import Path

import numpy as np


def ensure_json_serializable(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_result_json(path, payload: dict):
    """Write a result.json file with JSON-safe contents.

    Parameters
    ----------
    path : str or Path
        Destination file path (parent dirs created automatically).
    payload : dict
        Result dictionary (may contain numpy types).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = ensure_json_serializable(payload)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
