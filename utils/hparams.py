# -*- coding: utf-8 -*-
"""
hparams.py - Hyperparameter merge utility for KAO v3.2.1

Provides a single `merge_hparams()` function that enforces:
    defaults < json_overrides < cli_overrides
with ``None`` values meaning "not specified" (i.e. they do NOT override).
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Optional


def _parse_hparams_source(source: str) -> dict:
    """Parse a hparams string: inline JSON, ``@filepath``, or plain filepath.

    Returns an empty dict on empty / None input.
    Raises ``ValueError`` with a clear message on parse failure.
    """
    if not source:
        return {}
    # Accept @file or plain existing path
    path: Optional[str] = None
    if source.startswith("@"):
        path = source[1:]
    elif os.path.exists(source):
        path = source
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            raise ValueError(
                f"Failed to load hparams from file '{path}': {exc}"
            ) from exc
    # Otherwise treat as inline JSON
    try:
        return json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse hparams JSON string: {exc}\n  Input was: {source!r}"
        ) from exc


def merge_hparams(
    defaults: Dict[str, Any],
    json_overrides: Optional[Dict[str, Any] | str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge hyperparameters with priority: *defaults < json_overrides < cli_overrides*.

    Parameters
    ----------
    defaults : dict
        Base defaults defined in the script.
    json_overrides : dict | str | None
        Overrides from ``--hparams_json``.  Can be a dict (already parsed)
        or a raw string (will be parsed via :func:`_parse_hparams_source`).
        ``None`` means no overrides.
    cli_overrides : dict | None
        Overrides from explicit CLI arguments (e.g. ``--time_budget``).
        Keys whose value is ``None`` are **skipped** (meaning "user did not
        specify this flag").

    Returns
    -------
    dict
        The merged *effective* hyperparameters (a fresh copy — callers may
        mutate freely).
    """
    effective = copy.deepcopy(defaults)

    # --- layer 2: json_overrides ---
    if json_overrides is not None:
        if isinstance(json_overrides, str):
            json_overrides = _parse_hparams_source(json_overrides)
        for k, v in json_overrides.items():
            effective[k] = v

    # --- layer 3: cli_overrides (None values are skipped) ---
    if cli_overrides is not None:
        for k, v in cli_overrides.items():
            if v is not None:
                effective[k] = v

    return effective
