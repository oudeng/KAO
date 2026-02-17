# -*- coding: utf-8 -*-
"""Baseline method registry — discover and instantiate available SR baselines."""


def get_all_baselines(include_optional=True):
    """Return instances of all available baselines.

    Parameters
    ----------
    include_optional : bool
        If True, include baselines whose backend may not be installed
        (they are silently skipped when the import fails).
    """
    baselines = []

    from baselines.pysr_wrapper import PySRSR
    baselines.append(PySRSR())

    from baselines.rils_rols_wrapper import RILSROLSSR
    baselines.append(RILSROLSSR())

    from baselines.gplearn_wrapper import GPLearnSR
    baselines.append(GPLearnSR())

    if include_optional:
        try:
            from baselines.operon_wrapper import OperonSR, OPERON_AVAILABLE
            if OPERON_AVAILABLE:
                baselines.append(OperonSR())
        except Exception:
            pass

    return baselines


def get_baseline_by_name(name: str):
    """Look up a specific baseline by its ``name`` property.

    Raises ``ValueError`` if the name is not found.
    """
    for b in get_all_baselines():
        if b.name.lower() == name.lower():
            return b
    available = [b.name for b in get_all_baselines()]
    raise ValueError(
        f"Unknown baseline: {name!r}. Available: {available}"
    )
