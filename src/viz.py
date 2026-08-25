# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Unified plotting facade over the two visualization stacks.

Usage:
    from src.viz import plot_cka_heatmap

The repo carries two visualization modules with partially overlapping
APIs (src/utils/visualization.py and src/interp/visualization.py). This
facade merges their PUBLIC names into one namespace so callers do not
need to know which stack owns a function.

Name-conflict policy: first wins (utils/visualization over
interp/visualization). The losing symbol stays reachable via its owning
module — query ``origin_of``. Prefer adding new plots to
src/utils/visualization.py.
"""

import importlib

_modules = [
    importlib.import_module("src.utils.visualization"),
    importlib.import_module("src.interp.visualization"),
]

_public = {}
_for_modules = {}
for _mod in _modules:
    for _name in dir(_mod):
        _obj = getattr(_mod, _name)
        if _name.startswith("_") or not (callable(_obj) or isinstance(_obj, type)):
            continue
        if _name not in _public:
            _public[_name] = _obj
            _for_modules[_name] = _mod.__name__

globals().update(_public)
__all__ = tuple(sorted(_public))


def origin_of(name: str) -> str:
    """Return the owning module's dotted path for a facade symbol."""
    return _for_modules.get(name, "<unknown>")
