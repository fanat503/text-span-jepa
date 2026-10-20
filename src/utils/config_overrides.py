# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Config overrides applied at the CLI boundary, before training starts.

Kept free of torch so the precedence rules are testable without importing the
trainer: a seed override that is silently dropped is exactly the kind of defect
that turns a nominal multi-seed sweep into one run repeated N times.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from collections.abc import Mapping

__all__ = ["apply_cli_overrides"]


def apply_cli_overrides(
    config: Mapping[str, Any],
    *,
    seed: int | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Return a deep copy of ``config`` with CLI overrides applied.

    Seed precedence becomes: CLI ``--seed`` > ``meta.seed`` > top-level
    ``seed`` > 42.  The last two are read by ``src.train.main``; writing the
    CLI value into ``meta.seed`` makes it win over both without changing the
    documented resolution order.

    ``seed=0`` is a valid seed and is applied (the check is ``is not None``,
    not truthiness).  The input mapping is never mutated, so a caller can keep
    the file's config for logging and compare it with the resolved one.
    """
    resolved = deepcopy(dict(config))

    if output_dir is not None:
        resolved.setdefault("logging", {})["folder"] = output_dir

    if seed is not None:
        resolved.setdefault("meta", {})["seed"] = int(seed)

    return resolved
