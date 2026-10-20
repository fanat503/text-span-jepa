# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Aggregate final metrics across seeds: mean, sample std, and a t interval.

`scripts/plotting_engine.plot_seed_band` draws *curves*, one band per logged
column, and only when at least two measured seeds exist.  This module is for
the other half of a multi-seed paper: the *final scalar* table, where each cell
needs one number, one spread, and an explicit sample size.

Conventions, stated once so a reader can check the table without reading code:

* The spread is the **sample** standard deviation (``ddof=1``).  Population
  std would understate the uncertainty of a mean over seeds.
* The interval is a two-sided **Student-t** interval at 95%:
  ``mean ± t(0.975, n-1) · std / sqrt(n)``.  With two or three seeds this is
  much wider than the normal ``1.96 · SEM`` approximation, which is the point:
  a 2-seed paper must not print an interval narrower than its data supports.
* A cell with ``n = 1`` reports the mean and **no interval** (NaN in the
  numeric fields, ``"no interval"`` in :func:`format_summary`).  It is not an
  error and it is not a zero-width interval.
* Non-finite inputs (NaN, +-inf) are dropped and the surviving ``n`` is
  reported.  A metric that failed on one seed must not silently poison the
  mean of the others.

The t table covers ``df = 1..30`` at 95%; beyond that the normal critical value
1.96 is used (the difference is below 0.2% at df=30 and shrinks from there).
Only 95% is implemented: requesting another level raises instead of quietly
returning a differently-calibrated interval.
"""

from __future__ import annotations

import math
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

__all__ = ["summarize", "aggregate_runs", "paired_delta", "format_summary"]

#: Two-sided 95% Student-t critical values, df = 1..30.
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

_NAN = float("nan")


def _t_critical(df: int, confidence: float = 0.95) -> float:
    """Two-sided critical value for the requested confidence level."""
    if confidence != 0.95:
        raise ValueError(
            f"only the 95% interval is implemented (got confidence={confidence!r}); "
            "add a table rather than silently using the wrong calibration"
        )
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    return _T95.get(df, 1.96)


def _finite(values: Iterable[Any]) -> list[float]:
    """Float values that are not None and not NaN/inf, in input order."""
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        x = float(value)
        if math.isfinite(x):
            out.append(x)
    return out


def summarize(
    values: Sequence[float] | Iterable[float], *, confidence: float = 0.95
) -> dict[str, float]:
    """Mean, sample std, SEM, and the two-sided t interval over seeds.

    Args:
        values: per-seed values.  ``None`` and non-finite entries are dropped.
        confidence: only 0.95 is implemented; any other value raises.

    Returns:
        ``n``, ``mean``, ``std`` (sample, ddof=1), ``sem``, ``ci_low``,
        ``ci_high``, ``min``, ``max``.  With ``n < 2`` the spread fields are
        NaN.  With ``n == 0`` every numeric field is NaN.
    """
    xs = _finite(values)
    n = len(xs)
    if n == 0:
        return {
            "n": 0,
            "mean": _NAN,
            "std": _NAN,
            "sem": _NAN,
            "ci_low": _NAN,
            "ci_high": _NAN,
            "min": _NAN,
            "max": _NAN,
        }

    mean = math.fsum(xs) / n
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": _NAN,
            "sem": _NAN,
            "ci_low": _NAN,
            "ci_high": _NAN,
            "min": xs[0],
            "max": xs[0],
        }

    variance = math.fsum((x - mean) ** 2 for x in xs) / (n - 1)
    std = math.sqrt(variance)
    sem = std / math.sqrt(n)
    t = _t_critical(n - 1, confidence)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": mean - t * sem,
        "ci_high": mean + t * sem,
        "min": min(xs),
        "max": max(xs),
    }


def aggregate_runs(
    runs: Sequence[Mapping[str, float]],
    *,
    keys: Sequence[str] | None = None,
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Summarize each metric across runs, keeping a per-key sample size.

    Metrics are not required to appear in every run: an ablation arm may log a
    key that another arm does not, and a run that failed before logging it must
    lower ``n`` for that key rather than abort the table.  ``keys=None`` uses
    the sorted union of all keys.

    Args:
        runs: one mapping per seed, metric name -> value.
        keys: metrics to report; default is the sorted union.
        confidence: passed to :func:`summarize`.

    Returns:
        metric -> summary dict (see :func:`summarize`).
    """
    if keys is None:
        keys = sorted({key for run in runs for key in run})
    return {key: summarize([run.get(key) for run in runs], confidence=confidence) for key in keys}


def paired_delta(
    treatment: Sequence[float],
    baseline: Sequence[float],
    *,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Per-seed paired difference ``treatment - baseline`` and its interval.

    Pairing is by position and requires equal lengths: seed *i* of the
    treatment must be the same seed as seed *i* of the baseline.  Pairing by
    sorted value would fabricate a comparison that was never run, so unequal
    lengths raise instead of being truncated.

    Returns:
        The :func:`summarize` fields over the deltas, plus ``wins``,
        ``losses`` and ``ties`` (counts of delta > 0, < 0, == 0).
    """
    if len(treatment) != len(baseline):
        raise ValueError(
            "paired_delta needs one baseline value per treatment seed; "
            f"got {len(treatment)} treatment and {len(baseline)} baseline values"
        )
    deltas: list[float] = []
    wins = losses = ties = 0
    for a, b in zip(treatment, baseline):
        if a is None or b is None:
            continue
        x, y = float(a), float(b)
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        delta = x - y
        deltas.append(delta)
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            ties += 1

    out = summarize(deltas, confidence=confidence)
    out.update({"wins": float(wins), "losses": float(losses), "ties": float(ties)})
    return out


def format_summary(summary: Mapping[str, float], *, digits: int = 3) -> str:
    """One paper-table cell: ``mean ± std (n=.., 95% CI .. to ..)``."""
    n = int(summary["n"])
    if n == 0:
        return "n/a (n=0)"
    mean = summary["mean"]
    if n == 1:
        return f"{mean:.{digits}f} (n=1, no interval)"
    return (
        f"{mean:.{digits}f} ± {summary['std']:.{digits}f} "
        f"(n={n}, 95% CI {summary['ci_low']:.{digits}f} to {summary['ci_high']:.{digits}f})"
    )
