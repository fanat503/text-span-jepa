# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Publication figures for the paper, drawn from what `results_apparatus` emits.

The one rule
------------
**This module never draws a curve it was not given.**

`scripts/results_apparatus.py` already decided what is a measurement: a `Cell`
is either measured, or it renders as `NOT-RUN` / `NOT-APPLICABLE`, and there is
no third state. This module does not relax that. A plottable value has to walk
through the same door a table cell does -- `Cell.measured_flag` -- before it can
reach an artist, and every value that does arrives beside the provenance string
that explains where it came from.

Why a figure needs the discipline more than a table does
--------------------------------------------------------
A table cell that renders a placeholder is read as a mistake and investigated.
A figure with empty axes and a flat line through them is read as a result, and
two reviewers will agree on what it shows. The fabrication is not merely
unnoticed -- it is *believed*, which is the worst possible outcome for an
artifact whose entire job is to be believed. So:

* with no measured data at all, every entry point **refuses**: it returns a
  `Figure.refused(...)`, writes no file, and never constructs a `matplotlib`
  `Figure` at all. There is nothing on disk to mistake for a result;
* with some measured data, only the measured part is drawn, and every arm that
  was left out is named in the figure's own footnote and in `figures.json`,
  using the apparatus's vocabulary;
* the refused case is not a soft warning that a caller may ignore. `Figure`
  cannot represent "a path was written but nothing was drawn" -- that
  combination raises `FabricationError` in `__init__` -- so the invariant is
  structural rather than a matter of discipline at each `savefig`.

Two refusers, kept apart
------------------------
`Figure.reason_kind` distinguishes refusing to draw because there is **no
measurement** (`NO-MEASUREMENT`) from refusing because a figure would be
unreadable (`UNREADABLE`: 40 configs have measurements and a legend with 40
entries is not a publication figure). Both refuse; only the first is about
fabrication, and the tests that guard against invention target that one so they
cannot be satisfied by the other.

What this module will not do
----------------------------
* measure anything. Parameter counts come from the cache
  `results_apparatus table --recompute-params` writes; a config with no cached
  count has no x-coordinate, and is reported as missing rather than estimated;
* interpolate, smooth, extrapolate or pad. A `Curve` is a list of logged rows
  in file order;
* draw a partial series as if it were complete. A one-row run is a dot, not a
  curve, so it is refused as a curve and counted separately;
* invent an axis range. Limits come from the measured values, or from the
  matplotlib default when there are none -- and when there are none, no figure
  is written at all.

The four figures
----------------
* `compute_vs_performance` -- measured trainable parameters against a measured
  metric, one marker per config, never joined across arms (`loss` is a
  different objective in each arm, so a line through them would assert a
  ranking the apparatus explicitly refuses to make);
* `ablation_matrix` -- ablation arms by result column. A cell with no
  measurement is a hatched slot with the marker written in it, not a blank and
  not a zero;
* `training_stability` -- the logged metric curves that exist, for every config
  that has a `train_log.csv`;
* `baseline_comparison` -- one slot per non-ablation arm, bars only for the
  measured ones.

Usage
-----
    python scripts/plotting_engine.py figures     # every figure the data supports
    python scripts/plotting_engine.py all         # figures + the manifest
    python scripts/plotting_engine.py verify       # fabrication self-check
    python scripts/plotting_engine.py howto        # what would fill each refusal

CPU-only, no network, no training. matplotlib is imported lazily so that
`import plotting_engine` costs nothing for the tests that only exercise the
guards.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import re
import statistics
import sys
import tempfile
import yaml
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import results_apparatus as RA
from results_apparatus import (
    NOT_APPLICABLE,
    NOT_RUN,
    Cell,
    Config,
    FabricationError,
    RunState,
    assert_marker_honest,
)

# ═══════════════════════════════════════════════════════════════════
#  The measured-value door
# ═══════════════════════════════════════════════════════════════════


def measured_number(cell: Cell) -> float:
    """The float inside a cell that declares itself measured, or `FabricationError`.

    The single place a number enters this module. It refuses an unmeasured cell
    outright rather than returning `None`, because a caller that reaches here
    with an unmeasured cell has a bug: the `measured_flag` check belongs at the
    call site where the decision to plot is made, and returning `None` here
    would push that decision back onto every caller as a `None` check somebody
    can forget. A measured cell whose text does not parse is also an error --
    the apparatus formats numbers with `:.6g`, and a measured cell that is not a
    number is a contract change, not a formatting quirk.
    """
    if not cell.measured_flag:
        raise FabricationError(
            f"asked for the number inside an unmeasured cell ({cell.render()!r}); why: {cell.why}"
        )
    try:
        return float(cell.render().replace(",", ""))
    except ValueError as exc:
        raise FabricationError(
            f"cell claims to be measured but its text {cell.render()!r} is not a "
            f"number; why: {cell.why}"
        ) from exc


def read_param_counts(path: Path | None = None) -> dict[str, tuple[int, int]]:
    """The trainable/total counts the apparatus measured, keyed by config path.

    Read straight out of `docs/results/param_counts.json`, which
    `results_apparatus table --recompute-params` writes. **This never
    recomputes.** A parameter count this engine did not receive is not data, and
    measuring 62 models to decorate a figure is not a plotting engine's job. A
    config absent from the cache simply has no x-coordinate and is reported as
    missing, the same as an arm with no run.
    """
    target = Path(path) if path else RA.ParamCache.PATH
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, tuple[int, int]] = {}
    for rel, entry in (payload.get("configs") or {}).items():
        try:
            out[rel] = (int(entry["total"]), int(entry["trainable"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ═══════════════════════════════════════════════════════════════════
#  Refusal
# ═══════════════════════════════════════════════════════════════════


class PlotRefusal(RuntimeError):
    """A figure could not be drawn honestly.

    Raised by nothing in this module, and that is deliberate: the `plot_*`
    functions return a `Figure` instead, because a batch run has to render every
    figure and record every refusal in one manifest, and the first refusal would
    otherwise abort the rest. The type is exported so a caller that wants the
    refusal to stop it has something to catch, rather than inventing its own.
    """


#: `Figure.reason_kind` values. The distinction is load-bearing: the tests that
#: prove the engine refuses to invent target `NO-MEASUREMENT`, and a refusal to
#: draw an unreadable figure must not be able to satisfy them.
NO_MEASUREMENT = "NO-MEASUREMENT"
UNREADABLE = "UNREADABLE"


class Figure:
    """The outcome of one figure: drawn, or refused with a reason.

    The invariant is in `__init__`, not in discipline at the `savefig` calls:
    a `Figure` with a written path must have at least one measured datum behind
    it, and a `Figure` with no measured datum must have no path. There is no
    third combination, so "a figure file exists but nothing was measured in it"
    is a state this class cannot represent.

    `drawn` holds `(x, y, label, provenance)` per plotted datum. It exists so
    the manifest can be audited against the artist rather than trusted: the
    numbers in `figures.json` are the numbers that were handed to matplotlib.

    `missing` holds `(label, why, marker)` per thing left out, where `marker` is
    the apparatus's own `NOT_RUN` or `NOT_APPLICABLE`. Carrying the marker
    matters in a figure manifest in a way it does not in a prose refusal: the
    two mean opposite things about what the reader should do next, and a list of
    reason strings with no marker collapses "run this" and "this can never be
    measured" into one undifferentiated pile of absence.
    """

    __slots__ = ("drawn", "kind", "missing", "note", "paths", "reason", "reason_kind", "stem")

    def __init__(
        self,
        kind: str,
        stem: str,
        paths: Sequence[Path],
        drawn: Sequence[tuple[float, float, str, str]],
        missing: Sequence[tuple[str, str, str]],
        reason: str,
        reason_kind: str,
        note: str = "",
    ) -> None:
        if paths and not drawn:
            raise FabricationError(
                f"{kind}: {len(paths)} file(s) would be written with no measured datum "
                "behind them. A figure file with nothing measured in it is the exact "
                "artifact this module exists not to produce."
            )
        if drawn and not paths:
            raise FabricationError(
                f"{kind}: {len(drawn)} datum/ata were drawn but no path was written. "
                "Either the figure was not saved or the save failed silently."
            )
        self.kind = kind
        self.stem = stem
        self.paths = list(paths)
        self.drawn = list(drawn)
        self.missing = list(missing)
        self.reason = reason
        self.reason_kind = reason_kind
        self.note = note

    @classmethod
    def made(
        cls,
        kind: str,
        stem: str,
        paths: Sequence[Path],
        drawn: Sequence[tuple[float, float, str, str]],
        missing: Sequence[tuple[str, str, str]] = (),
        note: str = "",
    ) -> Figure:
        return cls(kind, stem, paths, drawn, missing, "", "", note)

    @classmethod
    def refused(
        cls,
        kind: str,
        stem: str,
        reason: str,
        missing: Sequence[tuple[str, str, str]] = (),
        reason_kind: str = NO_MEASUREMENT,
    ) -> Figure:
        return cls(kind, stem, (), (), missing, reason, reason_kind)

    @property
    def refused_flag(self) -> bool:
        return not self.drawn

    @property
    def wrote_anything(self) -> bool:
        return bool(self.paths)

    def status_cell(self) -> Cell:
        """This figure's outcome as a `Cell`, so the manifest reuses the apparatus's
        own vocabulary and its digit-free rule for unmeasured things."""
        if self.refused_flag:
            return Cell.not_run(self.reason or f"{self.kind}: no measurement")
        return Cell.measured(
            f"PLOTTED {len(self.drawn)} datum/ata",
            f"{self.kind}: {len(self.paths)} file(s) in {[p.name for p in self.paths]}",
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "stem": self.stem,
            "status": "REFUSED" if self.refused_flag else "PLOTTED",
            "reason_kind": self.reason_kind,
            "reason": self.reason,
            "note": self.note,
            "files": [p.as_posix() for p in self.paths],
            "plotted": [
                {"label": label, "x": x, "y": y, "provenance": why}
                for x, y, label, why in self.drawn
            ],
            "missing": [
                {"label": label, "why": why, "marker": marker}
                for label, why, marker in self.missing
            ],
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "REFUSED" if self.refused_flag else f"PLOTTED {len(self.drawn)}"
        return f"Figure({self.kind!r}, {state})"


def _summarise_missing(missing: Sequence[tuple[str, str, str]], limit: int = 6) -> str:
    """A readable account of what was left out, with an explicit count.

    Eliding with `+N more` rather than truncating silently: a reader who is told
    about 6 of 66 arms can reason about the 60, and one who is shown 6 with no
    note has been misled into thinking 6 is the number.
    """
    if not missing:
        return ""
    shown = "; ".join(f"{label} [{marker}] -- {why}" for label, why, marker in missing[:limit])
    extra = len(missing) - limit
    return shown + (f" (+{extra} more; the full list is in figures.json)" if extra > 0 else "")


# ═══════════════════════════════════════════════════════════════════
#  Theme and palette
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Swatch:
    """One series' full identity: hue plus two redundant channels.

    Colour is never the only channel. Marker shape and dash pattern come from the
    same tuple, so a figure survives being printed in greyscale, read by a
    dichromat, or projected badly -- and so two entries cannot be told apart by
    hue alone if the palette is edited badly.
    """

    color: str
    marker: str
    linestyle: str


#: Okabe & Ito's colour-blind-friendly qualitative palette, which is the standard
#: for exactly this problem: every pair is separable under the common forms of
#: colour-vision deficiency, and the hues are chosen to stay apart in greyscale
#: too. `black` leads because the most important series should be the one that
#: prints. `yellow` is last because #F0E442 is the weakest entry against white
#: paper; when it is reached it is a marker with an edge, never a hairline.
PALETTE: tuple[Swatch, ...] = (
    Swatch("#000000", "o", "-"),
    Swatch("#E69F00", "s", "--"),
    Swatch("#0072B2", "^", "-."),
    Swatch("#009E73", "D", ":"),
    Swatch("#CC79A7", "v", (0, (5, 1))),
    Swatch("#56B4E9", "P", (0, (3, 1, 1, 1))),
    Swatch("#D55E00", "X", (0, (7, 2, 1, 2))),
    Swatch("#F0E442", "*", (0, (1, 1))),
)

#: Distinct markers and dash patterns, so a greyscale print still separates them.
assert len({s.marker for s in PALETTE}) == len(PALETTE), "markers must be unique"
assert len({s.linestyle for s in PALETTE}) == len(PALETTE), "linestyles must be unique"


def swatches(n: int) -> list[Swatch]:
    """The first `n` swatches, cycling if asked for more than exist."""
    if n <= 0:
        return []
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


#: Cell fills for the two matrix-like figures. Chosen so that the *presence* of a
#: measurement is legible and its *size* is not: a flat tint with black text in
#: it, so the only thing a reader can take a number from is the number itself.
ACCENT_FILL = "#DCE9F4"
ACCENT_EDGE = "#0072B2"
ABSENT_FILL = "#F4F4F4"
ABSENT_EDGE = "#AFAFAF"


@dataclass(frozen=True)
class PlotTheme:
    """Publication defaults.

    `dpi` is for the raster fallback only -- PDF and SVG are resolution-free, and
    `pdf.fonttype: 42` / `svg.fonttype: 'none'` keep the text as text so the
    figure can be edited in a vector editor instead of arriving as outlines.
    300 dpi is the floor a journal asks for; 400 leaves room for the figure being
    scaled up in a two-column layout.
    """

    name: str = "publication"
    dpi: int = 400
    figsize: tuple[float, float] = (6.5, 4.0)
    font_family: str = "serif"
    font_size: float = 9.0
    label_size: float = 10.0
    tick_size: float = 8.0
    legend_size: float = 8.0
    line_width: float = 1.4
    marker_size: float = 5.0
    vector_formats: tuple[str, ...] = ("pdf", "svg")
    raster_formats: tuple[str, ...] = ("png",)
    grid_alpha: float = 0.25

    def formats(self, vector_only: bool = False) -> tuple[str, ...]:
        return self.vector_formats if vector_only else self.vector_formats + self.raster_formats

    def rcparams(self) -> dict[str, Any]:
        return {
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
            "savefig.bbox": "tight",
            "figure.constrained_layout.use": True,
            "font.family": self.font_family,
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "font.size": self.font_size,
            "axes.labelsize": self.label_size,
            "axes.titlesize": self.label_size,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "legend.fontsize": self.legend_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": self.grid_alpha,
            "grid.linewidth": 0.5,
            "lines.linewidth": self.line_width,
            "lines.markersize": self.marker_size,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }


PUBLICATION = PlotTheme()

#: The floor on how far apart two palette entries must sit in CIELAB, by the
#: CIE76 distance. The convention in the plotting literature is that below ~10
#: two adjacent series in a chart start to merge, and the measured minimum
#: pairwise separation of this palette is ~26, so 20 is a threshold with margin
#: rather than a number fitted to the answer. `test_palette_pairs_are_separable`
#: computes the distances; this constant is the bar they are held to.
MIN_PALETTE_SEPARATION = 20.0


def _hex_to_lab(colour: str) -> tuple[float, float, float]:
    """sRGB hex -> CIELAB (D65). Small enough to keep the property test honest."""
    h = colour.lstrip("#")
    srgb = [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]

    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(c) for c in srgb)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def palette_separation(a: str, b: str) -> float:
    """CIE76 distance between two hex colours, in Lab units."""
    return math.dist(_hex_to_lab(a), _hex_to_lab(b))


def _pyplot():
    """matplotlib.pyplot on the Agg backend, imported on first use.

    Lazy so that importing this module -- which the guard tests do, and which
    should not cost a font-cache build -- stays cheap, and so no test can
    accidentally open a window.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def apply_theme(theme: PlotTheme = PUBLICATION) -> None:
    """Install the rcParams. Explicit rather than a style sheet on disk, so the
    figure a reviewer opens renders the same as the one a CI run produced."""
    import matplotlib

    # `pdf.fonttype: 42` subsets the font on every save, and fontTools narrates
    # each subset at INFO. Fifteen lines of glyph bookkeeping per figure would
    # bury the engine's own refusals in the log, which is the one thing this
    # module needs a reader to be able to find.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    matplotlib.rcParams.update(theme.rcparams())


def save_figure(fig, out_dir: Path, stem: str, theme: PlotTheme, vector_only: bool) -> list[Path]:
    """Write every requested format and return the paths.

    Vector first: PDF is what a publisher takes, and the raster fallback exists
    only so a README or a slide deck has something to show.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in theme.formats(vector_only=vector_only):
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt, dpi=theme.dpi, metadata={"Creator": "plotting_engine"})
        paths.append(path)
    return paths


# ═══════════════════════════════════════════════════════════════════
#  Measured data
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Datum:
    """One measured (x, y), with the label it is plotted under and where it came from."""

    x: float
    y: float
    label: str
    provenance: str


@dataclass(frozen=True)
class Curve:
    """One measured series: logged rows in file order, nothing else.

    Constructed from a `RunState`, which holds rows only when a `train_log.csv`
    was found and parsed. There is no constructor that takes a list of floats
    from a caller, so the only way to get a plottable curve is to have read one
    off disk.

    `complete` is False when the run holds fewer logged rows than the config
    declares epochs. Such a curve is drawn, and drawn as a partial run, with a
    dashed tail -- but never padded forward to the declared length, because a
    padded line is a straight segment nobody measured.
    """

    label: str
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    column: str
    unit: str
    provenance: str
    complete: bool = True

    @property
    def n_points(self) -> int:
        return len(self.xs)

    @property
    def is_curveable(self) -> bool:
        """Two points is the least that is a curve. One point is a dot, and a
        line through a single dot is a segment with no second end."""
        return self.n_points >= 2

    @classmethod
    def from_run(
        cls,
        cfg: Config,
        state: RunState,
        column: str,
        unit: str = "",
        log_freq: int | None = None,
    ) -> Curve | None:
        """The curve this run logged for `column`, or None if it logged none.

        `None` is returned rather than an empty `Curve` so that "no run" and "a
        run with no rows" cannot be confused with a series that has points.
        """
        cell = RA.metric_cell(cfg, state, column)
        if not cell.measured_flag or not state.rows:
            return None
        index = RA.CSV_COLUMNS.index(column)
        ys = tuple(float(row[index]) for row in state.rows)
        step = log_freq if log_freq is not None else cfg.log_freq
        return cls(
            label=cfg.label,
            xs=tuple(i * step for i in range(len(ys))),
            ys=ys,
            column=column,
            unit=unit,
            provenance=f"{state.run_dir}/{RA.TRAIN_LOG}, rows 1-{len(ys)}",
            complete=state.epochs_done >= cfg.epochs if state.epochs_done else False,
        )


# ═══════════════════════════════════════════════════════════════════
#  Figure 1 — compute vs performance
# ═══════════════════════════════════════════════════════════════════

#: `results_apparatus.build_main_table`'s own warning, quoted rather than
#: paraphrased: the three arms optimise three different objectives, so `loss`
#: does not rank them and a line through the arms would assert that it does.
LOSS_RANKS_ARMS = (
    "The arms are not joined by a line: `loss` is a JEPA span loss, an MLM "
    "cross-entropy and a data2vec regression, so it does not rank them against "
    "one another (results_apparatus.build_main_table)."
)

#: Why a config with a measured metric still has no x-coordinate. The remedy is a
#: command in another script, not a run, which is why it is its own sentence.
NO_PARAM_COUNT = (
    f"no measured trainable-parameter count in {RA.ParamCache.PATH.name}; run "
    "`python scripts/results_apparatus.py table --recompute-params` to measure one"
)


def plot_compute_vs_performance(
    cfgs: Sequence[Config],
    states: Mapping[str, RunState],
    params: Mapping[str, tuple[int, int]],
    column: str = "loss",
    *,
    out_dir: Path,
    stem: str = "compute_vs_performance",
    theme: PlotTheme = PUBLICATION,
    vector_only: bool = False,
    title: str = "",
) -> Figure:
    """Measured trainable parameters against a measured metric, one marker per config.

    A config needs *both* coordinates to appear. Its x-coordinate comes from the
    measured parameter count, its y-coordinate from the apparatus's gated
    `metric_cell`, so a JEPA-only column on an MLM arm cannot contribute a point
    just because its CSV happens to hold a `0.000000`.
    """
    kind = "compute-vs-performance"
    points: list[Datum] = []
    missing: list[tuple[str, str]] = []

    for cfg in cfgs:
        state = states[cfg.rel]
        cell = RA.metric_cell(cfg, state, column)
        counted = params.get(cfg.rel)
        if cell.measured_flag and counted is not None:
            points.append(
                Datum(
                    x=float(counted[1]),
                    y=measured_number(cell),
                    # The group is the ARM, not a prefix of the label: `all_core`
                    # and `no_cgn` share no prefix but are the same objective, and
                    # grouping them by string would put four JEPA configs in four
                    # legend entries that mean nothing.
                    label=cfg.arm,
                    provenance=cell.why,
                )
            )
        else:
            missing.append(
                (
                    cfg.label,
                    (cell.why if not cell.measured_flag else NO_PARAM_COUNT),
                    cell.render() if not cell.measured_flag else NOT_RUN,
                )
            )

    if not points:
        return Figure.refused(
            kind,
            stem,
            reason=(
                f"{kind}: none of the {len(cfgs)} configs has both a measured "
                f"trainable-parameter count and a measured {column!r}, so there is no "
                f"point to draw. No file was written. An axes with markers on it and "
                f"nothing in them is read as a result, which is why the figure is "
                f"absent rather than empty. Nothing measured: {_summarise_missing(missing)}"
            ),
            missing=missing,
        )

    apply_theme(theme)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=theme.figsize)

    # One marker identity per arm, so the legend says which objective a point is
    # a measurement of rather than which config file it came from.
    groups: dict[str, list[Datum]] = {}
    for point in points:
        groups.setdefault(point.label, []).append(point)
    ordered = sorted(groups.items())
    for index, (arm, group) in enumerate(ordered):
        swatch = swatches(len(ordered))[index]
        ax.plot(
            [p.x for p in group],
            [p.y for p in group],
            linestyle="none",
            marker=swatch.marker,
            markersize=theme.marker_size + 1.0,
            markerfacecolor=swatch.color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=f"{arm} ({len(group)})",
        )

    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (measured)")
    ax.set_ylabel(_column_label(column))
    ax.set_title(title or f"{_column_label(column)} against compute")
    if len(ordered) > 1:
        ax.legend(loc="best", title="arm (configs)")
    _footnote(fig, theme, _compute_footnote(len(points), len(missing), column))
    fig.canvas.draw()
    paths = save_figure(fig, out_dir, stem, theme, vector_only)
    plt.close(fig)

    return Figure.made(
        kind,
        stem,
        paths,
        [(p.x, p.y, p.label, p.provenance) for p in points],
        missing,
        note=LOSS_RANKS_ARMS if column == "loss" else "",
    )


def _compute_footnote(plotted: int, missing: int, column: str) -> str:
    tail = (
        f"  {NOT_RUN}: {missing} config(s) had no measured {column!r} and are not on "
        f"this figure; figures.json lists each one with the reason."
        if missing
        else ""
    )
    return f"{plotted} measured point(s).{tail}"


def _footnote(fig, theme: PlotTheme, text: str) -> None:
    """The figure's own footnote.

    Written onto the figure rather than left in the log, because the figure is
    what gets pasted into a draft: a reader looking at the picture alone must be
    able to see that some arms are missing and why.
    """
    if not text:
        return
    fig.supxlabel(text, fontsize=theme.tick_size, color="#333333", wrap=True)


def _column_label(column: str) -> str:
    return dict(RA.CURVE_SERIES).get(column, column)


# ═══════════════════════════════════════════════════════════════════
#  Figure 2 — the ablation matrix
# ═══════════════════════════════════════════════════════════════════

#: The result columns the matrix is built from. `val_loss` is in the list
#: deliberately: `results_apparatus.val_loss_cell` documents that no CSV carries
#: it, so its column is permanently empty until the owner supplies the extraction
#: command. Leaving it out would make the matrix look complete when it is not.
MATRIX_COLUMNS: tuple[str, ...] = (
    "loss",
    "decoder_accuracy",
    "effective_rank",
    "collapsed_dim_ratio",
    "val_loss",
)


def plot_ablation_matrix(
    cfgs: Sequence[Config],
    states: Mapping[str, RunState],
    *,
    out_dir: Path,
    stem: str = "ablation_matrix",
    theme: PlotTheme = PUBLICATION,
    vector_only: bool = False,
    columns: Sequence[str] = MATRIX_COLUMNS,
) -> Figure:
    """Ablation arms by result column, with the unmeasured cells hatched.

    A measured cell gets the measurement printed in it and one flat accent fill.
    **There is no colour scale, and that is deliberate.** The five columns are
    five different quantities -- a loss, an accuracy, a rank, a ratio and a
    validation loss -- so one colour map across them either normalises away the
    value (leaving a number nobody can compare to anything) or, worse, implies a
    comparability that does not exist between an accuracy and a rank. A
    perceptually uniform map does not fix that; it makes the false comparison
    prettier. So the colour carries no numeric claim, the number carries the
    measurement, and an unmeasured cell is hatched with the marker written across
    it -- never a colour, because a colour on a map is a value, and never a blank,
    because a blank in a matrix reads as zero.
    """
    kind = "ablation-matrix"
    arms = [c for c in cfgs if c.ablation]
    cells: dict[tuple[int, int], Cell] = {}
    drawn: list[tuple[float, float, str, str]] = []
    missing: list[tuple[str, str, str]] = []
    measured_count = 0

    for r, cfg in enumerate(arms):
        state = states[cfg.rel]
        for c, column in enumerate(columns):
            cell = (
                RA.val_loss_cell(state)
                if column == "val_loss"
                else RA.metric_cell(cfg, state, column)
            )
            cells[(r, c)] = cell
            if cell.measured_flag:
                measured_count += 1
                drawn.append((float(c), float(r), f"{cfg.ablation}/{column}", cell.why))
            else:
                missing.append((f"{cfg.ablation}/{column}", cell.why, cell.render()))

    if not arms:
        return Figure.refused(
            kind,
            stem,
            reason=f"{kind}: no ablation configs resolved, so there are no rows to lay out.",
        )
    if measured_count == 0:
        return Figure.refused(
            kind,
            stem,
            reason=(
                f"{kind}: none of the {len(arms)} ablation arms has a measured value in "
                f"any of the {len(columns)} result columns, so every cell would be "
                f"unmeasured. No file was written. A grid of empty boxes is not a "
                f"smaller result than a grid of curves -- it is a grid that says the "
                f"ablation was run. Nothing measured: {_summarise_missing(missing)}"
            ),
            missing=missing,
        )

    apply_theme(theme)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(theme.figsize[0], max(2.5, 0.34 * len(arms) + 1.6)))
    from matplotlib.patches import Rectangle

    tight = len(arms) <= 12
    for (r, c), cell in cells.items():
        measured = cell.measured_flag
        ax.add_patch(
            Rectangle(
                (c, r),
                1,
                1,
                facecolor=ACCENT_FILL if measured else ABSENT_FILL,
                edgecolor=ACCENT_EDGE if measured else ABSENT_EDGE,
                hatch=None if measured else "////",
                linewidth=0.5,
                zorder=1,
            )
        )
        ax.text(
            c + 0.5,
            r + 0.5,
            cell.render(),
            ha="center",
            va="center",
            fontsize=theme.tick_size - 1.0,
            color="black" if measured else "#333333",
            rotation=0 if tight else 90,
        )

    ax.set_xlim(0, len(columns))
    ax.set_ylim(len(arms), 0)  # first ablation at the top, as the config tree reads
    ax.set_xticks([c + 0.5 for c in range(len(columns))])
    ax.set_xticklabels([_column_label(c) for c in columns], rotation=20, ha="right")
    ax.set_yticks([r + 0.5 for r in range(len(arms))])
    ax.set_yticklabels([c.ablation or c.rel for c in arms], fontsize=theme.tick_size - 1)
    ax.grid(False)
    _footnote(
        fig,
        theme,
        f"Filled = {measured_count} measured cell(s); hatched = {len(missing)} with no "
        f"measurement, {NOT_RUN} = a run would fill it, {NOT_APPLICABLE} = no run ever "
        f"could. No colour scale: the columns are different quantities.",
    )
    fig.canvas.draw()
    paths = save_figure(fig, out_dir, stem, theme, vector_only)
    plt.close(fig)

    return Figure.made(kind, stem, paths, drawn, missing)


# ═══════════════════════════════════════════════════════════════════
#  Figure 3 — training stability
# ═══════════════════════════════════════════════════════════════════

#: Above this many series a legend stops being readable, and the honest response
#: is to refuse rather than to silently plot a subset: a figure showing 8 of 40
#: arms without saying so is the same fabrication as plotting a curve nobody ran.
MAX_SERIES = 8


def plot_training_stability(
    cfgs: Sequence[Config],
    states: Mapping[str, RunState],
    *,
    out_dir: Path,
    stem: str = "training_stability",
    theme: PlotTheme = PUBLICATION,
    vector_only: bool = False,
    columns: Sequence[tuple[str, str]] = tuple(RA.CURVE_SERIES),
    only: Sequence[str] | None = None,
    max_series: int = MAX_SERIES,
) -> Figure:
    """Every logged metric curve that exists, one subplot per column.

    A config with one logged row contributes a refusal note, not a line: a
    segment needs two ends. A config whose run is shorter than the config's
    declared epochs is drawn with a dashed tail and labelled partial, because
    the short run is real and the missing epochs are not.
    """
    kind = "training-stability"
    wanted = set(only) if only else None
    curves: list[Curve] = []
    #: A one-row run is NOT a `NOT_RUN` cell -- a run happened -- but it is also not
    #: a curve. It gets its own bucket so the manifest can say which of the two it
    #: is instead of blurring "not measured" and "measured once".
    short: list[tuple[str, str, str]] = []
    missing: list[tuple[str, str, str]] = []

    for cfg in cfgs:
        if wanted is not None and cfg.rel not in wanted:
            continue
        state = states[cfg.rel]
        if not state.has_log:
            missing.append((cfg.label, state.reason or "no completed run", NOT_RUN))
            continue
        for column, unit in columns:
            curve = Curve.from_run(cfg, state, column, unit)
            if curve is None:
                gate = RA.metric_cell(cfg, state, column)
                missing.append((f"{cfg.label}/{column}", gate.why or "not measured", gate.render()))
            elif not curve.is_curveable:
                short.append(
                    (
                        f"{cfg.label}/{column}",
                        f"{curve.n_points} logged row(s): a single point is a dot, not a curve",
                        NOT_RUN,
                    )
                )
            else:
                curves.append(curve)

    if not curves:
        if not short:
            reason = (
                f"{kind}: no config has a logged curve for any of "
                f"{[c for c, _ in columns]}, so nothing could be plotted. No file was "
                f"written. With no run, the axes would be empty and the legend would "
                f"list arms nobody trained: readable as a result, which is the failure "
                f"this module exists to prevent. Nothing measured: "
                f"{_summarise_missing(missing)}"
            )
        else:
            reason = (
                f"{kind}: {len(short)} series have a logged value but only one point each, "
                f"so none is a curve. No file was written. {_summarise_missing(short)}"
            )
        return Figure.refused(kind, stem, reason=reason, missing=missing + short)

    labels = sorted({c.label for c in curves})
    if len(labels) > max_series:
        return Figure.refused(
            kind,
            stem,
            reason=(
                f"{kind}: {len(labels)} configs have logged curves, and a legend of "
                f"{len(labels)} entries is not a publication figure. No file was written. "
                f"Pass `only=[...]` to name the configs to plot rather than letting this "
                f"module choose which arms the reader never sees."
            ),
            missing=missing + short,
            reason_kind=UNREADABLE,
        )

    apply_theme(theme)
    plt = _pyplot()
    used_columns = [c for c, _ in columns if any(cu.column == c for cu in curves)]
    units = dict(columns)
    height = max(2.0, 2.1 * len(used_columns))
    fig, axes = plt.subplots(
        len(used_columns), 1, figsize=(theme.figsize[0], height), sharex=True, squeeze=False
    )

    for ax, column in zip(axes[:, 0], used_columns):
        here = [cu for cu in curves if cu.column == column]
        for index, curve in enumerate(here):
            swatch = swatches(len(labels))[labels.index(curve.label)]
            ax.plot(
                curve.xs,
                curve.ys,
                color=swatch.color,
                marker=swatch.marker,
                markevery=max(1, len(curve.xs) // 6),
                markerfacecolor="white",
                markeredgewidth=0.8,
                linestyle=swatch.linestyle if curve.complete else (0, (2, 2)),
                label=curve.label + ("" if curve.complete else " (partial)"),
            )
        ax.set_ylabel(_column_label(column) or units.get(column, ""))
        ax.legend(loc="best", ncol=min(3, len(here)))

    axes[-1, 0].set_xlabel("optimizer iteration")
    _footnote(
        fig,
        theme,
        f"{len(curves)} measured series from {len(labels)} config(s). "
        + ("Dashed = partial run. " if any(not cu.complete for cu in curves) else "")
        + f"Not plotted: {len(missing) + len(short)} config/series with no logged value.",
    )
    fig.canvas.draw()
    paths = save_figure(fig, out_dir, stem, theme, vector_only)
    plt.close(fig)

    drawn = [
        (float(i), float(cu.n_points), f"{cu.label}/{cu.column}", cu.provenance)
        for i, cu in enumerate(curves)
    ]
    return Figure.made(kind, stem, paths, drawn, missing + short)


# ═══════════════════════════════════════════════════════════════════
#  Figure 4 — baseline comparison
# ═══════════════════════════════════════════════════════════════════


def plot_baseline_comparison(
    cfgs: Sequence[Config],
    states: Mapping[str, RunState],
    *,
    column: str = "loss",
    out_dir: Path,
    stem: str = "baseline_comparison",
    theme: PlotTheme = PUBLICATION,
    vector_only: bool = False,
) -> Figure:
    """One slot per non-ablation arm; a bar only where a measurement exists.

    An unmeasured slot is drawn as a hatched band across the plot with the
    marker written above it and no rectangle. A zero-height rectangle would be
    worse than nothing: a bar chart's zero and its absent bar are the same
    picture, and the caption would have to be read to tell them apart.
    """
    kind = "baseline-comparison"
    arms = [c for c in cfgs if not c.ablation]
    bars: list[Datum] = []
    missing: list[tuple[str, str, str]] = []
    #: The slot index each missing arm occupies, kept alongside `missing` because
    #: the two lists have different lengths: enumerating `missing` and using that
    #: index would hatch slot 1 for whichever arm happens to be missing first.
    slots: list[int] = []
    for index, cfg in enumerate(arms):
        state = states[cfg.rel]
        cell = RA.metric_cell(cfg, state, column)
        if cell.measured_flag:
            bars.append(
                Datum(
                    x=float(index),
                    y=measured_number(cell),
                    # The config's own label, not its arm: the scaling ladder has a
                    # dozen configs that are all `text_span_jepa`, and a manifest
                    # that calls every bar the same thing cannot say which one a
                    # reviewer is looking at.
                    label=cfg.label or cfg.rel,
                    provenance=cell.why,
                )
            )
        else:
            missing.append((cfg.arm, cell.why, cell.render()))
            slots.append(index)

    if not bars:
        return Figure.refused(
            kind,
            stem,
            reason=(
                f"{kind}: none of the {len(arms)} non-ablation arms has a measured "
                f"{column!r}, so no bar can be drawn. No file was written. A row of "
                f"zero-height bars reads as a comparison in which every arm scored zero, "
                f"which is a claim, not an absence. Nothing measured: "
                f"{_summarise_missing(missing)}"
            ),
            missing=missing,
        )

    apply_theme(theme)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(theme.figsize[0], 0.5 * len(arms) + 1.5), theme.figsize[1]))
    from matplotlib.patches import Rectangle

    # Limits come from the measured values, not from a hardcoded floor. An
    # earlier version forced `set_ylim(0, max*1.25)`, which silently clipped any
    # negative measurement out of the picture while leaving a set of axes that
    # looked perfectly deliberate. A figure that hides its own data is worse than
    # one that is refused.
    values = [bar.y for bar in bars]
    lo = min([0.0, *values])
    hi = max([0.0, *values])
    pad = (hi - lo) * 0.18 or 1.0
    bottom, top = lo - pad * 0.2, hi + pad
    order = sorted(int(bar.x) for bar in bars)

    for bar in bars:
        swatch = swatches(max(1, len(order)))[order.index(int(bar.x))]
        ax.add_patch(
            Rectangle(
                (bar.x - 0.3, 0.0),
                0.6,
                bar.y,
                facecolor=swatch.color,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
        )
        # Inside the bar when it is tall enough to hold the text, outside when it
        # is not. A value label floating over the axis line is readable but reads
        # as an axis annotation rather than as this bar's number.
        inside = abs(bar.y) > (top - bottom) * 0.12
        ax.text(
            bar.x,
            bar.y - (pad * 0.1 if bar.y >= 0 else -pad * 0.1) if inside else bar.y + pad * 0.15,
            f"{bar.y:.4g}",
            ha="center",
            va="top" if (inside and bar.y >= 0) else ("bottom" if inside else "top"),
            fontsize=theme.tick_size,
            color="white" if inside else "black",
        )

    for slot in slots:
        # Full-height hatch, no rectangle of any measured height: the slot is
        # visibly present and visibly valueless.
        ax.add_patch(
            Rectangle(
                (slot - 0.3, bottom),
                0.6,
                top - bottom,
                facecolor=ABSENT_FILL,
                edgecolor=ABSENT_EDGE,
                hatch="////",
                linewidth=0.4,
                zorder=2,
            )
        )
        ax.text(
            slot,
            (bottom + top) / 2.0,
            NOT_RUN if arms[slot].is_jepa else NOT_APPLICABLE,
            ha="center",
            va="center",
            rotation=90,
            fontsize=theme.tick_size - 1,
            color="#333333",
        )

    ax.axhline(0.0, color="black", linewidth=0.6, zorder=4)
    # Explicit limits. A `Rectangle` handed to `add_patch` does not feed
    # matplotlib's data limits, so autoscaling leaves the x-axis at its 0..1
    # default and quietly clips the first and last bars off the plot -- which
    # looks exactly like an arm that measured nothing.
    ax.set_xlim(-0.5, len(arms) - 0.5)
    ax.set_ylim(bottom, top)
    ax.set_xticks([float(i) for i in range(len(arms))])
    ax.set_xticklabels([c.label or c.rel for c in arms], rotation=15, ha="right")
    ax.set_ylabel(_column_label(column))
    ax.set_title(f"{_column_label(column)} by arm")
    _footnote(
        fig,
        theme,
        f"{len(bars)} measured bar(s); {len(missing)} arm(s) hatched with no value. "
        + (LOSS_RANKS_ARMS if column == "loss" else ""),
    )
    fig.canvas.draw()
    paths = save_figure(fig, out_dir, stem, theme, vector_only)
    plt.close(fig)

    return Figure.made(
        kind, stem, paths, [(b.x, b.y, b.label, b.provenance) for b in bars], missing
    )


# ═══════════════════════════════════════════════════════════════════
#  Figure 5 — seed-variance band
# ═══════════════════════════════════════════════════════════════════


def _measured_seed(state: RunState) -> int | None:
    """The seed this run actually used, read from its own dumped params.

    `src/train.py` resolves the seed as `meta.seed`, else the legacy top-level
    `seed`, else 42 (train.py:1395-1399), and dumps the resolved config next to
    the CSV (`params-text-span-jepa.yaml`, train.py:1476-1478). Reading the dump
    rather than trusting a filename convention is what makes the grouping in
    `plot_seed_band` a measurement instead of a convention.
    """
    if not state.run_dir:
        return None
    dump = Path(state.run_dir) / RA.CONFIG_DUMP
    if not dump.is_file():
        return None
    try:
        dumped = yaml.safe_load(dump.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(dumped, dict):
        return None
    meta_seed = (
        (dumped.get("meta") or {}).get("seed") if isinstance(dumped.get("meta"), dict) else None
    )
    if meta_seed is not None:
        return int(meta_seed)
    top_seed = dumped.get("seed")
    return int(top_seed) if top_seed is not None else None


def _band_group_key(cfg: Config) -> str:
    """Content identity of an experiment, ignoring how it is named and seeded.

    Two runs are the same experiment when their resolved configs are equal
    after removing the three keys a seed sweep is allowed to vary: the
    top-level `seed`, `meta.seed`, and `logging.folder` (two seeds of one arm
    must not write one CSV -- tests/test_config_system.py contract 8). Grouping
    by content rather than by a name suffix means a seed twin cannot collide
    by accident, and a genuinely different arm cannot collide by construction.
    """
    stripped = copy.deepcopy(cfg.merged)
    stripped.pop("seed", None)
    meta = stripped.get("meta")
    if isinstance(meta, dict):
        meta.pop("seed", None)
    logging_block = stripped.get("logging")
    if isinstance(logging_block, dict):
        logging_block.pop("folder", None)
    return json.dumps(stripped, sort_keys=True, default=str)


#: The suffix `config/<...>_s3.yaml`-style seed twins carry. Only used for the
#: legend display name; grouping never reads names.
_BAND_SEED_SUFFIX = re.compile(r"^(?P<base>.+?)[-_](?:s|seed)\d+$")


def _band_display_label(labels: Sequence[str]) -> str:
    """The group's legend name: the shortest member label, seed suffix stripped."""
    shortest = min(labels, key=len)
    match = _BAND_SEED_SUFFIX.match(shortest)
    return match.group("base") if match else shortest


def plot_seed_band(
    cfgs: Sequence[Config],
    states: Mapping[str, RunState],
    *,
    out_dir: Path,
    stem: str = "seed_band",
    theme: PlotTheme = PUBLICATION,
    vector_only: bool = False,
    columns: Sequence[tuple[str, str]] = tuple(RA.CURVE_SERIES),
    only: Sequence[str] | None = None,
    max_series: int = MAX_SERIES,
) -> Figure:
    """Mean ± 1σ across the measured seeds of one experiment, per logged column.

    Variance across seeds is the one thing a single-run curve cannot show, and
    the one claim reviewers rightly discount without it. This figure exists so
    the day three seeds of one arm are on disk, the band is drawn under the
    same discipline as every other figure here -- from files, with provenance
    -- rather than assembled by hand in a spreadsheet.

    A group is the set of runs whose resolved configs are equal once `seed`,
    `meta.seed` and `logging.folder` are removed (`_band_group_key`). A band is
    drawn for a group only where ≥2 runs carry ≥2 DISTINCT measured seeds (each
    read from that run's own params dump, `_measured_seed`); a member with no
    recorded seed is excluded and named, because a run whose seed nobody knows
    cannot enter a seed band. Members share a log grid by construction
    (`log_freq` is part of the group identity), the band spans their common
    prefix, and a member shorter than two rows blocks the band rather than
    being silently dropped. A ±σ band over one seed would assert a spread
    nobody measured, so a single-seed group is a manifest note, never a line.
    """
    kind = "seed-band"
    wanted = set(only) if only else None

    groups: dict[str, list[tuple[Config, RunState]]] = {}
    missing: list[tuple[str, str, str]] = []
    for cfg in cfgs:
        if wanted is not None and cfg.rel not in wanted:
            continue
        state = states[cfg.rel]
        if not state.has_log:
            missing.append((cfg.label, state.reason or "no completed run", NOT_RUN))
            continue
        groups.setdefault(_band_group_key(cfg), []).append((cfg, state))

    bands: list[dict[str, Any]] = []
    for _, members in sorted(groups.items()):
        seeds = [_measured_seed(state) for _, state in members]
        kept = [
            (cfg, state, seed) for (cfg, state), seed in zip(members, seeds) if seed is not None
        ]
        excluded = [cfg.label for (cfg, _), seed in zip(members, seeds) if seed is None]
        if len({seed for _, _, seed in kept}) < 2:
            distinct = sorted({seed for _, _, seed in kept})
            label = _band_display_label([c.label for c, _ in members])
            missing.append(
                (
                    label,
                    (
                        f"{len(members)} run(s), distinct measured seeds: "
                        f"{distinct if distinct else 'none recorded in any params dump'}. A ±σ band "
                        "over this would assert a spread nobody measured; run the arm with "
                        "more distinct seeds."
                    ),
                    NOT_RUN,
                )
            )
            continue
        for seedless in excluded:
            missing.append(
                (
                    seedless,
                    (
                        "seed not recorded in this run's params dump; excluded from the "
                        "band -- a run whose seed nobody knows cannot enter a seed band"
                    ),
                    NOT_RUN,
                )
            )
        base = _band_display_label([c.label for c, _ in members])
        for column, unit in columns:
            curves = []
            for cfg, state, _ in kept:
                curve = Curve.from_run(cfg, state, column, unit)
                if curve is None:
                    gate = RA.metric_cell(cfg, state, column)
                    missing.append(
                        (f"{cfg.label}/{column}", gate.why or "not measured", gate.render())
                    )
                elif not curve.is_curveable:
                    missing.append(
                        (
                            f"{cfg.label}/{column}",
                            f"{curve.n_points} logged row(s): a single point is a dot, not a band",
                            NOT_RUN,
                        )
                    )
                else:
                    curves.append(curve)
            if len(curves) < 2:
                continue
            length = min(c.n_points for c in curves)
            if length < 2:
                missing.append(
                    (
                        f"{base}/{column}",
                        (
                            "the shortest member run logged 1 row; a band needs a common "
                            "prefix of ≥2 points, and no member is dropped to make one"
                        ),
                        NOT_RUN,
                    )
                )
                continue
            xs = curves[0].xs[:length]
            if any(c.xs[:length] != xs for c in curves[1:]):
                missing.append(
                    (
                        f"{base}/{column}",
                        (
                            "members log on different grids; refusing to interpolate -- an "
                            "interpolated point is a point nobody measured"
                        ),
                        NOT_RUN,
                    )
                )
                continue
            per_point = list(zip(*(c.ys[:length] for c in curves)))
            mean = [sum(v) / len(v) for v in per_point]
            std = [statistics.stdev(v) for v in per_point]
            seeds_used = sorted({seed for _, _, seed in kept})
            bands.append(
                {
                    "label": base,
                    "column": column,
                    "unit": unit,
                    "xs": xs,
                    "mean": mean,
                    "std": std,
                    "n": len(curves),
                    "seeds": seeds_used,
                    "complete": all(c.complete for c in curves),
                    "provenance": (
                        f"mean±1σ over seeds {seeds_used} from "
                        f"{[c.provenance for c in curves]}; rows 1-{length}"
                    ),
                }
            )

    if not bands:
        if not missing:
            reason = f"{kind}: no config has a logged curve, so nothing could be banded."
        else:
            reason = (
                f"{kind}: no group of runs carries ≥2 distinct measured seeds, so no "
                f"variance is measurable and no band could be drawn. No file was "
                f"written -- a band over one seed is a spread nobody measured. "
                f"{_summarise_missing(missing)}"
            )
        return Figure.refused(kind, stem, reason=reason, missing=missing)

    labels = sorted({b["label"] for b in bands})
    if len(labels) > max_series:
        return Figure.refused(
            kind,
            stem,
            reason=(
                f"{kind}: {len(labels)} experiment groups have bandable seed sets, and a "
                f"legend of {len(labels)} entries is not a publication figure. No file was "
                f"written. Pass `only=[...]` to name the groups to plot."
            ),
            missing=missing,
            reason_kind=UNREADABLE,
        )

    apply_theme(theme)
    plt = _pyplot()
    used_columns = [c for c, _ in columns if any(b["column"] == c for b in bands)]
    height = max(2.0, 2.1 * len(used_columns))
    fig, axes = plt.subplots(
        len(used_columns), 1, figsize=(theme.figsize[0], height), sharex=True, squeeze=False
    )

    for ax, column in zip(axes[:, 0], used_columns):
        here = [b for b in bands if b["column"] == column]
        for band in here:
            color = swatches(len(labels))[labels.index(band["label"])].color
            lower = [m - s for m, s in zip(band["mean"], band["std"])]
            upper = [m + s for m, s in zip(band["mean"], band["std"])]
            ax.fill_between(band["xs"], lower, upper, color=color, alpha=0.22, linewidth=0)
            ax.plot(
                band["xs"],
                band["mean"],
                color=color,
                linestyle="-" if band["complete"] else (0, (2, 2)),
                label=f"{band['label']} (n={band['n']} seeds, ±1σ)"
                + ("" if band["complete"] else " (partial)"),
            )
        ax.set_ylabel(_column_label(column) or dict(columns).get(column, ""))
        ax.legend(loc="best", ncol=min(3, len(here)))

    axes[-1, 0].set_xlabel("optimizer iteration")
    _footnote(
        fig,
        theme,
        f"{len(bands)} band(s) across {len(labels)} experiment group(s). Each band is the "
        f"sample σ (ddof=1) across the seeds named in its legend, over the common log "
        f"prefix of the member runs; the line is their mean. Seeds are read from each "
        f"run's own params dump. Not drawn: {len(missing)} group/series with fewer than "
        f"two distinct recorded seeds or no logged value.",
    )
    fig.canvas.draw()
    paths = save_figure(fig, out_dir, stem, theme, vector_only)
    plt.close(fig)

    drawn = [
        (
            float(band["xs"][-1]),
            float(band["mean"][-1]),
            f"{band['label']}/{band['column']} mean",
            band["provenance"],
        )
        for band in bands
    ]
    return Figure.made(kind, stem, paths, drawn, missing=missing)


# ═══════════════════════════════════════════════════════════════════
#  Manifest
# ═══════════════════════════════════════════════════════════════════


def manifest_markdown(figures: Sequence[Figure]) -> str:
    """`FIGURES.md`: what was drawn, and the loud reason for what was not.

    The manifest is the figure the paper can be written against. A refused
    figure appears in it with its reason, so "why is there no ablation matrix"
    has an answer that is a file rather than a recollection.
    """
    cells = [f.status_cell() for f in figures]
    assert_marker_honest(cells)
    lines = ["# Figures", ""]
    lines += [
        "Every row is either a figure on disk or a refusal with its reason. There is no",
        "third state: a figure file exists only where at least one measured value was",
        "handed to matplotlib, and `scripts/plotting_engine.py` cannot construct a",
        f"`Figure` that has a path and no data. `{NOT_RUN}` is the apparatus's own word",
        "for 'a run would fill this and none has'; it is never a number and never a dash.",
        "",
        "| figure | status | files | measured data points |",
        "|---|---|---|---|",
    ]
    for figure in figures:
        status = figure.status_cell().render()
        count = str(len(figure.drawn)) if figure.drawn else NOT_RUN
        files = ", ".join(p.name for p in figure.paths) if figure.paths else NOT_RUN
        lines.append(f"| {figure.kind} | {status} | {files} | {count} |")
    lines.append("")
    refused = [f for f in figures if f.refused_flag]
    if refused:
        lines += ["## Refusals", ""]
        for figure in refused:
            lines += [f"### {figure.kind}", "", figure.reason, ""]
            for label, why, marker in figure.missing[:40]:
                lines.append(f"- `{marker}` {label} -- {why}")
            if len(figure.missing) > 40:
                lines.append(f"- (+{len(figure.missing) - 40} more, in figures.json)")
            lines.append("")
    else:
        lines += [
            "No figure was refused: every figure the data supported was written.",
            "",
        ]
    return "\n".join(lines)


def write_manifest(
    figures: Sequence[Figure], out_dir: Path, theme: PlotTheme = PUBLICATION
) -> Path:
    """Write `figures.json` and `FIGURES.md` next to the figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "what": "One record per figure: drawn, or refused with the reason.",
            "how": (
                "plotted[] holds exactly the (x, y, label, provenance) tuples handed to "
                "matplotlib. Every entry with status REFUSED has an empty files[] and an "
                "empty plotted[], which is the invariant tests/test_plotting_engine.py "
                "asserts structurally."
            ),
            "source": "scripts/results_apparatus.py — this module measures nothing",
            "dpi": theme.dpi,
            "vector_formats": list(theme.vector_formats),
        },
        "figures": [f.to_json() for f in figures],
    }
    json_path = out_dir / "figures.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = out_dir / "FIGURES.md"
    md_path.write_text(manifest_markdown(figures), encoding="utf-8")
    return json_path


# ═══════════════════════════════════════════════════════════════════
#  The batch
# ═══════════════════════════════════════════════════════════════════


def render_all(
    cfgs: Sequence[Config],
    states: Mapping[str, RunState],
    params: Mapping[str, tuple[int, int]],
    *,
    out_dir: Path,
    theme: PlotTheme = PUBLICATION,
    vector_only: bool = False,
) -> list[Figure]:
    """Every figure the data supports, refusals included."""
    return [
        plot_compute_vs_performance(
            cfgs, states, params, out_dir=out_dir, theme=theme, vector_only=vector_only
        ),
        plot_ablation_matrix(cfgs, states, out_dir=out_dir, theme=theme, vector_only=vector_only),
        plot_training_stability(
            cfgs, states, out_dir=out_dir, theme=theme, vector_only=vector_only
        ),
        plot_baseline_comparison(
            cfgs, states, out_dir=out_dir, theme=theme, vector_only=vector_only
        ),
        plot_seed_band(cfgs, states, out_dir=out_dir, theme=theme, vector_only=vector_only),
    ]


# ═══════════════════════════════════════════════════════════════════
#  Fabrication self-check
# ═══════════════════════════════════════════════════════════════════


class _LeakyCell:
    """A stand-in that renders an unmeasured cell as a number.

    Mirrors `results_apparatus._LeakyCell`, for the same reason: the realistic way
    a fabricated figure ships is a change to something that reads a value, not a
    change to the guarded type. `Cell.render` ignores its text when the state is
    unmeasured, so a real `Cell` cannot leak a number and a self-test aimed at it
    would prove nothing.
    """

    measured_flag = False

    def __init__(self, text: str) -> None:
        self._text = text

    def render(self) -> str:
        return self._text

    @property
    def why(self) -> str:
        return "leaky stand-in"


#: How many attempts `guard_selftest` makes. Named so the PASS message quotes the
#: self-test rather than a number typed beside it.
GUARD_SELFTEST_ATTEMPTS = 12


def guard_selftest(tmp_dir: Path | None = None) -> list[str]:
    """Try to get a figure drawn without a measurement. Every attempt must fail.

    A refusal that has never been observed to refuse is indistinguishable from a
    refusal that cannot fail, which is the whole argument for a self-check. Three
    things have to hold:

    1. the STRUCTURAL property -- a `Figure` with no measured datum cannot carry a
       path, so "an empty figure file on disk" is not representable;
    2. the GUARD -- each way of smuggling a number past the measured-value door
       is rejected;
    3. the POSITIVE CONTROL -- a real measurement still produces a real figure,
       so a self-test that refuses everything would not pass here.
    """
    failures: list[str] = []

    # 1. Structural: a path with nothing measured behind it.
    for paths in ([Path("x.pdf")], [Path("a.pdf"), Path("b.svg")]):
        try:
            Figure("selftest", "selftest", paths, [], [], "why", NO_MEASUREMENT)
        except FabricationError:
            continue
        failures.append(f"a Figure accepted {len(paths)} output path(s) with no measured datum")

    # 2. The guard, against a renderer that leaks.
    for text in ("0.0", "0.000000", "—", "–", "-", ".", "", "N/A", "n/a", "?", "TBD", "1e-3")[
        :GUARD_SELFTEST_ATTEMPTS
    ]:
        try:
            measured_number(_LeakyCell(text))  # type: ignore[arg-type]
        except FabricationError:
            continue
        failures.append(f"measured_number let an unmeasured cell render as {text!r}")

    # 3. Positive control: a real measurement survives, and produces a figure.
    for good in ("2.1", "0.204", "137.9"):
        try:
            measured_number(Cell.measured(good, "a file on disk"))  # type: ignore[arg-type]
        except FabricationError as exc:
            failures.append(f"measured_number rejected a genuine measurement {good!r}: {exc}")

    # 4. The end-to-end refusal: no runs at all means no figure, on disk or not.
    cfgs = _synthetic_configs()
    states = {c.rel: RunState(c.label, c.epochs, c.run_dir) for c in cfgs}
    # A scratch directory, never the repo. The self-check has to work on the day
    # the tree does have runs, and writing a figure into `output/` from a command
    # whose job is to check that figures are not written is a contradiction that
    # would only show up after the owner had trained something.
    with tempfile.TemporaryDirectory(prefix="plotting-selftest-") as scratch:
        out = Path(tmp_dir) if tmp_dir is not None else Path(scratch)
        before = sorted(p.name for p in out.glob("*")) if out.is_dir() else []
        figures = render_all(cfgs, states, read_param_counts(), out_dir=out)
        for figure in figures:
            if not figure.refused_flag:
                failures.append(f"{figure.kind} drew a figure from configs with no runs at all")
            if figure.wrote_anything:
                failures.append(f"{figure.kind} wrote {len(figure.paths)} file(s) with no data")
            for rel in (f"{figure.stem}.pdf", f"{figure.stem}.svg", f"{figure.stem}.png"):
                if (out / rel).exists():
                    failures.append(
                        f"{figure.kind} left {rel} on disk with no measurement behind it"
                    )
        after = sorted(p.name for p in out.glob("*")) if out.is_dir() else []
        if after != before:
            failures.append(f"the refusal pass changed {out}: {set(after) - set(before)}")

    return failures


def _synthetic_configs() -> list[Config]:
    """Three configs with no runs, built without touching the config tree.

    The self-check must not depend on what the repo currently ships -- a test that
    passes because the repository happens to be empty is a test that will fail
    for the wrong reason the day it is not.
    """
    base = {
        "model": {"embed_dim": 64, "encoder_depth": 2, "num_heads": 2},
        "data": {"batch_size": 4, "max_seq_len": 128},
        "optimization": {"epochs": 3, "grad_accum_steps": 2},
        "logging": {"folder": "output/_selftest", "log_freq": 10},
        "meta": {"model_name": "text_span_jepa"},
    }
    raw = {"_meta": {"ablation": "all_core"}}
    return [
        Config("config/_selftest/full.yaml", {**base}, raw),
        Config("config/_selftest/other.yaml", {**base}, {**raw, "_meta": {"ablation": "no_cgn"}}),
        Config("config/_selftest/arm.yaml", {**base}, {}),
    ]


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════


def collect(run_roots: Sequence[str]) -> tuple[list[Config], dict[str, RunState]]:
    cfgs = RA.resolve_configs()
    roots = [Path(r) for r in (run_roots or ["output"])]
    return cfgs, {c.rel: RA.discover(c, roots) for c in cfgs}


def _common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--run-root",
        action="append",
        default=None,
        help="root that 'logging.folder: output/...' maps onto (default: output)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="where figures go (default: docs/results/figures)",
    )
    parser.add_argument(
        "--vector-only", action="store_true", help="skip the raster fallback (default: writes both)"
    )
    parser.add_argument("--dpi", type=int, default=PUBLICATION.dpi, help="raster dpi (default 400)")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plotting_engine",
        description=(
            "Publication figures drawn from what scripts/results_apparatus.py emits. "
            "Refuses to draw a curve it was not given: with no measurement it writes no "
            "file and says why."
        ),
    )
    # `dest` is not optional here: without it argparse dispatches but stores
    # nothing, and `main` cannot tell which subcommand ran.
    sub = parser.add_subparsers(dest="command")
    for name, help_text in (
        ("figures", "draw every figure the data supports"),
        ("all", "draw every figure and write figures.json + FIGURES.md"),
        ("verify", "fabrication self-check: every attempt to fake a figure must fail"),
        ("howto", "what would fill each refusal"),
    ):
        _common(sub.add_parser(name, help=help_text))
    return parser


def cmd_render(args, with_manifest: bool) -> int:
    theme = PlotTheme(dpi=args.dpi)
    out_dir = Path(args.out_dir) if args.out_dir else RA.REPO / "docs" / "results" / "figures"
    cfgs, states = collect(args.run_root or [])
    figures = render_all(
        cfgs,
        states,
        read_param_counts(),
        out_dir=out_dir,
        theme=theme,
        vector_only=args.vector_only,
    )
    for figure in figures:
        if figure.refused_flag:
            print(f"{NOT_RUN} {figure.kind}: {figure.reason}", file=sys.stderr)
        else:
            print(
                f"wrote {figure.kind}: {len(figure.paths)} file(s), "
                f"{len(figure.drawn)} measured datum/ata -> {out_dir}",
                file=sys.stderr,
            )
    if with_manifest:
        path = write_manifest(figures, out_dir, theme)
        print(f"wrote {path}", file=sys.stderr)
    refused = sum(1 for f in figures if f.refused_flag)
    print(f"\n{len(figures) - refused}/{len(figures)} figure(s) written.", file=sys.stderr)
    return 0


def cmd_verify(args) -> int:
    failures = guard_selftest()
    if failures:
        print("\nverify: FAILED -- the guard is not load-bearing:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(
        f"verify: PASS -- {GUARD_SELFTEST_ATTEMPTS} attempts to smuggle a number past the "
        "measured-value door were rejected, a Figure with a path and no data was "
        "refused, all five figures refused to draw from configs with no runs and left "
        "nothing on disk, and a genuine measurement still produced a figure. The "
        "refusal is load-bearing, not decorative."
    )
    return 0


def cmd_howto(args) -> int:
    cfgs, states = collect(args.run_root or [])
    # Into a scratch directory: `howto` reports, it does not produce. On a tree
    # with runs it would otherwise write a set of figures nobody asked for into
    # a directory nobody named.
    with tempfile.TemporaryDirectory(prefix="plotting-howto-") as scratch:
        figures = render_all(cfgs, states, read_param_counts(), out_dir=Path(scratch))
    drawn = [f for f in figures if not f.refused_flag]
    print("Figures, and what would fill each refusal")
    print("=" * 42)
    print()
    if drawn:
        print(f"{len(drawn)} of {len(figures)} figure(s) are written and the data supports")
        print("them. Re-run `python scripts/plotting_engine.py all` to produce them.")
        print()
    else:
        print("Nothing below was drawn. The figures do not exist because no run exists.")
        print()
    for figure in figures:
        print(f"* {figure.kind}: {figure.status_cell().render()}")
        if figure.refused_flag:
            print(f"    {figure.reason}")
    print()
    print("The run order is `python scripts/results_apparatus.py howto`. Once a")
    print("train_log.csv exists, re-run this script and the figures appear:")
    print()
    print("    python scripts/plotting_engine.py all")
    print()
    print("No figure is ever drawn from anything else. In particular `val loss` stays")
    print("empty even after a run: no CSV column carries it, and")
    print("`results_apparatus.val_loss_cell` holds the extraction command.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = getattr(args, "command", None)
    if command == "figures":
        return cmd_render(args, with_manifest=False)
    if command == "all":
        return cmd_render(args, with_manifest=True)
    if command == "verify":
        return cmd_verify(args)
    if command == "howto":
        return cmd_howto(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
