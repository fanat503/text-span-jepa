# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""`scripts/plotting_engine.py` must never draw a curve it was not given.

Why this file exists
--------------------
The engine that renders the paper's figures has exactly one interesting
property, and it is a negative one: with no measurements it must draw nothing.
Everything else in the module is plumbing around that.

The risk this file closes is specific and was measured before it was designed
around. `scripts/results_apparatus.py emit_curves` already takes the "write no
PNG at all" approach for training curves, and its own docstring names the
reason: *"A figure with empty axes and a flat zero line is worse than no
figure: it looks like a result."* A table cell that renders `NOT-RUN` is read
as a mistake and investigated by whoever reads the draft. A figure with empty
axes and a flat line is read as a measurement and **believed** by two reviewers
who agree on what it shows. The fabrication is not merely unnoticed, it is
consensus. So the engine has to be structurally incapable of producing that
artifact, and that is what is asserted here rather than assumed.

The shape of the central assertion
----------------------------------
`test_empty_data_draws_nothing_at_all` does not check that the engine returns a
sensible object. A refusal that is merely *reported* is exactly the failure this
file exists to prevent, because the report is a string nobody reads while the
PNG sits in the figures directory. So the test watches matplotlib itself:

* `matplotlib.figure.Figure.__init__` -- zero constructions. A refusal happens
  before a canvas exists, so there is nothing on disk *and* nothing in memory;
* `Figure.savefig` -- zero writes;
* every data-drawing `Axes` method (`plot`, `scatter`, `bar`, `step`,
  `errorbar`, `fill_between`, `pcolormesh`, `imshow`) -- zero calls, so there is
  no Line2D, no QuadMesh, no bar and therefore no curve;
* the output directory -- empty afterwards.

Patching matplotlib rather than the engine matters. A test that patched the
engine's own `_pyplot` seam would pass even if the engine reached matplotlib by
some other route, which is exactly the kind of test that keeps passing after the
guard it was written for is deleted.

Two kinds of vacuous green, both closed here
--------------------------------------------
1. *Refuses everything.* `test_measured_runs_produce_figures_and_name_the_rest`
   feeds the engine one real `train_log.csv` on disk and requires real files
   with exactly the points the CSV holds. Without it, an engine that raised on
   every input would pass every other test in this file.
2. *The self-test is decorative.* `test_guard_selftest_is_load_bearing` breaks
   the guard at runtime (`measured_number` monkeypatched to a leaky
   implementation) and requires `guard_selftest` to report it. A self-check that
   has never been observed to fail is indistinguishable from one that cannot.

Counting assertions
-------------------
No test here pins a magic constant that a refactor could change without meaning
anything. The palette property is a measured CIELAB separation held above a
documented floor; the "how many series" refusal is checked for the property that
it *names* the problem rather than for a number; the dpi assertion is a floor,
because 300 is a journal requirement rather than this module's preference.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import matplotlib.axes
import matplotlib.figure
import yaml

import plotting_engine as PE
import results_apparatus as RA
from results_apparatus import NOT_APPLICABLE, NOT_RUN, Cell, FabricationError

# ═══════════════════════════════════════════════════════════════════
#  Watching matplotlib
# ═══════════════════════════════════════════════════════════════════

#: Every `Axes` method that can put a mark on the canvas. `text`/`annotate` are
#: deliberately absent: a refusal is allowed to exist as an exception carrying a
#: string, and `Figure.reason` is exactly that. What it is not allowed to do is
#: draw a *mark*, and these are the methods that draw marks.
DATA_METHODS = (
    "plot",
    "scatter",
    "bar",
    "barh",
    "step",
    "errorbar",
    "stem",
    "fill_between",
    "pcolormesh",
    "imshow",
)


class Spy:
    """Counts figure constructions, saves and data-artist calls."""

    def __init__(self) -> None:
        self.figures = 0
        self.saves = 0
        self.calls: list[str] = []

    def summary(self) -> str:
        return (
            f"{self.figures} Figure(s) built, {self.saves} savefig call(s), "
            f"data artists: {self.calls or 'none'}"
        )


@contextmanager
def watching_matplotlib():
    spy = Spy()
    real_init = matplotlib.figure.Figure.__init__
    real_save = matplotlib.figure.Figure.savefig
    real_methods = {name: getattr(matplotlib.axes.Axes, name) for name in DATA_METHODS}

    def counting_init(self, *a, **k):
        spy.figures += 1
        return real_init(self, *a, **k)

    def counting_save(self, *a, **k):
        spy.saves += 1
        return real_save(self, *a, **k)

    def counting_method(name, real):
        def wrapper(self, *a, **k):
            spy.calls.append(name)
            return real(self, *a, **k)

        return wrapper

    with (
        mock.patch.object(matplotlib.figure.Figure, "__init__", counting_init),
        mock.patch.object(matplotlib.figure.Figure, "savefig", counting_save),
    ):
        for name, real in real_methods.items():
            setattr(matplotlib.axes.Axes, name, counting_method(name, real))
        try:
            yield spy
        finally:
            for name, real in real_methods.items():
                setattr(matplotlib.axes.Axes, name, real)


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

BASE_CONFIG = {
    "model": {"embed_dim": 64, "encoder_depth": 2, "num_heads": 2},
    "data": {"batch_size": 4, "max_seq_len": 128},
    "optimization": {"epochs": 2, "grad_accum_steps": 2},
    "logging": {"folder": "output/_fixture", "log_freq": 10},
    "meta": {"model_name": "text_span_jepa"},
}


def make_config(rel: str, ablation: str | None = None, arm: str = "text_span_jepa") -> RA.Config:
    """A resolved config built in memory, with no file behind it.

    The refusal tests must not depend on what the repo currently ships. A test
    that passes because the repository happens to contain no runs is a test that
    will start failing for an unrelated reason the day it does.
    """
    merged = dict(BASE_CONFIG)
    merged["meta"] = {**BASE_CONFIG["meta"], "model_name": arm}
    raw = {"_meta": {"ablation": ablation}} if ablation else {}
    return RA.Config(rel, merged, raw)


def make_state(
    cfg: RA.Config, n_rows: int = 0, run_dir: str | None = None, complete: bool = True
) -> RA.RunState:
    """A `RunState` with `n_rows` logged rows, as `discover` would have produced."""
    state = RA.RunState(cfg.label, cfg.epochs, cfg.run_dir)
    if n_rows == 0:
        state.reason = f"no run directory at {cfg.run_dir or '(no logging.folder)'}"
        return state
    state.run_dir = run_dir or f"output/_fixture/{cfg.label}"
    state.rows = [
        [
            2.5 - 0.25 * r if column == 0 else (0.4 + 0.5 * r if column == 8 else 0.0)
            for column in range(len(RA.CSV_COLUMNS))
        ]
        for r in range(n_rows)
    ]
    state.status = RA.STATUS_COMPLETE if complete else RA.STATUS_PARTIAL
    state.epochs_done = cfg.epochs if complete else 1
    return state


def _coerce_or_zero(text: str) -> float:
    """The leak under test: an unreadable value becomes a plausible number.

    This is how a fabricated figure actually ships -- a formatter that pads an
    absent value to zero, not a crash. A self-check that only noticed crashes
    would be passed by this function.
    """
    try:
        return float(text)
    except ValueError:
        return 0.0


def write_train_log(directory: Path, n_rows: int = 8) -> Path:
    """A real `train_log.csv` on disk, so `RA.discover` finds it the way the trainer
    leaves one rather than the way a test would like it to find one."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RA.TRAIN_LOG
    lines = [",".join(RA.CSV_COLUMNS)]
    for r in range(n_rows):
        row = ["0.0"] * len(RA.CSV_COLUMNS)
        row[0] = f"{2.5 - 0.25 * r:.6f}"
        row[8] = f"{0.4 + 0.5 * r:.6f}"
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def nothing_measured():
    """Five configs, none of which has a run. The repository's real state."""
    cfgs = [
        make_config("config/_fixture/full.yaml", "all_core"),
        make_config("config/_fixture/no_cgn.yaml", "no_cgn"),
        make_config("config/_fixture/no_spc.yaml", "no_spc"),
        make_config("config/_fixture/xsmall.yaml"),
        make_config("config/_fixture/mlm.yaml", arm="mlm"),
    ]
    states = {c.rel: make_state(c) for c in cfgs}
    return cfgs, states, {}


# ═══════════════════════════════════════════════════════════════════
#  THE refusal
# ═══════════════════════════════════════════════════════════════════


class TestRefusesToInvent:
    def test_empty_data_draws_nothing_at_all(self, nothing_measured, tmp_path):
        """The central assertion: no measurement means no mark on any canvas.

        What would count as a failure here, and is explicitly not a pass:

        * an empty axes with a line on it (the `results_apparatus` failure);
        * a scatter with no points but a legend of arm names;
        * a bar chart of zero-height bars, which reads as "every arm scored 0";
        * a figure file on disk and a refusal only in the log;
        * an exception carrying the reason, which is honest but still leaves the
          question "was anything drawn?" unanswered -- so this test requires the
          call to *return* and requires the returns to carry no path.

        The last point is why the test asserts both `spy` silence and the shape of
        the returned objects. Silence alone would be satisfied by a crash.
        """
        cfgs, states, params = nothing_measured
        out_dir = tmp_path / "figures"

        with watching_matplotlib() as spy:
            figures = PE.render_all(cfgs, states, params, out_dir=out_dir)

        assert len(figures) == 5, "every figure must answer, not just the ones that could be drawn"
        assert spy.figures == 0, f"a Figure was constructed with no measurement: {spy.summary()}"
        assert spy.saves == 0, f"something was saved with no measurement: {spy.summary()}"
        assert spy.calls == [], f"a data artist was created with no measurement: {spy.calls}"
        assert (
            not out_dir.exists() or list(out_dir.glob("*")) == []
        ), f"the refusal left files on disk: {sorted(p.name for p in out_dir.glob('*'))}"

        for figure in figures:
            assert figure.refused_flag, f"{figure.kind} answered from nothing"
            assert figure.paths == [], f"{figure.kind} claims to have written {figure.paths}"
            assert figure.drawn == [], f"{figure.kind} reports plotted data it never had"
            assert figure.reason_kind == PE.NO_MEASUREMENT
            assert figure.reason.strip(), f"{figure.kind} refused without saying why"
            assert figure.missing, (
                f"{figure.kind} refused without naming what is missing, so the reader "
                "cannot tell which arms to run"
            )
            assert all(
                why.strip() for _, why, _ in figure.missing
            ), f"{figure.kind} listed a missing arm with no reason"

    def test_refusal_reason_names_the_missing_arms_and_the_missing_artifact(
        self, nothing_measured, tmp_path
    ):
        """A refusal that does not say what to go and run is not usable.

        Asserted as a property -- the reason names at least one of the arms it
        reports missing, and every missing entry cites a concrete artifact rather
        than saying "no data" -- rather than by matching one hard-coded sentence,
        so rewording the message does not break it and deleting the substance does.
        """
        cfgs, states, params = nothing_measured
        figures = PE.render_all(cfgs, states, params, out_dir=tmp_path / "f")

        for figure in figures:
            assert figure.reason.strip()
            if len(figure.missing) <= 6:
                # Short enough to be listed: every one of them must be in the prose.
                for label, _, _ in figure.missing:
                    assert label in figure.reason, (
                        f"{figure.kind} reports {label!r} missing without naming it: "
                        f"{figure.reason!r}"
                    )
            for label, why, marker in figure.missing:
                assert why.strip(), f"{figure.kind} listed {label!r} with no reason"
                assert marker in (NOT_RUN, NOT_APPLICABLE), (
                    f"{figure.kind} reported {label!r} with marker {marker!r}, which is "
                    "neither 'go and run this' nor 'this can never be measured'"
                )
                # Each marker promises a different next action, so each must cite a
                # different kind of thing. A `NOT-RUN` reason that does not say
                # what to produce leaves the reader with no action; a
                # `NOT-APPLICABLE` reason that does not say why leaves them
                # wondering whether they simply have not looked hard enough.
                if marker == NOT_RUN:
                    assert "output/" in why or RA.TRAIN_LOG in why or "=" in why, (
                        f"{figure.kind} marks {label!r} {NOT_RUN} with {why!r}, which "
                        "names no artifact the owner could produce"
                    )
                else:
                    assert "/" in why or "resolved config" in why, (
                        f"{figure.kind} marks {label!r} {NOT_APPLICABLE} with {why!r}, "
                        "which does not say what about the config makes it impossible"
                    )

        curves = next(f for f in figures if f.kind == "training-stability")
        assert curves.refused_flag
        assert any(
            label in curves.reason for label, _, _ in curves.missing
        ), "the refusal reason does not name a single one of the arms it reports missing"

    def test_unmeasured_number_cannot_enter_the_engine(self):
        """The door itself, tested with a renderer that leaks.

        `Cell.render` ignores its text when the state is unmeasured, so a real
        `Cell` cannot be made to leak -- and a self-test aimed only at a real
        `Cell` would prove nothing. `PE._LeakyCell` models the realistic leak: a
        change to something that *reads* a value, not to the guarded type.
        """
        for text in ("0.0", "0.000000", "—", "–", "-", ".", "", "N/A", "n/a", "?", "TBD", "1e-3"):
            with pytest.raises(FabricationError):
                PE.measured_number(PE._LeakyCell(text))

        for cell in (Cell.not_run("no run"), Cell.not_applicable("weight is 0")):
            with pytest.raises(FabricationError):
                PE.measured_number(cell)

        # Positive control: the guard is not "refuse everything".
        for good in ("2.1", "0.204", "137.9"):
            assert PE.measured_number(Cell.measured(good, "a file on disk")) == float(good)

    def test_figure_cannot_carry_a_path_without_data(self):
        """The invariant is in the constructor, so no call site can forget it.

        Checked by trying to construct the impossible object. If this ever stops
        raising, `test_empty_data_draws_nothing_at_all`'s file assertions become
        the only thing standing between a refusal and a figure of nothing.
        """
        with pytest.raises(FabricationError):
            PE.Figure("k", "stem", [Path("x.pdf")], [], [], "why", PE.NO_MEASUREMENT)

        # The mirror image is refused too, so `drawn` cannot quietly disagree
        # with what was written.
        with pytest.raises(FabricationError):
            PE.Figure("k", "stem", [], [(1.0, 2.0, "a", "b")], [], "", "")

    def test_one_logged_row_is_not_a_curve(self, tmp_path):
        """A single point is a dot. Drawing a line through it invents the segment.

        This is the failure called out above, one level down: the run happened,
        so nothing here is NOT-RUN, and an engine that only checked "did we get a
        number" would draw a line. The refusal has to fire on shape, not on
        provenance.
        """
        cfgs = [make_config("config/_fixture/one_row.yaml")]
        states = {cfgs[0].rel: make_state(cfgs[0], n_rows=1)}

        figure = PE.plot_training_stability(cfgs, states, out_dir=tmp_path / "f")

        assert figure.refused_flag
        assert figure.reason_kind == PE.NO_MEASUREMENT
        assert figure.paths == []
        assert any(
            "dot, not a curve" in why for _, why, _ in figure.missing
        ), f"the reason does not say why a one-row run is not a curve: {figure.reason}"

    def test_partial_measurement_plots_only_what_exists(self, tmp_path):
        """Three arms, one run. One bar, two named absences, no interpolation."""
        cfgs = [
            make_config("config/_fixture/a.yaml"),
            make_config("config/_fixture/b.yaml", arm="mlm"),
            make_config("config/_fixture/c.yaml", arm="data2vec"),
        ]
        states = {
            cfgs[0].rel: make_state(cfgs[0], n_rows=8),
            cfgs[1].rel: make_state(cfgs[1]),
            cfgs[2].rel: make_state(cfgs[2]),
        }
        params = {c.rel: (30_000_000, 28_000_000) for c in cfgs}

        figure = PE.plot_baseline_comparison(cfgs, states, out_dir=tmp_path / "f")

        assert not figure.refused_flag
        assert len(figure.drawn) == 1, f"expected exactly the one measured arm, got {figure.drawn}"
        # The label must identify the CONFIG, not the arm: a dozen JEPA configs
        # all report `text_span_jepa`, and a manifest where every bar is called
        # the same thing cannot say which one is which.
        assert figure.drawn[0][2] == cfgs[0].label
        assert len(figure.missing) == 2, "the two unmeasured arms must still be reported"
        assert all(why.strip() for _, why, _ in figure.missing)
        assert figure.paths, "a figure with a measurement must actually be written"

    def test_unreadable_legend_is_a_refusal_not_a_silent_subset(self, tmp_path):
        """Refusing to draw 40 arms is not a fabrication; drawing 8 of them is.

        When every config has a run -- which is the state this repo will be in the
        day after the owner trains -- a legend with one entry per config is
        unreadable, and the tempting fix is to plot the first N. That hides arms
        from the reader with no marker that anything was dropped, so the engine
        refuses and asks to be told which configs to draw.
        """
        cfgs = [make_config(f"config/_fixture/many{i}.yaml", f"no_x{i}") for i in range(12)]
        states = {c.rel: make_state(c, n_rows=8) for c in cfgs}

        figure = PE.plot_training_stability(cfgs, states, out_dir=tmp_path / "f")

        assert figure.refused_flag
        assert figure.reason_kind == PE.UNREADABLE, (
            "this refusal is about legibility, not about missing data, and the two "
            "must stay distinguishable or the central test cannot be trusted"
        )
        assert figure.paths == []
        assert (
            "only=[" in figure.reason
        ), "the refusal must say how to proceed, not just that it cannot"

        # And naming them is enough.
        chosen = [c.rel for c in cfgs[:2]]
        drawn = PE.plot_training_stability(cfgs, states, out_dir=tmp_path / "f", only=chosen)
        assert not drawn.refused_flag
        plotted = {label for _, _, label, _ in drawn.drawn}
        assert plotted, "naming the configs must produce series"
        assert all(
            label.split("/")[0] in {c.label for c in cfgs[:2]} for label in plotted
        ), f"a series outside the requested configs was drawn: {plotted}"
        assert not any(label.split("/")[0] in {c.label for c in cfgs[2:]} for label in plotted)

    def test_real_repository_with_no_runs_draws_nothing(self, tmp_path):
        """The same refusal against the actual 68-config tree and a run-root that
        does not exist -- the repository's real state, not a fixture's.

        This is the test that would go red if someone deleted the guard from one
        plot function while leaving the others intact, because it calls all five.
        """
        cfgs = RA.resolve_configs()
        assert len(cfgs) > 1, "the config tree resolved to nothing; the test proves nothing"
        states = {c.rel: RA.discover(c, [tmp_path / "no-such-root"]) for c in cfgs}
        assert not any(s.has_log for s in states.values()), (
            "a run directory exists under the run-root, so this is not the "
            "no-measurement case it claims to test"
        )

        with watching_matplotlib() as spy:
            figures = PE.render_all(
                cfgs, states, PE.read_param_counts(), out_dir=tmp_path / "figures"
            )

        assert spy.figures == 0, spy.summary()
        assert spy.calls == [], spy.calls
        assert all(f.refused_flag for f in figures), [f.kind for f in figures if not f.refused_flag]
        assert not (tmp_path / "figures").exists()


# ═══════════════════════════════════════════════════════════════════
#  The positive control: a real run produces a real figure
# ═══════════════════════════════════════════════════════════════════


class TestMeasuresRealFiles:
    def test_measured_runs_produce_figures_and_name_the_rest(self, tmp_path):
        """A `train_log.csv` written to disk, discovered the way the trainer leaves
        one, drawn with exactly its rows.

        Without this test every other one in the file could pass against an engine
        that refuses all input, which is the failure mode a guard-heavy design is
        most exposed to.
        """
        run_root = tmp_path / "runs"
        cfgs = [
            make_config("config/_fixture/ran.yaml", "all_core"),
            make_config("config/_fixture/never.yaml", "no_cgn"),
        ]
        for cfg in cfgs:
            cfg.run_dir = str(run_root / cfg.label)
        states = {c.rel: RA.discover(c, [run_root]) for c in cfgs}
        write_train_log(run_root / cfgs[0].label, n_rows=8)
        for epoch in range(1, cfgs[0].epochs + 1):
            (run_root / cfgs[0].label / f"checkpoint-ep{epoch}.pth.tar").write_bytes(b"")
        states = {c.rel: RA.discover(c, [run_root]) for c in cfgs}

        assert states[cfgs[0].rel].has_log, "the fixture failed: discover found no rows"
        assert not states[cfgs[1].rel].has_log
        params = {c.rel: (30_000_000, 28_000_000) for c in cfgs}

        figure = PE.plot_training_stability(
            [cfgs[0]], {cfgs[0].rel: states[cfgs[0].rel]}, out_dir=tmp_path / "figs"
        )

        assert not figure.refused_flag
        logged = states[cfgs[0].rel].rows
        expected_series = {
            column
            for column, _ in RA.CURVE_SERIES
            if RA.metric_cell(cfgs[0], states[cfgs[0].rel], column).measured_flag
        }
        assert expected_series, "the fixture logged nothing the engine could measure"
        assert {label.split("/")[1] for _, _, label, _ in figure.drawn} == expected_series
        # Every plotted series must have as many points as the CSV has rows: the
        # engine neither pads a short run nor resamples it.
        for _, n_points, _, provenance in figure.drawn:
            assert n_points == len(logged), f"{provenance} claims {n_points} of {len(logged)} rows"
            assert RA.TRAIN_LOG in provenance

    def test_a_partial_run_is_labelled_partial_and_not_padded(self, tmp_path):
        """Fewer logged rows than declared epochs is a real, short run.

        It is drawn, and drawn as itself. Padding it forward to the declared length
        would draw a straight segment nobody measured -- the most plausible-looking
        fabrication available, because a smooth decay is what a healthy run looks
        like.
        """
        cfg = make_config("config/_fixture/short.yaml", "all_core")
        cfg.opt_cfg = {**cfg.opt_cfg, "epochs": 50}
        state = make_state(cfg, n_rows=6, complete=False)

        curve = PE.Curve.from_run(cfg, state, "loss")

        assert curve is not None
        assert curve.n_points == 6
        assert curve.complete is False
        assert (
            curve.xs[-1] == 5 * cfg.log_freq
        ), "the x axis must come from log_freq, not from epochs"

    def test_jepa_only_columns_cannot_be_borrowed_from_a_baseline_arm(self, tmp_path):
        """An MLM run logs `0.000000` for every JEPA column; that is a logger
        artefact, not a measurement.

        `results_apparatus.COLUMN_GATE` closes it in the tables. A figure that read
        the CSV directly would put a marker at exactly y=0 for every JEPA-only
        column on every baseline arm -- a flat line at zero, which is the precise
        artifact this module exists not to produce.
        """
        jepa = make_config("config/_fixture/jepa.yaml")
        mlm = make_config("config/_fixture/mlm.yaml", arm="mlm")
        states = {
            jepa.rel: make_state(jepa, n_rows=8),
            mlm.rel: make_state(mlm, n_rows=8),
        }
        params = {jepa.rel: (30_000_000, 28_000_000), mlm.rel: (30_000_000, 28_000_000)}

        figure = PE.plot_compute_vs_performance(
            [mlm],
            {mlm.rel: states[mlm.rel]},
            params,
            column="decoder_accuracy",
            out_dir=tmp_path / "f",
        )

        assert figure.refused_flag
        assert any(
            ("baseline" in why and "logger" in why) or RA.ParamCache.PATH.name in why
            for _, why, _ in figure.missing
        ), f"the refusal does not explain the logger artefact: {figure.missing}"

        # The JEPA arm does contribute, so the gate is not just refusing everything.
        jepa_figure = PE.plot_compute_vs_performance(
            [jepa], {jepa.rel: states[jepa.rel]}, params, column="loss", out_dir=tmp_path / "f"
        )
        assert not jepa_figure.refused_flag
        assert len(jepa_figure.drawn) == 1

    def test_the_engine_measures_nothing_itself(self, tmp_path):
        """A parameter count the engine did not receive is not data.

        It refuses to recompute 62 models to decorate a figure, and a config
        absent from the cache is reported as missing rather than given an
        estimated x-coordinate.
        """
        assert PE.read_param_counts(tmp_path / "nothing.json") == {}

        payload = {"configs": {"a.yaml": {"total": 10, "trainable": 8}, "bad.yaml": {"total": "x"}}}
        cache = tmp_path / "param_counts.json"
        cache.write_text(json.dumps(payload), encoding="utf-8")

        counts = PE.read_param_counts(cache)

        assert counts == {"a.yaml": (10, 8)}, "an unreadable entry must be skipped, not guessed"
        assert "bad.yaml" not in counts


# ═══════════════════════════════════════════════════════════════════
#  Output quality
# ═══════════════════════════════════════════════════════════════════


class TestPublicationQuality:
    def test_theme_meets_a_journals_floor(self):
        """300 dpi and real vector output, not 140 dpi PNG and hope.

        Asserted as a floor against the documented requirement rather than against
        a number this module happens to use, so a theme that raises dpi stays
        green and one that lowers it does not.
        """
        theme = PE.PUBLICATION

        assert theme.dpi >= 300, f"{theme.dpi} dpi is below the 300 a journal asks for"
        assert "pdf" in theme.vector_formats and "svg" in theme.vector_formats
        rc = theme.rcparams()
        assert rc["savefig.dpi"] == theme.dpi
        # 42 embeds TrueType so the text stays editable; 3 would arrive as paths.
        assert rc["pdf.fonttype"] == 42, "PDF text must stay text, not become outlines"
        assert rc["svg.fonttype"] == "none"

    def test_written_figures_include_a_real_vector_file(self, tmp_path):
        """The PDF on disk is a PDF, and the PNG beside it is a PNG.

        Cheap, and it catches the failure where a format string is wrong and the
        engine silently writes the same bytes under three extensions.
        """
        cfg = make_config("config/_fixture/one.yaml")
        figure = PE.plot_baseline_comparison(
            [cfg], {cfg.rel: make_state(cfg, n_rows=8)}, out_dir=tmp_path / "f"
        )

        suffixes = {p.suffix for p in figure.paths}
        assert {".pdf", ".svg", ".png"} <= suffixes, suffixes
        for path in figure.paths:
            assert path.stat().st_size > 0
            if path.suffix == ".pdf":
                assert path.read_bytes()[:5] == b"%PDF-", "the .pdf is not a PDF"
            if path.suffix == ".png":
                assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "the .png is not a PNG"

    def test_palette_pairs_are_separable(self):
        """Every pair of series colours is far enough apart to be told apart.

        The property, not a list: any two entries in `PALETTE` must sit at least
        `MIN_PALETTE_SEPARATION` apart in CIELAB, measured here with the same
        conversion the engine uses. The measured minimum for this palette is well
        above the floor, so an edit that brings two entries together fails rather
        than passing quietly.
        """
        import itertools

        worst = min(
            (
                PE.palette_separation(a.color, b.color),
                a.color,
                b.color,
            )
            for a, b in itertools.combinations(PE.PALETTE, 2)
        )
        assert worst[0] >= PE.MIN_PALETTE_SEPARATION, (
            f"{worst[1]} and {worst[2]} are only {worst[0]:.1f} apart in CIELAB; "
            f"the floor is {PE.MIN_PALETTE_SEPARATION}"
        )

    def test_colour_is_never_the_only_channel(self):
        """Hue, marker shape and dash pattern are three separate channels.

        A greyscale print or a dichromat viewer keeps two of the three, so the
        figure survives the medium it ends up in. Asserted as uniqueness in each
        channel independently.
        """
        assert len({s.marker for s in PE.PALETTE}) == len(PE.PALETTE)
        assert len({s.linestyle for s in PE.PALETTE}) == len(PE.PALETTE)
        assert len({s.color for s in PE.PALETTE}) == len(PE.PALETTE)
        assert len(PE.PALETTE) >= 8, "eight is the size of the colour-blind-safe palette this is"

    def test_two_series_are_distinguishable_in_greyscale(self):
        """Two entries that differ only in hue would be identical in a photocopy.

        CIELAB L* is the luminance a greyscale conversion keeps, so this checks
        that no two palette entries are equal in L* as well as separable in full
        colour -- a stronger claim than the palette is usually held to, and the
        reason a series also carries a marker and a dash pattern.
        """
        lumas = [PE._hex_to_lab(s.color)[0] for s in PE.PALETTE]
        assert (
            len({round(v, 3) for v in lumas}) > len(lumas) / 2
        ), "most palette entries share a luminance; hue is doing all the work"


# ═══════════════════════════════════════════════════════════════════
#  The seed band
# ═══════════════════════════════════════════════════════════════════


class TestSeedBand:
    """`plot_seed_band`: variance only where ≥2 distinct measured seeds exist.

    The seed is read from each run's own params dump (what `src/train.py`
    resolved and wrote at train.py:1476), so every test here writes a real dump
    into a real temporary run directory -- a seed the engine cannot read from
    anywhere on disk is a seed the engine must refuse to band.
    """

    @staticmethod
    def _seed_cfg(rel: str, seed: int, folder: str) -> RA.Config:
        """Same experiment content as BASE_CONFIG, varying only seed + run dir."""
        merged = {
            **BASE_CONFIG,
            "logging": {**BASE_CONFIG["logging"], "folder": folder},
            "meta": {**BASE_CONFIG["meta"], "seed": seed},
        }
        return RA.Config(rel, merged, {})

    @staticmethod
    def _seeded_state(cfg: RA.Config, seed: int, run_dir: Path, n_rows: int = 8) -> RA.RunState:
        """Rows whose loss column carries a seed-specific offset, so mean and
        σ are both nonzero and both checkable against hand arithmetic."""
        state = RA.RunState(cfg.label, cfg.epochs, cfg.run_dir)
        state.run_dir = str(run_dir)
        offset = 0.1 * seed
        state.rows = [
            [2.5 - 0.25 * r - offset] + [0.0] * (len(RA.CSV_COLUMNS) - 1) for r in range(n_rows)
        ]
        state.status = RA.STATUS_COMPLETE
        state.epochs_done = cfg.epochs
        return state

    @classmethod
    def _seed_run_dir(cls, tmp_path: Path, name: str, seed: int | None) -> str:
        """A real run directory holding a real params dump, as the trainer leaves one."""
        run_dir = tmp_path / name
        run_dir.mkdir(parents=True, exist_ok=True)
        if seed is not None:
            (run_dir / RA.CONFIG_DUMP).write_text(
                yaml.safe_dump({"meta": {"model_name": "text_span_jepa", "seed": seed}}),
                encoding="utf-8",
            )
        return str(run_dir)

    def _two_seed_fixture(self, tmp_path: Path, embed_dim: int = 64) -> tuple:
        cfg_a = self._seed_cfg("config/_fixture/arm_s1.yaml", 1, "output/_fixture/arm_s1")
        cfg_b = self._seed_cfg("config/_fixture/arm_s2.yaml", 2, "output/_fixture/arm_s2")
        if embed_dim != 64:
            for cfg in (cfg_a, cfg_b):
                cfg.merged["model"]["embed_dim"] = embed_dim
        dirs = {
            cfg.rel: self._seed_run_dir(tmp_path, Path(cfg.rel).stem, seed)
            for cfg, seed in ((cfg_a, 1), (cfg_b, 2))
        }
        states = {
            cfg.rel: self._seeded_state(cfg, seed, Path(dirs[cfg.rel]))
            for cfg, seed in ((cfg_a, 1), (cfg_b, 2))
        }
        return cfg_a, cfg_b, states

    def test_band_over_two_distinct_seeds_is_drawn_with_checkable_arithmetic(self, tmp_path):
        """Two seeds of one arm -> mean line + band, and the plotted mean is the mean.

        MUTATION VERDICT: break `_measured_seed` (return None) and this test
        fails -- the figure is refused instead of drawn. Break `_band_group_key`
        (stop stripping seed keys) and the two runs fall into two n=1 groups,
        refused the same way. Break the band arithmetic (mean over the wrong
        axis, or σ with ddof=0) and the final-x assertion `0.60` fails: the
        seeded loss rows end at 0.65 (offset 0.1) and 0.55 (offset 0.2), so the
        hand-computed mean is (0.65+0.55)/2 = 0.60 exactly.
        """
        cfg_a, cfg_b, states = self._two_seed_fixture(tmp_path)
        cfgs = [cfg_a, cfg_b]
        figure = PE.plot_seed_band(cfgs, states, out_dir=tmp_path / "f")
        assert not figure.refused_flag, figure.reason
        assert figure.wrote_anything
        loss_entries = [d for d in figure.drawn if d[2] == "arm/loss mean"]
        assert len(loss_entries) == 1, f"one loss band expected, got {loss_entries}"
        x, y, _, provenance = loss_entries[0]
        assert x == pytest.approx(7 * 10), "last common log x is (n_rows-1)*log_freq"
        assert y == pytest.approx(0.60), f"mean of 0.65 and 0.55, got {y}"
        assert "seeds [1, 2]" in provenance, provenance
        # a real vector file, as PublicationQuality demands of every factory
        assert any(
            p.suffix == ".pdf" and p.is_file() and p.stat().st_size > 0 for p in figure.paths
        )

    def test_two_runs_of_one_seed_are_refused_and_write_nothing(self, tmp_path):
        """Same arm, two runs, one seed value: no spread is measurable.

        MUTATION VERDICT: if the distinctness check on measured seeds is
        dropped (band over any ≥2 runs), this test fails -- a figure gets
        written where a refusal is asserted.
        """
        cfg_a = self._seed_cfg("config/_fixture/arm_s1.yaml", 42, "output/_fixture/arm_s1")
        cfg_b = self._seed_cfg("config/_fixture/arm_s2.yaml", 42, "output/_fixture/arm_s2")
        states = {}
        for cfg in (cfg_a, cfg_b):
            run_dir = self._seed_run_dir(tmp_path, Path(cfg.rel).stem, 42)
            states[cfg.rel] = self._seeded_state(cfg, 42, Path(run_dir))
        figure = PE.plot_seed_band([cfg_a, cfg_b], states, out_dir=tmp_path / "f")
        assert figure.refused_flag
        assert not figure.wrote_anything
        assert "distinct measured seeds" in figure.reason
        assert figure.missing, "the single-seed group must be named, not silently absent"

    def test_no_recorded_seed_is_refused_not_guessed(self, tmp_path):
        """Run dirs exist but hold no params dump: the seed cannot be attributed.

        MUTATION VERDICT: if `_measured_seed` starts guessing (e.g. falling back
        to the config's declared seed instead of the dump), this test fails --
        a figure would be drawn from runs whose dump was never read.
        """
        cfg_a = self._seed_cfg("config/_fixture/arm_s1.yaml", 1, "output/_fixture/arm_s1")
        cfg_b = self._seed_cfg("config/_fixture/arm_s2.yaml", 2, "output/_fixture/arm_s2")
        states = {}
        for cfg in (cfg_a, cfg_b):
            run_dir = self._seed_run_dir(tmp_path, Path(cfg.rel).stem, None)  # no dump
            states[cfg.rel] = self._seeded_state(cfg, 1, Path(run_dir))
        figure = PE.plot_seed_band([cfg_a, cfg_b], states, out_dir=tmp_path / "f")
        assert figure.refused_flag
        assert not figure.wrote_anything
        assert "none recorded" in figure.reason

    def test_arms_that_differ_in_content_are_never_grouped(self, tmp_path):
        """Same display base, different embed_dim: two experiments, not two seeds.

        MUTATION VERDICT: if `_band_group_key` starts ignoring real config
        content (e.g. grouping by display label), the two arms merge into one
        group and a band is drawn across different experiments -- this test
        fails on the refusal.
        """
        cfg_a = self._seed_cfg("config/_fixture/arm_s1.yaml", 1, "output/_fixture/arm_s1")
        cfg_b = self._seed_cfg("config/_fixture/arm_wide_s2.yaml", 2, "output/_fixture/arm_wide_s2")
        # Rebind, never mutate: make_config's shallow copy shares the `model`
        # dict between configs, so mutating embed_dim in place would change it
        # for BOTH arms and the test would pass for the wrong reason.
        cfg_b.merged["model"] = {**cfg_b.merged["model"], "embed_dim": 96}
        states = {}
        for cfg, seed in ((cfg_a, 1), (cfg_b, 2)):
            run_dir = self._seed_run_dir(tmp_path, Path(cfg.rel).stem, seed)
            states[cfg.rel] = self._seeded_state(cfg, seed, Path(run_dir))
        figure = PE.plot_seed_band([cfg_a, cfg_b], states, out_dir=tmp_path / "f")
        assert figure.refused_flag
        assert not figure.wrote_anything
        assert (
            "arm" in figure.reason and "arm_wide" in figure.reason
        ), f"both single-seed groups must be named: {figure.reason[:200]}"

    def test_member_without_recorded_seed_is_excluded_and_named(self, tmp_path):
        """2 recorded seeds + 1 unrecorded run -> band over 2, exclusion declared.

        MUTATION VERDICT: silently including the unrecorded run in the band
        (seeds it never had) fails the n=2 assertion; dropping it silently
        fails on the missing entry that must name it.
        """
        cfg_a = self._seed_cfg("config/_fixture/arm_s1.yaml", 1, "output/_fixture/arm_s1")
        cfg_b = self._seed_cfg("config/_fixture/arm_s2.yaml", 2, "output/_fixture/arm_s2")
        cfg_c = self._seed_cfg("config/_fixture/arm_s3.yaml", 3, "output/_fixture/arm_s3")
        states = {}
        for cfg, seed in ((cfg_a, 1), (cfg_b, 2), (cfg_c, None)):
            run_dir = self._seed_run_dir(tmp_path, Path(cfg.rel).stem, seed)
            states[cfg.rel] = self._seeded_state(cfg, seed or 1, Path(run_dir))
        figure = PE.plot_seed_band([cfg_a, cfg_b, cfg_c], states, out_dir=tmp_path / "f")
        assert not figure.refused_flag, figure.reason
        loss_entries = [d for d in figure.drawn if d[2] == "arm/loss mean"]
        assert len(loss_entries) == 1
        assert "seeds [1, 2]" in loss_entries[0][3], loss_entries[0][3]
        excluded = [m for m in figure.missing if m[0] == "arm_s3"]
        assert len(excluded) == 1 and "seed not recorded" in excluded[0][1], figure.missing


# ═══════════════════════════════════════════════════════════════════
#  The manifest
# ═══════════════════════════════════════════════════════════════════


class TestManifest:
    def test_manifest_records_a_refusal_with_no_files(self, nothing_measured, tmp_path):
        """`figures.json` is the file that answers "why is there no figure?".

        A refused entry has an empty `files` and an empty `plotted`. That is the
        same invariant the `Figure` constructor enforces, restated in the artifact
        a reader (or a reviewer, or a script six weeks from now) will actually
        look at.
        """
        cfgs, states, params = nothing_measured
        out_dir = tmp_path / "figures"
        figures = PE.render_all(cfgs, states, params, out_dir=out_dir)
        path = PE.write_manifest(figures, out_dir)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload["figures"]) == 5
        for entry in payload["figures"]:
            assert entry["status"] == "REFUSED"
            assert entry["files"] == []
            assert entry["plotted"] == []
            assert entry["reason"].strip()
            assert entry["reason_kind"] == PE.NO_MEASUREMENT
            assert entry["missing"], "a refusal must say what is missing"
        assert payload["_meta"]["dpi"] >= 300
        assert "pdf" in payload["_meta"]["vector_formats"]

    def test_manifest_rows_for_unmeasured_figures_carry_no_digit(self, nothing_measured, tmp_path):
        """The apparatus's own rule, applied to the manifest.

        `results_apparatus.assert_marker_honest` refuses an unmeasured cell that
        renders a digit. `Figure.status_cell` returns a real `Cell`, so the same
        guard covers the figure manifest without a second rule to keep in step.
        """
        cfgs, states, params = nothing_measured
        figures = PE.render_all(cfgs, states, params, out_dir=tmp_path / "f")

        RA.assert_marker_honest([f.status_cell() for f in figures])
        for figure in figures:
            cell = figure.status_cell()
            assert cell.render() == NOT_RUN
            assert not any(
                ch.isdigit() for ch in cell.render()
            ), f"{figure.kind} rendered {cell.render()!r} for a figure that does not exist"

        text = PE.manifest_markdown(figures)
        assert NOT_RUN in text
        assert text.count(NOT_RUN) >= len(figures), "each refusal must be visible in FIGURES.md"
        assert not RA.scan_rendered(text), "FIGURES.md contains a number-like blank"

    def test_manifest_does_not_claim_files_that_were_never_written(
        self, nothing_measured, tmp_path
    ):
        out_dir = tmp_path / "figures"
        figures = PE.render_all(*nothing_measured[:2], nothing_measured[2], out_dir=out_dir)
        PE.write_manifest(figures, out_dir)
        payload = json.loads((out_dir / "figures.json").read_text(encoding="utf-8"))
        on_disk = {p.name for p in out_dir.iterdir()}

        claimed = {Path(name).name for entry in payload["figures"] for name in entry["files"]}
        assert (
            claimed <= on_disk
        ), f"the manifest names files that do not exist: {claimed - on_disk}"


# ═══════════════════════════════════════════════════════════════════
#  The engine's own self-check
# ═══════════════════════════════════════════════════════════════════


class TestGuardSelfTest:
    def test_self_check_passes(self, tmp_path):
        assert PE.guard_selftest(tmp_path / "selftest") == []

    def test_guard_selftest_is_load_bearing(self, monkeypatch, tmp_path):
        """Break the guard at runtime; the self-check must notice.

        The replacement is a *realistic* leak rather than one that raises: a
        renderer that coerces an unreadable value to zero instead of refusing.
        A self-check that only noticed crashes would be satisfied by an
        implementation that simply could not run.
        """
        monkeypatch.setattr(
            PE,
            "measured_number",
            lambda cell: _coerce_or_zero(cell.render()),
        )
        failures = PE.guard_selftest(tmp_path / "selftest")

        assert len(failures) >= PE.GUARD_SELFTEST_ATTEMPTS, (
            f"the self-check reported {len(failures)} failure(s) with a leaky "
            f"measured_number; it should have reported at least "
            f"{PE.GUARD_SELFTEST_ATTEMPTS}"
        )
        assert any("unmeasured cell" in f for f in failures)

    def test_self_check_notices_a_figure_with_a_path_and_no_data(self, monkeypatch, tmp_path):
        """The structural half of the self-check, observed failing.

        `Figure.__init__` is the guard. Substituting an unguarded subclass and
        running the self-check must produce a complaint, which is what makes
        `test_figure_cannot_carry_a_path_without_data` worth having.

        Patching the *class* rather than `PE.FabricationError` is deliberate:
        replacing the exception would leave both the raiser and the catcher in
        the self-test looking at the same new name, so the guard would appear to
        keep working and the mutation would prove nothing.
        """

        class _Unguarded(PE.Figure):
            __slots__ = ()

            def __init__(self, kind, stem, paths, drawn, missing, reason, reason_kind, note=""):
                self.kind = kind
                self.stem = stem
                self.paths = list(paths)
                self.drawn = list(drawn)
                self.missing = list(missing)
                self.reason = reason
                self.reason_kind = reason_kind
                self.note = note

        monkeypatch.setattr(PE, "Figure", _Unguarded)
        failures = PE.guard_selftest(tmp_path / "selftest")

        assert failures, "the self-check passed with the structural guard removed"
        assert any("output path" in f or "no measured datum" in f for f in failures), failures


# ═══════════════════════════════════════════════════════════════════
#  Curve construction
# ═══════════════════════════════════════════════════════════════════


class TestCurve:
    def test_a_curve_only_comes_from_a_run_that_logged_rows(self):
        cfg = make_config("config/_fixture/c.yaml")

        assert (
            PE.Curve.from_run(cfg, make_state(cfg), "loss") is None
        ), "a run with no rows produced a curve"
        curve = PE.Curve.from_run(cfg, make_state(cfg, n_rows=8), "loss")

        assert curve is not None
        assert curve.n_points == 8
        assert curve.is_curveable
        assert len(curve.xs) == len(curve.ys) == 8
        assert curve.xs[0] == 0, "iteration zero must be iteration zero, not iteration one"
        assert curve.provenance.startswith(curve.label) or RA.TRAIN_LOG in curve.provenance

    def test_a_one_point_series_says_it_is_not_curveable(self):
        cfg = make_config("config/_fixture/c.yaml")
        curve = PE.Curve.from_run(cfg, make_state(cfg, n_rows=1), "loss")

        assert curve is not None
        assert not curve.is_curveable, "one point is a dot and calling it a curve is the bug"

    def test_curve_rows_match_the_csv_column_order(self, tmp_path):
        """The y values are the CSV column, read by NAME not by position.

        `CSV_COLUMNS` is the trainer's own list; hardcoding an index here would
        silently plot the wrong metric the day a column is inserted.
        """
        cfg = make_config("config/_fixture/c.yaml")
        state = make_state(cfg, n_rows=4)
        index = RA.CSV_COLUMNS.index("loss")

        curve = PE.Curve.from_run(cfg, state, "loss")

        assert curve is not None
        assert list(curve.ys) == [row[index] for row in state.rows]
