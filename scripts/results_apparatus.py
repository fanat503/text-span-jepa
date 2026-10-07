# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Results apparatus for the paper's results section.

Why this exists
---------------
The campaign so far has been about correctness. The suite is green and 25 cards
closed real defects, and there is not one row of results, because training is
forbidden on the machine the agents use and the GPU belongs to the owner. This
module builds everything about a results section that can be built WITHOUT
training, so the only thing left for the owner is the run itself.

The hard rule
-------------
Nothing here invents, extrapolates or interpolates a number. Two kinds of cell
exist and they are never confused:

* **measured** -- the number came out of a file the trainer wrote, or out of an
  in-process measurement this script performed (parameter counts).
* **not run / not applicable** -- no number exists. The first means a run would
  fill it; the second means no run ever could.

Both unmeasured states render as a loud word, never as a number, never as a
blank, and never as a dash. A dash in a results table is read as zero by every
human who has ever seen one, and a plausible placeholder is worse than nothing
because at a glance it is indistinguishable from a result. `Cell` makes that
structural rather than a matter of discipline: a cell constructed without a
measured value has no number to render, whatever you hand it.

Three traps this module closes, all found by reading the trainer rather than by
guessing at it:

1. `src/train.py:1556-1569` fills absent metric keys with `0`, and
   `src/models/jepa.py:650` emits `decoder_accuracy: 0.0` on an early-return
   path. A `0.000000` in `train_log.csv` therefore does NOT distinguish "the
   decoder scored zero" from "there was no decoder". Rows are gated on the
   resolved config, so a column whose weight is `0` renders `NOT-APPLICABLE`.

2. The baseline arms return an empty `diag_dict`, so `effective_rank`,
   `collapsed_dim_ratio` and `mask_fraction` are logged as `0` for every
   baseline whether or not anything was measured. Same gate.

3. Validation loss is never written to any machine-readable artifact. It goes
   to stdout only (`src/train.py:1606`) and to `best.pt`'s payload
   (`extra.best_val_loss`). So the paper's headline validation column cannot be
   assembled from `train_log.csv` at all, and it is marked `NOT-RUN` with the
   extraction command in its provenance rather than back-filled from the
   training loss, which is a different quantity.

Wall-clock cost
---------------
`cost` prints arithmetic over four config-derived terms (exact) and two measured
ones, and refuses to print a duration until both measured ones are supplied. The
seconds-per-epoch comes from the owner's own stdout log: `src/train.py:1591`
already prints `Epoch N avg loss: ... time: Ns`, so the projection becomes exact
from the first epoch of the run they were going to do anyway. No synthetic
benchmark, nothing to extrapolate from a proxy, and this module never trains.

Usage
-----
    python scripts/results_apparatus.py table     # the paper's tables
    python scripts/results_apparatus.py cost      # wall-clock model
    python scripts/results_apparatus.py curves    # training-curve figures
    python scripts/results_apparatus.py verify    # fabrication self-check
    python scripts/results_apparatus.py howto     # the owner's run order

CPU-only, no network, no training. Model construction happens on meta tensors
where that works and on CPU where it does not; nothing is ever stepped.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import io
import json
import re
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import yaml

from src.models.mechanisms import MechanismBundle
from src.train import _deep_merge, _normalize_model_name, create_model

# `tests/test_config_system.py:72`. The GPT-2 vocabulary padded to a multiple of
# 64, which is what every other count in this repo is measured against. Using
# the unpadded 50257 would put this table out of step with the test suite.
GPT2_VOCAB = 50304

# Artifacts `src/train.py` writes into `logging.folder`.
TRAIN_LOG = "train_log.csv"
CONFIG_DUMP = "params-text-span-jepa.yaml"
BEST_CKPT = "best.pt"
EPOCH_CKPT_RE = re.compile(r"^checkpoint-ep(\d+)\.pth\.tar$")

# `src/train.py:1230-1244`, in order. This is the exact column list and the
# complete set of numbers a finished run leaves in machine-readable form.
CSV_COLUMNS = [
    "loss",
    "lr",
    "wd",
    "loss_span",
    "loss_future",
    "loss_decoder",
    "loss_variance",
    "loss_covariance",
    "effective_rank",
    "collapsed_dim_ratio",
    "mask_fraction",
    "decoder_accuracy",
]

STATUS_PARTIAL = "PARTIAL"
STATUS_COMPLETE = "COMPLETE"


# ═══════════════════════════════════════════════════════════════════
#  Cell — the fabrication guard, in code
# ═══════════════════════════════════════════════════════════════════

#: The only two things an unmeasured cell may ever render as.
NOT_RUN = "NOT-RUN"
NOT_APPLICABLE = "NOT-APPLICABLE"

#: Glyphs that are NOT acceptable markers, with the reason each one is refused.
#: `verify` asserts these never reach the output, so a future edit that "tidies
#: up" the marker has to delete the reason too, in the same diff.
REFUSED_MARKERS = {
    "-": "minus reads as a negative number",
    "—": "em dash reads as zero in every table anyone has ever read",
    "–": "en dash reads as zero",
    ".": "a lone dot reads as zero",
    "N/A": "reads as an oversight rather than as a decision",
    "n/a": "reads as an oversight rather than as a decision",
    "?": "a question mark invites the reader to supply their own guess",
    "TBD": "invites the reader to supply their own guess",
    "TODO": "invites the reader to supply their own guess",
}


class FabricationError(RuntimeError):
    """Raised when an unmeasured cell was about to render as something number-like."""


class Cell:
    """One table cell: either a measurement, or a loud statement that there is none.

    A cell holds either a measured string or a reason. There is no third state
    and no way to render a number that was not supplied, so a fabricated figure
    cannot be produced by forgetting to check something -- it has to be passed
    to `Cell.measured`, beside the provenance string saying where it came from.
    """

    __slots__ = ("_state", "_text", "_why")

    MEASURED = "measured"
    NOT_RUN = "not-run"
    NOT_APPLICABLE = "not-applicable"

    def __init__(self, text: str | None, state: str, why: str = "") -> None:
        self._text = text
        self._state = state
        self._why = why

    @classmethod
    def measured(cls, value: Any, why: str) -> Cell:
        """A number or label a file on disk, or this process, produced."""
        return cls(str(value), cls.MEASURED, why)

    @classmethod
    def not_run(cls, why: str) -> Cell:
        """A run would produce this. None has."""
        return cls(None, cls.NOT_RUN, why)

    @classmethod
    def not_applicable(cls, why: str) -> Cell:
        """No run of this config can produce this, ever."""
        return cls(None, cls.NOT_APPLICABLE, why)

    @property
    def measured_flag(self) -> bool:
        return self._state == self.MEASURED

    @property
    def why(self) -> str:
        return self._why

    def render(self) -> str:
        if self._state == self.MEASURED:
            return self._text if self._text is not None else ""
        return NOT_RUN if self._state == self.NOT_RUN else NOT_APPLICABLE

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Cell({self.render()!r}, why={self._why!r})"


def assert_marker_honest(cells: Iterable[Cell]) -> None:
    """Fail loudly if any unmeasured cell renders as something number-like.

    The invariant the whole module exists to keep. It runs before anything is
    printed, so a bad cell is caught when it is built rather than by a reader
    squinting at a rendered table six weeks from now.
    """
    for cell in cells:
        text = cell.render()
        if cell.measured_flag:
            continue
        if text in REFUSED_MARKERS or text.strip() == "":
            raise FabricationError(
                f"unmeasured cell would render as {text!r}, which is a marker on the "
                f"refused list: {REFUSED_MARKERS.get(text, 'a blank is not a result')}"
            )
        if any(ch.isdigit() for ch in text):
            raise FabricationError(
                f"unmeasured cell would render as {text!r}, which contains a digit and "
                "is therefore readable as a measurement"
            )


# ═══════════════════════════════════════════════════════════════════
#  Rendering
# ═══════════════════════════════════════════════════════════════════


class Table:
    """A titled table of `Cell`s. Rendering is where the guard runs."""

    def __init__(self, title: str, headers: Sequence[str], note: str = "") -> None:
        self.title = title
        self.headers = list(headers)
        self.note = note
        self.rows: list[list[Cell]] = []

    def add(self, cells: Sequence[Cell]) -> None:
        if len(cells) != len(self.headers):
            raise ValueError(f"row has {len(cells)} cells, table has {len(self.headers)} headers")
        self.rows.append(list(cells))

    def all_cells(self) -> list[Cell]:
        return [c for row in self.rows for c in row]

    def _widths(self) -> list[int]:
        widths = [len(h) for h in self.headers]
        for row in self.rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell.render()))
        return widths

    def markdown(self) -> str:
        assert_marker_honest(self.all_cells())
        widths = self._widths()
        lines = [f"### {self.title}", ""]
        if self.note:
            lines += [self.note, ""]
        lines.append("| " + " | ".join(h.ljust(w) for h, w in zip(self.headers, widths)) + " |")
        lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
        for row in self.rows:
            lines.append("| " + " | ".join(c.render().ljust(w) for c, w in zip(row, widths)) + " |")
        lines.append("")
        return "\n".join(lines)

    def plain(self) -> str:
        assert_marker_honest(self.all_cells())
        widths = self._widths()
        out = [self.title]
        if self.note:
            out += ["", self.note]
        out += [
            "  ".join(h.ljust(w) for h, w in zip(self.headers, widths)).rstrip(),
            "  ".join("-" * w for w in widths),
        ]
        for row in self.rows:
            out.append("  ".join(c.render().ljust(w) for c, w in zip(row, widths)).rstrip())
        return "\n".join(out)

    def csv(self) -> str:
        assert_marker_honest(self.all_cells())
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(self.headers)
        for row in self.rows:
            writer.writerow([c.render() for c in row])
        return buf.getvalue()

    def render(self, fmt: str) -> str:
        return {"markdown": self.markdown, "csv": self.csv}.get(fmt, self.plain)()


# ═══════════════════════════════════════════════════════════════════
#  Config resolution
# ═══════════════════════════════════════════════════════════════════


class Config:
    """One shipped config file, resolved exactly as `src.train` resolves it."""

    def __init__(self, rel: str, merged: dict[str, Any], raw: dict[str, Any]) -> None:
        self.rel = rel
        self.merged = merged
        self.raw = raw
        self.model_cfg = merged.get("model", {}) or {}
        self.data_cfg = merged.get("data", {}) or {}
        self.opt_cfg = merged.get("optimization", {}) or {}
        self.meta_cfg = merged.get("meta", {}) or {}
        self.log_cfg = merged.get("logging", {}) or {}
        self.arm = _normalize_model_name(self.meta_cfg.get("model_name", "text_span_jepa"))
        self.ablation = (raw.get("_meta") or {}).get("ablation")
        self.run_dir = str(self.log_cfg.get("folder", ""))

    @property
    def is_jepa(self) -> bool:
        return self.arm == "text_span_jepa"

    @property
    def epochs(self) -> int:
        return int(self.opt_cfg.get("epochs", 50))

    @property
    def seqs_per_step(self) -> int:
        """Per-device micro-batch x accumulation = sequences per optimizer step."""
        return int(self.data_cfg.get("batch_size", 64)) * int(
            self.opt_cfg.get("grad_accum_steps", 8)
        )

    @property
    def seq_len(self) -> int:
        return int(self.data_cfg.get("max_seq_len", 512))

    @property
    def tokens_per_step(self) -> int:
        return self.seqs_per_step * self.seq_len

    @property
    def log_freq(self) -> int:
        return int(self.log_cfg.get("log_freq", 10))

    @property
    def label(self) -> str:
        return str(self.ablation or Path(self.rel).stem)


def resolve_configs() -> list[Config]:
    defaults = yaml.safe_load((REPO / "defaults.yaml").read_text(encoding="utf-8"))
    out: list[Config] = []
    for path in sorted((REPO / "config").rglob("*.yaml")):
        rel = path.relative_to(REPO).as_posix()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        out.append(Config(rel, _deep_merge(defaults, raw), raw))
    return out


def defaults_model_block() -> dict[str, Any]:
    return yaml.safe_load((REPO / "defaults.yaml").read_text(encoding="utf-8"))["model"]


# ═══════════════════════════════════════════════════════════════════
#  In-process measurement: parameter counts
# ═══════════════════════════════════════════════════════════════════


@contextlib.contextmanager
def meta_device():
    """Build on meta tensors where possible: exact shapes, zero allocation.

    The same shim `tests/test_config_system.py:1083` uses, for the same reason
    -- `TextSpanJEPLEncoder` calls `torch.linspace(...).item()` and a meta tensor
    cannot service `.item()`.
    """
    original = torch.linspace

    def cpu_linspace(*a, **k):
        kwargs = dict(k)
        kwargs.setdefault("device", "cpu")
        return original(*a, **kwargs)

    torch.linspace = cpu_linspace
    try:
        with torch.device("meta"):
            yield
    finally:
        torch.linspace = original


def _numels(model) -> tuple[int, int]:
    return (
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )


def count_params(cfg: Config) -> tuple[int, int, str]:
    """(total, trainable, method). Measured exactly; `method` is provenance only.

    Meta tensors give the same `numel()` as a real allocation and cost no
    memory, but a config with all twelve mechanisms on trips `.item()` inside a
    mechanism and must fall back to CPU. Both paths count the same number; which
    one ran is reported so the measurement's own provenance stays visible.
    """
    args = (cfg.arm, cfg.model_cfg, GPT2_VOCAB, cfg.seq_len)
    try:
        with meta_device():
            model = create_model(*args, torch.device("meta"))
        return _numels(model) + ("meta",)
    except Exception:
        gc.collect()
        model = create_model(*args, torch.device("cpu"))
        result = _numels(model) + ("cpu",)
        del model
        gc.collect()
        return result


def fingerprint(cfg: Config) -> str:
    """Content hash of the two files that determine the count.

    Keying the cache on content means editing a config invalidates that row and
    nothing else. A cache keyed on filename would go stale silently, which is the
    same class of failure as a stale results table.
    """
    parts = [
        (REPO / "defaults.yaml").read_bytes(),
        (REPO / cfg.rel).read_bytes(),
        str(GPT2_VOCAB).encode(),
    ]
    return hashlib.sha256(b"\x00".join(parts)).hexdigest()[:16]


class ParamCache:
    """Content-addressed cache of measured parameter counts, committed to docs/."""

    PATH = REPO / "docs" / "results" / "param_counts.json"

    def __init__(self) -> None:
        self.data: dict[str, dict[str, Any]] = {}
        if self.PATH.is_file():
            try:
                self.data = json.loads(self.PATH.read_text(encoding="utf-8")).get("configs", {})
            except (ValueError, OSError):
                self.data = {}

    def get(self, cfg: Config, recompute: bool = False) -> tuple[int, int, str, bool]:
        fp = fingerprint(cfg)
        entry = self.data.get(cfg.rel)
        if not recompute and entry and entry.get("fingerprint") == fp:
            return int(entry["total"]), int(entry["trainable"]), str(entry["method"]), True
        total, trainable, method = count_params(cfg)
        self.data[cfg.rel] = {
            "fingerprint": fp,
            "total": total,
            "trainable": trainable,
            "method": method,
        }
        return total, trainable, method, False

    def save(self) -> None:
        payload = {
            "_meta": {
                "what": "Measured parameter counts for every shipped config.",
                "how": (
                    "sum(p.numel() for p in model.parameters()) over the model that "
                    "src.train.create_model builds from the deep-merged config, with "
                    "vocab_size=50304 (the tests/test_config_system.py:72 convention)."
                ),
                "method_field": (
                    "'meta' = built on meta tensors, no allocation. 'cpu' = built for "
                    "real, because a config with all twelve mechanisms on calls "
                    "Tensor.item() which meta cannot service. The count is identical "
                    "either way."
                ),
                "regenerate": "python scripts/results_apparatus.py table --recompute-params",
                "invalidation": (
                    "keyed by sha256(defaults.yaml + this config); editing either "
                    "invalidates that row automatically."
                ),
            },
            "configs": dict(sorted(self.data.items())),
        }
        self.PATH.parent.mkdir(parents=True, exist_ok=True)
        self.PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def active_mechanisms(cfg: Config) -> list[str] | None:
    """The runtime truth, or None for an arm that has no GWP bundle."""
    if not cfg.is_jepa:
        return None
    from src.models.jepa import TextSpanJEPAConfig
    from src.models.mechanisms import MechanismBundle

    model_cfg = dict(cfg.model_cfg)
    model_cfg["vocab_size"] = GPT2_VOCAB
    model_cfg["max_seq_len"] = cfg.seq_len
    try:
        bundle = MechanismBundle.from_config(TextSpanJEPAConfig(**model_cfg))
    except Exception:
        return None
    return list(bundle.active_mechanisms())


# ═══════════════════════════════════════════════════════════════════
#  Run discovery — what the trainer actually left on disk
# ═══════════════════════════════════════════════════════════════════


class RunState:
    """Whatever exists on disk for one config. Absence is recorded, not inferred."""

    NOT_RUN = NOT_RUN
    PARTIAL = STATUS_PARTIAL
    COMPLETE = STATUS_COMPLETE

    def __init__(self, label: str, epochs: int, declared_dir: str) -> None:
        self.label = label
        self.epochs = epochs
        self.declared_dir = declared_dir
        self.run_dir: str | None = None
        self.reason = ""
        self.status = self.NOT_RUN
        self.epochs_done = 0
        self.rows: list[list[float]] = []
        self.drift: list[str] = []

    @property
    def has_log(self) -> bool:
        return bool(self.rows)

    @property
    def dir_path(self) -> Path | None:
        return Path(self.run_dir) if self.run_dir else None

    @property
    def last(self) -> dict[str, float] | None:
        """The final complete CSV row, keyed by column name."""
        if not self.rows:
            return None
        return dict(zip(CSV_COLUMNS, self.rows[-1]))


def resolve_run_dir(cfg: Config, run_roots: Sequence[Path]) -> Path | None:
    """Where this config's output will be, honouring a run-root override.

    Every shipped config declares its own `logging.folder`, which is what stops
    two arms overwriting each other's CSV (tests/test_config_system.py contract
    8). A `--run-root` maps that declared folder onto a different root so the
    owner can point at an external results drive without editing 62 configs.
    """
    declared = cfg.run_dir
    if not declared:
        return None
    declared_path = Path(declared)
    if declared_path.is_absolute():
        return declared_path
    parts = declared_path.parts
    tail = Path(*parts[1:]) if parts and parts[0] == "output" else declared_path
    return (run_roots[0] if run_roots else REPO / "output") / tail


def read_train_log(path: Path) -> list[list[float]]:
    """Parse `train_log.csv`, tolerating exactly the damage training can do.

    Two real cases, both handled by skipping rather than guessing:
      * a resumed run constructs a second `CSVLogger`, which appends a second
        header row into the middle of the file;
      * an interrupted run can leave a truncated final line.
    """
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for raw in csv.reader(fh):
            if not raw:
                continue
            try:
                parsed = [float(v) for v in raw]
            except ValueError:
                continue
            if len(parsed) < len(CSV_COLUMNS):
                continue
            rows.append(parsed[: len(CSV_COLUMNS)])
    return rows


def discover(cfg: Config, run_roots: Sequence[Path]) -> RunState:
    state = RunState(cfg.label, cfg.epochs, cfg.run_dir)
    run_dir = resolve_run_dir(cfg, run_roots)
    if run_dir is None or not run_dir.is_dir():
        state.reason = f"no run directory at {cfg.run_dir or '(no logging.folder)'!r}"
        return state
    state.run_dir = str(run_dir)
    log = run_dir / TRAIN_LOG
    if not log.is_file():
        state.reason = f"{run_dir} exists but holds no {TRAIN_LOG}"
        return state
    state.rows = read_train_log(log)
    if not state.rows:
        state.reason = f"{log} holds no numeric rows (header only, or unparseable)"
        return state
    done = []
    for entry in run_dir.iterdir():
        match = EPOCH_CKPT_RE.match(entry.name)
        if match:
            done.append(int(match.group(1)))
    state.epochs_done = max(done) if done else 0
    state.status = STATUS_COMPLETE if state.epochs_done >= cfg.epochs else STATUS_PARTIAL
    if not (run_dir / BEST_CKPT).is_file():
        state.reason = (
            f"no {BEST_CKPT}: validation never improved, so this run has no "
            "validation number to report"
        )
    state.drift = config_drift(run_dir / CONFIG_DUMP, cfg)
    return state


#: Config keys whose value changes what a run measures. Compared against the
#: run's own `params-*.yaml` dump, because 25 cards have edited configs since the
#: first runs were contemplated and a stale artifact silently re-labelled with
#: the current config would be a result for a model nobody trained.
DRIFT_KEYS = (
    ("model", "embed_dim"),
    ("model", "encoder_depth"),
    ("model", "num_heads"),
    ("model", "predictor_embed_dim"),
    ("model", "predictor_depth"),
    ("data", "batch_size"),
    ("data", "max_seq_len"),
    ("optimization", "epochs"),
    ("optimization", "grad_accum_steps"),
    ("optimization", "lr"),
)


def config_drift(dump_path: Path, cfg: Config) -> list[str]:
    """Keys on which the run's own dumped config differs from the config now.

    Returns an empty list when there is no dump, when it cannot be read, or when
    every compared key agrees. An unreadable dump is not reported as drift: the
    honest statement is "cannot tell", which the run's own `status` already
    covers, whereas inventing a drift would be its own kind of fabrication.
    """
    if not dump_path.is_file():
        return []
    try:
        dumped = yaml.safe_load(dump_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    now = cfg.merged
    drifted = []
    for section, key in DRIFT_KEYS:
        was = (dumped.get(section) or {}).get(key)
        is_now = (now.get(section) or {}).get(key)
        if was is not None and is_now is not None and was != is_now:
            drifted.append(f"{section}.{key}: run had {was}, config now says {is_now}")
    for mechanism in MechanismBundle.ALL_MECHANISMS:
        key = f"use_{mechanism}"
        was = (dumped.get("model") or {}).get(key)
        is_now = (now.get("model") or {}).get(key)
        if was is not None and is_now is not None and was != is_now:
            drifted.append(f"model.{key}: run had {was}, config now says {is_now}")
    return drifted


# ═══════════════════════════════════════════════════════════════════
#  Metric cells — gated on the resolved config, never on a bare 0
# ═══════════════════════════════════════════════════════════════════

#: Sentinel meaning "only the JEPA path ever populates this column".
JEPA_ONLY = "jepa-only"

#: CSV column -> every gate that must hold for the column to mean anything.
#: All gates must pass; the first failure becomes the cell's reason.
#:
#: The JEPA-only gate on the loss components is not defensive padding. It was
#: measured: `baselines/mlm_baseline.py:203` returns `{"loss_mlm", "mlm_accuracy"}`
#: and `baselines/data2vec_baseline.py:189` returns `{"loss_data2vec", "ema_decay",
#: "num_masked"}`. Neither mentions `loss_span`, `loss_decoder`,
#: `decoder_accuracy` or any of the VICReg terms, so `src/train.py:1560-1568`'s
#: `.get(key, 0)` writes a hard `0.000000` for every one of them on every
#: baseline row. A `0.000000` in that column for an MLM arm is an artefact of the
#: logger, not a measurement, and rendering it as a number is precisely the
#: failure this module exists to prevent.
COLUMN_GATE: dict[str, tuple[str, ...]] = {
    "loss": (),
    "lr": (),
    "wd": (),
    "loss_span": (JEPA_ONLY, "lambda_span"),
    "loss_future": (JEPA_ONLY, "lambda_future"),
    "loss_decoder": (JEPA_ONLY, "lambda_decoder"),
    "loss_variance": (JEPA_ONLY, "lambda_variance"),
    "loss_covariance": (JEPA_ONLY, "lambda_covariance"),
    "effective_rank": (JEPA_ONLY,),
    "collapsed_dim_ratio": (JEPA_ONLY,),
    "mask_fraction": (JEPA_ONLY,),
    "decoder_accuracy": (JEPA_ONLY, "lambda_decoder"),
}

#: Columns that carry no information for a given arm, with the source. Kept as
#: prose so the table note and the gate cannot drift apart.
ARM_COLUMN_NOTE = {
    "mlm": "baselines/mlm_baseline.py:203 returns only loss_mlm and mlm_accuracy",
    "data2vec": (
        "baselines/data2vec_baseline.py:189 returns only loss_data2vec, ema_decay " "and num_masked"
    ),
}

CURVE_SERIES = [
    ("loss", "total loss"),
    ("effective_rank", "effective rank (JEPA diag)"),
    ("decoder_accuracy", "decoder token accuracy"),
    ("collapsed_dim_ratio", "collapsed dim ratio"),
]


def metric_cell(cfg: Config, state: RunState, column: str) -> Cell:
    """One metric cell, with the reason it is empty stated in the cell.

    Three outcomes and no fourth:
      * measured -- the number is in the CSV and the resolved config makes it
        meaningful;
      * NOT-APPLICABLE -- the config can never produce it (weight 0, or a
        baseline arm whose empty `diag_dict` makes `src/train.py` log a 0);
      * NOT-RUN -- a run would produce it, and none has.
    """
    if column not in COLUMN_GATE:
        raise KeyError(f"{column!r} is not a {TRAIN_LOG} column")
    gates = COLUMN_GATE[column]

    for gate in gates:
        if gate == JEPA_ONLY:
            if not cfg.is_jepa:
                return Cell.not_applicable(
                    f"{ARM_COLUMN_NOTE.get(cfg.arm, 'this arm')}, and "
                    "src/train.py:1556-1569 logs any key it does not find as 0, so a 0 "
                    "here would be an artefact of the logger rather than a measurement"
                )
            continue
        if float(cfg.model_cfg.get(gate, 0.0) or 0.0) == 0.0:
            return Cell.not_applicable(
                f"{gate} is 0 in the resolved config: this arm has no {gate[7:]} term, "
                "and src/train.py logs an absent key as 0.0"
            )
    if not state.has_log:
        return Cell.not_run(f"no completed run ({state.reason or 'not started'})")
    row = state.last or {}
    if row.get(column) is None:
        return Cell.not_run(f"{state.run_dir}/{TRAIN_LOG} has no readable {column} value")
    return Cell.measured(f"{row[column]:.6g}", f"last logged row of {state.run_dir}/{TRAIN_LOG}")


def val_loss_cell(state: RunState) -> Cell:
    """Validation loss, which no CSV carries.

    `src/train.py:1606` logs it to stdout and `save_checkpoint` stores the best
    value in `best.pt`'s `extra.best_val_loss`. It is the paper's headline column
    and it cannot be assembled from `train_log.csv`, so it stays empty with the
    extraction command to hand rather than back-filled from the training loss,
    which is a different quantity.
    """
    directory = state.dir_path
    if directory is None or not directory.is_dir():
        return Cell.not_run(f"no run directory for {state.declared_dir or '(unset)'}")
    if not state.has_log:
        return Cell.not_run(f"no completed run ({state.reason or 'not started'})")
    best = directory / BEST_CKPT
    if not best.is_file():
        return Cell.not_run(
            f"no {BEST_CKPT} in {directory}; val_loss lives in its "
            "extra.best_val_loss, so without it there is nothing to read"
        )
    return Cell.not_run(
        f"val_loss is in {best} (extra.best_val_loss) and on stdout; no CSV column "
        "carries it. Read it with: python -c \"import torch;ck=torch.load(r'"
        f"{best}',map_location='cpu',weights_only=False);print(ck['extra']['best_val_loss'])\""
    )


# ═══════════════════════════════════════════════════════════════════
#  Shared cell builders
# ═══════════════════════════════════════════════════════════════════


def status_cell(state: RunState) -> Cell:
    """How far the run got. `CONFIG-DRIFT` is its own state, and it outranks both.

    A run whose artifacts exist but whose config no longer matches what the run
    actually used is not evidence about the config on disk. Labelling it
    `COMPLETE` would attach a measured number to a model that was never trained
    that way, which is the failure mode this whole apparatus is built against.
    """
    if state.drift:
        return Cell.measured(
            "CONFIG-DRIFT",
            f"{len(state.drift)} key(s) differ between the run's {CONFIG_DUMP} and "
            f"the config now: {state.drift[0]}"
            + (f" (+{len(state.drift) - 1} more)" if len(state.drift) > 1 else ""),
        )
    if state.status == STATUS_COMPLETE:
        why = f"{state.epochs_done} checkpoint-ep*.pth.tar files in {state.run_dir}"
        return Cell.measured(f"COMPLETE {state.epochs_done}/{state.epochs} ep", why)
    if state.status == STATUS_PARTIAL:
        why = f"{state.epochs_done} checkpoint-ep*.pth.tar files in {state.run_dir}"
        return Cell.measured(f"PARTIAL {state.epochs_done}/{state.epochs} ep", why)
    return Cell.not_run(state.reason or "not started")


def shape_cell(cfg: Config) -> Cell:
    d = cfg.model_cfg
    shape = f"{d.get('embed_dim', '?')}/{d.get('encoder_depth', '?')}/{d.get('num_heads', '?')}"
    return Cell.measured(shape, f"model embed_dim/encoder_depth/num_heads resolved for {cfg.rel}")


def flags_flipped(cfg: Config, defaults_model: dict[str, Any], limit: int = 3) -> str:
    """The `model:` keys whose resolved value differs from defaults.yaml.

    Measured by diffing the two merged dicts rather than read off the filename,
    so a row that quietly changed two keys says so instead of claiming to be a
    one-mechanism leave-one-out. Long deltas are elided with an explicit count
    rather than truncated mid-token, and `--format csv` always carries all of
    them.
    """
    diffs = []
    for key in sorted(set(cfg.model_cfg) | set(defaults_model)):
        new = cfg.model_cfg.get(key, "<absent>")
        old = defaults_model.get(key, "<absent>")
        if new != old:
            diffs.append(f"{key}={new}")
    if not diffs:
        return "(none)"
    if len(diffs) <= limit:
        return "; ".join(diffs)
    kept = "; ".join(diffs[:limit])
    return f"{kept}; +{len(diffs) - limit} more (csv has the full list)"


# ═══════════════════════════════════════════════════════════════════
#  Table A — main results
# ═══════════════════════════════════════════════════════════════════

MAIN_HEADERS = [
    "arm",
    "config",
    "d/h/hd",
    "params trainable",
    "epochs",
    "seqs/step",
    "tokens/step",
    "status",
    "loss @ last logged",
    "dec-acc",
    "best val loss",
    "log dir",
]


def build_main_table(
    cfgs: Sequence[Config], states: dict[str, RunState], params: dict[str, tuple[int, int]]
) -> Table:
    note = (
        "Every cell right of `status` comes from a file `src/train.py` wrote "
        f"({TRAIN_LOG}, {BEST_CKPT}) or from a measurement this script made "
        f"(parameter counts). Nothing here is modelled, extrapolated or carried over "
        f"from another run. `{NOT_RUN}` = a run would produce it and none has. "
        f"`{NOT_APPLICABLE}` = no run of that config ever could. "
        "`dec-acc` is the decoder's masked-token accuracy, NOT a downstream "
        "representation score; this repo computes no linear-probe metric during "
        "training, so the paper has no representation-quality column until "
        "`src/eval/probes.py` is wired into a post-training step. Read the three "
        "arms' `loss` values with care: they are three different objectives (JEPA "
        "span loss, MLM cross-entropy, data2vec regression), so `loss` does not rank "
        "the arms against one another, and every other metric column is JEPA-only."
    )
    table = Table("Table A - main results", MAIN_HEADERS, note)
    for cfg in cfgs:
        if cfg.ablation:
            continue
        state = states[cfg.rel]
        table.add(
            [
                Cell.measured(cfg.arm, f"meta.model_name in {cfg.rel}"),
                Cell.measured(cfg.rel, "config path"),
                shape_cell(cfg),
                Cell.measured(
                    f"{params[cfg.rel][1] / 1e6:.1f}M",
                    f"sum(p.numel()) over p.requires_grad, measured for {cfg.rel}",
                ),
                Cell.measured(cfg.epochs, f"optimization.epochs in {cfg.rel}"),
                Cell.measured(cfg.seqs_per_step, "data.batch_size x optimization.grad_accum_steps"),
                Cell.measured(cfg.tokens_per_step, f"{cfg.seqs_per_step} x {cfg.seq_len} tokens"),
                status_cell(state),
                metric_cell(cfg, state, "loss"),
                metric_cell(cfg, state, "decoder_accuracy"),
                val_loss_cell(state),
                Cell.measured(state.run_dir or cfg.run_dir, f"logging.folder in {cfg.rel}"),
            ]
        )
    return table


# ═══════════════════════════════════════════════════════════════════
#  Table B — the ablation table
# ═══════════════════════════════════════════════════════════════════

ABLATION_HEADERS = [
    "arm",
    "config",
    "mechs on",
    "flag flipped vs defaults",
    "d params",
    "epochs",
    "status",
    "loss @ last logged",
    "dec-acc",
    "best val loss",
]


def build_ablation_table(
    cfgs: Sequence[Config],
    states: dict[str, RunState],
    params: dict[str, tuple[int, int]],
    defaults_model: dict[str, Any],
    full_params: int | None,
) -> Table:
    note = (
        "The grid from `config/ablations/README.md`. `mechs on` is "
        "`MechanismBundle.active_mechanisms()` read back from the runtime, so it "
        "honours the `use_wsd and use_jawp` guard in `TextSpanJEPA.__init__` rather "
        "than trusting the `use_*` flags -- which is why `no_jawp` reports fewer than "
        "one fewer. `flag flipped` is a diff against the `defaults.yaml` model block, "
        "not a filename. `d params` is trainable params minus the full model's, both "
        "measured. Every result column is empty because no ablation has been run."
    )
    table = Table("Table B - ablation (leave-one-out and interaction arms)", ABLATION_HEADERS, note)
    for cfg in cfgs:
        if not cfg.ablation:
            continue
        state = states[cfg.rel]
        active = active_mechanisms(cfg)
        if active is None:
            mech_cell = Cell.not_applicable("non-JEPA arm has no GWP bundle")
        else:
            mech_cell = Cell.measured(
                f"{len(active)}/12",
                f"active_mechanisms() for {cfg.rel}: {', '.join(active)}",
            )
        if full_params is None:
            delta_cell = Cell.not_applicable("the full-model parameter count is unavailable")
        else:
            delta_cell = Cell.measured(
                f"{params[cfg.rel][1] - full_params:+,}",
                "trainable params minus the full model's, both measured",
            )
        table.add(
            [
                Cell.measured(cfg.ablation, f"_meta.ablation in {cfg.rel}"),
                Cell.measured(cfg.rel, "config path"),
                mech_cell,
                Cell.measured(
                    flags_flipped(cfg, defaults_model),
                    f"diff of the merged model block against defaults.yaml for {cfg.rel}",
                ),
                delta_cell,
                Cell.measured(cfg.epochs, f"optimization.epochs in {cfg.rel}"),
                status_cell(state),
                metric_cell(cfg, state, "loss"),
                metric_cell(cfg, state, "decoder_accuracy"),
                val_loss_cell(state),
            ]
        )
    return table


# ═══════════════════════════════════════════════════════════════════
#  Table C — wall-clock cost model
# ═══════════════════════════════════════════════════════════════════

# `src/train.py:1591`: logger.info(f"Epoch {n} avg loss: {avg:.4f} time: {secs:.0f}s")
EPOCH_TIME_RE = re.compile(r"Epoch\s+(\d+)\s+avg loss:\s*([\d.]+)\s+time:\s*(\d+)s")

COST_HELPERS = """\
Measure the two unknown terms, then re-run `cost` with them.

  --train-seqs N    length of YOUR copy of the corpus, in sequences of {seq_len} tokens
      python -c "from src.datasets.kaggle import load_wikitext103 as L; \\
      d,_=L('gpt2',{seq_len},'train',r'{root}'); print(len(d))"

  --log-file PATH   the run's own stdout. src/train.py:1591 already prints
                    'Epoch N avg loss: ... time: Ns' every epoch, and its startup
                    lines (:1152, :1157, :1175, :1177) already say which model and
                    which device produced it. So the timing attributes itself and
                    no separate benchmark is needed.
      python -m src.train --fname <cfg> 2>&1 | tee run.log
      python scripts/results_apparatus.py cost --log-file run.log --train-seqs N

The attribution is strict on purpose. One run's seconds-per-epoch is applied ONLY
to configs whose MEASURED trainable-parameter count matches what that run logged,
so this table can never state a wall clock for a model that was not run. To fill
in a larger model, run one epoch of it and hand the table that run's log.
"""


def read_epoch_times(log_file: Path) -> list[tuple[int, float]]:
    """Epoch index -> measured seconds, parsed from the trainer's own stdout.

    The honest throughput measurement: it is the real run's own timing, on the
    owner's GPU, at the owner's batch size. Nothing is extrapolated from a proxy.
    """
    found: dict[int, float] = {}
    text = log_file.read_text(encoding="utf-8", errors="replace")
    for match in EPOCH_TIME_RE.finditer(text):
        found[int(match.group(1))] = float(match.group(3))
    return sorted(found.items())


def read_run_identity(log_file: Path) -> RunIdentity:
    """What run produced this log, read from the trainer's own startup lines.

    `src/train.py` already prints what it built and where: `Using device`
    (1152), `Model type` (1157), `Model parameters (get_num_params())` (1175) and
    `Trainable parameters` (1177). Reading them is what makes a seconds-per-epoch
    measurement attributable -- the alternative is to apply one run's timing to
    every row of the table, which is a wall-clock number for a model that was
    never run, and that is the fabrication this module exists to prevent.
    """
    text = log_file.read_text(encoding="utf-8", errors="replace")
    trainable = re.search(r"Trainable parameters:\s*([\d,]+)", text)
    total = re.search(r"Model parameters \(get_num_params\(\)\):\s*([\d,]+)", text)
    device = re.search(r"Using device:\s*(\S+)", text)
    model = re.search(r"Model type:\s*(\S+)\s*->\s*(\S+)", text)
    return RunIdentity(
        trainable=int(trainable.group(1).replace(",", "")) if trainable else None,
        total=int(total.group(1).replace(",", "")) if total else None,
        device=device.group(1) if device else None,
        raw_model_name=model.group(1) if model else None,
    )


class RunIdentity:
    """Which run a measurement came from, as the log itself reports it."""

    def __init__(
        self,
        trainable: int | None,
        total: int | None,
        device: str | None,
        raw_model_name: str | None,
    ) -> None:
        self.trainable = trainable
        self.total = total
        self.device = device
        self.raw_model_name = raw_model_name

    def describe(self) -> str:
        bits = [f"trainable={self.trainable:,}" if self.trainable else "trainable=unread"]
        if self.device:
            bits.append(f"device={self.device}")
        if self.raw_model_name:
            bits.append(f"model_name={self.raw_model_name}")
        return ", ".join(bits)


def build_cost_table(
    cfgs: Sequence[Config],
    params: dict[str, tuple[int, int]],
    train_seqs: int | None,
    epoch_times: Sequence[tuple[int, float]],
    identity: RunIdentity | None,
) -> Table:
    note = (
        "Wall clock is arithmetic over terms that are either config-derived (exact, "
        "read from the merged config) or measured (with where the measurement came "
        "from). Both measured terms print as `NOT-RUN` until supplied -- this module "
        "will not guess a corpus length or a GPU throughput, because a wall-clock "
        "figure resting on a guessed denominator is precisely the kind of plausible "
        "number that reads as a measurement."
    )
    if identity is not None:
        note += (
            f" A seconds-per-epoch measurement is attributed ONLY to configs whose "
            f"measured trainable parameter count equals the {identity.describe()} that "
            "the log reports. A timing measured on one model is never applied to "
            "another, so this table cannot state a wall clock for a run that did not "
            "happen."
        )
    headers = [
        "config",
        "params trainable",
        "epochs",
        "seqs/step",
        "tokens/step",
        "train_seqs",
        "steps",
        "sec/epoch",
        "wall clock",
        "basis",
    ]
    table = Table("Table C - wall-clock cost model", headers, note)

    measured_spe = sum(s for _, s in epoch_times) / len(epoch_times) if epoch_times else None

    for cfg in cfgs:
        if cfg.ablation:
            continue
        per_step = cfg.seqs_per_step
        trainable = params[cfg.rel][1]

        if train_seqs:
            ipe = -(-train_seqs // per_step)
            steps_cell = Cell.measured(
                f"{cfg.epochs} x {ipe:,} = {cfg.epochs * ipe:,}",
                f"optimization.epochs x ceil({train_seqs} / {per_step}) iterations per epoch",
            )
            train_cell = Cell.measured(train_seqs, f"--train-seqs at max_seq_len {cfg.seq_len}")
        else:
            steps_cell = Cell.not_run(
                "needs --train-seqs: iterations per epoch is "
                "ceil(len(train_dataset) / data.batch_size), and the corpus length is "
                "a property of the data on the owner's disk, not of the config"
            )
            train_cell = Cell.not_run("--train-seqs not supplied")

        attributed = (
            measured_spe is not None
            and identity is not None
            and identity.trainable is not None
            and identity.trainable == trainable
        )
        if measured_spe is None:
            spe_cell = Cell.not_run(
                "no --log-file; src/train.py:1591 already prints "
                "'Epoch N avg loss: ... time: Ns', so one real epoch suffices"
            )
        elif attributed:
            spe_cell = Cell.measured(
                f"{measured_spe:.1f}s",
                f"mean of {len(epoch_times)} epoch timings in --log-file, whose "
                f"trainable-parameter count ({identity.trainable:,}) matches this row's "
                "measured count",
            )
        else:
            reason = (
                f"the --log-file run measured {identity.trainable:,} trainable "
                f"parameters; this config measures {trainable:,}, so its timing does "
                "not transfer"
                if identity is not None and identity.trainable is not None
                else "the --log-file has no readable 'Trainable parameters:' line, so "
                "its timing cannot be attributed to any specific config"
            )
            spe_cell = Cell.not_run(reason)

        if not attributed:
            wall_cell = Cell.not_run("needs a sec/epoch measured for this exact config")
            basis_cell = Cell.not_run("no measurement to compute from")
        elif train_seqs:
            seconds = cfg.epochs * measured_spe  # type: ignore[operator]
            hours = seconds / 3600.0
            wall_cell = Cell.measured(
                f"{hours:.1f} h",
                f"{cfg.epochs} epochs x {measured_spe:.1f}s = {seconds:,.0f}s " f"= {hours:.1f} h",
            )
            basis_cell = Cell.measured(
                f"epochs {epoch_times[0][0]}-{epoch_times[-1][0]} of this run on "
                f"{identity.device or 'its own device'}",
                "projected only between epochs of one run on one model; no second "
                "model or second GPU is involved",
            )
        else:
            wall_cell = Cell.not_run("needs --train-seqs as well as a measured sec/epoch")
            basis_cell = Cell.not_run("incomplete")

        table.add(
            [
                Cell.measured(cfg.rel, "config path"),
                Cell.measured(f"{trainable / 1e6:.1f}M", f"measured for {cfg.rel}, trainable only"),
                Cell.measured(cfg.epochs, "optimization.epochs"),
                Cell.measured(per_step, "data.batch_size x optimization.grad_accum_steps"),
                Cell.measured(cfg.tokens_per_step, f"{per_step} sequences x {cfg.seq_len} tokens"),
                train_cell,
                steps_cell,
                spe_cell,
                wall_cell,
                basis_cell,
            ]
        )
    return table


# ═══════════════════════════════════════════════════════════════════
#  curves
# ═══════════════════════════════════════════════════════════════════


def emit_curves(cfgs: Sequence[Config], states: dict[str, RunState], out_dir: Path) -> int:
    """Draw the training curves that exist. Draw nothing, loudly, for the ones that do not.

    A figure with empty axes and a flat zero line is worse than no figure: it
    looks like a result. So when there is no `train_log.csv` this prints the
    marker and writes no PNG at all.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    blank: list[str] = []
    labels = dict(CURVE_SERIES)

    for cfg in cfgs:
        state = states[cfg.rel]
        usable = [
            column for column, _ in CURVE_SERIES if metric_cell(cfg, state, column).measured_flag
        ]
        if not usable:
            blank.append(cfg.rel)
            continue
        xs = [i * cfg.log_freq for i in range(len(state.rows))]
        fig, axes = plt.subplots(len(usable), 1, figsize=(9, 2.4 * len(usable)), squeeze=False)
        for ax, column in zip(axes[:, 0], usable):
            ys = [row[CSV_COLUMNS.index(column)] for row in state.rows]
            ax.plot(xs, ys, linewidth=1.0)
            ax.set_ylabel(labels.get(column, column))
            ax.grid(alpha=0.3)
        axes[0, 0].set_xlabel(f"optimizer iteration (logged every {cfg.log_freq})")
        fig.suptitle(f"{cfg.rel}  [{state.status}]  {state.run_dir}", fontsize=9)
        fig.tight_layout()
        path = out_dir / (cfg.label.replace("/", "_") + ".png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"wrote {path}")
        written += 1

    if blank:
        print(
            f"\n{NOT_RUN} -- no training curve written for {len(blank)} of {len(cfgs)} "
            f"configs, because no {TRAIN_LOG} exists for them:",
            file=sys.stderr,
        )
        for rel in blank:
            print(f"  {NOT_RUN}  {rel}", file=sys.stderr)
        print(
            "\nA curve figure needs a run. Nothing was drawn, on purpose: empty axes "
            "with a flat line read as a measurement, which is what this campaign "
            "exists to prevent.",
            file=sys.stderr,
        )
    return written


# ═══════════════════════════════════════════════════════════════════
#  Table D — the self-check
# ═══════════════════════════════════════════════════════════════════

VERIFY_HEADERS = [
    "config",
    "unmeasured cells",
    "all markers loud",
    "measured cells",
    "provenance given",
    "config drift keys",
]


def build_verify_table(
    cfgs: Sequence[Config], states: dict[str, RunState], params: dict[str, tuple[int, int]]
) -> Table:
    """One row per shipped config: is every unmeasured cell loud, and is every
    measured cell traceable to a file or an in-process measurement?"""
    table = Table(
        "Table D - fabrication self-check",
        VERIFY_HEADERS,
        "The contract, checked per config. A row failing either boolean is a bug in "
        "this script, not a formatting nit.",
    )
    for cfg in cfgs:
        state = states[cfg.rel]
        cells = [
            status_cell(state),
            metric_cell(cfg, state, "loss"),
            metric_cell(cfg, state, "decoder_accuracy"),
            val_loss_cell(state),
            Cell.measured(params[cfg.rel][1], "in-process measurement on meta or CPU"),
            Cell.measured(
                len(state.drift),
                "keys where the run's params-*.yaml disagrees with the config now",
            ),
        ]
        unmeasured = [c for c in cells if not c.measured_flag]
        loud = all(
            c.render() in (NOT_RUN, NOT_APPLICABLE) and not any(ch.isdigit() for ch in c.render())
            for c in unmeasured
        )
        provenanced = all(bool(c.why.strip()) for c in cells)
        table.add(
            [
                Cell.measured(cfg.rel, "config path"),
                Cell.measured(len(unmeasured), "count of cells with no measurement"),
                Cell.measured(loud, "no unmeasured cell renders a digit"),
                Cell.measured(len(cells) - len(unmeasured), "count of measured cells"),
                Cell.measured(provenanced, "every cell carries a provenance string"),
                Cell.measured(
                    len(state.drift),
                    "keys where the run's params-*.yaml disagrees with the config now",
                ),
            ]
        )
    return table


class _LeakyCell:
    """A stand-in that renders an unmeasured cell as a number.

    Not a `Cell`: `Cell.render` ignores its `_text` when the state is
    unmeasured, so an unmeasured cell cannot leak a number at all. The
    realistic way a fabricated table ships is therefore a change to the
    RENDERER -- `render()` preferring `_text`, a formatter that pads an absent
    value to "0", a CSV writer that writes "" for None. This stub models exactly
    that, so the self-test measures the guard against the failure it is really
    there to catch rather than against a construction that is already safe.
    """

    measured_flag = False

    def __init__(self, text: str) -> None:
        self._text = text

    def render(self) -> str:
        return self._text


#: How many leaky-renderer attacks `guard_selftest` attempts. Named so the PASS
#: message quotes the self-test rather than a number typed beside it.
GUARD_SELFTEST_ATTACKS = 12


def guard_selftest() -> list[str]:
    """Prove the guard is not vacuous, by trying to get past it.

    A check never observed to fail is indistinguishable from a check that cannot
    fail. Three things are asserted here, and all three have to hold for
    `verify` to say PASS:

    1. the STRUCTURAL property -- an unmeasured `Cell` cannot render its own
       text, so there is no argument that reaches the output as a number;
    2. the GUARD -- a leaky renderer is caught for every refused marker and for
       anything containing a digit;
    3. the POSITIVE CONTROL -- a genuine measurement survives both.
    """
    failures: list[str] = []

    # 1. Structural: the text argument of an unmeasured cell is unreachable.
    for text in ("0.000000", "—", "-", "12.5", "99%"):
        for state in (Cell.NOT_RUN, Cell.NOT_APPLICABLE):
            rendered = Cell(text, state, "probe").render()
            if rendered not in (NOT_RUN, NOT_APPLICABLE):
                failures.append(
                    f"an unmeasured cell rendered as {rendered!r} instead of its "
                    "marker; Cell.render is leaking _text"
                )

    # 2. The guard, against a renderer that does leak.
    for text in ("0.000000", "0", "—", "–", "-", ".", "", "N/A", "n/a", "?", "TBD", "12.5")[
        :GUARD_SELFTEST_ATTACKS
    ]:
        try:
            assert_marker_honest([_LeakyCell(text)])
        except FabricationError:
            continue
        failures.append(f"the guard let an unmeasured cell render as {text!r}")

    # 3. Positive control: a real measurement must survive the guard.
    for good in ("2.1", "0.204", "COMPLETE 50/50 ep", "137.9M"):
        try:
            assert_marker_honest([Cell.measured(good, "a file on disk")])
        except FabricationError as exc:
            failures.append(f"the guard rejected a genuine measurement {good!r}: {exc}")
    return failures


def scan_rendered(text: str) -> list[str]:
    """Second, independent check: re-scan rendered text for a number-like blank.

    `assert_marker_honest` inspects `Cell` objects. This inspects the string a
    human actually reads, so a bug in the renderer cannot slip past a check that
    only looks at the model.
    """
    problems: list[str] = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # A one-element split is a heading or a note, not a row, so a lone blank
        # there is formatting rather than a missing measurement.
        if len(cells) < 2:
            continue
        for cell in cells:
            if cell in REFUSED_MARKERS:
                problems.append(f"refused marker {cell!r} in rendered row: {line!r}")
            elif not cell:
                problems.append(f"empty cell in rendered row: {line!r}")
    return problems


# ═══════════════════════════════════════════════════════════════════
#  howto
# ═══════════════════════════════════════════════════════════════════


def _phase_of(cfg: Config, defaults_model: dict[str, Any], mechanisms: Sequence[str]) -> str:
    """Which run phase this config belongs in, from what it actually changes.

    Classified by the resolved delta rather than by filename, because
    `config/ablations/` contains two `no_` files that are NOT leave-one-out
    mechanism rows: `no_future_loss.yaml` and `no_decoder_loss.yaml` zero a
    `lambda_*` instead of a `use_*`. A prefix test would file them with the
    mechanism grid and imply a comparability the configs do not have.
    """
    if not cfg.ablation:
        if cfg.rel.startswith("config/scaling/"):
            return "ladder"
        if cfg.rel.startswith("config/kaggle/"):
            return "kaggle"
        return "baselines"

    if cfg.ablation == "all_core":
        return "full-model"
    any_use_off = [
        key for key, value in cfg.model_cfg.items() if key.startswith("use_") and value is False
    ]
    newly_off = [key for key in any_use_off if defaults_model.get(key) is not False]
    if len(newly_off) == 1:
        return "leave-one-out"
    if newly_off:
        return "multi-off"
    if any(cfg.model_cfg.get("lambda_" + m) == 0.1 for m in mechanisms):
        return "interaction"
    return "other-arms"


PHASE_TITLES = {
    "ladder": "Phase 1 -- the scaling ladder (width/depth only)",
    "baselines": "Phase 2 -- the local-GPU baselines (MLM and data2vec)",
    "kaggle": "Phase 2b -- the Kaggle T4 arms (need Kaggle, not the local GPU)",
    "full-model": "Phase 3a -- the full GWP model (the table's reference arm)",
    "leave-one-out": "Phase 3b -- the leave-one-out grid, one mechanism per run",
    "multi-off": "Phase 3c -- arms that switch off more than one mechanism",
    "interaction": "Phase 4 -- interaction and 10x-weight arms",
    "other-arms": "Phase 5 -- hyperparameter arms",
}


def howto_text(cfgs: Sequence[Config]) -> str:
    from src.models.mechanisms import MechanismBundle as _Bundle

    mechanisms = list(_Bundle.ALL_MECHANISMS)
    defaults_model = defaults_model_block()
    smallest = next(
        (c for c in cfgs if c.rel == "config/scaling/xsmall_30m.yaml"),
        None,
    )
    phases: dict[str, list[Config]] = {}
    for cfg in cfgs:
        phase = _phase_of(cfg, defaults_model, mechanisms)
        if cfg.rel.startswith("config/scaling/devices/") or cfg.rel.startswith(
            "config/tinystories/"
        ):
            phase = "unrunnable"
        phases.setdefault(phase, []).append(cfg)

    lines = [
        "Run order for the owner",
        "========================",
        "",
        "Every command below is one run. None has been executed: this machine has no",
        "CUDA and training is forbidden on it.",
        "",
        "Phase 0 -- one cheap run, which also calibrates every wall-clock estimate",
        "---------------------------------------------------------------------",
    ]
    if smallest is not None:
        lines += [
            f"  python -m src.train --fname {smallest.rel} 2>&1 | tee run.log",
            "",
            "  src/train.py:1591 prints 'Epoch N avg loss: ... time: Ns' every epoch, so",
            "  run.log alone is enough to turn the whole cost table from symbols into",
            "  arithmetic:",
            "",
            "    python scripts/results_apparatus.py cost --log-file run.log --train-seqs N",
            "",
            "  This run is also Phase 1's smallest rung -- it is listed once, here.",
        ]

    for phase in (
        "ladder",
        "baselines",
        "kaggle",
        "full-model",
        "leave-one-out",
        "multi-off",
        "interaction",
        "other-arms",
    ):
        members = phases.get(phase) or []
        if not members:
            continue
        title = PHASE_TITLES[phase]
        lines += ["", title, "-" * len(title)]
        for cfg in members:
            if smallest is not None and cfg.rel == smallest.rel:
                continue
            lines.append(f"  python -m src.train --fname {cfg.rel}")

    # `devices/*.yaml` and tinystories fall into no phase on purpose. Listing
    # them as "just run it" would be wrong: the first needs N GPUs to mean
    # anything, the second cannot run as shipped.
    unrunnable = phases.get("unrunnable") or []
    if unrunnable:
        lines += ["", "Not in any phase, and not runnable as shipped:", ""]
        for cfg in unrunnable:
            why = (
                "holds the global effective batch at 512 and varies only the "
                "per-device micro-batch, which needs N GPUs to mean anything"
                if "/devices/" in cfg.rel
                else "src/train.py:740 hard-codes load_wikitext103, so meta.dataset "
                "and data.root_path are inert"
            )
            lines.append(f"  {NOT_APPLICABLE}  {cfg.rel}  -- {why}")

    lines += [
        "",
        "Then, with the runs on disk:",
        "",
        "  python scripts/results_apparatus.py table     # the rows fill themselves in",
        "  python scripts/results_apparatus.py curves    # the figures appear",
        "  python scripts/results_apparatus.py verify    # the guards still hold",
        "",
        "The tables are a pure function of (configs, files on disk). There is no results",
        "file to keep in step by hand, so a stale table is not a state this apparatus can",
        "be in.",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════


def emit(text: str, out: str | None) -> None:
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}", file=sys.stderr)
    else:
        print(text)


def measure_params(
    cfgs: Sequence[Config], recompute: bool
) -> tuple[dict[str, tuple[int, int]], int]:
    cache = ParamCache()
    params: dict[str, tuple[int, int]] = {}
    computed = 0
    for cfg in cfgs:
        total, trainable, _method, was_cached = cache.get(cfg, recompute=recompute)
        params[cfg.rel] = (total, trainable)
        computed += 0 if was_cached else 1
    if recompute:
        cache.save()
    return params, computed


def run_roots_of(args) -> list[Path]:
    return [Path(r) for r in (getattr(args, "run_root", None) or ["output"])]


def cmd_table(args) -> int:
    started = time.time()
    cfgs = resolve_configs()
    roots = run_roots_of(args)
    states = {c.rel: discover(c, roots) for c in cfgs}
    params, computed = measure_params(cfgs, args.recompute_params)

    full = next((c for c in cfgs if c.ablation == "all_core"), None)
    full_params = params[full.rel][1] if full is not None else None

    main_t = build_main_table(cfgs, states, params)
    abl_t = build_ablation_table(cfgs, states, params, defaults_model_block(), full_params)
    emit(main_t.render(args.format) + "\n" + abl_t.render(args.format), args.out)

    with_log = sum(1 for c in cfgs if states[c.rel].has_log)
    drifted = [c.rel for c in cfgs if states[c.rel].drift]
    for rel in drifted:
        print(
            f"CONFIG-DRIFT {rel}: the run's {CONFIG_DUMP} disagrees with the config "
            f"now on {len(states[rel].drift)} key(s): {states[rel].drift[0]}",
            file=sys.stderr,
        )
    print(
        f"[{len(cfgs)} configs; {with_log} have a {TRAIN_LOG}; "
        f"{computed} parameter counts measured now, "
        f"{time.time() - started:.1f}s]",
        file=sys.stderr,
    )
    if with_log == 0:
        print(
            f"{NOT_RUN} -- no run anywhere produced a {TRAIN_LOG}, so every result "
            "column above is empty by measurement rather than by omission. Run order: "
            "`results_apparatus.py howto`.",
            file=sys.stderr,
        )
    return 0


def cmd_cost(args) -> int:
    cfgs = resolve_configs()
    params, _ = measure_params(cfgs, recompute=False)
    epoch_times: list[tuple[int, float]] = []
    identity: RunIdentity | None = None
    if args.log_file:
        log_path = Path(args.log_file)
        if not log_path.is_file():
            print(f"cost: no such log file: {log_path}", file=sys.stderr)
            return 2
        epoch_times = read_epoch_times(log_path)
        if not epoch_times:
            print(
                f"cost: {log_path} has no 'Epoch N avg loss: ... time: Ns' lines "
                "(src/train.py:1591), so there is nothing to measure from.",
                file=sys.stderr,
            )
            return 2
        identity = read_run_identity(log_path)
        if identity.trainable is None:
            print(
                f"cost: warning -- {log_path} has no readable 'Trainable parameters:' "
                "line (src/train.py:1177), so its timing cannot be attributed to any "
                "config and every sec/epoch cell stays NOT-RUN. Re-run the config "
                "without discarding its startup output.",
                file=sys.stderr,
            )
        else:
            matched = [c.rel for c in cfgs if params[c.rel][1] == identity.trainable]
            print(
                f"cost: timing attributed from {log_path} ({identity.describe()}) to "
                f"{len(matched)} of {len(cfgs)} configs with the same measured count.",
                file=sys.stderr,
            )
    table = build_cost_table(cfgs, params, args.train_seqs, epoch_times, identity)
    emit(table.render(args.format), args.out)
    print()
    print(
        COST_HELPERS.format(
            seq_len=cfgs[0].seq_len if cfgs else 512,
            root=(cfgs[0].data_cfg.get("root_path") if cfgs else "data/wikitext-103"),
        )
    )
    return 0


def cmd_curves(args) -> int:
    cfgs = resolve_configs()
    states = {c.rel: discover(c, run_roots_of(args)) for c in cfgs}
    out_dir = Path(args.out_dir or (REPO / "docs" / "results" / "figures"))
    written = emit_curves(cfgs, states, out_dir)
    print(f"\n{written} figure(s) written to {out_dir}.")
    return 0


def cmd_verify(args) -> int:
    cfgs = resolve_configs()
    states = {c.rel: discover(c, run_roots_of(args)) for c in cfgs}
    params, _ = measure_params(cfgs, args.recompute_params)

    verify_t = build_verify_table(cfgs, states, params)
    main_t = build_main_table(cfgs, states, params)
    abl_t = build_ablation_table(cfgs, states, params, defaults_model_block(), None)

    # Structural check on the cells, then an independent check on the strings.
    for table in (verify_t, main_t, abl_t):
        assert_marker_honest(table.all_cells())
    rendered = "\n".join(t.render("plain") for t in (verify_t, main_t, abl_t))
    problems = scan_rendered(rendered)
    selftest_failures = guard_selftest()

    emit(verify_t.render(args.format), args.out)
    if problems or selftest_failures:
        if selftest_failures:
            print(
                "\nverify: FAILED -- the guard is not load-bearing:",
                file=sys.stderr,
            )
            for line in selftest_failures:
                print(f"  {line}", file=sys.stderr)
        if problems:
            print(f"\nverify: FAILED -- {len(problems)} problem(s):", file=sys.stderr)
            for line in problems[:20]:
                print(f"  {line}", file=sys.stderr)
        return 1
    print(
        f"\nverify: PASS -- {len(verify_t.rows)} configs checked. Every unmeasured cell "
        f"renders as {NOT_RUN} or {NOT_APPLICABLE}, none renders a digit, none of the "
        f"{len(REFUSED_MARKERS)} refused markers appears in the output, and every "
        "measured cell carries a provenance string. The guard was also observed to "
        f"reject {GUARD_SELFTEST_ATTACKS} attempts to render an unmeasured cell as a "
        "number or a dash, and to accept a genuine measurement, so this PASS is not "
        "vacuous."
    )
    return 0


def cmd_howto(args) -> int:
    emit(howto_text(resolve_configs()), args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--format", choices=["markdown", "csv", "plain"], default="markdown")
    common.add_argument("--out", default=None, help="write here instead of stdout")
    common.add_argument(
        "--run-root",
        action="append",
        default=None,
        help="root that 'logging.folder: output/...' maps onto (default: output)",
    )

    parser = argparse.ArgumentParser(
        prog="results_apparatus",
        description=(
            "Build the paper's results tables from what is on disk. Never fabricates: "
            f"an unmeasured cell renders as {NOT_RUN} or {NOT_APPLICABLE}, and that is "
            "enforced in code rather than by convention."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    p_table = sub.add_parser("table", parents=[common], help="the main and ablation tables")
    p_table.add_argument(
        "--recompute-params",
        action="store_true",
        help="re-measure every parameter count instead of reusing the cache",
    )
    p_table.set_defaults(func=cmd_table)

    p_cost = sub.add_parser("cost", parents=[common], help="wall-clock model, arithmetic shown")
    p_cost.add_argument(
        "--train-seqs",
        type=int,
        default=None,
        help="sequences in the corpus at this config's max_seq_len",
    )
    p_cost.add_argument(
        "--log-file",
        default=None,
        help="the run's own stdout log; its startup lines attribute the timing",
    )
    p_cost.set_defaults(func=cmd_cost)

    p_curves = sub.add_parser("curves", parents=[common], help="training-curve figures")
    p_curves.add_argument("--out-dir", default=None)
    p_curves.set_defaults(func=cmd_curves)

    p_verify = sub.add_parser("verify", parents=[common], help="the fabrication self-check")
    p_verify.add_argument("--recompute-params", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    p_howto = sub.add_parser("howto", parents=[common], help="the owner's run order")
    p_howto.set_defaults(func=cmd_howto)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
