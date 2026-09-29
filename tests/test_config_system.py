# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Config-system contracts for text-span-jepa.

`config/**/*.yaml` is deep-merged over `defaults.yaml` by `src.train._deep_merge`
and the merged result is handed to `create_model`. Every contract below is
therefore a property of the *merged* config, not of any single file, and the
merge itself is imported from `src.train` rather than reimplemented here.

Pinned contracts
----------------
1.  Every shipped config survives the real deep merge and passes the validation
    `create_model` applies to it. (This is the test 18 of 57 files failed: they
    set `ema_tau_end: 1.0`, which `TextSpanJEPAConfig.validate()` rejects and
    which freezes the target encoder.)
2.  The leave-one-out ablation grid is complete: for every mechanism in
    `MechanismBundle.ALL_MECHANISMS` there is a `no_<mech>.yaml` whose ACTIVE
    set is the full set minus that one, and a `<mech>_on.yaml` differing from
    it by exactly one mechanism. Active sets are read back from
    `MechanismBundle.active_mechanisms()` -- the runtime truth -- never from the
    config's own `use_*` flags, so the WSD-needs-JAWP guard in
    `TextSpanJEPA.__init__` is honoured rather than assumed.
3.  Every config key exists at its exact dotted path in `defaults.yaml`, and
    `src/train.py`'s own startup warning now checks paths rather than bare
    leaf names, so a misplaced key or a misspelled section is reported instead
    of silently ignored.
4.  No config restates a default it does not need to change, beyond one
    documented per-file exception (see `_allowed_repeats`).
5.  `ema_tau_end < 1.0` everywhere and the target encoder provably still moves
    on the final scheduled step.
6.  The scaling ladder varies width/depth and nothing else.
7.  No dead keys, and the scaling filenames match measured parameter counts.

CPU-only, no network, no training. Model construction for parameter counting is
on `meta` tensors; mechanism activation uses a shrunk shape.
"""

from __future__ import annotations

import contextlib
import logging
import re
import warnings
from contextvars import ContextVar
from pathlib import Path

import pytest
import torch
import yaml

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
from src.models.mechanisms import MechanismBundle
from src.train import _deep_merge, _normalize_model_name, _warn_unknown_config_keys
from src.utils.schedulers import EMATauSchedule

REPO = Path(__file__).resolve().parent.parent
DEFAULTS_PATH = REPO / "defaults.yaml"
CONFIG_ROOT = REPO / "config"
ABLATIONS = "config/ablations"
SCALING_DIR = CONFIG_ROOT / "scaling"
# NOT `config/scaling/dist/`: this repo's .gitignore carries a blanket
# `dist/` rule (a build-artifact convention), which would silently drop the
# whole family from version control.
DEVICES_DIR = SCALING_DIR / "devices"

ALL_MECHANISMS = tuple(MechanismBundle.ALL_MECHANISMS)
GPT2_VOCAB = 50304

# Keys `src.train._warn_unknown_config_keys` accepts without a defaults.yaml
# entry, as full dotted paths. Kept in step with the trainer's own list by
# `test_trainer_extra_known_matches_the_trainer`, so a new opt-in there fails
# here instead of silently changing what this file claims the trainer accepts.
# See `TestTrainerTypoDetectorGap`.
_TRAINER_EXTRA_KNOWN = {
    "model.average_top_k_layers",
    "model.loss_beta",
    "model.loss_scale",
    "model.ema_decay",
    "model.ema_end_decay",
    "model.ema_anneal_end_step",
    "model.head_layers",
    "meta.dataset",
    "data.allow_missing_validation",
}

# Shape/curriculum fields shrunk to something constructible in microseconds.
# Mechanism flags, hyperparameters and lambdas are NOT touched, so the
# active-mechanism set is the real one.
_TINY_SHAPE = {
    "vocab_size": 64,
    "max_seq_len": 16,
    "embed_dim": 32,
    "encoder_depth": 1,
    "num_heads": 2,
    "predictor_embed_dim": 16,
    "predictor_depth": 1,
    "future_offsets": [1],
    "num_refine_steps": 1,
    "jawk_k_start": 2,
    "jawk_k_end": 4,
    "jawk_curriculum_steps": 0,
    "drop_path_rate": 0.0,
    "spc_n_bands": 4,
    "cgn_n_groups": 2,
    "jspace_k_workspace": 4,
    "wsd_k": 4,
    "swip_k_workspace": 4,
    "rdc_k_workspace": 4,
    "sigreg_n_sketches": 8,
    "sigreg_n_integration_points": 5,
    "pcr_level_dims": None,
}


# ══════════════════════════════════════════════════════════════════════
#  Loading / merging helpers (cached: the suite walks 60+ files repeatedly)
# ══════════════════════════════════════════════════════════════════════


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_DEFAULTS = _read_yaml(DEFAULTS_PATH)
CONFIG_IDS = sorted(_rel(p) for p in CONFIG_ROOT.rglob("*.yaml") if p.is_file())

_RAW: dict = {}
_MERGED: dict = {}
_ACTIVE: dict = {}


def _raw(rel: str) -> dict:
    if rel not in _RAW:
        _RAW[rel] = _read_yaml(REPO / rel)
    return _RAW[rel]


def _merged(rel: str) -> dict:
    """The config exactly as `src/train.py::__main__` would build it."""
    if rel not in _MERGED:
        _MERGED[rel] = _deep_merge(_DEFAULTS, _raw(rel))
    return _MERGED[rel]


def _is_jepa(rel: str) -> bool:
    return _normalize_model_name(
        _merged(rel).get("meta", {}).get("model_name", "text_span_jepa")
    ) == ("text_span_jepa")


def _jepa_config(rel: str, shrink: bool = False) -> TextSpanJEPAConfig:
    """Build the `TextSpanJEPAConfig` `create_model` would build for this config."""
    merged = _merged(rel)
    model = dict(merged.get("model", {}))
    if shrink:
        model.update(_TINY_SHAPE)
    else:
        model.setdefault("vocab_size", GPT2_VOCAB)
        model.setdefault("max_seq_len", merged.get("data", {}).get("max_seq_len", 512))
    return TextSpanJEPAConfig(**model)


def _active(rel: str) -> frozenset:
    """Mechanisms actually constructed for this config.

    Ground truth is `MechanismBundle.active_mechanisms()`, built from the same
    config object `create_model` builds. This honours the WSD guard in
    `TextSpanJEPA.__init__` (`if config.use_wsd and config.use_jawp`), which a
    naive read of the `use_*` flags would miss.
    """
    if rel not in _ACTIVE:
        bundle = MechanismBundle.from_config(_jepa_config(rel, shrink=True))
        _ACTIVE[rel] = frozenset(bundle.active_mechanisms())
    return _ACTIVE[rel]


def _leaves(d, prefix=""):
    """Yield (dotted_path, value) for every leaf of a nested mapping."""
    if not isinstance(d, dict):
        return
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            yield from _leaves(v, p)
        else:
            yield p, v


def _node_paths(d, prefix=""):
    """Every dotted path in a nested mapping, leaves and interior nodes."""
    if not isinstance(d, dict):
        return set()
    out = set()
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        out.add(p)
        if isinstance(v, dict):
            out |= _node_paths(v, p)
    return out


_DEFAULTS_LEAF_MAP = dict(_leaves(_DEFAULTS))
_DEFAULTS_LEAVES = set(_DEFAULTS_LEAF_MAP)
_DEFAULTS_NODES = _node_paths(_DEFAULTS)
# No `_DEFAULTS_LEAF_NAMES`. It existed only to mirror the trainer's
# leaf-NAME comparison, which is the defect `TestTrainerTypoDetectorGap`
# pins shut; a bare-name set in this file would invite the mirror back.


def _get(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return KeyError
        node = node[part]
    return node


def _effective_batch(cfg: dict) -> int:
    return int(_get(cfg, "data.batch_size")) * int(_get(cfg, "optimization.grad_accum_steps"))


# Everything a capacity sweep must hold fixed: only width and depth may move.
LADDER_CONSTANTS = (
    "data.max_seq_len",
    "data.batch_size",
    "data.mask_ratio",
    "optimization.grad_accum_steps",
    "optimization.epochs",
    "optimization.lr",
    "optimization.start_lr",
    "optimization.final_lr",
    "optimization.warmup",
    "optimization.weight_decay",
    "optimization.final_weight_decay",
    "model.num_refine_steps",
    "model.refine_step_size",
    "model.drop_path_rate",
    "model.ema_tau_start",
    "model.ema_tau_end",
    "model.jawk_curriculum_steps",
    "model.jawk_k_start",
    "model.jawk_alpha",
    "model.future_warmup_steps",
    "model.mask_ratio_start",
    "model.mask_ratio_end",
    "model.lambda_span",
    "model.lambda_future",
    "model.lambda_decoder",
    "model.lambda_variance",
    "model.lambda_covariance",
)

# Only width/depth (and the derived predictor width/depth) may vary.
LADDER_VARYING = (
    "model.embed_dim",
    "model.encoder_depth",
    "model.num_heads",
    "model.predictor_embed_dim",
    "model.predictor_depth",
)


# ══════════════════════════════════════════════════════════════════════
#  1. Every config must survive the merge and validate
# ══════════════════════════════════════════════════════════════════════


class TestEveryConfigRuns:
    @pytest.mark.parametrize("rel", CONFIG_IDS)
    def test_deep_merge_then_validate(self, rel):
        """A config must produce the config object `create_model` builds.

        `create_model` builds a `TextSpanJEPAConfig` (and calls `validate()`)
        only for the JEPA family; `mlm` / `data2vec` dispatch to baseline
        classes that never see a JEPA config, so those are checked against the
        shape invariants their baseline actually uses.
        """
        cfg = _merged(rel)
        if _is_jepa(rel):
            _jepa_config(rel).validate()
        else:
            model = cfg.get("model", {})
            assert model["embed_dim"] % model["num_heads"] == 0, (
                f"{rel}: embed_dim={model['embed_dim']} not divisible by "
                f"num_heads={model['num_heads']}"
            )
            assert model["encoder_depth"] >= 1, f"{rel}: encoder_depth must be >= 1"

    @pytest.mark.parametrize("rel", CONFIG_IDS)
    def test_has_every_section_the_trainer_reads(self, rel):
        merged = _merged(rel)
        for section in ("meta", "data", "model", "optimization", "logging"):
            assert section in merged, f"{rel}: missing required section '{section}'"

    def test_inventory_is_not_shrunk_by_accident(self):
        assert len(CONFIG_IDS) >= 57, (
            f"only {len(CONFIG_IDS)} configs found under config/; the pre-fix tree "
            f"had 57 files and this file is not a licence to delete experiments"
        )


# ══════════════════════════════════════════════════════════════════════
#  2. The leave-one-out ablation grid
# ══════════════════════════════════════════════════════════════════════

NO_CFG = {m: f"{ABLATIONS}/no_{m}.yaml" for m in ALL_MECHANISMS}
ON_CFG = {m: f"{ABLATIONS}/{m}_on.yaml" for m in ALL_MECHANISMS}
NONE_CFG = f"{ABLATIONS}/none.yaml"
FULL_SET = frozenset(ALL_MECHANISMS)

# The one documented exception to "turning one flag off removes one mechanism".
# `TextSpanJEPA.__init__` builds WSD only under `if config.use_wsd and
# config.use_jawp:`, so removing the workspace root removes WSD with it. There
# is no configuration in which WSD is on and JAWP is off -- the code forbids it.
JAWP_DOWNSTREAM = frozenset({"jawp", "wsd"})

# Each mechanism's own loss weight. CGN's is the partition-of-unity
# orthogonality term, which is separate from the gate; PCR and JAWP are modules
# rather than regularisers and have no weight of their own.
WEIGHT_KEY = {m: f"lambda_{m}" for m in ALL_MECHANISMS}
WEIGHT_KEY["cgn"] = "lambda_cgn_ortho"
WEIGHT_KEY["pcr"] = None
WEIGHT_KEY["jawp"] = None


def _downstream(mech: str) -> frozenset:
    """Mechanisms that stop being constructed when `use_<mech>` goes false.

    Measured from `MechanismBundle` with the full model otherwise on, so the
    leave-one-out row is defined as "the full set minus exactly what this one
    flag removes" rather than a number someone typed.
    """
    model = dict(_DEFAULTS["model"])
    model[f"use_{mech}"] = False
    cfg = TextSpanJEPAConfig(**{**model, **_TINY_SHAPE})
    return FULL_SET - frozenset(MechanismBundle.from_config(cfg).active_mechanisms())


class TestAblationGridComplete:
    def test_every_mechanism_has_both_arms(self):
        wanted = list(NO_CFG.values()) + list(ON_CFG.values())
        missing = [p for p in wanted if not (REPO / p).is_file()]
        assert not missing, (
            f"the grid needs {len(ALL_MECHANISMS)} x 2 = {2 * len(ALL_MECHANISMS)} "
            f"files (ALL_MECHANISMS); missing: {missing}"
        )

    def test_all_mechanisms_length_is_stable(self):
        """AGENTS.md 'counts must agree': grid, ALL_MECHANISMS, audit matrix."""
        assert len(ALL_MECHANISMS) == 12, (
            f"ALL_MECHANISMS is now {len(ALL_MECHANISMS)}: update the ablation "
            f"grid, the mechanisms.py header, GWP.N_MECHANISMS and "
            f"proofs/IMPLEMENTATION_STATUS.md together"
        )


class TestLeaveOneOut:
    @pytest.mark.parametrize("mech", ALL_MECHANISMS)
    def test_no_arm_is_full_set_minus_exactly_what_its_flag_removes(self, mech):
        active = _active(NO_CFG[mech])
        removed = _downstream(mech)
        assert active == FULL_SET - removed, (
            f"no_{mech}.yaml activates {sorted(active)}, expected exactly "
            f"{sorted(FULL_SET - removed)} (the full set minus {sorted(removed)}, "
            f"which is what turning use_{mech} off actually removes). A "
            f"leave-one-out row that drops anything else is a confounded ablation."
        )

    @pytest.mark.parametrize("mech", ALL_MECHANISMS)
    def test_the_flag_removes_one_mechanism_except_for_the_root(self, mech):
        removed = _downstream(mech)
        if mech == "jawp":
            assert removed == JAWP_DOWNSTREAM, (
                f"turning use_jawp off removes {sorted(removed)}, not "
                f"{sorted(JAWP_DOWNSTREAM)}. The WSD guard in "
                f"TextSpanJEPA.__init__ changed, or the grid needs re-auditing."
            )
        else:
            assert removed == {mech}, (
                f"turning use_{mech} off also removes {sorted(removed - {mech})}. "
                f"Add it to JAWP_DOWNSTREAM with a reason, or fix the guard."
            )

    @pytest.mark.parametrize("mech", ALL_MECHANISMS)
    def test_on_arm_actually_activates_its_mechanism(self, mech):
        active = _active(ON_CFG[mech])
        assert mech in active, (
            f"{mech}_on.yaml does not activate {mech!r} (active: {sorted(active)}). "
            f"A dead '_on' arm trains with the mechanism off while being reported "
            f"as its on-arm."
        )

    @pytest.mark.parametrize("mech", ALL_MECHANISMS)
    def test_off_on_pair_differs_by_exactly_the_downstream_set(self, mech):
        off, on = _active(NO_CFG[mech]), _active(ON_CFG[mech])
        assert off ^ on == _downstream(mech), (
            f"no_{mech}.yaml {sorted(off)} vs {mech}_on.yaml {sorted(on)} differ by "
            f"{sorted(off ^ on)}, expected exactly {sorted(_downstream(mech))}"
        )

    @pytest.mark.parametrize("mech", ALL_MECHANISMS)
    def test_pair_differs_by_one_mechanism_for_every_non_root(self, mech):
        if mech == "jawp":
            pytest.skip("JAWP is the root: WSD is constructed only when JAWP is on")
        off, on = _active(NO_CFG[mech]), _active(ON_CFG[mech])
        assert len(off ^ on) == 1, f"{mech} pair differs by {sorted(off ^ on)}"

    @pytest.mark.parametrize("mech", ALL_MECHANISMS)
    def test_on_arm_is_the_full_model(self, mech):
        active = _active(ON_CFG[mech])
        assert active == FULL_SET, (
            f"the on-arm of a leave-one-out table is the full model; "
            f"{mech}_on.yaml resolves to {sorted(active)}"
        )

    @pytest.mark.parametrize("mech", ALL_MECHANISMS)
    def test_no_arm_never_collapses_to_the_none_baseline(self, mech):
        active = _active(NO_CFG[mech])
        assert active != _active(NONE_CFG), (
            f"no_{mech}.yaml resolves to the same active set as none.yaml "
            f"({sorted(active)}) -- a two-variable delta, not a one-mechanism "
            f"ablation"
        )

    @pytest.mark.parametrize("mech", ALL_MECHANISMS)
    def test_no_arm_differs_from_the_full_model_only_in_its_keys(self, mech):
        """The strongest form of "one-mechanism delta": compare MERGED blocks.

        The full model IS defaults.yaml, so a leave-one-out row must differ
        from it in the mechanism's own flag and, where it has one, its own
        weight -- and in nothing else. This is the assertion that would have
        caught the pre-fix `no_swip.yaml`, which also switched JAWP off.
        """
        ref = dict(_leaves(_DEFAULTS["model"]))
        off = dict(_leaves(_merged(NO_CFG[mech])["model"]))
        differing = {p for p, v in off.items() if ref.get(p) != v}
        allowed = {f"use_{mech}"} | ({WEIGHT_KEY[mech]} if WEIGHT_KEY[mech] else set())
        assert differing <= allowed, (
            f"no_{mech}.yaml changes {sorted(differing - allowed)} as well as its "
            f"own mechanism {sorted(differing & allowed)}; a leave-one-out row "
            f"that changes anything else is a confounded ablation"
        )
        assert f"use_{mech}" in differing, f"no_{mech}.yaml does not turn {mech} off"

    @pytest.mark.parametrize("mech", ALL_MECHANISMS)
    def test_on_arm_differs_from_the_full_model_only_in_its_weight(self, mech):
        ref = dict(_leaves(_DEFAULTS["model"]))
        on = dict(_leaves(_merged(ON_CFG[mech])["model"]))
        differing = {p for p, v in on.items() if ref.get(p) != v}
        allowed = {WEIGHT_KEY[mech]} if WEIGHT_KEY[mech] else set()
        assert differing <= allowed, (
            f"{mech}_on.yaml changes {sorted(differing - allowed)} on top of its own "
            f"weight key {sorted(allowed)}"
        )

    def test_none_baseline_activates_nothing(self):
        assert _active(NONE_CFG) == frozenset(), sorted(_active(NONE_CFG))

    def test_every_ablated_flag_is_off_in_its_no_arm(self):
        """Belt and braces: the flag itself, not just the derived active set."""
        for mech in ALL_MECHANISMS:
            model = _raw(NO_CFG[mech]).get("model", {})
            assert (
                model.get(f"use_{mech}") is False
            ), f"no_{mech}.yaml must explicitly set model.use_{mech}: false"


# ══════════════════════════════════════════════════════════════════════
#  3. Key paths -- the trainer's own detector compares dotted PATHS
# ══════════════════════════════════════════════════════════════════════


#: The path quoted out of one `_warn_unknown_config_keys` warning line.
_WARNED_PATH_RE = re.compile(r"Unknown config key '([^']+)'")

#: `_trainer_would_warn` is a plain function, but it needs the `caplog`
#: fixture, so the handle is threaded through a ContextVar rather than
#: through every call site.
_CAPLOG: ContextVar = ContextVar("caplog", default=None)


def _bad_paths(cfg: dict) -> list:
    """Config leaves whose exact dotted path is absent from defaults.yaml."""
    return sorted(
        p for p, _v in _leaves(cfg) if not p.startswith("_meta.") and p not in _DEFAULTS_LEAVES
    )


def _bad_nodes(cfg: dict) -> list:
    """Config keys holding a mapping that do not name a defaults.yaml subtree."""
    bad = []
    for p in _node_paths(cfg):
        if p in _DEFAULTS_NODES or p.startswith("_meta"):
            continue
        bad.append(p)
    return sorted(bad)


def _trainer_would_warn(cfg: dict) -> set:
    """Paths `src.train._warn_unknown_config_keys` actually warns about.

    Calls the shipped function and reads its warnings back, rather than
    mirroring it: a mirror pins the copy, not the trainer, and the three cases
    this file exists to catch are exactly the ones where the copy and the
    trainer disagree about what a "key" is. The config is deep-merged over an
    empty dict first, so the argument is the resolved tree a run would see.
    """
    caplog = _CAPLOG.get()
    if caplog is None:  # pragma: no cover - only reachable outside a test
        raise RuntimeError("_trainer_would_warn needs the caplog fixture")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with caplog.at_level(logging.WARNING, logger=""):
            caplog.clear()
            _warn_unknown_config_keys(_deep_merge({}, cfg))
    warned = set()
    for record in caplog.records:
        m = _WARNED_PATH_RE.search(record.getMessage())
        if m:
            warned.add(m.group(1))
    return warned


class TestKeyPaths:
    @pytest.mark.parametrize("rel", CONFIG_IDS)
    def test_every_key_exists_at_its_exact_path(self, rel):
        bad = _bad_paths(_raw(rel)) + _bad_nodes(_raw(rel))
        assert not bad, (
            f"{rel}: {bad} do not exist at those paths in defaults.yaml. The "
            f"trainer now warns about each of them by path, but a warning is a log "
            f"line, not a stop: a key in the wrong subtree is still silently inert "
            f"at runtime, so it must not be committed in the first place."
        )

    def test_defaults_yaml_declares_every_section_a_config_may_use(self):
        for section in ("meta", "data", "model", "optimization", "logging"):
            assert section in _DEFAULTS_NODES, f"defaults.yaml is missing section '{section}'"


class TestTrainerTypoDetectorGap:
    """The `src/train.py` detector compares full dotted PATHS, not leaf names.

    History, because it is the reason these three tests exist at all. The
    detector used to build `{p.split('.')[-1] for p in _leaves(defaults)}` and
    test membership of the leaf *name*. It therefore could not see a key in the
    wrong subtree, nor a wholly misspelled section, and each of the three
    cases below was accepted silently: the value was never read, so the run
    trained something other than what its config said.

    They were negative controls -- asserting the gap still existed -- so the
    fix turned them red first; this is the rewrite they asked for, and the
    assertions are inverted to pin the fixed behaviour. They stay three
    separate tests because the three cases are three different ways of being
    wrong, and a partial fix (paths for leaves, names for sections) has to
    fail.
    """

    @pytest.fixture(autouse=True)
    def _capture(self, caplog):
        """Publish this test's `caplog` to `_trainer_would_warn`."""
        token = _CAPLOG.set(caplog)
        try:
            yield
        finally:
            _CAPLOG.reset(token)

    def test_misplaced_key_is_detected(self):
        """`batch_size` is a real key name; it just lives at `data.batch_size`."""
        misplaced = {"model": {"batch_size": 64}}
        assert _trainer_would_warn(misplaced) == {"model.batch_size"}, (
            "the trainer cannot see a real key sitting in the wrong subtree. "
            "`model.batch_size` is read by nobody, so the run trains at the "
            "defaults.yaml micro-batch whatever the config says."
        )
        assert _bad_paths(misplaced) == ["model.batch_size"]

    def test_misspelled_section_is_detected(self):
        """A new top-level namespace is the widest form of the same hole."""
        assert _trainer_would_warn({"modle": {"embed_dim": 8}}) == {"modle.embed_dim"}
        assert _trainer_would_warn({"optimisation": {"lr": 1.0}}) == {"optimisation.lr"}
        assert _bad_paths({"modle": {"embed_dim": 8}}) == ["modle.embed_dim"]

    def test_nested_typo_inside_a_known_section_is_still_caught(self):
        """The one case leaf-name matching got right, for free. Must not regress."""
        assert _trainer_would_warn({"model": {"lamda_swip": 0.1}}) == {"model.lamda_swip"}

    def test_a_correctly_pathed_key_is_not_warned_about(self):
        """The false-positive guard. A stricter detector is not a better one.

        `TestKeyPaths` already proves every shipped config is path-clean, but
        against this file's own predicate. This pins the trainer's log, so
        over-strictness surfaces as a spurious startup warning on a correct
        config rather than as a quiet slowdown in the check's usefulness.
        """
        assert _trainer_would_warn({"model": {"embed_dim": 64, "use_swip": True}}) == set()
        assert _trainer_would_warn({"data": {"batch_size": 64}}) == set()
        assert _trainer_would_warn({"optimization": {"grad_accum_steps": 8}}) == set()

    def test_meta_and_documented_exemptions_stay_exempt(self):
        """`_meta.*` provenance and the CLI-only opt-in must not become noise."""
        cfg = {
            "_meta": {"note": "free-form", "devices": 8},
            "data": {"allow_missing_validation": True},
            "model": {"description": "one line of prose", "head_layers": 2},
        }
        assert _trainer_would_warn(cfg) == set(), (
            "an exemption in src/train.py's `extra_known` / `metadata_prefixes` "
            "stopped covering a documented key, so every config using it would now "
            "log a spurious 'possible typo' at startup"
        )
        # ...and the exemptions are PATH-scoped, so they cannot be borrowed
        # from the wrong subtree to smuggle a key past the check.
        assert _trainer_would_warn({"model": {"allow_missing_validation": True}}) == {
            "model.allow_missing_validation"
        }

    def test_trainer_extra_known_matches_the_trainer(self):
        """Keeps the two lists from drifting apart silently.

        `_TRAINER_EXTRA_KNOWN` is this file's copy of the trainer's opt-in
        list. It is read out of the source rather than restated, so an entry
        added on the `src/` side without being documented here fails instead of
        quietly changing what this file claims the trainer accepts.
        """
        block = re.search(r"extra_known = \{(.*?)\}", _TRAIN_SRC, re.DOTALL)
        assert block is not None, "src/train.py no longer names an `extra_known` set"
        found = set(re.findall(r'"([^"]+)"', block.group(1)))
        assert all("." in k for k in found), (
            f"src/train.py's extra_known went back to bare leaf names: {sorted(found)}. "
            f"A name matches in every section, which is the hole this class pins shut."
        )
        drift = sorted(found ^ _TRAINER_EXTRA_KNOWN)
        assert not drift, f"src/train.py's extra_known changed: {drift}"


# ══════════════════════════════════════════════════════════════════════
#  4. Delta purity
# ══════════════════════════════════════════════════════════════════════

_ON_RE = re.compile(rf"^{ABLATIONS}/({'|'.join(ALL_MECHANISMS)})_on\.yaml$")
_LADDER_SHAPE_RE = re.compile(r"^config/scaling/[^/]+\.yaml$")

# Two documented exception classes, 12 + 5 = 17 leaves in the whole tree.
PURITY_BUDGET = len(ALL_MECHANISMS) + len(LADDER_VARYING)
_LADDER_SHAPE = frozenset(LADDER_VARYING)


def _allowed_repeats(rel: str) -> set:
    """Documented per-file exceptions to "restate nothing you do not change".

    Class 1 -- `config/ablations/<mech>_on.yaml` restates exactly one key,
    `model.use_<mech>`, and deliberately so. The on-arm of a leave-one-out
    table IS the full model, so the file would otherwise carry no model delta
    at all; restating the flag it is named for makes the row provably the
    on-arm by inspection, and `test_on_arm_is_the_full_model` proves the
    resolved active set really is the full set.

    Class 2 -- `config/scaling/<rung>.yaml` restates its own width and depth.
    Those five keys ARE the capacity ladder's experiment variable; a ladder
    whose middle rung has no dimensions in its own file is unreadable in a
    paper appendix, and tests/test_v025_integration.py reads them unmerged.
    Only the largest rung is affected, because it is the only one whose shape
    coincides with the defaults.yaml reference.
    """
    m = _ON_RE.match(rel)
    if m:
        return {f"model.use_{m.group(1)}"}
    if _LADDER_SHAPE_RE.match(rel):
        return set(_LADDER_SHAPE)
    return set()


def _repeats(rel: str) -> list:
    allowed = _allowed_repeats(rel)
    return [
        f"{p}={v!r}"
        for p, v in _leaves(_raw(rel))
        if p not in allowed
        and not p.startswith("_meta.")
        and p in _DEFAULTS_LEAF_MAP
        and _DEFAULTS_LEAF_MAP[p] == v
    ]


class TestDeltaPurity:
    @pytest.mark.parametrize("rel", CONFIG_IDS)
    def test_no_unnecessary_restatements(self, rel):
        repeats = _repeats(rel)
        assert not repeats, (
            f"{rel} restates {len(repeats)} default(s) it does not need to change: "
            f"{repeats[:8]}{' ...' if len(repeats) > 8 else ''}. A copied defaults "
            f"block keeps the value it had the day it was copied, so the "
            f"experiment silently stops matching its own description."
        )

    def test_whole_tree_redundancy_budget(self):
        """Whole-tree count. Was 3739 redundant leaves over 36 of 57 files.

        `test_no_unnecessary_restatements` already asserts zero per file, so
        this asserts the *exemption surface* stays exactly as documented: 12
        `<mech>_on.yaml` self-descriptions + the 5 shape keys in the one ladder
        rung that matches the reference shape. If a new file starts copying
        defaults, the per-file test catches it; if the exemptions start being
        handed out to new files, this catches that.
        """
        offenders = [f"{rel}:{p}" for rel in CONFIG_IDS for p in _repeats(rel)]
        assert not offenders, f"{len(offenders)} unexempted redundant leaves: {offenders[:20]}"

        exempt = []
        for rel in CONFIG_IDS:
            allowed = _allowed_repeats(rel)
            for path, value in _leaves(_raw(rel)):
                if path in allowed and _DEFAULTS_LEAF_MAP.get(path) == value:
                    exempt.append((rel, path))
        assert len(exempt) == PURITY_BUDGET, (
            f"the documented exemption surface is {len(exempt)} leaves, expected "
            f"{PURITY_BUDGET} ({len(ALL_MECHANISMS)} on-arm self-descriptions + "
            f"{len(LADDER_VARYING)} shape keys in config/scaling/base_140m.yaml). "
            f"Got: {sorted(exempt)}"
        )
        permitted = set(ON_CFG.values()) | {"config/scaling/base_140m.yaml"}
        assert {rel for rel, _ in exempt} <= permitted, (
            f"an exemption was granted outside the two documented classes: "
            f"{sorted({rel for rel, _ in exempt} - permitted)}"
        )

    def test_leave_one_out_arms_are_one_key_deltas(self):
        for mech in ALL_MECHANISMS:
            model = _raw(NO_CFG[mech]).get("model", {})
            flags = {k: v for k, v in model.items() if k.startswith("use_")}
            assert flags == {f"use_{mech}": False}, (
                f"no_{mech}.yaml sets the mechanism flags {sorted(flags)}; a "
                f"leave-one-out arm flips exactly one"
            )

    def test_the_two_exception_classes_are_bounded(self):
        """The budget is only honest if the classes stay the size they claim."""
        on_arms = [rel for rel in CONFIG_IDS if _ON_RE.match(rel)]
        assert len(on_arms) == len(ALL_MECHANISMS), on_arms
        ladder = [rel for rel in CONFIG_IDS if _LADDER_SHAPE_RE.match(rel)]
        assert ladder, "config/scaling/*.yaml must exist for the shape exception to mean anything"
        for rel in ladder:
            repeats = [p for p, _ in _leaves(_raw(rel)) if p in _LADDER_SHAPE]
            assert len(repeats) == len(_LADDER_SHAPE), (
                f"{rel} declares {len(repeats)} of the {len(_LADDER_SHAPE)} shape "
                f"keys; a capacity rung must state its own width and depth"
            )
        for rel in CONFIG_IDS:
            if not _ON_RE.match(rel):
                assert not (_LADDER_SHAPE_RE.match(rel) and rel.startswith(ABLATIONS))

    def test_baseline_arm_is_not_a_verbatim_defaults_copy(self):
        """Regression guard for the worst offender in the pre-fix tree."""
        repeats = _repeats(f"{ABLATIONS}/no_cgn.yaml")
        assert not repeats, f"no_cgn.yaml restates {repeats}"


# ══════════════════════════════════════════════════════════════════════
#  5. EMA endpoint: the target encoder must keep moving
# ══════════════════════════════════════════════════════════════════════


def _ema_total_steps(rel: str) -> int:
    """`total_steps` as `_build_optimization` computes it (ipe := 1)."""
    opt = _merged(rel).get("optimization", {})
    return max(int(opt.get("ipe_scale", 1.0) * opt.get("epochs", 50)), 1)


def _final_tau(rel: str) -> float:
    cfg = _jepa_config(rel)
    total = _ema_total_steps(rel)
    sched = EMATauSchedule(tau_start=cfg.ema_tau_start, tau_end=cfg.ema_tau_end, total_steps=total)
    tau = None
    for _ in range(total):
        tau = sched.step()
    return tau


class TestEMATargetStillMoves:
    @pytest.mark.parametrize("rel", CONFIG_IDS)
    def test_ema_tau_end_is_below_one(self, rel):
        if not _is_jepa(rel):
            pytest.skip("data2vec / MLM baselines own their EMA")
        cfg = _jepa_config(rel)
        assert cfg.ema_tau_end < 1.0, (
            f"{rel}: ema_tau_end={cfg.ema_tau_end}. At tau == 1.0 "
            f"update_target_encoder() becomes mul_(1.0).add_(q, alpha=0.0): the "
            f"target encoder freezes. In self-distillation the target IS the "
            f"training signal, so the run degrades to a fixed random teacher."
        )
        assert 0.0 < cfg.ema_tau_start < cfg.ema_tau_end

    @pytest.mark.parametrize("rel", CONFIG_IDS)
    def test_final_scheduled_tau_still_updates_the_target(self, rel):
        if not _is_jepa(rel):
            pytest.skip("data2vec / MLM baselines own their EMA")
        tau = _final_tau(rel)
        assert tau < 1.0, f"{rel}: scheduled tau reaches {tau} on the last step"

        model = TextSpanJEPA(_jepa_config(rel, shrink=True))
        with torch.no_grad():
            for p in model.encoder.parameters():
                p.fill_(1.0)
            for p in model.target_encoder.parameters():
                p.zero_()
            model.update_target_encoder(tau)
        for p in model.target_encoder.parameters():
            assert torch.allclose(
                p, torch.full_like(p, 1.0 - tau), atol=1e-6
            ), f"{rel}: target encoder absorbed no online signal at tau={tau}"

    def test_no_config_restates_the_ema_endpoint(self):
        restated = [
            rel
            for rel in CONFIG_IDS
            if _is_jepa(rel) and "ema_tau_end" in _raw(rel).get("model", {})
        ]
        assert not restated, (
            f"defaults.yaml declares the EMA endpoint once; these configs restate "
            f"model.ema_tau_end: {restated}"
        )


# ══════════════════════════════════════════════════════════════════════
#  6. The scaling ladder and the fixed-size distributed family
# ══════════════════════════════════════════════════════════════════════

LADDER = sorted(p.name for p in SCALING_DIR.glob("*.yaml"))
DIST = sorted(p.name for p in DEVICES_DIR.glob("*.yaml"))

# The ladder is ordered by capacity, not by filename: the shipped filenames are
# xsmall / small / base / large, which is not alphabetical.
LADDER_BY_SIZE = sorted(
    LADDER, key=lambda n: _get(_merged(f"config/scaling/{n}"), "model.embed_dim")
)


class TestScalingLadder:
    def test_ladder_has_four_rungs(self):
        assert len(LADDER) >= 4, f"config/scaling/*.yaml found: {LADDER}"

    @pytest.mark.parametrize("dotted", LADDER_CONSTANTS)
    def test_constant_along_the_ladder(self, dotted):
        values = {n: _get(_merged(f"config/scaling/{n}"), dotted) for n in LADDER}
        assert len({repr(v) for v in values.values()}) == 1, (
            f"{dotted} varies along the scaling ladder: {values}. This is a capacity "
            f"sweep -- a hyperparameter that moves with size makes the curve measure "
            f"size and that hyperparameter jointly."
        )

    @pytest.mark.parametrize("dotted", LADDER_VARYING)
    def test_width_and_depth_increase_strictly(self, dotted):
        values = {n: _get(_merged(f"config/scaling/{n}"), dotted) for n in LADDER_BY_SIZE}
        seq = [values[n] for n in LADDER_BY_SIZE]
        assert seq == sorted(seq) and len(set(seq)) == len(
            seq
        ), f"{dotted} is not strictly increasing with capacity: {values}"

    def test_effective_batch_is_one_number(self):
        batches = {n: _effective_batch(_merged(f"config/scaling/{n}")) for n in LADDER}
        assert len(set(batches.values())) == 1, (
            f"effective batch (batch_size x grad_accum_steps) varies along the "
            f"ladder: {batches}. Two points per batch size cannot separate model "
            f"size from effective batch."
        )

    def test_tokens_per_step_are_identical(self):
        tokens = {
            n: _effective_batch(_merged(f"config/scaling/{n}"))
            * _get(_merged(f"config/scaling/{n}"), "data.max_seq_len")
            for n in LADDER
        }
        assert len(set(tokens.values())) == 1, f"tokens/step vary along the ladder: {tokens}"

    def test_mechanism_set_is_held_fixed_along_the_ladder(self):
        sets = {n: _active(f"config/scaling/{n}") for n in LADDER}
        assert len(set(sets.values())) == 1, (
            f"the capacity ladder must hold the mechanism set fixed; got "
            f"{ {n: sorted(s) for n, s in sets.items()} }"
        )


class TestFixedSizeDistributedFamily:
    """Weak scaling: model size and global batch fixed, device count varies.

    Device count is a launcher flag (the TPU/DDP side is owned elsewhere), so
    the only device-dependent quantity the config system can express is the
    per-device micro-batch. `data.batch_size` is therefore the sweep axis and
    `optimization.grad_accum_steps` is pinned, holding the global effective
    batch -- and hence the number of optimizer steps per epoch -- identical.
    """

    def test_family_exists(self):
        assert (
            len(DIST) >= 3
        ), f"config/scaling/devices/ must hold a fixed-size family; found {DIST}"

    def test_model_shape_is_held_fixed(self):
        shapes = {
            n: tuple(_get(_merged(f"config/scaling/devices/{n}"), d) for d in LADDER_VARYING)
            for n in DIST
        }
        assert (
            len(set(shapes.values())) == 1
        ), f"the distributed family must hold model size fixed; got {shapes}"
        reference = tuple(_get(_DEFAULTS, d) for d in LADDER_VARYING)
        assert set(shapes.values()) == {reference}, (
            f"the distributed family must use the defaults.yaml reference shape "
            f"{reference}; got {shapes}"
        )

    @pytest.mark.parametrize(
        "dotted",
        [
            d
            for d in LADDER_CONSTANTS
            if d not in ("data.batch_size", "optimization.grad_accum_steps")
        ],
    )
    def test_matches_the_reference_except_the_batch(self, dotted):
        ref = _get(_DEFAULTS, dotted)
        for name in DIST:
            got = _get(_merged(f"config/scaling/devices/{name}"), dotted)
            assert got == ref, f"devices/{name}: {dotted}={got!r}, reference={ref!r}"

    def test_accumulation_is_pinned_across_the_family(self):
        acc = {
            n: _get(_merged(f"config/scaling/devices/{n}"), "optimization.grad_accum_steps")
            for n in DIST
        }
        assert len(set(acc.values())) == 1, (
            f"the sweep is over the per-device micro-batch; accumulation must be "
            f"pinned so the global batch does not move: {acc}"
        )

    def test_device_count_is_recorded_and_increases(self):
        devices = {}
        for name in DIST:
            meta = _raw(f"config/scaling/devices/{name}").get("_meta", {})
            assert "devices" in meta, (
                f"config/scaling/devices/{name} must record _meta.devices: the device "
                f"count is a launcher argument, so the only way the sweep is "
                f"reproducible is if the config says what it is meant to be run on"
            )
            devices[name] = int(meta["devices"])
        assert list(devices.values()) == sorted(
            devices.values()
        ), f"device count must increase across the family: {devices}"
        assert len(set(devices.values())) == len(devices), f"device count repeats: {devices}"

    def test_global_effective_batch_is_held_fixed(self):
        """micro-batch x devices x accumulation is the token budget per step."""
        totals = {}
        for name in DIST:
            cfg = _merged(f"config/scaling/devices/{name}")
            devices = int(_raw(f"config/scaling/devices/{name}")["_meta"]["devices"])
            totals[name] = (
                int(_get(cfg, "data.batch_size"))
                * devices
                * int(_get(cfg, "optimization.grad_accum_steps"))
            )
        ref = _effective_batch(_DEFAULTS)
        assert set(totals.values()) == {ref}, (
            f"the fixed-size family is a WEAK-scaling sweep: every rung must see "
            f"the reference global effective batch of {ref} sequences per "
            f"optimizer step. Got {totals}"
        )

    def test_micro_batch_strictly_decreases(self):
        micros = [
            int(_get(_merged(f"config/scaling/devices/{n}"), "data.batch_size")) for n in DIST
        ]
        assert micros == sorted(micros, reverse=True) and len(set(micros)) == len(
            micros
        ), f"per-device micro-batch must strictly decrease as device count grows: {micros}"


# ══════════════════════════════════════════════════════════════════════
#  7. Dead keys
# ══════════════════════════════════════════════════════════════════════

DEAD_KEYS = ("optimization.ema", "logging.write_tag")
_TRAIN_SRC = (REPO / "src" / "train.py").read_text(encoding="utf-8")


class TestNoDeadKeys:
    @pytest.mark.parametrize("path", DEAD_KEYS)
    def test_dead_key_not_declared_in_defaults(self, path):
        assert path not in _DEFAULTS_LEAVES, (
            f"defaults.yaml declares '{path}' but nothing reads it. The trainer "
            f"uses model.ema_tau_start/end and logging.folder; a key that looks "
            f"live but is not is worse than no key."
        )

    def test_dead_key_set_by_no_config(self):
        used = {p for rel in CONFIG_IDS for p, _v in _leaves(_raw(rel))}
        still = sorted(set(DEAD_KEYS) & used)
        assert not still, f"configs still set the dead keys {still}"

    def test_meta_seed_is_declared(self):
        """`src.train.main` reads `meta.seed` first, then top-level `seed`."""
        assert "meta.seed" in _DEFAULTS_LEAVES, (
            "configs set meta.seed; it survives only because src/train.py reads it "
            "explicitly, so defaults.yaml must declare it"
        )

    def test_meta_dataset_is_declared(self):
        assert "meta.dataset" in _DEFAULTS_LEAVES, (
            "meta.dataset is provenance every experiment config records; declare it "
            "so the path check covers it"
        )

    def test_ema_schedule_is_inert_in_the_trainer(self):
        """Documents a `src/train.py` gap for that file's owner.

        `TextSpanJEPAConfig.validate()` accepts `model.ema_schedule`, but
        `_build_optimization` constructs an `EMATauSchedule` and the training
        loop only ever calls `.step()`: `step_cosine()` has no caller, so a
        config asking for the cosine ramp silently gets the linear one. The key
        is read (so it stays declared) but currently has no effect.

        Reported as a skip, not a failure: if `src/train.py` grows the
        `step_cosine()` call, this turns into a reminder to re-check the shipped
        `ema_tau_start`/`ema_tau_end` values against a schedule that now
        actually bends.
        """
        assert "model.ema_schedule" in _DEFAULTS_LEAVES
        if "step_cosine" in _TRAIN_SRC:
            pytest.skip(
                "src/train.py now calls EMATauSchedule.step_cosine(): "
                "model.ema_schedule is live. Re-check every config's "
                "ema_tau_start/ema_tau_end against the now-bending schedule."
            )

    def test_trainer_ema_fallback_is_not_the_frozen_value(self):
        """`_build_optimization` must not fall back to a frozen target.

        `_build_optimization` reads
            tau_end=model_cfg.get("ema_tau_end", <fallback>)
        so any run with `--no_defaults`, or any config that omits the key, gets
        whatever the literal is. At `1.0` the run is silently dead:
        `EMATauSchedule.step()` returns exactly `tau_end`, and `1 - 1.0 == 0`, so
        `update_target_encoder` performs `mul_(1.0).add_(q, alpha=0.0)` for the
        whole run and the target encoder never moves.

        Every shipped config supplies the key (see
        `test_every_config_supplies_the_ema_endpoint_so_the_fallback_is_unreachable`),
        which is why the tree was safe while the literal stayed broken.
        """
        m = re.search(r'model_cfg\.get\("ema_tau_end",\s*([0-9.]+)\)', _TRAIN_SRC)
        if m is None:
            pytest.skip("src/train.py no longer supplies an ema_tau_end fallback")
        assert float(m.group(1)) < 1.0

    def test_every_config_supplies_the_ema_endpoint_so_the_fallback_is_unreachable(self):
        """The property that makes the `src` fallback harmless today."""
        for rel in CONFIG_IDS:
            if not _is_jepa(rel):
                continue
            assert _jepa_config(rel).ema_tau_end < 1.0, rel
        assert "model.ema_tau_end" in _DEFAULTS_LEAVES, (
            "if defaults.yaml stopped declaring ema_tau_end, every config would "
            "silently fall through to src/train.py's frozen 1.0"
        )


# ══════════════════════════════════════════════════════════════════════
#  8. Scaling filenames vs measured parameter counts
# ══════════════════════════════════════════════════════════════════════


@contextlib.contextmanager
def _meta_device():
    """Build on meta tensors: exact parameter shapes, zero allocation.

    `TextSpanJEPLEncoder` calls `torch.linspace(...).item()`, which meta tensors
    cannot service, so `linspace` is pinned to CPU for the duration.
    """
    original = torch.linspace

    def cpu_linspace(*a, **k):
        k = dict(k)
        k.setdefault("device", "cpu")
        return original(*a, **k)

    torch.linspace = cpu_linspace
    try:
        with torch.device("meta"):
            yield
    finally:
        torch.linspace = original


def _count_params(rel: str):
    """(total, trainable, get_num_params) for a config, on meta tensors."""
    with _meta_device():
        model_cfg = dict(_merged(rel)["model"])
        model_cfg.setdefault("vocab_size", GPT2_VOCAB)
        model_cfg.setdefault("max_seq_len", _get(_merged(rel), "data.max_seq_len"))
        cfg = TextSpanJEPAConfig(**model_cfg)
        cfg.validate()
        model = TextSpanJEPA(cfg)
        return (
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            model.get_num_params(),
        )


LADDER_CLAIM = {
    "xsmall_30m.yaml": 30,
    "small_100m.yaml": 100,
    "base_140m.yaml": 140,
    "large_300m.yaml": 300,
}


class TestScalingParamCounts:
    """A scaling paper quoting the wrong number is a desk-reject item.

    Measured in-process on this machine with each config exactly as it resolves.
    The filename tracks TRAINABLE parameters (encoder + predictor + mechanisms;
    the target encoder is frozen). `get_num_params()` -- the number the trainer
    logs at startup -- excludes the target encoder and both embeddings, so it
    under-reports by ~2.5x. That is a `src/` issue and is reported, not fixed
    here.
    """

    @pytest.mark.parametrize("name,claim", sorted(LADDER_CLAIM.items()))
    def test_filename_tracks_trainable_parameters(self, name, claim):
        total, trainable, _reported = _count_params(f"config/scaling/{name}")
        measured_m = trainable / 1e6
        assert 0.85 * claim <= measured_m <= 1.15 * claim, (
            f"config/scaling/{name} claims {claim}M but the config resolves to "
            f"{measured_m:.1f}M trainable ({total / 1e6:.1f}M total) parameters"
        )

    @pytest.mark.parametrize("name", sorted(LADDER_CLAIM))
    def test_config_documents_its_true_counts(self, name):
        text = (SCALING_DIR / name).read_text(encoding="utf-8")
        total, trainable, _reported = _count_params(f"config/scaling/{name}")
        for value in (f"{total:,}", f"{trainable:,}"):
            assert value in text, (
                f"config/scaling/{name} must state the measured counts "
                f"({trainable:,} trainable / {total:,} total) in a comment, and say "
                f"which one the filename refers to"
            )

    def test_get_num_params_under_reports_the_true_total(self):
        """Negative control on a `src/` defect this file cannot fix."""
        _total, _trainable, reported = _count_params("config/scaling/base_140m.yaml")
        assert reported < 0.5 * _total, (
            "TextSpanJEPA.get_num_params() now covers the target encoder; the "
            "comments in config/scaling/*.yaml must be recomputed"
        )
