# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
#
# Regression tests for the held-out split in src/eval/probes.py.
#
# The bug these guard: LinearProbe.evaluate fit the classifier on `dataset` and
# then reported accuracy by iterating that same dataset; FutureTokenProbe
# accumulated a running accuracy from logits computed *before* opt.step().
# Both numbers were training-set scores wearing an evaluation label.

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from src.eval import probes

# Token id space of the fake encoder / future-probe vocabulary.
_TABLE = 32
_EMBED = 8
_SEQ = 6
_N = 32
_SEED = 0
# Token at t+1 is token at t plus this, so the future-token task is learnable
# and a scrambled sequence is visibly not predicted.
_STEP = 3
# A different permutation per row: scrambling the *order* of a held-out sequence
# destroys the offset relationship without changing its token multiset.
_PERM_GEN = torch.Generator().manual_seed(99)
_PERM = torch.stack([torch.randperm(_SEQ, generator=_PERM_GEN) for _ in range(_N)])


# --------------------------------------------------------------------------------------
# Fixtures: a parameter-free, perfectly deterministic encoder
# --------------------------------------------------------------------------------------


class _FrozenEncoder(nn.Module):
    """Maps token id -> a fixed row of a frozen table. No parameters, no randomness."""

    def __init__(self, vocab_size=_TABLE, embed_dim=_EMBED):
        super().__init__()
        g = torch.Generator(device="cpu")
        g.manual_seed(20260927)
        self.register_buffer("table", torch.randn(vocab_size, embed_dim, generator=g))

    def forward(self, input_ids):
        return self.table[input_ids], None


class _FakeModel(nn.Module):
    def __init__(self, vocab_size=_TABLE, embed_dim=_EMBED):
        super().__init__()
        self.encoder = _FrozenEncoder(vocab_size, embed_dim)


class _LabeledDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


class _TokenDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return self.input_ids[idx]


def _separable_dataset():
    """Label == the single token used throughout the sequence.

    Every pooled representation is then one of exactly two well-separated points,
    so a linear probe reaches 100% on every split. That is what makes the
    relabelling tests below exact instead of statistical.
    """
    labels = torch.arange(_N) % 2
    input_ids = labels.view(-1, 1).repeat(1, _SEQ)
    return _LabeledDataset(input_ids, labels.clone())


def _token_dataset():
    """Arithmetic sequences: token at t+d is a deterministic function of token at t."""
    rows = torch.arange(_N).view(-1, 1) + torch.arange(_SEQ).view(1, -1) * _STEP
    return _TokenDataset(rows % _TABLE)


def _linear(**kwargs):
    """Build a LinearProbe.

    Deliberately passes no seed by default so the two decisive leakage tests
    below run unchanged against the pre-fix implementation, where the number they
    report is a training score.
    """
    kwargs.setdefault("embed_dim", _EMBED)
    kwargs.setdefault("num_classes", 2)
    return probes.LinearProbe(**kwargs)


def _future(**kwargs):
    kwargs.setdefault("embed_dim", _EMBED)
    kwargs.setdefault("vocab_size", _TABLE)
    kwargs.setdefault("offsets", (1, 2))
    kwargs.setdefault("seed", _SEED)
    return probes.FutureTokenProbe(**kwargs)


def _reported_eval_acc(result, key):
    """The number the module presents as the held-out accuracy named ``key``.

    Falls back to the pre-fix unsplit keys so the leakage tests can report a
    *number* for the old implementation instead of erroring on a missing key.
    """
    if key in result:
        return result[key]
    for candidate in (key.replace("_test_top1", ""), "accuracy"):
        if candidate in result:
            return result[candidate]
    raise AssertionError(f"{key} not reported; keys={sorted(result)}")


def _split_of(module, n):
    """The module must publish the split it claims to have used."""
    fn = getattr(probes, "split_indices", None)
    if fn is None:
        raise AssertionError(
            "src.eval.probes publishes no split: it does not say which samples it "
            "evaluated on, so no reported number can be shown to come from held-out data."
        )
    return fn(n, seed=getattr(module, "seed", _SEED))


def _scramble(input_ids, indices):
    """Permute the token order of the given rows only."""
    out = input_ids.clone()
    rows = out[indices]
    out[indices] = rows[torch.arange(rows.size(0)).unsqueeze(1), _PERM[indices]]
    return out


# --------------------------------------------------------------------------------------
# split_indices
# --------------------------------------------------------------------------------------


def test_split_indices_is_a_disjoint_partition():
    train, val, test = probes.split_indices(_N, seed=_SEED)
    all_idx = torch.cat([train, val, test])
    assert all_idx.numel() == _N
    assert torch.equal(all_idx.sort().values, torch.arange(_N)), "split must cover every sample"
    assert len(set(all_idx.tolist())) == _N, "split must not repeat a sample"
    for split in (train, val, test):
        assert split.numel() >= 1, "an empty split would silently score nothing"


def test_split_indices_is_deterministic_and_seed_dependent():
    a = probes.split_indices(_N, seed=1)
    b = probes.split_indices(_N, seed=1)
    c = probes.split_indices(_N, seed=2)
    assert all(torch.equal(x, y) for x, y in zip(a, b)), "same seed must reproduce the split"
    assert any(not torch.equal(x, y) for x, y in zip(a, c)), "seed must actually be used"


def test_split_indices_fractions_are_respected():
    train, val, test = probes.split_indices(1000, seed=_SEED, train_frac=0.6, val_frac=0.2)
    assert train.numel() == 600
    assert val.numel() == 200
    assert test.numel() == 200


def test_split_indices_rejects_impossible_fractions():
    with pytest.raises(ValueError):
        probes.split_indices(10, train_frac=0.9, val_frac=0.9)
    with pytest.raises(ValueError):
        probes.split_indices(2, seed=_SEED)


# --------------------------------------------------------------------------------------
# LinearProbe
# --------------------------------------------------------------------------------------


def test_linear_probe_test_accuracy_responds_to_relabelling_the_test_split():
    """Decisive: destroy the test labels; the reported test accuracy must move."""
    probe = _linear(max_epochs=20)
    dataset = _separable_dataset()
    # device is passed explicitly so that, against the pre-fix code, this test
    # fails on the leakage claim and not on its cuda default (which has its own test).
    clean = probe.evaluate(_FakeModel(), dataset, device="cpu")
    clean_test = _reported_eval_acc(clean, "test_acc")

    _train, _val, test_idx = _split_of(probe, len(dataset))
    labels = dataset.labels.clone()
    labels[test_idx] = (labels[test_idx] + 1) % 2
    dirty = probe.evaluate(_FakeModel(), _LabeledDataset(dataset.input_ids, labels), device="cpu")
    dirty_test = _reported_eval_acc(dirty, "test_acc")

    assert clean_test > 0.9, f"sanity: the probe should solve a separable task, got {clean}"
    assert clean_test - dirty_test > 0.5, (
        "relabelling the test split left the reported accuracy at "
        f"{dirty_test} (was {clean_test}): it was scored on the data it was fit on."
    )


def test_linear_probe_train_and_test_are_measurements_of_different_data():
    """train/val/test must be three different measurements, not one number thrice.

    Proven exactly: corrupting only the test labels leaves train and val
    bit-identical and moves test. A single shared accuracy cannot do that.
    """
    probe = _linear(max_epochs=20)
    dataset = _separable_dataset()
    clean = probe.evaluate(_FakeModel(), dataset, device="cpu")

    for key in ("train_acc", "val_acc", "test_acc"):
        assert key in clean, f"missing {key}: a reader cannot tell which is which ({clean})"
        assert 0.0 <= clean[key] <= 1.0
    assert (
        "accuracy" not in clean
    ), "'accuracy' is an unsplit name that hides whether it is train or held-out"
    sizes = [clean[k] for k in ("n_train", "n_val", "n_test") if k in clean]
    if sizes:
        assert sum(sizes) == _N, f"reported split sizes do not cover the dataset: {sizes}"

    _train, _val, test_idx = _split_of(probe, len(dataset))
    labels = dataset.labels.clone()
    labels[test_idx] = (labels[test_idx] + 1) % 2
    dirty = probe.evaluate(_FakeModel(), _LabeledDataset(dataset.input_ids, labels), device="cpu")

    assert dirty["train_acc"] == clean["train_acc"], (
        "training accuracy moved when only test labels changed: the fit path "
        f"touches the test split ({clean['train_acc']} -> {dirty['train_acc']})"
    )
    assert dirty["val_acc"] == clean["val_acc"], "model selection moved with the test labels"
    assert dirty["test_acc"] != clean["test_acc"], "the test metric ignored the test labels"


def test_linear_probe_hyperparameters_are_selected_on_val_only():
    result = _linear(max_epochs=20, seed=_SEED, lr_candidates=(1e-3, 1e-2, 1e-1)).evaluate(
        _FakeModel(), _separable_dataset()
    )
    assert "selected_lr" in result, "no record of which hyperparameter was chosen"
    assert result["selected_lr"] in (1e-3, 1e-2, 1e-1)
    assert result["test_acc"] > 0.9, f"sanity: no candidate should break the task ({result})"


def test_linear_probe_is_deterministic():
    model = _FakeModel()
    dataset = _separable_dataset()
    first = _linear(max_epochs=20, seed=_SEED).evaluate(model, dataset)
    second = _linear(max_epochs=20, seed=_SEED).evaluate(model, dataset)
    assert first == second, f"same seed must reproduce the run: {first} != {second}"


def test_linear_probe_does_not_perturb_the_global_rng():
    torch.manual_seed(5)
    before = torch.rand(4)
    torch.manual_seed(5)
    _linear(max_epochs=5, seed=_SEED).evaluate(_FakeModel(), _separable_dataset())
    after = torch.rand(4)
    assert torch.equal(before, after), "evaluate() reseeded the caller's global RNG"


# --------------------------------------------------------------------------------------
# FutureTokenProbe
# --------------------------------------------------------------------------------------

_STEPS = 300
_EVAL_EVERY = 100


@pytest.fixture(scope="module")
def future_clean():
    """One trained run, reused read-only by the tests that need the baseline."""
    return _future().evaluate(
        _FakeModel(), _token_dataset(), max_steps=_STEPS, eval_every=_EVAL_EVERY
    )


def test_future_probe_reports_held_out_top1_and_top5(future_clean):
    for d in (1, 2):
        for split in ("train", "val", "test"):
            assert f"future_probe_d{d}_{split}_top1" in future_clean, f"missing d{d} {split} top1"
            assert f"future_probe_d{d}_{split}_top5" in future_clean, f"missing d{d} {split} top5"
            assert 0.0 <= future_clean[f"future_probe_d{d}_{split}_top1"] <= 1.0
            assert (
                future_clean[f"future_probe_d{d}_{split}_top5"]
                >= future_clean[f"future_probe_d{d}_{split}_top1"]
            ), "top-5 accuracy can never be below top-1"
        assert (
            future_clean[f"future_probe_d{d}_n_test_positions"] > 0
        ), "an accuracy with no denominator"


def test_future_probe_has_no_ambiguous_train_accuracy_key(future_clean):
    for d in (1, 2):
        assert f"future_probe_d{d}" not in future_clean, (
            f"'future_probe_d{d}' is an unsplit name, and the value the old code put "
            f"under it was a running TRAINING accuracy. keys={sorted(future_clean)}"
        )


def test_future_probe_actually_trains_across_more_than_one_pass(future_clean):
    """max_steps is a step budget, not a single pass over the loader."""
    untrained = _future().evaluate(_FakeModel(), _token_dataset(), max_steps=0, eval_every=0)
    trained = future_clean
    assert trained["future_probe_d1_train_top1"] > untrained["future_probe_d1_train_top1"] + 0.3, (
        "the readout barely moved after "
        f"{_STEPS} steps: {untrained['future_probe_d1_train_top1']} -> "
        f"{trained['future_probe_d1_train_top1']}"
    )
    assert trained["future_probe_d1_test_top1"] > 0.5, (
        "a learnable task should transfer to held-out data: "
        f"{trained['future_probe_d1_test_top1']}"
    )


def test_future_probe_test_metric_responds_to_scrambling_the_test_split(future_clean):
    """Decisive: scramble the held-out sequences; the reported test accuracy must move."""
    probe = _future()
    dataset = _token_dataset()
    clean_test = _reported_eval_acc(future_clean, "future_probe_d1_test_top1")

    _train, _val, test_idx = _split_of(probe, len(dataset))
    scrambled = _scramble(dataset.input_ids, test_idx)
    dirty = probe.evaluate(
        _FakeModel(), _TokenDataset(scrambled), max_steps=_STEPS, eval_every=_EVAL_EVERY
    )
    dirty_test = _reported_eval_acc(dirty, "future_probe_d1_test_top1")

    assert clean_test - dirty_test > 0.1, (
        "scrambling the held-out sequences did not move the reported test accuracy "
        f"({clean_test} -> {dirty_test}): it is a training score."
    )


def test_future_probe_fit_is_unaffected_by_test_data(future_clean):
    probe = _future()
    dataset = _token_dataset()
    _train, _val, test_idx = _split_of(probe, len(dataset))
    scrambled = _scramble(dataset.input_ids, test_idx)
    dirty = probe.evaluate(
        _FakeModel(), _TokenDataset(scrambled), max_steps=_STEPS, eval_every=_EVAL_EVERY
    )

    for d in (1, 2):
        for split in ("train", "val"):
            key = f"future_probe_d{d}_{split}_top1"
            assert dirty[key] == future_clean[key], (
                f"{key} was perturbed by held-out data: the fit path iterates the "
                f"whole dataset ({future_clean[key]} -> {dirty[key]})"
            )


def test_future_probe_is_deterministic():
    model = _FakeModel()
    dataset = _token_dataset()
    first = _future().evaluate(model, dataset, max_steps=_STEPS, eval_every=_EVAL_EVERY)
    second = _future().evaluate(model, dataset, max_steps=_STEPS, eval_every=_EVAL_EVERY)
    assert first == second, f"same seed must reproduce the run: {first} != {second}"


def test_future_probe_terminates_when_no_sequence_is_long_enough():
    """A dataset with T <= offset has no usable batch; the loop must not spin."""
    short = torch.zeros(_N, 2, dtype=torch.long)
    probe = probes.FutureTokenProbe(embed_dim=_EMBED, vocab_size=_TABLE, offsets=(8,), seed=_SEED)
    result = probe.evaluate(_FakeModel(), _TokenDataset(short), max_steps=5, eval_every=1)
    assert result["future_probe_d8_n_test_positions"] == 0.0, (
        "an offset longer than the sequence scores nothing, and must say so "
        f"rather than report a 0/1 sentinel: {result}"
    )
    assert result["future_probe_d8_test_top1"] == 0.0


def test_future_probe_does_not_perturb_the_global_rng():
    dataset = _token_dataset()
    torch.manual_seed(5)
    before = torch.rand(4)
    torch.manual_seed(5)
    _future().evaluate(_FakeModel(), dataset, max_steps=20, eval_every=10)
    after = torch.rand(4)
    assert torch.equal(before, after), "evaluate() reseeded the caller's global RNG"


# --------------------------------------------------------------------------------------
# Repo constraints
# --------------------------------------------------------------------------------------


def test_probes_default_to_cpu_device():
    """This repo is CPU-only; a cuda default makes every default call a crash."""
    assert inspect.signature(probes.LinearProbe.evaluate).parameters["device"].default == "cpu"
    assert inspect.signature(probes.FutureTokenProbe.evaluate).parameters["device"].default == "cpu"


@pytest.mark.parametrize("cls", [probes.LinearProbe, probes.FutureTokenProbe])
def test_probes_expose_explicit_split_and_seed_parameters(cls):
    params = inspect.signature(cls.__init__).parameters
    assert "seed" in params, "no seed parameter: the probes are unseeded"
    assert "train_frac" in params and "val_frac" in params, "split fractions are not explicit"
