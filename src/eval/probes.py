# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Evaluation: linear probes, future-token probes, geometry metrics
# Following NextLat (Microsoft, 2025) / I-JEPA evaluation protocols
#
# Both probes fit on a TRAIN split only, select on a VAL split only, and report
# on a TEST split only. The split is drawn once per `evaluate` call from an
# explicit seed and is published by `split_indices`, so any reported number can
# be reproduced from the returned split sizes and seed.

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.models.collapse import CollapseDiagnostics

DEFAULT_SEED = 0
DEFAULT_TRAIN_FRAC = 0.6
DEFAULT_VAL_FRAC = 0.2
# A top-5 score is undefined for a readout with fewer than 5 outputs.
MIN_TOPK = 5


def split_indices(
    n: int,
    seed: int = DEFAULT_SEED,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic disjoint train/val/test index split of ``range(n)``.

    The permutation is drawn from a private ``torch.Generator`` so the caller's
    global RNG stream is neither consumed nor reseeded.

    Args:
        n: number of samples in the dataset.
        seed: master seed for the permutation.
        train_frac: fraction held out for fitting, in (0, 1).
        val_frac: fraction held out for model selection, in [0, 1). train_frac +
            val_frac must leave at least one sample for testing.

    Returns:
        (train_idx, val_idx, test_idx) — disjoint, together covering
        ``range(n)``, and non-empty whenever ``n >= 3``.

    """
    n = int(n)
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    if train_frac + val_frac >= 1.0:
        raise ValueError(f"train_frac + val_frac = {train_frac + val_frac} leaves no test split")
    if n < 3:
        raise ValueError(f"a 3-way split needs at least 3 samples, got {n}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator)

    n_train = min(max(1, round(train_frac * n)), n - 2)
    n_val = min(max(1, round(val_frac * n)), n - n_train - 1)
    return (
        perm[:n_train],
        perm[n_train : n_train + n_val],
        perm[n_train + n_val :],
    )


@contextlib.contextmanager
def _isolated_rng(seed: int):
    """Seed a private RNG stream for the duration of the block only."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield


def _loader(dataset, indices: torch.Tensor, batch_size: int, seed: int, shuffle: bool):
    """DataLoader over a split, with a private RNG so shuffling is reproducible."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _input_ids(batch) -> torch.Tensor:
    """Accept both the bare-tensor and the ``(input_ids, ...)`` dataset shapes."""
    if isinstance(batch, torch.Tensor):
        return batch
    first = batch[0]
    if isinstance(first, dict):
        return first["input_ids"]
    return first


class LinearProbe:
    """Linear probe: train a linear classifier on frozen representations.

    The classifier is fit on the train split, the learning rate is chosen on the
    val split, and accuracy is reported separately for all three splits. There is
    no unsplit ``accuracy`` key on purpose: a single number named that way hides
    whether it is a training score, which is the mistake this class used to make.
    """

    def __init__(
        self,
        embed_dim=768,
        num_classes=2,
        lr=1e-3,
        max_epochs=100,
        lr_candidates: Sequence[float] = (1e-3, 1e-2),
        train_frac: float = DEFAULT_TRAIN_FRAC,
        val_frac: float = DEFAULT_VAL_FRAC,
        seed: int = DEFAULT_SEED,
        batch_size: int = 64,
    ):
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.lr = lr
        self.max_epochs = max_epochs
        self.lr_candidates = tuple(lr_candidates) if lr_candidates else (lr,)
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.seed = int(seed)
        self.batch_size = batch_size

    # -- internals ---------------------------------------------------------------------

    def _pooled(self, model, input_ids, device) -> torch.Tensor:
        h, _ = model.encoder(input_ids)
        return h.mean(dim=1)

    def _fit(self, model, dataset, train_idx, lr, device) -> nn.Linear:
        with _isolated_rng(self.seed):
            classifier = nn.Linear(self.embed_dim, self.num_classes).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        loader = _loader(dataset, train_idx, self.batch_size, self.seed, shuffle=True)
        for _ in range(self.max_epochs):
            for batch in loader:
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)
                with torch.no_grad():
                    features = self._pooled(model, input_ids, device)
                loss = F.cross_entropy(classifier(features), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return classifier

    @torch.no_grad()
    def _accuracy(self, model, classifier, dataset, indices, device) -> float:
        if indices.numel() == 0:
            return 0.0
        correct = 0
        total = 0
        loader = _loader(dataset, indices, self.batch_size, self.seed, shuffle=False)
        for batch in loader:
            labels = batch[1].to(device)
            features = self._pooled(model, batch[0].to(device), device)
            correct += (classifier(features).argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
        return correct / total

    # -- public API --------------------------------------------------------------------

    def evaluate(self, model, dataset, device: str = "cpu") -> dict[str, float]:
        """Fit on train, select on val, report all three splits separately.

        Args:
            model: frozen encoder wrapper; ``model.encoder(input_ids)`` must
                return ``(h, aux)``.
            dataset: ``(input_ids, labels)`` dataset.
            device: torch device. Defaults to ``cpu``; this repo has no CUDA.

        Returns:
            Dict with ``train_acc`` / ``val_acc`` / ``test_acc`` (the only three
            accuracies worth quoting is ``test_acc``), plus ``selected_lr``, the
            split sizes and the seed that produced them.

        """
        model.eval()
        train_idx, val_idx, test_idx = split_indices(
            len(dataset), self.seed, self.train_frac, self.val_frac
        )

        best_lr = None
        best_val = -1.0
        best_classifier = None
        for lr in self.lr_candidates:
            classifier = self._fit(model, dataset, train_idx, lr, device)
            val_acc = self._accuracy(model, classifier, dataset, val_idx, device)
            if val_acc > best_val:
                best_val = val_acc
                best_lr = lr
                best_classifier = classifier
        if best_classifier is None:  # empty candidate list
            best_classifier = self._fit(model, dataset, train_idx, self.lr, device)
            best_lr = self.lr

        return {
            "train_acc": self._accuracy(model, best_classifier, dataset, train_idx, device),
            "val_acc": self._accuracy(model, best_classifier, dataset, val_idx, device),
            "test_acc": self._accuracy(model, best_classifier, dataset, test_idx, device),
            "selected_lr": float(best_lr),
            "n_train": float(train_idx.numel()),
            "n_val": float(val_idx.numel()),
            "n_test": float(test_idx.numel()),
            "seed": float(self.seed),
        }


class FutureTokenProbe:
    """Future-token probe from NextLat: predictive information in representations.

    Each offset readout is fit on the train split, the best step is chosen on the
    val split, and top-1 / top-5 are reported on a fresh ``torch.no_grad`` forward
    pass over each split. The logits used for reporting are never the ones that
    produced the loss, which is what used to make ``future_probe_d{d}`` a running
    training accuracy rather than a measurement.
    """

    def __init__(
        self,
        embed_dim=768,
        vocab_size=50304,
        offsets=(1, 4, 16),
        lr=1e-3,
        train_frac: float = DEFAULT_TRAIN_FRAC,
        val_frac: float = DEFAULT_VAL_FRAC,
        seed: int = DEFAULT_SEED,
        batch_size: int = 32,
    ):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.offsets = tuple(offsets)
        self.lr = lr
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.seed = int(seed)
        self.batch_size = batch_size
        with _isolated_rng(self.seed):
            self.probes = nn.ModuleDict(
                {f"offset_{d}": nn.Linear(embed_dim, vocab_size) for d in self.offsets}
            )
        self.reset_probes()

    def reset_probes(self) -> None:
        """Redraw all readouts from ``self.seed`` (does not touch the global RNG)."""
        with _isolated_rng(self.seed):
            for d in self.offsets:
                nn.init.normal_(self.probes[f"offset_{d}"].weight, std=0.02)
                nn.init.zeros_(self.probes[f"offset_{d}"].bias)

    # -- internals ---------------------------------------------------------------------

    def _pairs(self, model, probe, input_ids, d, device):
        """(source representation, target token) pairs for offset ``d``."""
        _B, T = input_ids.shape
        if d >= T:
            return None
        with torch.no_grad():
            h, _ = model.encoder(input_ids)
        h_src = h[:, : T - d, :].reshape(-1, h.size(-1))
        target = input_ids[:, d:].reshape(-1)
        return probe(h_src), target

    @torch.no_grad()
    def _topk_accuracy(self, model, probe, dataset, indices, d, device) -> tuple[float, float, int]:
        top1 = 0
        top5 = 0
        n = 0
        k5 = min(MIN_TOPK, self.vocab_size)
        loader = _loader(dataset, indices, self.batch_size, self.seed, shuffle=False)
        for batch in loader:
            pairs = self._pairs(model, probe, _input_ids(batch).to(device), d, device)
            if pairs is None:
                continue
            logits, target = pairs
            top1 += (logits.argmax(dim=-1) == target).sum().item()
            hits = logits.topk(k5, dim=-1).indices == target.unsqueeze(-1)
            top5 += hits.any(dim=-1).sum().item()
            n += target.size(0)
        if n == 0:
            return 0.0, 0.0, 0
        return top1 / n, top5 / n, n

    # -- public API --------------------------------------------------------------------

    def evaluate(
        self,
        model,
        dataset,
        device: str = "cpu",
        max_steps: int = 5000,
        eval_every: int = 100,
    ) -> dict[str, float]:
        """Fit each offset readout on train and report held-out top-1 / top-5.

        The readouts are redrawn from ``self.seed`` first, so the call is
        hermetic: it does not depend on how the probe happened to be constructed
        and it does not consume the caller's global RNG.

        Args:
            model: frozen encoder wrapper; ``model.encoder(input_ids)`` must
                return ``(h, aux)``.
            dataset: bare-token dataset (``Tensor`` or dict with ``input_ids``).
            device: torch device. Defaults to ``cpu``; this repo has no CUDA.
            max_steps: number of train batches per offset (the budget spans
                epochs; batches too short for the offset consume budget but do
                not produce a loss).
            eval_every: val-checkpoint cadence in steps; 0 disables selection
                and leaves the readout at its final train state.

        Returns:
            Dict of ``future_probe_d{d}_{train,val,test}_{top1,top5}`` plus
            ``future_probe_d{d}_n_test_positions``. There is deliberately no
            bare ``future_probe_d{d}`` key.

        """
        model.eval()
        train_idx, val_idx, test_idx = split_indices(
            len(dataset), self.seed, self.train_frac, self.val_frac
        )
        self.reset_probes()
        self.probes = self.probes.to(device)

        results: dict[str, float] = {}
        for d in self.offsets:
            probe = self.probes[f"offset_{d}"]
            optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr)
            loader = _loader(dataset, train_idx, self.batch_size, self.seed, shuffle=True)

            best_state = None
            best_val = -1.0
            step = 0
            # The step budget spans epochs: a single pass over the loader is one
            # batch when the split fits in batch_size, so a bare `enumerate(loader)`
            # would silently make max_steps meaningless. `step` counts every batch
            # consumed, including batches too short for this offset, so that a
            # dataset with no usable batch terminates instead of spinning.
            while step < max_steps:
                for batch in loader:
                    if step >= max_steps:
                        break
                    step += 1
                    pairs = self._pairs(model, probe, _input_ids(batch).to(device), d, device)
                    if pairs is None:  # sequence shorter than the offset
                        continue
                    logits, target = pairs
                    loss = F.cross_entropy(logits, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if eval_every and step % eval_every == 0:
                        val_top1, _val_top5, _n = self._topk_accuracy(
                            model, probe, dataset, val_idx, d, device
                        )
                        if val_top1 > best_val:
                            best_val = val_top1
                            best_state = {
                                k: v.detach().clone() for k, v in probe.state_dict().items()
                            }
            if best_state is not None:
                probe.load_state_dict(best_state)

            for split, indices in (
                ("train", train_idx),
                ("val", val_idx),
                ("test", test_idx),
            ):
                top1, top5, n = self._topk_accuracy(model, probe, dataset, indices, d, device)
                results[f"future_probe_d{d}_{split}_top1"] = top1
                results[f"future_probe_d{d}_{split}_top5"] = top5
                if split == "test":
                    results[f"future_probe_d{d}_n_test_positions"] = float(n)
        return results


class GeometryMetrics:
    """Representation geometry metrics — reuses CollapseDiagnostics.

    From I-JEPA, NextLat, VICReg, Barlow Twins, DINO, C-JEPA,
    BYOL, Kornblith CKA, LeCun JEPA, Ansuini intrinsic dim.
    Exception handling follows NextLat: try/except returns zeros/infs.
    """

    _diag = CollapseDiagnostics()

    @staticmethod
    @torch.no_grad()
    def compute(representations):
        if representations.dim() == 3:
            B, T, D = representations.shape
            N = B * T
        else:
            N, D = representations.shape

        metrics = {}
        try:
            d = GeometryMetrics._diag
            # NextLat metrics
            metrics["effective_rank"] = d._effective_rank(representations)
            metrics["participation_ratio"] = d._participation_ratio(representations)
            metrics["condition_number"] = d._condition_number(representations)
            metrics["numerical_rank"] = d._numerical_rank(representations)
            metrics["rank_utilization"] = (
                metrics["numerical_rank"] / min(N, D) if min(N, D) > 0 else 0.0
            )
            metrics["coherence"] = d._coherence(representations)

            # I-JEPA metrics
            metrics["collapsed_dim_ratio"] = d._collapsed_dim_ratio(representations)
            metrics["sv_entropy"] = d._singular_value_entropy(representations)

            # C-JEPA / BYOL
            metrics["svd_sharpness"] = d._svd_sharpness(representations)

            # LeCun 2022
            metrics["alpha_norm"] = d._alpha_norm(representations)

            # Ansuini et al. 2019
            metrics["intrinsic_dim"] = d._intrinsic_dim_score(representations)

            # DINOv2
            flat = representations.reshape(-1, representations.size(-1))
            metrics["mean_pairwise_cosine"] = d._mean_pairwise_cosine(flat)

            # Wang & Isola (ICLR 2022)
            metrics["uniformity"] = d._uniformity(flat)

            # DINO
            metrics["cov_trace"] = d._feature_covariance_trace(representations)

        except Exception as e:
            metrics["error"] = str(e)
            for key in [
                "effective_rank",
                "participation_ratio",
                "numerical_rank",
                "rank_utilization",
                "coherence",
                "collapsed_dim_ratio",
                "sv_entropy",
                "svd_sharpness",
                "alpha_norm",
                "intrinsic_dim",
                "mean_pairwise_cosine",
            ]:
                metrics.setdefault(key, 0.0)
            metrics.setdefault("condition_number", float("inf"))

        return metrics
