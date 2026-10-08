# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
#
# Probing Complexity Curve (PCC)
#
# THE KEY METRIC FOR THE PAPER'S CENTRAL CLAIM
#
# Not "can a linear probe extract X?" but "what's the MINIMUM probe
# complexity needed?" If JEPA needs a linear probe and MLM needs a
# 2-layer MLP for the same linguistic feature, that's quantitative
# evidence that JEPA representations are more accessible/structured.
#
# Inspired by:
# - Hewitt & Manning (2019): structural probing
# - Alain & Bengio (2017): understanding intermediate layers
# - Pimentel et al. (2023): probing pareto frontier
# - Conneau et al. (2018): probing linguistic features across layers
#
# ── WHAT IS HELD OUT HERE ────────────────────────────────────────────────────
# `min_extracting_depth`, which the header above calls "THE KEY METRIC FOR THE
# PAPER'S CENTRAL CLAIM", used to be decided by `max over epochs of validation
# accuracy` on the only partition that existed. There was no test set, so the
# number that decided a depth was the maximum of a noisy quantity rather than
# the accuracy of the probe that depth actually selected.
#
# The partition was also redrawn for EVERY depth and EVERY model, from the
# process-global RNG, so a JEPA-vs-MLM comparison compared two unrelated
# experiments. Measured on byte-identical data and a fixed global seed, two
# consecutive `evaluate` calls returned different numbers; and
# `compare_models(reps, reps.clone())` -- the same representations in both
# arms -- reported a per-depth advantage of up to 0.625. A comparison whose
# two arms are the same array must return exactly zero.
#
# Now: one partition per `evaluate()` call, drawn from a private generator and
# shared by every depth; the same partition and the same per-depth initial
# weights handed to both arms of `compare_models`; the epoch selected on a
# validation third; and the reported accuracy computed once on a test third the
# optimiser never saw. `compare_models(reps, reps.clone())` is now exactly
# 0.0, and two `evaluate` calls at a fixed seed are identical.
#
# KNOWN CONFOUND, reported not fixed. `min_extracting_depth` is still set by
# the early-stopping budget, not only by the representation. `patience` counts
# non-improvements of a validation accuracy measured on `n_val` rows, so with
# the shipped defaults (patience 5, max_epochs 30) the probe stops before it
# converges. Measured on linearly separable data where a linear probe reaches
# 0.925 held out: patience 5 / max_epochs 30 gives 0.700 and
# `min_extracting_depth` = 3 (i.e. "not extractable at any depth"); patience 25
# / max_epochs 400 gives 0.875 and `min_extracting_depth` = 1. Raising
# max_epochs alone changes nothing, because patience fires first. The
# threshold crossing is therefore a statement about the selection budget as
# well as about the representation. Raising the budget is a science decision
# and is not made here; `val_depths` and `max_val_accuracy` are reported so the
# selection score is visible next to the reported one.
#
# Also unfixed, and reported in .agent-notes/task-11.md: the threshold
# comparison is `test_accuracy >= min_accuracy`, where the accuracy is a float32
# mean and the threshold is a Python float. An accuracy that is exactly 0.7 in
# exact arithmetic comes out of that mean as 0.69999998, so it fails a
# `min_accuracy=0.7` threshold. `min_extracting_depth` can therefore flip on a
# one-ulp difference.

import torch
import torch.nn.functional as F
from torch import nn

#: Offset added to the seed to derive the probe-weight stream. Kept clear of
#: the split stream's seeds so changing one partition cannot move the other.
#: Mirrors `layer_analysis.LayerwiseProbe`.
WEIGHT_STREAM_OFFSET = 1

#: Fractions of the rows given to training, epoch selection, and the reported
#: test score, in that order. The two tail fractions are equal so neither the
#: epoch choice nor the reported number gets the larger share.
DEFAULT_SPLIT_FRACTIONS = (0.6, 0.2, 0.2)


class ProbingComplexityCurve:
    """Probing Complexity Curve: minimum probe depth needed to extract
    linguistic information from representations.

    For each linguistic feature (POS, syntactic depth, entity type, etc.),
    train probes at depths 1 (linear), 2 (1-hidden MLP), 3, 4, etc.
    and record the accuracy at each depth.

    The "probing complexity gap" between JEPA and MLM at each feature
    directly tests the hypothesis: JEPA representations encode
    linguistic information more accessibly (requiring simpler probes).

    Key output: ProbingComplexityGap = min_depth(JEPA) - min_depth(MLM)
    Positive = JEPA needs deeper probe (bad)
    Negative = JEPA needs shallower probe (good - evidence for hypothesis)

    The reported per-depth accuracies are TEST accuracies, on rows the probe
    was neither trained nor epoch-selected on. See the module header for what
    that replaces and for the measurement of the split-resampling noise.
    """

    def __init__(
        self,
        embed_dim=768,
        num_classes=None,
        depths=(1, 2, 3, 4),
        hidden_mult=2,
        lr=1e-3,
        max_epochs=50,
        patience=5,
        min_accuracy=0.7,
        device="cpu",
        seed=0,
        split_fractions=DEFAULT_SPLIT_FRACTIONS,
    ):
        """
        Args:
            embed_dim: dimension of representations
            num_classes: number of output classes (None = auto-detect)
            depths: tuple of probe depths to evaluate
            hidden_mult: hidden layer width multiplier (embed_dim * hidden_mult)
            lr: learning rate
            max_epochs: max training epochs per probe
            patience: early stopping patience
            min_accuracy: minimum accuracy threshold for "extractable"
            device: compute device
            seed: seeds both the row partition and the probe weights, each
                from its own private stream, so the whole curve is
                reproducible without consuming the caller's global RNG
            split_fractions: (train, validation, test) row fractions

        """
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.depths = depths
        self.hidden_mult = hidden_mult
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_accuracy = min_accuracy
        self.device = device
        self.seed = int(seed)
        self.split_fractions = tuple(split_fractions)

    def train_val_test_split(self, n, seed=None):
        """Draw ONE (train, val, test) row partition.

        Public and deterministic so the partition behind a reported curve can
        be reproduced, and reused, without retraining the probes.

        Args:
            n: number of rows to partition
            seed: overrides ``self.seed``; pass it when the partition should
                be pinned independently of the instance

        Returns:
            ``(train_idx, val_idx, test_idx)``, disjoint 1-D int64 tensors
            whose union is ``range(n)``.

        Raises:
            ValueError: if any of the three sides would be empty. An empty
                test side has no accuracy to report and an empty validation
                side leaves the epoch choice arbitrary, so both are refused
                rather than silently collapsed.

        """
        fr = tuple(float(f) for f in self.split_fractions)
        if len(fr) != 3 or any(f <= 0.0 for f in fr) or sum(fr) > 1.0:
            raise ValueError(
                f"split_fractions must be three positive shares summing to at "
                f"most 1, got {self.split_fractions!r}"
            )
        n = int(n)
        n_train = round(fr[0] * n)
        n_val = round(fr[1] * n)
        n_test = n - n_train - n_val
        if min(n_train, n_val, n_test) < 1:
            raise ValueError(
                f"split_fractions={self.split_fractions!r} on n={n} leave "
                f"{n_train} train / {n_val} validation / {n_test} test rows; "
                f"every side must be non-empty. Use more samples, or coarser "
                f"fractions."
            )
        gen = torch.Generator().manual_seed(self.seed if seed is None else int(seed))
        idx = torch.randperm(n, generator=gen)
        return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]

    def weight_generator(self, depth):
        """Private generator for one depth's probe weights.

        Keyed on depth rather than advanced in a loop, so the initialisation
        a given depth gets does not depend on which other depths were
        evaluated, and so the two arms of `compare_models` receive identical
        initial weights at every depth.

        """
        return torch.Generator().manual_seed(self.seed + WEIGHT_STREAM_OFFSET + int(depth))

    def seeded_probe(self, depth, num_classes, generator=None):
        """Build the depth-`depth` probe with weights from `generator`.

        `_build_probe` constructs stock `nn.Linear` layers, which draw their
        initialisation from the process-global RNG. The two arms of
        `compare_models` therefore started from different points, which is a
        second noise channel on top of the redrawn split. This reseeds the
        layers in place, in order, reproducing torch's own law from a private
        generator.

        """
        probe = self._build_probe(depth, num_classes)
        if generator is None:
            return probe
        for module in probe.modules():
            if not isinstance(module, nn.Linear):
                continue
            bound = module.in_features**-0.5
            with torch.no_grad():
                module.weight.copy_(
                    torch.empty_like(module.weight).uniform_(-bound, bound, generator=generator)
                )
                if module.bias is not None:
                    module.bias.copy_(
                        torch.empty_like(module.bias).uniform_(-bound, bound, generator=generator)
                    )
        return probe

    def _build_probe(self, depth, num_classes):
        """Build a probe at the given depth.

        depth=1: Linear(D, C)
        depth=2: Linear(D, D*2) → ReLU → Linear(D*2, C)
        depth=3: Linear(D, D*2) → ReLU → Linear(D*2, D) → ReLU → Linear(D, C)
        etc.
        """
        layers = []
        in_dim = self.embed_dim
        for i in range(depth - 1):
            out_dim = self.embed_dim * self.hidden_mult if i == 0 else in_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_classes))
        return nn.Sequential(*layers)

    def _train_probe(self, probe, representations, labels, split):
        """Train one probe with early stopping on a validation partition.

        The epoch and the weights are selected on the validation split. The
        test split is scored exactly once, afterwards, on the restored best
        state, so the returned ``test_accuracy`` is not the maximum of
        anything and neither the optimiser nor the model selection has seen
        it.

        Args:
            probe: nn.Module probe, already carrying its final weights
            representations: (N, D)
            labels: (N,) class indices
            split: ``(train_idx, val_idx, test_idx)``, from
                :meth:`train_val_test_split`. **Supply this whenever you
                compare two or more representations**; omitting it draws a
                fresh partition, which is only correct for a standalone
                probe.

        Returns:
            dict with ``test_accuracy``, ``val_accuracy`` (the best-epoch
            selection score), and ``train_accuracy``.

        """
        N = representations.size(0)
        if N < 10:
            return {"test_accuracy": 0.0, "val_accuracy": 0.0, "train_accuracy": 0.0}

        train_idx, val_idx, test_idx = split

        # Ensure representations are float and require_grad compatible
        representations = representations.detach().float()
        # Probe needs to be trainable
        probe = probe.to(self.device)
        probe.train()

        train_reps = representations[train_idx].to(self.device)
        train_labels = labels[train_idx].to(self.device)
        val_reps = representations[val_idx].to(self.device)
        val_labels = labels[val_idx].to(self.device)
        test_reps = representations[test_idx].to(self.device)
        test_labels = labels[test_idx].to(self.device)

        optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        def _acc(x, y):
            with torch.no_grad():
                return (probe(x).argmax(dim=-1) == y).float().mean().item()

        best_acc = 0.0
        best_state = None
        no_improve = 0

        for _epoch in range(self.max_epochs):
            # Train
            probe.train()
            logits = probe(train_reps)
            loss = F.cross_entropy(logits, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Validate
            probe.eval()
            acc = _acc(val_reps, val_labels)

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                break

        if best_state is not None:
            probe.load_state_dict(best_state)

        probe.eval()
        return {
            "test_accuracy": _acc(test_reps, test_labels),
            "val_accuracy": best_acc,
            "train_accuracy": _acc(train_reps, train_labels),
        }

    def evaluate(self, representations, labels, task_name="default", split=None):
        """Evaluate probing complexity for a single task.

        All depths are trained and scored on ONE partition, drawn here unless
        the caller supplies it. The accuracies are therefore comparable to
        each other, which is the entire point of a complexity curve:
        ``min_extracting_depth`` is a threshold crossing over them, and a
        threshold crossing over measurements taken on different partitions is
        a statistic of the partitions.

        Args:
            representations: (N, D) representation vectors
            labels: (N,) integer class labels
            task_name: name of the linguistic task
            split: optional ``(train_idx, val_idx, test_idx)`` from
                :meth:`train_val_test_split`

        Returns:
            dict with per-depth TEST accuracy, the best-epoch validation
            score alongside it, the minimum extracting depth, and the
            partition's provenance

        """
        num_classes = self.num_classes or labels.max().item() + 1
        num_classes = max(int(num_classes), 2)

        if split is None:
            split = self.train_val_test_split(representations.size(0))
        train_idx, val_idx, test_idx = split

        results = {
            "task": task_name,
            # The number the module exists to produce. Now a held-out score.
            "depths": {},
            "val_depths": {},
            "min_extracting_depth": None,
        }

        for depth in self.depths:
            probe = self.seeded_probe(depth, num_classes, self.weight_generator(depth))
            # _train_probe needs gradients - use enable_grad context
            with torch.enable_grad():
                scores = self._train_probe(probe, representations, labels, split)
            results["depths"][depth] = scores["test_accuracy"]
            results["val_depths"][depth] = scores["val_accuracy"]

            # First depth whose HELD-OUT accuracy exceeds the threshold
            if (
                scores["test_accuracy"] >= self.min_accuracy
                and results["min_extracting_depth"] is None
            ):
                results["min_extracting_depth"] = depth

        # If no depth reached threshold, set to max + 1
        if results["min_extracting_depth"] is None:
            results["min_extracting_depth"] = max(self.depths) + 1

        results["max_accuracy"] = max(results["depths"].values())
        # The same statistic as `max_accuracy` was before this module had a
        # test split: the best-epoch validation score. Reported next to it
        # because the gap between the two IS the optimism the old number
        # carried, and it is the size of that optimism a reader needs.
        results["max_val_accuracy"] = max(results["val_depths"].values())
        results["n_train"] = int(train_idx.numel())
        results["n_val"] = int(val_idx.numel())
        results["n_test"] = int(test_idx.numel())
        results["split_seed"] = self.seed

        return results

    def compare_models(self, jepa_reps, baseline_reps, labels, task_name="default"):
        """Compare probing complexity between JEPA and baseline.

        THE CORE COMPARISON for the paper.

        Both arms are evaluated on the SAME partition and with the SAME
        per-depth initial weights, because they carry the same row count and
        the same architecture at each depth; the only thing that differs is
        the representation. Previously each arm drew its own split per depth,
        so this returned two unrelated experiments.

        Args:
            jepa_reps: (N, D) JEPA representations
            baseline_reps: (N, D) baseline representations
            labels: (N,) class labels
            task_name: name of linguistic task

        Returns:
            dict with complexity gap, per-depth comparison, and the shared
            partition's provenance

        """
        split = self.train_val_test_split(jepa_reps.size(0))

        jepa_result = self.evaluate(jepa_reps, labels, f"{task_name}_jepa", split=split)
        baseline_result = self.evaluate(baseline_reps, labels, f"{task_name}_baseline", split=split)

        # Probing Complexity Gap: negative = JEPA is more accessible
        complexity_gap = (
            jepa_result["min_extracting_depth"] - baseline_result["min_extracting_depth"]
        )

        # Per-depth accuracy comparison
        depth_comparison = {}
        for depth in self.depths:
            j_acc = jepa_result["depths"].get(depth, 0)
            b_acc = baseline_result["depths"].get(depth, 0)
            depth_comparison[depth] = {
                "jepa_accuracy": j_acc,
                "baseline_accuracy": b_acc,
                "jepa_advantage": j_acc - b_acc,
            }

        return {
            "task": task_name,
            "jepa_min_depth": jepa_result["min_extracting_depth"],
            "baseline_min_depth": baseline_result["min_extracting_depth"],
            "complexity_gap": complexity_gap,
            "jepa_more_accessible": complexity_gap < 0,
            "depth_comparison": depth_comparison,
            "jepa_max_acc": jepa_result["max_accuracy"],
            "baseline_max_acc": baseline_result["max_accuracy"],
            "jepa_max_val_acc": jepa_result["max_val_accuracy"],
            "baseline_max_val_acc": baseline_result["max_val_accuracy"],
            "n_train": jepa_result["n_train"],
            "n_val": jepa_result["n_val"],
            "n_test": jepa_result["n_test"],
            "split_seed": self.seed,
        }

    def multi_task_comparison(self, jepa_reps_dict, baseline_reps_dict, labels_dict):
        """Compare probing complexity across multiple linguistic tasks.

        Args:
            jepa_reps_dict: {task_name: (N, D)} JEPA representations per task
            baseline_reps_dict: {task_name: (N, D)} baseline representations per task
            labels_dict: {task_name: (N,)} labels per task

        Returns:
            dict with per-task results and aggregate summary

        """
        results = {}
        complexity_gaps = []

        for task_name in jepa_reps_dict:
            if task_name not in baseline_reps_dict or task_name not in labels_dict:
                continue
            result = self.compare_models(
                jepa_reps_dict[task_name],
                baseline_reps_dict[task_name],
                labels_dict[task_name],
                task_name,
            )
            results[task_name] = result
            complexity_gaps.append(result["complexity_gap"])

        # Aggregate
        if complexity_gaps:
            avg_gap = sum(complexity_gaps) / len(complexity_gaps)
            n_jepa_better = sum(1 for g in complexity_gaps if g < 0)
        else:
            avg_gap = 0.0
            n_jepa_better = 0

        results["_summary"] = {
            "avg_complexity_gap": avg_gap,
            "n_tasks_jepa_more_accessible": n_jepa_better,
            "n_tasks_total": len(complexity_gaps),
            "fraction_jepa_better": n_jepa_better / max(len(complexity_gaps), 1),
        }

        return results


class LinguisticProbeTasks:
    """Standard linguistic probing tasks for Probing Complexity Curve.

    Following Conneau et al. (2018) and Tenney et al. (2019):
    - Surface: token length, word frequency
    - Syntactic: POS tags, dependency depth
    - Semantic: entity type, sentiment
    """

    @staticmethod
    def pos_tagging(representations, tokens_list, pos_tags_list):
        """POS tagging probe task.

        Args:
            representations: (N, D) pooled word representations
            tokens_list: list of token strings (for reference)
            pos_tags_list: list of POS tag indices (N,)

        Returns:
            dict ready for ProbingComplexityCurve.evaluate()

        """
        labels = torch.tensor(pos_tags_list, dtype=torch.long)
        return {
            "representations": representations,
            "labels": labels,
            "task_name": "pos_tagging",
            "num_classes": int(labels.max().item()) + 1,
        }

    @staticmethod
    def syntactic_depth(representations, depth_values, n_bins=5):
        """Syntactic tree depth probe task.

        Args:
            representations: (N, D)
            depth_values: (N,) continuous depth values
            n_bins: number of bins for discretization

        Returns:
            dict ready for evaluate()

        """
        depths = (
            torch.tensor(depth_values, dtype=torch.float32)
            if not isinstance(depth_values, torch.Tensor)
            else depth_values.clone().detach().float()
        )
        # Discretize into bins
        percentiles = torch.linspace(0, 100, n_bins + 1)[1:-1]
        bins = torch.tensor([torch.quantile(depths, p / 100).item() for p in percentiles])
        labels = torch.bucketize(depths, bins)
        return {
            "representations": representations,
            "labels": labels,
            "task_name": "syntactic_depth",
            "num_classes": n_bins,
        }

    @staticmethod
    def word_length(representations, token_lengths, n_bins=5):
        """Word length probe (surface-level baseline).

        Args:
            representations: (N, D)
            token_lengths: (N,) word lengths
            n_bins: bins for discretization

        Returns:
            dict ready for evaluate()

        """
        lengths = torch.tensor(token_lengths, dtype=torch.float32)
        percentiles = torch.linspace(0, 100, n_bins + 1)[1:-1]
        bins = torch.tensor([torch.quantile(lengths, p / 100).item() for p in percentiles])
        labels = torch.bucketize(lengths, bins)
        return {
            "representations": representations,
            "labels": labels,
            "task_name": "word_length",
            "num_classes": n_bins,
        }

    @staticmethod
    def entity_type(representations, entity_labels):
        """Named entity type classification probe.

        Args:
            representations: (N, D)
            entity_labels: (N,) integer entity type labels

        Returns:
            dict ready for evaluate()

        """
        labels = torch.tensor(entity_labels, dtype=torch.long)
        return {
            "representations": representations,
            "labels": labels,
            "task_name": "entity_type",
            "num_classes": int(labels.max().item()) + 1,
        }

    @staticmethod
    def sentiment(representations, sentiment_labels):
        """Sentiment classification probe.

        Args:
            representations: (N, D)
            sentiment_labels: (N,) integer sentiment labels (0=neg, 1=neu, 2=pos)

        Returns:
            dict ready for evaluate()

        """
        labels = torch.tensor(sentiment_labels, dtype=torch.long)
        return {
            "representations": representations,
            "labels": labels,
            "task_name": "sentiment",
            "num_classes": 3,
        }
