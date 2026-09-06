# Copyright 2026 Text-Span JEPA Authors
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


import torch
import torch.nn.functional as F
from torch import nn


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
    Negative = JEPA needs shallower probe (good — evidence for hypothesis)
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

    def _train_probe(self, probe, representations, labels):
        """Train a single probe with early stopping.

        Args:
            probe: nn.Module probe
            representations: (N, D)
            labels: (N,) class indices

        Returns:
            best_accuracy on validation set

        """
        N = representations.size(0)
        if N < 10:
            return 0.0

        # Ensure representations are float and require_grad compatible
        representations = representations.detach().float()
        # Probe needs to be trainable
        probe = probe.to(self.device)
        probe.train()

        # Split 80/20
        n_train = int(0.8 * N)
        idx = torch.randperm(N)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:]

        train_reps = representations[train_idx].to(self.device)
        train_labels = labels[train_idx].to(self.device)
        val_reps = representations[val_idx].to(self.device)
        val_labels = labels[val_idx].to(self.device)

        probe = probe.to(self.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        best_acc = 0.0
        best_state = None
        no_improve = 0

        for epoch in range(self.max_epochs):
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
            with torch.no_grad():
                logits = probe(val_reps)
                acc = (logits.argmax(dim=-1) == val_labels).float().mean().item()

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

        return best_acc

    def evaluate(self, representations, labels, task_name="default"):
        """Evaluate probing complexity for a single task.

        Args:
            representations: (N, D) representation vectors
            labels: (N,) integer class labels
            task_name: name of the linguistic task

        Returns:
            dict with per-depth accuracy and minimum extracting depth

        """
        num_classes = self.num_classes or labels.max().item() + 1
        num_classes = max(int(num_classes), 2)

        results = {
            "task": task_name,
            "depths": {},
            "min_extracting_depth": None,
        }

        for depth in self.depths:
            probe = self._build_probe(depth, num_classes)
            # _train_probe needs gradients — use enable_grad context
            with torch.enable_grad():
                acc = self._train_probe(probe, representations, labels)
            results["depths"][depth] = acc

            # First depth that exceeds threshold
            if acc >= self.min_accuracy and results["min_extracting_depth"] is None:
                results["min_extracting_depth"] = depth

        # If no depth reached threshold, set to max + 1
        if results["min_extracting_depth"] is None:
            results["min_extracting_depth"] = max(self.depths) + 1

        results["max_accuracy"] = max(results["depths"].values())

        return results

    def compare_models(self, jepa_reps, baseline_reps, labels, task_name="default"):
        """Compare probing complexity between JEPA and baseline.

        THE CORE COMPARISON for the paper.

        Args:
            jepa_reps: (N, D) JEPA representations
            baseline_reps: (N, D) baseline representations
            labels: (N,) class labels
            task_name: name of linguistic task

        Returns:
            dict with complexity gap and per-depth comparison

        """
        jepa_result = self.evaluate(jepa_reps, labels, f"{task_name}_jepa")
        baseline_result = self.evaluate(baseline_reps, labels, f"{task_name}_baseline")

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
