# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Probe generalization test: do probes trained on one dataset
# transfer to another?
#
# If JEPA probes generalize better than MLM probes, this is STRONG
# evidence that JEPA representations are more structured and universal.
# A probe trained on POS for WikiText that works on Penn TreeBank
# without retraining = the POS features are genuinely encoded,
# not just memorized from the training distribution.
#
# This is a critical test that most interpretability papers skip.
# Reviewers who know probing literature (Hewitt & Liang, 2019;
# Ravichander et al., 2020) will ask: "How do you know the probe
# isn't just learning the task from the labels?"

import torch
import torch.nn.functional as F
from torch import nn


class ProbeGeneralizationTest:
    """Test whether probes generalize across datasets.

    Method:
    1. Train probe on dataset A representations + labels
    2. Test on dataset B representations + labels (ZERO-SHOT)
    3. Compare JEPA vs baseline generalization gap

    If JEPA probe generalizes better → JEPA features are more
    universal and genuinely encode linguistic structure.
    """

    def __init__(
        self,
        embed_dim=768,
        num_classes=2,
        lr=1e-3,
        max_epochs=50,
        patience=5,
        device="cpu",
    ):
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device

    def _train_probe(self, representations, labels, num_classes=None):
        """Train a linear probe. Returns trained probe."""
        representations = representations.detach().float()
        N = representations.size(0)
        if N < 10:
            return None

        nc = num_classes or self.num_classes
        nc = max(int(labels.max().item()) + 1, 2) if N > 0 else 2

        probe = nn.Linear(self.embed_dim, nc).to(self.device)
        opt = torch.optim.Adam(probe.parameters(), lr=self.lr, weight_decay=0.01)

        best_state = None
        best_acc = 0.0
        no_improve = 0

        for epoch in range(self.max_epochs):
            probe.train()
            reps = representations.to(self.device)
            labs = labels.to(self.device)
            logits = probe(reps)
            loss = F.cross_entropy(logits, labs)
            opt.zero_grad()
            loss.backward()
            opt.step()

            probe.eval()
            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == labs).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break

        if best_state:
            probe.load_state_dict(best_state)
        return probe

    @torch.no_grad()
    def _evaluate_probe(self, probe, representations, labels):
        """Evaluate probe on held-out data."""
        if probe is None:
            return 0.0
        probe.eval()
        reps = representations.detach().float().to(self.device)
        labs = labels.to(self.device)
        logits = probe(reps)
        return (logits.argmax(dim=-1) == labs).float().mean().item()

    def cross_dataset_generalization(
        self,
        source_reps,
        source_labels,
        target_reps,
        target_labels,
        task_name="default",
    ):
        """Train probe on source, test on target (zero-shot transfer).

        Args:
            source_reps: (N_s, D) representations from source dataset
            source_labels: (N_s,) labels from source dataset
            target_reps: (N_t, D) representations from target dataset
            target_labels: (N_t,) labels from target dataset
            task_name: task name

        Returns:
            dict with source accuracy, target accuracy, generalization gap

        """
        # Train on source
        with torch.enable_grad():
            probe = self._train_probe(source_reps, source_labels)

        # Evaluate on source (train accuracy)
        source_acc = self._evaluate_probe(probe, source_reps, source_labels)

        # Evaluate on target (zero-shot transfer)
        target_acc = self._evaluate_probe(probe, target_reps, target_labels)

        # Generalization gap
        gen_gap = source_acc - target_acc
        # Generalization ratio: how much of source performance is preserved
        gen_ratio = target_acc / max(source_acc, 1e-10)

        return {
            "task": task_name,
            "source_accuracy": source_acc,
            "target_accuracy": target_acc,
            "generalization_gap": gen_gap,
            "generalization_ratio": gen_ratio,  # Higher = better generalization
            "probe_transfers": target_acc > 0.5,  # Better than random for binary
        }

    def compare_models(
        self,
        jepa_source,
        baseline_source,
        target_source,
        jepa_target,
        baseline_target,
        target_target,
        task_name="default",
    ):
        """Compare probe generalization between JEPA and baseline.

        THE KEY COMPARISON: if JEPA probes transfer better,
        JEPA representations are more universally structured.

        Args:
            jepa_source: (N_s, D) JEPA reps from source dataset
            baseline_source: (N_s, D) baseline reps from source dataset
            target_source: (N_s,) source labels
            jepa_target: (N_t, D) JEPA reps from target dataset
            baseline_target: (N_t, D) baseline reps from target dataset
            target_target: (N_t,) target labels

        Returns:
            dict with comparison

        """
        jepa_result = self.cross_dataset_generalization(
            jepa_source,
            target_source,
            jepa_target,
            target_target,
            f"{task_name}_jepa",
        )

        baseline_result = self.cross_dataset_generalization(
            baseline_source,
            target_source,
            baseline_target,
            target_target,
            f"{task_name}_baseline",
        )

        return {
            "task": task_name,
            "jepa_gen_ratio": jepa_result["generalization_ratio"],
            "baseline_gen_ratio": baseline_result["generalization_ratio"],
            "jepa_target_acc": jepa_result["target_accuracy"],
            "baseline_target_acc": baseline_result["target_accuracy"],
            "jepa_generalizes_better": jepa_result["generalization_ratio"]
            > baseline_result["generalization_ratio"],
            "jepa_gen_gap": jepa_result["generalization_gap"],
            "baseline_gen_gap": baseline_result["generalization_gap"],
        }


class ProbeSelectivityTest:
    """Test probe selectivity: does the probe learn the right thing?

    Hewitt & Liang (2019): a "selectivity" test measures whether
    a probe's accuracy drops when trained on STRUCTURED labels vs
    RANDOM control tasks.

    If JEPA probe has HIGHER selectivity → the probe is genuinely
    extracting linguistic structure, not just memorizing.
    """

    def __init__(self, embed_dim=768, num_classes=2, lr=1e-3, max_epochs=30, device="cpu"):
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.lr = lr
        self.max_epochs = max_epochs
        self.device = device

    def _train_probe(self, reps, labels):
        """Train linear probe."""
        reps = reps.detach().float()
        N = reps.size(0)
        if N < 10:
            return 0.0

        nc = max(int(labels.max().item()) + 1, 2)
        probe = nn.Linear(self.embed_dim, nc).to(self.device)
        opt = torch.optim.Adam(probe.parameters(), lr=self.lr, weight_decay=0.01)

        n_train = int(0.8 * N)
        idx = torch.randperm(N)

        best_acc = 0.0
        for epoch in range(self.max_epochs):
            probe.train()
            logits = probe(reps[idx[:n_train]].to(self.device))
            loss = F.cross_entropy(logits, labels[idx[:n_train]].to(self.device))
            opt.zero_grad()
            loss.backward()
            opt.step()

            probe.eval()
            with torch.no_grad():
                val_logits = probe(reps[idx[n_train:]].to(self.device))
                acc = (
                    (val_logits.argmax(dim=-1) == labels[idx[n_train:]].to(self.device))
                    .float()
                    .mean()
                    .item()
                )
            best_acc = max(best_acc, acc)

        return best_acc

    @torch.no_grad()
    def compute_selectivity(self, representations, real_labels, n_control=5):
        """Compute probe selectivity.

        Selectivity = accuracy(real_task) - accuracy(control_task)

        Control task: random labels with same label distribution
        as the real task. If probe gets high accuracy on control
        task, it's memorizing, not extracting structure.

        Higher selectivity = probe is genuinely extracting structure.

        Args:
            representations: (N, D)
            real_labels: (N,) real linguistic labels
            n_control: number of control experiments

        Returns:
            dict with selectivity metrics

        """
        with torch.enable_grad():
            real_acc = self._train_probe(representations, real_labels)

        control_accs = []
        for _ in range(n_control):
            # Random labels with same distribution
            perm = torch.randperm(real_labels.size(0))
            control_labels = real_labels[perm]
            with torch.enable_grad():
                ctrl_acc = self._train_probe(representations, control_labels)
            control_accs.append(ctrl_acc)

        mean_control = sum(control_accs) / len(control_accs)
        selectivity = real_acc - mean_control

        return {
            "real_task_accuracy": real_acc,
            "control_task_accuracy": mean_control,
            "selectivity": selectivity,  # Higher = more genuine
            "probe_is_genuine": selectivity > 0.1,
            "n_control_experiments": n_control,
        }

    def compare_selectivity(self, jepa_reps, baseline_reps, labels, n_control=5):
        """Compare probe selectivity between JEPA and baseline."""
        jepa_sel = self.compute_selectivity(jepa_reps, labels, n_control)
        baseline_sel = self.compute_selectivity(baseline_reps, labels, n_control)

        return {
            "jepa_selectivity": jepa_sel["selectivity"],
            "baseline_selectivity": baseline_sel["selectivity"],
            "jepa_more_selective": jepa_sel["selectivity"] > baseline_sel["selectivity"],
            "jepa_real_acc": jepa_sel["real_task_accuracy"],
            "baseline_real_acc": baseline_sel["real_task_accuracy"],
            "jepa_control_acc": jepa_sel["control_task_accuracy"],
            "baseline_control_acc": baseline_sel["control_task_accuracy"],
        }


class StructuralProbeGeneralization:
    """Does the structural probe (Hewitt & Manning 2019) generalize
    across treebanks?

    If JEPA's structural probe trained on Penn TreeBank also works
    on Universal Dependencies → JEPA encodes universal syntactic
    structure, not PTB-specific patterns.
    """

    @staticmethod
    def compute(probe, source_reps, source_tree_dists, target_reps, target_tree_dists):
        """Test structural probe generalization.

        Args:
            probe: trained StructuralProbe
            source_reps: source dataset representations
            source_tree_dists: source tree distances
            target_reps: target dataset representations
            target_tree_dists: target tree distances

        Returns:
            dict with source and target Spearman correlation

        """

        # Evaluate on source
        source_result = probe.evaluate([source_reps], [source_tree_dists])

        # Evaluate on target (without retraining)
        target_result = probe.evaluate([target_reps], [target_tree_dists])

        return {
            "source_spearman": source_result["spearman_r"],
            "target_spearman": target_result["spearman_r"],
            "generalization_ratio": target_result["spearman_r"]
            / max(source_result["spearman_r"], 1e-10),
            "probe_generalizes": target_result["spearman_r"] > 0.3,
        }
