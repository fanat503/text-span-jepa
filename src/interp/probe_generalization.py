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
#
# ── WHAT IS HELD OUT HERE ────────────────────────────────────────────────────
# Every metric this module reported before this revision was a TRAINING-set
# number. `source_accuracy` was `probe(source_reps)` on the very rows the
# probe had just been fitted on; `generalization_gap = source - target` was
# therefore a measure of overfitting, not of generalisation; and
# `generalization_ratio = target / source` was monotone in how much the probe
# overfit, so a probe that memorised its source scored BETTER on it. The
# early stopping compounded this: patience was counted on the in-sample
# accuracy, which saturates, so the probe stopped after one or two epochs and
# `source_accuracy` was both in-sample and badly undertrained.
#
# The source side is now partitioned three ways -- train / selection /
# held-out -- so `source_accuracy_heldout` is a number the optimiser never
# saw and never selected on. Both `compare_models` arms share that partition
# and the probe weights, because they are scored on the same rows.
#
# The TARGET side was always genuinely held out (it is a different dataset)
# and is unchanged. `ProbeSelectivityTest` had the same defect plus one
# worse: five control tasks, five different random splits.
#
# KNOWN CONFOUND, reported not fixed. `generalization_ratio` is a quotient of
# two quantities that can each be near zero, so it is unbounded and its sign
# flips with the sign of the target correlation. Measured on the structural
# probe's own held-out source correlation: denominator 0.0365, numerator
# -0.0672, ratio -1.84. It is not "how much generalisation was preserved" and
# was never that; replacing it with a difference, or with a bootstrap
# interval on the difference, is a decision about what the paper should claim
# and is not made here.
#
# Also unfixed: the same quotient pattern appears in
# `ProbingComplexityCurve` and `StructuralProbeGeneralization`.

import math

import torch
import torch.nn.functional as F
from torch import nn

#: Offset added to the seed to derive the probe-weight stream, so one seed
#: yields two independent, reproducible streams: which rows are held out, and
#: which weights the probe starts from. Mirrors
#: `layer_analysis.LayerwiseProbe`; the two modules stay independent because
#: neither should have to know about the other to be correct.
WEIGHT_STREAM_OFFSET = 1

#: Source-side partition, as (train, selection, held-out) fractions. The two
#: tail fractions are equal so neither the epoch selection nor the reported
#: score gets the larger share.
DEFAULT_SOURCE_SPLIT = (0.6, 0.2, 0.2)


def _seeded_linear(in_features, out_features, generator, bias=True):
    """An ``nn.Linear`` whose ``reset_parameters`` law comes from `generator`.

    ``nn.Linear`` has no generator argument, so the plain constructor draws
    from the process-global RNG. That was the second unseeded draw per probe
    here, and it is the one that makes a JEPA-vs-baseline comparison two
    experiments: the two arms got different initialisations, so a difference
    in final accuracy was partly a difference in the starting point. This
    reproduces torch's own law (kaiming_uniform with ``a=sqrt(5)`` is exactly
    ``uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))``; see
    ``torch.nn.Linear.reset_parameters``) from a private generator, in the
    same order, so it is bit-identical to the stock initialisation given the
    same seed.
    """
    layer = nn.Linear(in_features, out_features, bias=bias)
    bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 0.0
    with torch.no_grad():
        layer.weight.copy_(
            torch.empty_like(layer.weight).uniform_(-bound, bound, generator=generator)
        )
        if layer.bias is not None:
            layer.bias.copy_(
                torch.empty_like(layer.bias).uniform_(-bound, bound, generator=generator)
            )
    return layer


def _three_way_split(n, seed, fractions=DEFAULT_SOURCE_SPLIT):
    """Partition ``n`` rows into train / selection / held-out index tensors.

    All three sides must be non-empty: an empty held-out side has no score to
    report and an empty selection side leaves the epoch choice arbitrary.
    Every draw comes from a private generator, so this module never consumes
    the caller's global stream.

    Args:
        n: number of rows to partition
        seed: seeds the private ``torch.Generator``
        fractions: three positive shares summing to at most 1

    Returns:
        ``(train_idx, sel_idx, test_idx)``, disjoint 1-D int64 tensors whose
        union is ``range(n)``.
    """
    n = int(n)
    fr = tuple(float(f) for f in fractions)
    if len(fr) != 3 or any(f <= 0.0 for f in fr) or sum(fr) > 1.0:
        raise ValueError(
            f"fractions must be three positive shares summing to at most 1, got {fractions!r}"
        )
    n_train = round(fr[0] * n)
    n_sel = round(fr[1] * n)
    counts = (n_train, n_sel, n - n_train - n_sel)
    if min(counts) < 1:
        raise ValueError(
            f"fractions={fractions!r} on n={n} leave {counts[0]} train / "
            f"{counts[1]} selection / {counts[2]} held-out rows; every side must "
            f"be non-empty. Use more samples, or coarser fractions."
        )
    gen = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(n, generator=gen)
    return (
        idx[: counts[0]],
        idx[counts[0] : counts[0] + counts[1]],
        idx[counts[0] + counts[1] :],
    )


class ProbeGeneralizationTest:
    """Test whether probes generalize across datasets.

    Method:
    1. Partition the source representations into train / selection /
       held-out rows. One partition, drawn once from a private generator.
    2. Train the probe on the train rows, choosing the epoch on the
       selection rows.
    3. Score the held-out source rows (never seen, never selected on) and
       the target dataset (a different dataset, always zero-shot).

    If JEPA probe generalizes better, JEPA features are more universal and
    genuinely encode linguistic structure.
    """

    def __init__(
        self,
        embed_dim=768,
        num_classes=2,
        lr=1e-3,
        max_epochs=50,
        patience=5,
        device="cpu",
        seed=0,
        split_fractions=DEFAULT_SOURCE_SPLIT,
    ):
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.seed = int(seed)
        self.split_fractions = tuple(split_fractions)

    def source_split(self, n, seed=None):
        """The one source-side partition, public and reproducible.

        Both arms of :meth:`compare_models` are given the same ``n`` and the
        same ``seed``, so they receive the same partition and are scored on
        the same rows: the comparison is then a statement about the two
        representations rather than about two draws.

        Args:
            n: number of source rows
            seed: overrides ``self.seed``, for pinning the partition
                independently of the instance

        Returns:
            ``(train_idx, sel_idx, test_idx)``
        """
        return _three_way_split(n, self.seed if seed is None else seed, self.split_fractions)

    def weight_generator(self):
        """Private generator for the probe-weight stream.

        Separate from the split stream: which rows are held out is a property
        of the data, which weights the probe starts from is not.
        """
        return torch.Generator().manual_seed(self.seed + WEIGHT_STREAM_OFFSET)

    def _train_probe(self, representations, labels, split=None, generator=None):
        """Train a linear probe on the train side of `split`.

        The epoch is chosen on the selection side and the best state restored,
        so the held-out side is untouched by both the optimiser and the model
        selection. Returns a dict of probe plus the three accuracies, or
        ``None`` if there are too few rows to train on at all.
        """
        representations = representations.detach().float()
        N = representations.size(0)
        if N < 10:
            return None

        if split is None:
            split = self.source_split(N)
        train_idx, sel_idx, test_idx = split

        nc = max(int(labels.max().item()) + 1, 2)
        if generator is None:
            generator = self.weight_generator()
        probe = _seeded_linear(self.embed_dim, nc, generator).to(self.device)
        opt = torch.optim.Adam(probe.parameters(), lr=self.lr, weight_decay=0.01)

        tr_reps = representations[train_idx].to(self.device)
        tr_labs = labels[train_idx].to(self.device)
        sel_reps = representations[sel_idx].to(self.device)
        sel_labs = labels[sel_idx].to(self.device)
        te_reps = representations[test_idx].to(self.device)
        te_labs = labels[test_idx].to(self.device)

        def _acc(x, y):
            with torch.no_grad():
                return (probe(x).argmax(dim=-1) == y).float().mean().item()

        best_sel = 0.0
        best_state = None
        no_improve = 0

        for _epoch in range(self.max_epochs):
            probe.train()
            loss = F.cross_entropy(probe(tr_reps), tr_labs)
            opt.zero_grad()
            loss.backward()
            opt.step()

            probe.eval()
            sel_acc = _acc(sel_reps, sel_labs)
            if sel_acc > best_sel:
                best_sel = sel_acc
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
            "probe": probe,
            "train_accuracy": _acc(tr_reps, tr_labs),
            "selection_accuracy": best_sel,
            "heldout_accuracy": _acc(te_reps, te_labs),
        }

    @torch.no_grad()
    def _evaluate_probe(self, probe, representations, labels):
        """Score a probe on rows it was not fitted on."""
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
        split=None,
        generator=None,
    ):
        """Train probe on source, test on target (zero-shot transfer).

        Args:
            source_reps: (N_s, D) representations from source dataset
            source_labels: (N_s,) labels from source dataset
            target_reps: (N_t, D) representations from target dataset
            target_labels: (N_t,) labels from target dataset
            task_name: task name
            split: optional ``(train_idx, sel_idx, test_idx)`` over the source
                rows, from :meth:`source_split`. Supply it to compare two
                representations on identical rows; omit it and a fresh
                partition is drawn.
            generator: optional probe-weight generator, shared the same way

        Returns:
            dict with the held-out source accuracy, the selection and
            in-sample accuracies under names that say which is which, the
            transfer accuracy, the generalisation gap and ratio computed from
            the held-out source accuracy, and the split's provenance.
        """
        source_reps = source_reps.detach().float()
        if split is None:
            split = self.source_split(source_reps.size(0))
        train_idx, sel_idx, test_idx = split

        # Train on the source's train rows only.
        with torch.enable_grad():
            trained = self._train_probe(
                source_reps, source_labels, split=split, generator=generator
            )

        if trained is None:
            raise ValueError(
                f"source has {source_reps.size(0)} rows, too few to train on; the "
                f"in-sample number this replaces was only computable because it "
                f"never left the training set"
            )
        probe = trained["probe"]

        # Zero-shot transfer. The target is a different dataset, so it is
        # held out by construction; this part is unchanged.
        target_acc = self._evaluate_probe(probe, target_reps, target_labels)

        source_heldout = trained["heldout_accuracy"]
        # Generalisation gap: how much of the held-out source performance the
        # target preserves. On the held-out number, not the in-sample one --
        # otherwise this is a measure of overfitting.
        gen_gap = source_heldout - target_acc
        gen_ratio = target_acc / max(source_heldout, 1e-10)

        return {
            "task": task_name,
            # The number that replaced `source_accuracy`: out-of-sample on the
            # source, and never selected on.
            "source_accuracy_heldout": source_heldout,
            "source_selection_accuracy": trained["selection_accuracy"],
            "source_train_accuracy": trained["train_accuracy"],
            "target_accuracy": target_acc,
            "generalization_gap": gen_gap,
            "generalization_ratio": gen_ratio,  # Higher = better generalization
            "probe_transfers": target_acc > 0.5,  # Better than random for binary
            "source_overfit": trained["train_accuracy"] - source_heldout,
            "n_source_train": int(train_idx.numel()),
            "n_source_selection": int(sel_idx.numel()),
            "n_source_heldout": int(test_idx.numel()),
            "split_seed": self.seed,
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

        THE KEY COMPARISON: if JEPA probes transfer better, JEPA
        representations are more universally structured.

        Both arms are given the SAME source partition and the SAME probe
        weights, because they are scored on the same rows with the same
        architecture; the only thing that differs is the representation.
        Passing each arm its own generator, seeded identically, is what makes
        that true -- sharing one generator would make arm two replay arm one's
        draws, and giving each arm a fresh one from the global RNG would make
        the accuracy difference partly the starting point.

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
        # Both arms see the same rows and the same starting weights.
        split = self.source_split(jepa_source.size(0))
        gen_a = self.weight_generator()
        gen_b = self.weight_generator()

        jepa_result = self.cross_dataset_generalization(
            jepa_source,
            target_source,
            jepa_target,
            target_target,
            f"{task_name}_jepa",
            split=split,
            generator=gen_a,
        )

        baseline_result = self.cross_dataset_generalization(
            baseline_source,
            target_source,
            baseline_target,
            target_target,
            f"{task_name}_baseline",
            split=split,
            generator=gen_b,
        )

        return {
            "task": task_name,
            "jepa_gen_ratio": jepa_result["generalization_ratio"],
            "baseline_gen_ratio": baseline_result["generalization_ratio"],
            "jepa_target_acc": jepa_result["target_accuracy"],
            "baseline_target_acc": baseline_result["target_accuracy"],
            "jepa_source_acc_heldout": jepa_result["source_accuracy_heldout"],
            "baseline_source_acc_heldout": baseline_result["source_accuracy_heldout"],
            "jepa_generalizes_better": jepa_result["generalization_ratio"]
            > baseline_result["generalization_ratio"],
            "jepa_gen_gap": jepa_result["generalization_gap"],
            "baseline_gen_gap": baseline_result["generalization_gap"],
            "split_seed": self.seed,
            "n_source_heldout": jepa_result["n_source_heldout"],
        }


class ProbeSelectivityTest:
    """Test probe selectivity: does the probe learn the right thing?

    Hewitt & Liang (2019): a "selectivity" test measures whether
    a probe's accuracy drops when trained on STRUCTURED labels vs
    RANDOM control tasks.

    If JEPA probe has HIGHER selectivity, the probe is genuinely
    extracting linguistic structure, not just memorizing.

    One partition, drawn once, is shared by the real task and by every
    control draw: they are all fitted and scored on the same rows, so a
    difference between them is a difference between the label assignments and
    not between the draws. Before this revision each of the ``n_control``
    control tasks drew its own unseeded split, so the reported selectivity
    carried ``n_control`` partition noises on top of the label noise.
    """

    def __init__(
        self,
        embed_dim=768,
        num_classes=2,
        lr=1e-3,
        max_epochs=30,
        device="cpu",
        seed=0,
        split_fractions=DEFAULT_SOURCE_SPLIT,
    ):
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.lr = lr
        self.max_epochs = max_epochs
        self.device = device
        self.seed = int(seed)
        self.split_fractions = tuple(split_fractions)

    def source_split(self, n, seed=None):
        """The one train / selection / held-out partition. See
        :meth:`ProbeGeneralizationTest.source_split`."""
        return _three_way_split(n, self.seed if seed is None else seed, self.split_fractions)

    def control_label_generator(self, draw):
        """Private generator for the ``draw``-th control label permutation.

        Permuting the real labels leaves their marginal distribution
        untouched, which is what "random labels with the same distribution"
        means in the docstring below.
        """
        return torch.Generator().manual_seed(self.seed + WEIGHT_STREAM_OFFSET + draw)

    def _train_probe(self, reps, labels, split, generator):
        """Fit a linear probe on `split`'s train rows, select on selection.

        Returns ``(heldout_accuracy, selection_accuracy, train_accuracy)``.
        The held-out score is computed once, after the epoch has been chosen,
        so it is not the maximum of anything.
        """
        reps = reps.detach().float()
        N = reps.size(0)
        train_idx, sel_idx, test_idx = split

        nc = max(int(labels.max().item()) + 1, 2)
        probe = _seeded_linear(self.embed_dim, nc, generator).to(self.device)
        opt = torch.optim.Adam(probe.parameters(), lr=self.lr, weight_decay=0.01)

        tr_reps = reps[train_idx].to(self.device)
        tr_labs = labels[train_idx].to(self.device)
        sel_reps = reps[sel_idx].to(self.device)
        sel_labs = labels[sel_idx].to(self.device)
        te_reps = reps[test_idx].to(self.device)
        te_labs = labels[test_idx].to(self.device)

        def _acc(x, y):
            with torch.no_grad():
                return (probe(x).argmax(dim=-1) == y).float().mean().item()

        best_sel = 0.0
        best_state = None
        for _epoch in range(self.max_epochs):
            probe.train()
            loss = F.cross_entropy(probe(tr_reps), tr_labs)
            opt.zero_grad()
            loss.backward()
            opt.step()

            probe.eval()
            sel_acc = _acc(sel_reps, sel_labs)
            if sel_acc > best_sel:
                best_sel = sel_acc
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}

        if best_state is not None:
            probe.load_state_dict(best_state)
        probe.eval()
        return _acc(te_reps, te_labs), best_sel, _acc(tr_reps, tr_labs)

    def compute_selectivity(self, representations, real_labels, n_control=5, split=None):
        """Compute probe selectivity.

        Selectivity = held-out accuracy(real_task) - held-out
        accuracy(control_task).

        Control task: random labels with the same label distribution as the
        real task, permuted with a private generator. If the probe gets high
        accuracy on a control task it is memorising, not extracting structure.

        Higher selectivity = probe is genuinely extracting structure.

        Args:
            representations: (N, D)
            real_labels: (N,) real linguistic labels
            n_control: number of control experiments
            split: optional shared ``(train_idx, sel_idx, test_idx)``

        Returns:
            dict with selectivity metrics, all of them held out
        """
        reps = representations.detach().float()
        labels = real_labels.detach()
        N = reps.size(0)
        if N < 10:
            raise ValueError(
                f"selectivity needs at least 10 samples to partition into "
                f"train/selection/held-out, got {N}"
            )
        if split is None:
            split = self.source_split(N)
        train_idx, sel_idx, test_idx = split

        # A fresh generator per fit, all seeded the same, so the real task and
        # every control differ ONLY in their labels -- not in their starting
        # weights. Sharing one generator across fits would advance it and hand
        # each control its own initialisation.
        with torch.enable_grad():
            real_heldout, real_sel, real_train = self._train_probe(
                reps,
                labels,
                split,
                torch.Generator().manual_seed(self.seed),
            )

        control_heldouts = []
        for draw in range(int(n_control)):
            perm = torch.randperm(N, generator=self.control_label_generator(draw))
            control_labels = labels[perm]
            with torch.enable_grad():
                ctrl_heldout, _ctrl_sel, _ctrl_train = self._train_probe(
                    reps,
                    control_labels,
                    split,
                    torch.Generator().manual_seed(self.seed),
                )
            control_heldouts.append(ctrl_heldout)

        mean_control = sum(control_heldouts) / len(control_heldouts)
        selectivity = real_heldout - mean_control

        return {
            "real_task_accuracy": real_heldout,
            "control_task_accuracy": mean_control,
            "selectivity": selectivity,  # Higher = more genuine
            "probe_is_genuine": selectivity > 0.1,
            "n_control_experiments": n_control,
            "real_train_accuracy": real_train,
            "real_selection_accuracy": real_sel,
            "n_train": int(train_idx.numel()),
            "n_selection": int(sel_idx.numel()),
            "n_heldout": int(test_idx.numel()),
            "split_seed": self.seed,
        }

    def compare_selectivity(self, jepa_reps, baseline_reps, labels, n_control=5):
        """Compare probe selectivity between JEPA and baseline."""
        split = self.source_split(jepa_reps.size(0))
        jepa_sel = self.compute_selectivity(jepa_reps, labels, n_control, split=split)
        baseline_sel = self.compute_selectivity(baseline_reps, labels, n_control, split=split)

        return {
            "jepa_selectivity": jepa_sel["selectivity"],
            "baseline_selectivity": baseline_sel["selectivity"],
            "jepa_more_selective": jepa_sel["selectivity"] > baseline_sel["selectivity"],
            "jepa_real_acc": jepa_sel["real_task_accuracy"],
            "baseline_real_acc": baseline_sel["real_task_accuracy"],
            "jepa_control_acc": jepa_sel["control_task_accuracy"],
            "baseline_control_acc": baseline_sel["control_task_accuracy"],
            "split_seed": self.seed,
        }


class StructuralProbeGeneralization:
    """Does the structural probe (Hewitt & Manning 2019) generalize
    across treebanks?

    If JEPA's structural probe trained on Penn TreeBank also works
    on Universal Dependencies, JEPA encodes universal syntactic
    structure, not PTB-specific patterns.
    """

    @staticmethod
    def compute(
        probe,
        source_reps,
        source_tree_dists,
        target_reps,
        target_tree_dists,
        source_train_idx=None,
        seed=0,
        holdout_fraction=None,
    ):
        """Test structural probe generalization.

        The unit of held-out here is the SENTENCE, because ``evaluate`` pools
        every sentence it is given into one correlation: a "held-out" number
        means a held-out set of sentences.

        ``source_train_idx`` is the load-bearing argument. Pass the train side
        of the same partition the probe was fitted on --
        ``StructuralProbe.sentence_split(len(source_reps), seed,
        holdout_fraction)`` paired with
        ``probe.train_probe(..., train_idx=train_idx)`` -- and
        ``source_spearman_heldout`` is out-of-sample by construction. Omit it
        and a partition is drawn here; the number is then only as held-out as
        the probe's own training set happened to be, which this function
        cannot see. Either way the train-side figure is reported next to it
        under a name that says so.

        Args:
            probe: trained StructuralProbe
            source_reps: list of source sentence representations, (T, D) each
            source_tree_dists: list of source gold tree distances, (T, T) each
            target_reps: list of target sentence representations
            target_tree_dists: list of target gold tree distances
            source_train_idx: the sentences the probe was fitted on. Their
                complement is scored.
            seed: seeds the partition drawn when ``source_train_idx`` is None
            holdout_fraction: held-out fraction for that drawn partition

        Returns:
            dict with train and held-out source Spearman correlation, the
            target Spearman correlation, and the split's provenance
        """
        from src.interp.structural_probe import DEFAULT_HOLDOUT_FRACTION, StructuralProbe

        if source_train_idx is None:
            frac = DEFAULT_HOLDOUT_FRACTION if holdout_fraction is None else holdout_fraction
            train_idx, test_idx = StructuralProbe.sentence_split(
                len(source_reps), seed=seed, holdout_fraction=frac
            )
            provided = False
        else:
            train_idx = torch.as_tensor(source_train_idx, dtype=torch.long).reshape(-1)
            mask = torch.ones(len(source_reps), dtype=torch.bool)
            mask[train_idx] = False
            test_idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
            provided = True
        StructuralProbe.check_sentence_split((train_idx, test_idx), len(source_reps))

        def _take(seq, idx):
            return [seq[int(i)] for i in idx.tolist()]

        source_train = probe.evaluate(
            _take(source_reps, train_idx), _take(source_tree_dists, train_idx)
        )
        source_heldout = probe.evaluate(
            _take(source_reps, test_idx), _take(source_tree_dists, test_idx)
        )

        # The target is a different treebank, so it is held out by
        # construction: unchanged from before.
        target_result = probe.evaluate(target_reps, target_tree_dists)

        return {
            "source_spearman_train": source_train["spearman_r"],
            # The number that replaced `source_spearman`, which was this same
            # quantity computed on whichever sentences the probe had been
            # fitted on.
            "source_spearman_heldout": source_heldout["spearman_r"],
            "source_uuas_heldout": source_heldout["uuas"],
            "source_optimism": source_train["spearman_r"] - source_heldout["spearman_r"],
            "target_spearman": target_result["spearman_r"],
            "generalization_ratio": target_result["spearman_r"]
            / max(source_heldout["spearman_r"], 1e-10),
            "probe_generalizes": target_result["spearman_r"] > 0.3,
            "source_train_idx_provided": provided,
            "n_source_train": int(train_idx.numel()),
            "n_source_heldout": int(test_idx.numel()),
            "split_seed": seed,
        }
