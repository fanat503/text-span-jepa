# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
#
# Structural Probe: syntactic tree distance from representations
# Hewitt & Manning (2019) "A Structural Probe for Finding Syntax in Word Representations"
# Tests whether representation geometry encodes syntactic structure
#
# WHAT IS HELD OUT HERE. The unit of generalisation for a structural probe
# is the SENTENCE, not the token pair: `evaluate` pools the upper triangle of
# every sentence it is handed into one vector and correlates the two, so a
# "held-out" number means a held-out set of sentences. There was previously
# no way to ask for one -- `train_probe` saw every sentence it was given and
# `evaluate` scored whatever it was handed, so
# `StructuralProbeGeneralization.source_spearman` was the probe's own
# training set under a name that did not say so. Measured on 20 training vs
# 20 unseen sentences of identical construction: 0.2115 vs 0.0699, an
# optimism of +0.14 in the direction the name claimed.
#
# `StructuralProbeGeneralization.compute` was also unreachable: it wrapped the
# caller's already-per-sentence lists in another list, so
# `StructuralProbe.evaluate` received a list of lists and raised
# `AttributeError: 'list' object has no attribute 'to'`. No test called it.
#
# `sentence_split` exists so a caller can carve that partition BEFORE
# fitting, and `train_probe(train_idx=...)` so it can fit on the train side
# only. `StructuralProbeGeneralization.compute` takes the same indices.

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

#: Default fraction of sentences held out. Unchanged from the 80/20 shape the
#: module has always used for any internal split.
DEFAULT_HOLDOUT_FRACTION = 0.2


class StructuralProbe(nn.Module):
    """Structural probe from Hewitt & Manning (2019).

    Tests whether the squared L2 distance between word representations
    approximates the tree distance in the syntactic parse tree.

    The probe is a bilinear transform: B = L^T L (rank-d projection),
    then distance under B approximates parse tree distance.

    Key metric: Spearman correlation between predicted distances
    and gold tree distances. Higher = more syntactic structure encoded.

    That correlation is out-of-sample only if the sentences passed to
    :meth:`evaluate` are disjoint from the ones passed to
    :meth:`train_probe`. Neither method enforces it, because neither method
    can: it is the caller's partition to make. Use :meth:`sentence_split`.
    """

    @staticmethod
    def sentence_split(
        n: int,
        seed: int = 0,
        holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    ):
        """Draw ONE train/held-out partition over ``n`` sentences.

        Public and deterministic, so the partition behind a published
        correlation can be reproduced or reused without refitting the probe.

        The split is the most recent sentence list to be passed here: tokens
        within a sentence stay together on one side, because tree distances
        are only defined within a sentence.

        Args:
            n: number of sentences to partition
            seed: seeds a private ``torch.Generator``; the process-global RNG
                is never touched
            holdout_fraction: fraction of sentences kept out of training

        Returns:
            ``(train_idx, test_idx)``, disjoint 1-D int64 tensors whose union
            is ``range(n)``.

        """
        n = int(n)
        frac = float(holdout_fraction)
        if not 0.0 < frac < 1.0:
            raise ValueError(f"holdout_fraction must lie strictly in (0, 1), got {frac!r}")
        n_test = round(n * frac)
        if n - n_test < 1 or n_test < 1:
            raise ValueError(
                f"holdout_fraction={frac!r} on n={n} leaves {n - n_test} training "
                f"and {n_test} held-out sentences; both sides must be non-empty "
                f"for a held-out correlation to exist"
            )
        gen = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(n, generator=gen)
        return idx[: n - n_test], idx[n - n_test :]

    @staticmethod
    def check_sentence_split(split, n):
        """Reject a split that does not describe a partition of ``n`` sentences.

        Bounds, sizes and disjointness. The disjointness check is not
        redundant: an earlier version of :meth:`sentence_split` sliced
        ``idx[: n - n_test]`` against ``idx[n_test:]``, which overlaps whenever
        ``n_test < n / 2`` and still satisfies the size check. It is O(n log n)
        on the number of sentences, which is tens, not the number of tokens.

        """
        train_idx, test_idx = split
        n_train, n_test = int(train_idx.numel()), int(test_idx.numel())
        if n_train == 0 or n_test == 0:
            raise ValueError(
                f"split has {n_train} training and {n_test} held-out sentences; an "
                f"empty side has no score to report"
            )
        if n_train + n_test != n:
            raise ValueError(
                f"split covers {n_train + n_test} sentences but {n} were passed; it "
                f"was drawn for a different set"
            )
        for name, idx in (("train", train_idx), ("test", test_idx)):
            if int(idx.min()) < 0 or int(idx.max()) >= n:
                raise ValueError(
                    f"{name} indices fall outside [0, {n}); the split does not "
                    f"address these sentences"
                )
        both = torch.cat([train_idx.reshape(-1), test_idx.reshape(-1)]).sort().values
        if not torch.equal(both, torch.arange(n)):
            raise ValueError(
                f"split is not a partition of range({n}): it repeats or omits "
                f"{n - int(torch.unique(both).numel())} sentences, so the two "
                f"sides are not disjoint"
            )

    def __init__(self, embed_dim=768, probe_rank=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.probe_rank = probe_rank
        # Learnable projection matrix (rank probe_rank)
        self.proj = nn.Parameter(torch.randn(probe_rank, embed_dim) * 0.01)

    def forward(self, representations):
        """Compute predicted tree distances.

        Args:
            representations: (B, T, D) word representations

        Returns:
            distances: (B, T, T) predicted pairwise tree distances

        """
        _B, _T, _D = representations.shape
        # Project: h' = P h
        projected = representations @ self.proj.T  # (B, T, probe_rank)
        # Pairwise squared distances under projection
        # ||P h_i - P h_j||^2 = ||h'_i - h'_j||^2
        diff = projected.unsqueeze(2) - projected.unsqueeze(1)  # (B, T, T, R)
        distances = (diff**2).sum(dim=-1)  # (B, T, T)
        return distances

    def train_probe(
        self,
        representations_list,
        tree_distances_list,
        lr=0.001,
        epochs=30,
        device="cpu",
        train_idx=None,
    ):
        """Train the structural probe on gold parse tree distances.

        Args:
            representations_list: list of (T, D) tensors
            tree_distances_list: list of (T, T) tensors (gold tree distances)
            lr: learning rate
            epochs: number of training epochs
            device: compute device
            train_idx: optional 1-D index tensor selecting the sentences to
                fit on. Omitting it fits on every sentence given, which is
                only what you want when the sentences you later score are
                disjoint from these by construction -- pass
                :meth:`sentence_split`'s train side to make that explicit.

        Returns:
            Training loss history

        """
        self.to(device)
        if train_idx is not None:
            train_idx = torch.as_tensor(train_idx, dtype=torch.long).reshape(-1)
            representations_list = [representations_list[i] for i in train_idx.tolist()]
            tree_distances_list = [tree_distances_list[i] for i in train_idx.tolist()]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_history = []

        for epoch in range(epochs):
            total_loss = 0
            n_samples = 0
            for reps, gold_dist in zip(representations_list, tree_distances_list):
                reps = reps.to(device).unsqueeze(0)  # (1, T, D)
                gold_dist = gold_dist.to(device)

                pred_dist = self(reps).squeeze(0)  # (T, T)
                # L2 loss on upper triangle (symmetric matrix)
                mask = torch.ones_like(gold_dist, dtype=torch.bool)
                mask = mask.triu(diagonal=1)
                loss = F.mse_loss(pred_dist[mask], gold_dist[mask].float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_samples += 1

            loss_history.append(total_loss / max(n_samples, 1))

        return loss_history

    @torch.no_grad()
    def evaluate(self, representations_list, tree_distances_list, device="cpu"):
        """Evaluate probe: Spearman correlation with gold tree distances.

        Scores exactly the sentences it is given and says nothing about
        whether they were seen in training. A caller reporting this number
        as a probe's accuracy must have partitioned the sentences first; see
        the module header and :meth:`sentence_split`.

        Args:
            representations_list: list of (T, D) tensors
            tree_distances_list: list of (T, T) tensors

        Returns:
            dict with 'spearman_r' and 'uuas' metrics

        """
        self.eval()
        self.to(device)

        all_pred = []
        all_gold = []
        correct_edges = 0
        total_edges = 0

        for reps, gold_dist in zip(representations_list, tree_distances_list):
            reps = reps.to(device).unsqueeze(0)
            gold_dist = gold_dist.to(device)

            pred_dist = self(reps).squeeze(0)

            mask = torch.ones_like(gold_dist, dtype=torch.bool).triu(diagonal=1)
            all_pred.append(pred_dist[mask].cpu())
            all_gold.append(gold_dist[mask].float().cpu())

            # UUAS: fraction of gold tree edges correctly predicted
            # by minimum spanning tree on predicted distances
            T = reps.size(1)
            if T > 2:
                # Greedy: for each token, predict parent = nearest neighbor
                # (approximation of MST for speed)
                pred_parents = self._greedy_parents(pred_dist, T)
                gold_parents = self._greedy_parents(gold_dist, T)
                correct_edges += (pred_parents[1:] == gold_parents[1:]).sum().item()
                total_edges += T - 1

        # Spearman correlation
        if len(all_pred) > 0:
            pred_cat = torch.cat(all_pred)
            gold_cat = torch.cat(all_gold)
            spearman_r = self._spearman(pred_cat, gold_cat)
        else:
            spearman_r = 0.0

        uuas = correct_edges / max(total_edges, 1)

        return {
            "spearman_r": spearman_r,
            "uuas": uuas,
        }

    @staticmethod
    def _greedy_parents(dist_matrix, T):
        """Greedy parent assignment: each node's parent is its nearest neighbor."""
        parents = torch.zeros(T, dtype=torch.long)
        if T <= 1:
            return parents
        for i in range(1, T):
            # Nearest earlier position (approximate root-finding)
            min_dist = float("inf")
            best_j = 0
            for j in range(i):
                if dist_matrix[i, j].item() < min_dist:
                    min_dist = dist_matrix[i, j].item()
                    best_j = j
            parents[i] = best_j
        return parents

    @staticmethod
    def _spearman(x, y):
        """Compute Spearman rank correlation.

        KNOWN DEFECT, reported not fixed: the ranks come from
        ``argsort().argsort()``, which assigns *ordinal* ranks 0, 1, 2, ...
        to tied values in whatever order the sort happens to produce. Gold
        tree distances are heavily tied -- a chain parse has many pairs at
        the same distance -- so this is not Spearman's rho on this data, and
        it is not reproducible across orderings. A tie-corrected
        ``average`` rank is the fix; it is a separate decision because it
        moves every structural-probe number in the repo, and none of them is
        on this card.

        The bare ``except Exception: return 0.0`` is the same family of
        defect: a failed correlation is reported as "no structure" instead
        of as a failure. Also reported, also not fixed here.

        """
        try:
            if x.numel() < 2:
                return 0.0
            rx = x.argsort().argsort().float()
            ry = y.argsort().argsort().float()
            rx = rx - rx.mean()
            ry = ry - ry.mean()
            denom = rx.norm() * ry.norm()
            if denom == 0:
                return 0.0
            return (rx @ ry / denom).item()
        except Exception:
            return 0.0
