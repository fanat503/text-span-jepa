# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Structural Probe: syntactic tree distance from representations
# Hewitt & Manning (2019) "A Structural Probe for Finding Syntax in Word Representations"
# Tests whether representation geometry encodes syntactic structure

import torch
import torch.nn.functional as F
from torch import nn


class StructuralProbe(nn.Module):
    """Structural probe from Hewitt & Manning (2019).

    Tests whether the squared L2 distance between word representations
    approximates the tree distance in the syntactic parse tree.

    The probe is a bilinear transform: B = L^T L (rank-d projection),
    then distance under B approximates parse tree distance.

    Key metric: Spearman correlation between predicted distances
    and gold tree distances. Higher = more syntactic structure encoded.
    """

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
    ):
        """Train the structural probe on gold parse tree distances.

        Args:
            representations_list: list of (T, D) tensors
            tree_distances_list: list of (T, T) tensors (gold tree distances)
            lr: learning rate
            epochs: number of training epochs
            device: compute device

        Returns:
            Training loss history

        """
        self.to(device)
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
        """Compute Spearman rank correlation."""
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
