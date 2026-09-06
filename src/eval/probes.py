# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Evaluation: linear probes, future-token probes, geometry metrics
# Following NextLat (Microsoft, 2025) / I-JEPA evaluation protocols

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.models.collapse import CollapseDiagnostics


class LinearProbe:
    """Linear probe: train a linear classifier on frozen representations."""

    def __init__(self, embed_dim=768, num_classes=2, lr=1e-3, max_epochs=100):
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.lr = lr
        self.max_epochs = max_epochs

    def evaluate(self, model, dataset, device="cuda"):
        """Train linear classifier on frozen encoder and return accuracy."""
        model.eval()
        classifier = nn.Linear(self.embed_dim, self.num_classes).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.lr)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(self.max_epochs):
            for batch in loader:
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)
                with torch.no_grad():
                    h, _ = model.encoder(input_ids)
                    h_pooled = h.mean(dim=1)
                logits = classifier(h_pooled)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)
                h, _ = model.encoder(input_ids)
                h_pooled = h.mean(dim=1)
                logits = classifier(h_pooled)
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / max(total, 1)
        return {"accuracy": accuracy}


class FutureTokenProbe:
    """Future-token probe from NextLat: predictive information in representations."""

    def __init__(self, embed_dim=768, vocab_size=50304, offsets=(1, 4, 16)):
        self.vocab_size = vocab_size
        self.offsets = offsets
        self.probes = nn.ModuleDict(
            {f"offset_{d}": nn.Linear(embed_dim, vocab_size) for d in offsets},
        )

    def evaluate(self, model, dataset, device="cuda", max_steps=5000):
        model.eval()
        self.probes = self.probes.to(device)
        results = {}

        for d in self.offsets:
            probe = self.probes[f"offset_{d}"]
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            total_correct = 0
            total_samples = 0

            for step, batch in enumerate(loader):
                if step >= max_steps:
                    break
                input_ids = (
                    batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)
                )
                _B, T = input_ids.shape
                if d >= T:
                    continue
                with torch.no_grad():
                    h, _ = model.encoder(input_ids)
                h_src = h[:, : T - d, :].reshape(-1, h.size(-1))
                target = input_ids[:, d:].reshape(-1)
                logits = probe(h_src)
                loss = F.cross_entropy(logits, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_correct += (logits.argmax(dim=-1) == target).sum().item()
                total_samples += target.size(0)

            results[f"future_probe_d{d}"] = total_correct / max(total_samples, 1)
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
