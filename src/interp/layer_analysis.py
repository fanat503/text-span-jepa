# Copyright 2026 Text-Span-JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Layer-wise representation analysis
#
# Every interpretability paper shows HOW information is distributed
# across layers. Without this, the story is incomplete.
#
# Key questions:
# - Which layer encodes syntax? Which encodes semantics?
# - Is information uniformly distributed or bottlenecked?
# - Does JEPA have a different layer-wise profile than MLM?
#
# Hypothesis: JEPA distributes information more uniformly across
# layers (no "linguistic bottleneck") because the predictor forces
# each layer to maintain predictive information, whereas MLM allows
# layers to "forget" information that's not needed for reconstruction.

import torch
import torch.nn.functional as F
from torch import nn


class LayerwiseProbe:
    """Train probes at each layer to find where linguistic information
    is encoded.

    Following Tenney et al. (2019) "BERT Rediscovers the Classical NLP
    Pipeline": probe each layer for POS, syntax, semantics, etc.
    The layer where probe accuracy peaks = "where this info lives".

    JEPA hypothesis: JEPA layers have MORE UNIFORM probe accuracy
    (information distributed evenly) vs MLM (sharp peaks at specific layers).
    """

    def __init__(
        self, embed_dim=768, num_classes=2, lr=1e-3, max_epochs=30, patience=5, device="cpu"
    ):
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device

    def _train_linear_probe(self, representations, labels):
        """Train a single linear probe."""
        representations = representations.detach().float()
        N = representations.size(0)
        if N < 10:
            return 0.0

        n_train = int(0.8 * N)
        idx = torch.randperm(N)
        train_reps = representations[idx[:n_train]].to(self.device)
        train_labels = labels[idx[:n_train]].to(self.device)
        val_reps = representations[idx[n_train:]].to(self.device)
        val_labels = labels[idx[n_train:]].to(self.device)

        num_classes = max(int(labels.max().item()) + 1, 2)
        probe = nn.Linear(self.embed_dim, num_classes).to(self.device)
        opt = torch.optim.Adam(probe.parameters(), lr=self.lr)

        best_acc = 0.0
        no_improve = 0

        for epoch in range(self.max_epochs):
            probe.train()
            logits = probe(train_reps)
            loss = F.cross_entropy(logits, train_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            probe.eval()
            with torch.no_grad():
                logits = probe(val_reps)
                acc = (logits.argmax(dim=-1) == val_labels).float().mean().item()

            if acc > best_acc:
                best_acc = acc
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break

        return best_acc

    @torch.no_grad()
    def probe_all_layers(self, layer_representations, labels, task_name="default"):
        """Probe each layer for task-specific information.

        Args:
            layer_representations: list of (N, D) tensors, one per layer
            labels: (N,) class labels
            task_name: name of the probing task

        Returns:
            dict with per-layer accuracy and peak layer
        """
        with torch.enable_grad():
            accuracies = []
            for layer_idx, reps in enumerate(layer_representations):
                acc = self._train_linear_probe(reps, labels)
                accuracies.append(acc)

        # Find peak layer
        peak_layer = accuracies.index(max(accuracies)) if accuracies else 0

        # Layer uniformity: 1 - std(accuracies) / mean(accuracies)
        if accuracies and sum(accuracies) > 0:
            mean_acc = sum(accuracies) / len(accuracies)
            std_acc = (sum((a - mean_acc) ** 2 for a in accuracies) / len(accuracies)) ** 0.5
            uniformity = 1.0 - std_acc / mean_acc if mean_acc > 0 else 0.0
        else:
            uniformity = 0.0

        return {
            "task": task_name,
            "per_layer_accuracy": accuracies,
            "peak_layer": peak_layer,
            "peak_accuracy": max(accuracies) if accuracies else 0.0,
            "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "layer_uniformity": max(min(uniformity, 1.0), 0.0),
            "n_layers": len(accuracies),
        }

    def compare_layer_profiles(self, jepa_layers, baseline_layers, labels, task_name="default"):
        """Compare layer-wise profiles between JEPA and baseline.

        Args:
            jepa_layers: list of (N, D) JEPA layer representations
            baseline_layers: list of (N, D) baseline layer representations
            labels: (N,) class labels

        Returns:
            dict with comparison
        """
        jepa_result = self.probe_all_layers(jepa_layers, labels, f"{task_name}_jepa")
        baseline_result = self.probe_all_layers(baseline_layers, labels, f"{task_name}_baseline")

        return {
            "task": task_name,
            "jepa_uniformity": jepa_result["layer_uniformity"],
            "baseline_uniformity": baseline_result["layer_uniformity"],
            "jepa_more_uniform": jepa_result["layer_uniformity"]
            > baseline_result["layer_uniformity"],
            "jepa_peak_layer": jepa_result["peak_layer"],
            "baseline_peak_layer": baseline_result["peak_layer"],
            "jepa_peak_accuracy": jepa_result["peak_accuracy"],
            "baseline_peak_accuracy": baseline_result["peak_accuracy"],
        }


class LayerwiseCKA:
    """CKA similarity between layers of different models.

    If JEPA layer 3 is similar to MLM layer 6 → JEPA encodes
    the same information in fewer layers (more efficient).

    Also: CKA between adjacent layers within a model shows
    how much each layer transforms the representation.
    """

    @staticmethod
    @torch.no_grad()
    def inter_model_cka(jepa_layers, baseline_layers):
        """CKA between all pairs of layers across models.

        Args:
            jepa_layers: list of (N, D) tensors
            baseline_layers: list of (N, D) tensors

        Returns:
            (L_jepa, L_baseline) CKA matrix
        """
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()

        L_j = len(jepa_layers)
        L_b = len(baseline_layers)
        cka_matrix = torch.zeros(L_j, L_b)

        for i in range(L_j):
            for j in range(L_b):
                flat_j = jepa_layers[i].reshape(-1, jepa_layers[i].size(-1))
                flat_b = baseline_layers[j].reshape(-1, baseline_layers[j].size(-1))
                # Need same number of samples
                N = min(flat_j.size(0), flat_b.size(0))
                cka_matrix[i, j] = diag._cka_linear(flat_j[:N], flat_b[:N])

        return cka_matrix

    @staticmethod
    @torch.no_grad()
    def intra_model_cka(layers):
        """CKA between adjacent layers within a model.

        Shows how much each layer transforms the representation.
        Large CKA between adjacent layers = redundant.
        Small CKA = each layer adds new information.

        Args:
            layers: list of (N, D) tensors

        Returns:
            dict with per-layer CKA and mean
        """
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()

        adjacent_cka = []
        for i in range(len(layers) - 1):
            flat_a = layers[i].reshape(-1, layers[i].size(-1))
            flat_b = layers[i + 1].reshape(-1, layers[i + 1].size(-1))
            N = min(flat_a.size(0), flat_b.size(0))
            cka = diag._cka_linear(flat_a[:N], flat_b[:N])
            adjacent_cka.append(cka)

        return {
            "per_layer_cka": adjacent_cka,
            "mean_adjacent_cka": sum(adjacent_cka) / len(adjacent_cka) if adjacent_cka else 0.0,
            "min_cka": min(adjacent_cka) if adjacent_cka else 0.0,
            "redundant_layers": sum(1 for c in adjacent_cka if c > 0.95),
        }


class LayerwiseGeometry:
    """Track how representation geometry evolves across layers.

    Effective rank, anisotropy, etc. at each layer.
    JEPA hypothesis: more uniform geometry across layers
    (no sharp transitions) vs MLM (bottleneck at middle layers).
    """

    @staticmethod
    @torch.no_grad()
    def compute(layers):
        """Compute geometry metrics at each layer.

        Args:
            layers: list of (N, D) tensors

        Returns:
            dict with per-layer geometry
        """
        from src.interp.representation_geometry import RepresentationGeometry

        per_layer = {}
        for i, reps in enumerate(layers):
            geom = RepresentationGeometry.compute_all(reps)
            per_layer[f"layer_{i}"] = geom

        # Aggregate: how much does geometry change across layers?
        eff_dims = [per_layer[f"layer_{i}"]["effective_dimension"] for i in range(len(layers))]
        anisotropies = [per_layer[f"layer_{i}"]["anisotropy"] for i in range(len(layers))]

        # Coefficient of variation (lower = more uniform)
        if eff_dims and sum(eff_dims) > 0:
            mean_ed = sum(eff_dims) / len(eff_dims)
            std_ed = (sum((e - mean_ed) ** 2 for e in eff_dims) / len(eff_dims)) ** 0.5
            cv_eff_dim = std_ed / mean_ed if mean_ed > 0 else 0.0
        else:
            cv_eff_dim = 0.0

        return {
            "per_layer": per_layer,
            "effective_dims": eff_dims,
            "anisotropies": anisotropies,
            "cv_effective_dim": cv_eff_dim,  # Lower = more uniform
            "n_layers": len(layers),
        }

    @staticmethod
    def compare(jepa_layers, baseline_layers):
        """Compare layer-wise geometry between JEPA and baseline.

        JEPA should have LOWER cv_effective_dim (more uniform geometry).
        """
        jepa_geom = LayerwiseGeometry.compute(jepa_layers)
        baseline_geom = LayerwiseGeometry.compute(baseline_layers)

        return {
            "jepa_cv_eff_dim": jepa_geom["cv_effective_dim"],
            "baseline_cv_eff_dim": baseline_geom["cv_effective_dim"],
            "jepa_more_uniform": jepa_geom["cv_effective_dim"] < baseline_geom["cv_effective_dim"],
            "jepa_eff_dims": jepa_geom["effective_dims"],
            "baseline_eff_dims": baseline_geom["effective_dims"],
        }


class LayerRoutingAnalysis:
    """Analyze which layers route information to which downstream tasks.

    "Routing" = how much removing a layer's output affects
    performance on different tasks.

    If layer 3 of JEPA is critical for POS but not NER → that layer
    "routes" POS information. More specialized routing = more interpretable.
    """

    @staticmethod
    def routing_score(layer_representations, task_labels_dict, embed_dim=768, device="cpu"):
        """Compute routing scores: how critical is each layer for each task.

        Uses leave-one-out: remove each layer (ablate) and measure
        performance drop. Large drop = that layer is critical for that task.

        Args:
            layer_representations: list of (N, D) tensors
            task_labels_dict: {task_name: (N,) labels}
            embed_dim: embedding dimension
            device: compute device

        Returns:
            dict with (n_layers, n_tasks) routing matrix
        """
        n_layers = len(layer_representations)
        task_names = list(task_labels_dict.keys())
        n_tasks = len(task_names)
        if n_layers < 2:
            # Leave-one-out is undefined for a single layer.
            return {
                "routing_matrix": torch.zeros(n_layers, max(n_tasks, 0)),
                "task_names": task_names,
                "baseline_accs": {},
                "specialization": 0.0,
                "skipped": True,
            }

        # First, compute baseline accuracy using ALL layers (sum)
        combined = sum(layer_representations) / n_layers
        baseline_accs = {}
        for task_name, labels in task_labels_dict.items():
            from src.interp.layer_analysis import LayerwiseProbe

            probe = LayerwiseProbe(embed_dim=embed_dim, max_epochs=20, device=device)
            with torch.enable_grad():
                acc = probe._train_linear_probe(combined.detach().float(), labels)
            baseline_accs[task_name] = acc

        # Leave-one-out: remove each layer and recompute
        routing = torch.zeros(n_layers, n_tasks)
        for i in range(n_layers):
            # Sum all layers except i
            # Normalize by the FULL layer count so baseline and ablation share
            # one scale (dividing by n_layers-1 inflated LOO representations).
            loo_combined = sum(l for j, l in enumerate(layer_representations) if j != i) / (
                n_layers
            )
            for t_idx, (task_name, labels) in enumerate(task_labels_dict.items()):
                from src.interp.layer_analysis import LayerwiseProbe

                probe = LayerwiseProbe(embed_dim=embed_dim, max_epochs=20, device=device)
                with torch.enable_grad():
                    acc = probe._train_linear_probe(loo_combined.detach().float(), labels)
                # Routing score = baseline - leave-one-out (higher = more critical)
                routing[i, t_idx] = baseline_accs[task_name] - acc

        return {
            "routing_matrix": routing,
            "task_names": task_names,
            "baseline_accs": baseline_accs,
            "specialization": routing.max(dim=1).values.mean().item(),  # Mean max routing per layer
        }
