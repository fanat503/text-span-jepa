# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Representation comparison: JEPA vs baseline (MLM, data2vec)
# Systematic comparison pipeline with linguistic feature extraction,
# geometry comparison, and full reporting


import torch

from src.utils.cka_metrics import linear_cka, rbf_cka


class RepresentationComparator:
    """Systematic comparison of two representation spaces.

    Compares JEPA representations against a baseline (MLM/data2vec)
    across multiple dimensions:
    - Geometry (rank, anisotropy, uniformity)
    - Linguistic feature encoding (probing)
    - Disentanglement
    - Causal structure (intervention predictability)
    """

    def __init__(self, jepa_model, baseline_model, tokenizer=None, device="cpu"):
        self.jepa = jepa_model
        self.baseline = baseline_model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def extract_representations(self, dataloader, max_batches=100, layer_idx=-1):
        """Extract representations from both models.

        Args:
            dataloader: DataLoader yielding (input_ids, ...) batches
            max_batches: max number of batches to process
            layer_idx: which layer to extract (-1 = final)

        Returns:
            dict with 'jepa' and 'baseline' representation tensors
        """
        self.jepa.eval()
        self.baseline.eval()

        jepa_reps = []
        baseline_reps = []
        input_ids_list = []

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            ids = (
                batch.to(self.device)
                if isinstance(batch, torch.Tensor)
                else batch[0].to(self.device)
            )
            input_ids_list.append(ids.cpu())

            h_j, _ = self.jepa.encoder(ids)
            h_b, _ = self.baseline.encoder(ids)

            # Pool: mean over sequence
            jepa_reps.append(h_j.mean(dim=1).cpu())
            baseline_reps.append(h_b.mean(dim=1).cpu())

        return {
            "jepa": torch.cat(jepa_reps, dim=0) if jepa_reps else torch.tensor([]),
            "baseline": torch.cat(baseline_reps, dim=0) if baseline_reps else torch.tensor([]),
            "input_ids": torch.cat(input_ids_list, dim=0) if input_ids_list else torch.tensor([]),
        }

    @torch.no_grad()
    def geometry_comparison(self, jepa_reps, baseline_reps):
        """Compare representation geometry between models.

        Args:
            jepa_reps: (N, D) JEPA representations
            baseline_reps: (N, D) baseline representations

        Returns:
            dict with comparison metrics
        """
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()

        jepa_metrics = diag.compute(
            jepa_reps.unsqueeze(1),  # (N, 1, D)
            jepa_reps.unsqueeze(1),
        )
        baseline_metrics = diag.compute(
            baseline_reps.unsqueeze(1),
            baseline_reps.unsqueeze(1),
        )

        comparison = {}
        for key in jepa_metrics:
            if key.endswith("_online"):
                base_key = key.replace("_online", "")
                comparison[base_key] = {
                    "jepa": jepa_metrics.get(key, 0.0),
                    "baseline": baseline_metrics.get(key, 0.0),
                    "diff": jepa_metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0),
                    "jepa_better": None,  # depends on metric
                }

        # Determine "better" direction per metric
        higher_is_better = {
            "effective_rank",
            "participation_ratio",
            "rank_utilization",
            "sv_entropy",
            "intrinsic_dim",
        }
        lower_is_better = {
            "collapsed_dim_ratio",
            "condition_number",
            "mean_pairwise_cosine",
            "coherence",
            "svd_sharpness",
        }

        for key, vals in comparison.items():
            if key in higher_is_better:
                vals["jepa_better"] = vals["jepa"] > vals["baseline"]
            elif key in lower_is_better:
                vals["jepa_better"] = vals["jepa"] < vals["baseline"]

        return comparison

    @torch.no_grad()
    def cka_similarity(self, jepa_reps, baseline_reps):
        """CKA similarity between JEPA and baseline representations."""
        from src.models.collapse import CollapseDiagnostics

        diag = CollapseDiagnostics()
        cka_lin = linear_cka(jepa_reps, baseline_reps)
        cka_rbf = rbf_cka(jepa_reps, baseline_reps)
        return {
            "cka_linear": cka_lin,
            "cka_rbf": cka_rbf,
        }

    def full_comparison_report(self, dataloader, max_batches=50):
        """Run full comparison pipeline and generate report.

        Args:
            dataloader: DataLoader with text data
            max_batches: how many batches to process

        Returns:
            dict with all comparison results
        """
        # 1. Extract representations
        reps = self.extract_representations(dataloader, max_batches)
        jepa_reps = reps["jepa"]
        baseline_reps = reps["baseline"]

        if jepa_reps.numel() == 0 or baseline_reps.numel() == 0:
            return {"error": "No representations extracted"}

        # 2. Geometry comparison
        geometry = self.geometry_comparison(jepa_reps, baseline_reps)

        # 3. CKA similarity
        cka = self.cka_similarity(jepa_reps, baseline_reps)

        # 4. Linguistic features
        input_ids = reps["input_ids"]
        ling_features = self.extract_linguistic_features(input_ids)

        report = {
            "geometry": geometry,
            "cka": cka,
            "linguistic_features": ling_features,
            "n_samples": jepa_reps.size(0),
        }

        return report

    def extract_linguistic_features(self, input_ids):
        """Extract linguistic features from token IDs.

        Creates simple binary features: is_upper, has_digit, is_short,
        etc. For use with disentanglement metrics.

        Args:
            input_ids: (B, T) token IDs

        Returns:
            dict with feature tensors
        """
        features = {}

        # Token length feature (approximate via token ID magnitude)
        # Short tokens tend to have lower IDs in BPE
        _max_id = input_ids.float().max()
        features["token_magnitude"] = (
            input_ids.float() / _max_id if _max_id > 0 else torch.zeros_like(input_ids)
        )

        # Position feature
        B, T = input_ids.shape
        positions = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(B, T)
        features["position"] = positions / T

        # Boundary features
        features["seq_start"] = torch.zeros(B, T)
        features["seq_start"][:, :5] = 1.0

        features["seq_end"] = torch.zeros(B, T)
        features["seq_end"][:, -5:] = 1.0

        return features


def extract_linguistic_features(tokens):
    """Extract linguistic features from token strings.

    Standalone function for use with tokenized text.

    Args:
        tokens: list of token strings

    Returns:
        dict of feature_name -> float value
    """
    features = {}
    if not tokens:
        return features

    features["is_upper"] = (
        1.0 if (len(tokens) > 0 and len(tokens[0]) > 0 and tokens[0][0].isupper()) else 0.0
    )
    features["n_tokens"] = float(len(tokens))
    features["avg_token_len"] = sum(len(t) for t in tokens) / max(len(tokens), 1)
    features["has_digit"] = 1.0 if any(c.isdigit() for t in tokens for c in t) else 0.0
    features["frac_upper"] = sum(1 for t in tokens if t[0].isupper() if len(t) > 0) / max(
        len(tokens), 1
    )
    features["has_punct"] = 1.0 if any(not t.isalnum() for t in tokens) else 0.0

    return features
