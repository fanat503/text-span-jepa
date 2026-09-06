# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Information-theoretic metrics for representation analysis
#
# THE KEY THEORETICAL DIFFERENCE between JEPA and MLM:
# - JEPA predicts in latent space → encodes only PREDICTABLE information
# - MLM reconstructs tokens → encodes ALL information needed for reconstruction
#
# This predicts:
# - MI(h_jepa; abstract | surface) > MI(h_mlm; abstract | surface)
# - JEPA has LOWER MI with surface features (token identity)
# - JEPA has HIGHER MI with abstract features (syntax, semantics)
# - JEPA has lower TOTAL mutual information with input (more compressed)
#
# Methods:
# - MINE (Mutual Information Neural Estimation, Belghazi et al., 2018)
# - InfoNCE (van den Oord et al., 2018)
# - Conditional MI estimation
# - Representation entropy / information compression

import math

import torch
import torch.nn.functional as F
from torch import nn


class MINEEstimator(nn.Module):
    """MINE: Mutual Information Neural Estimation.

    Belghazi et al., "MINE: Mutual Information Neural Estimation",
    ICML 2018.

    Estimates MI(X; Y) by training a neural network to maximize:
        sup_T  E_P[T(x,y)] - log(E_Q[exp(T(x,y))])
    where T is the statistics network, P is joint, Q is product of marginals.

    Used to measure MI between representations and linguistic properties.
    """

    def __init__(self, dim_x, dim_y, hidden_dim=128):
        super().__init__()
        self.statistics_net = nn.Sequential(
            nn.Linear(dim_x + dim_y, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, y):
        """Compute MINE statistic."""
        return self.statistics_net(torch.cat([x, y], dim=-1))

    def compute_mi(self, x, y, n_steps=100, lr=1e-3, batch_size=None):
        """Estimate MI(X; Y) by training MINE.

        Args:
            x: (N, dx) samples from X
            y: (N, dy) samples from Y (paired with x)
            n_steps: training steps
            lr: learning rate
            batch_size: mini-batch size (None = full batch)

        Returns:
            float: estimated MI in nats

        """
        N = x.size(0)
        if batch_size is None:
            batch_size = min(N, 256)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        mi_estimates = []
        for step in range(n_steps):
            # Sample batch
            idx = torch.randperm(N)[:batch_size]
            x_batch = x[idx]
            y_batch = y[idx]

            # Marginal samples: shuffle y
            y_marginal = y_batch[torch.randperm(batch_size)]

            # MINE objective
            t_joint = self(x_batch, y_batch)
            t_marginal = self(x_batch, y_marginal)

            # MI estimate: E[T(x,y)] - log(E[exp(T(x,y'))])
            mi_lower_bound = t_joint.mean() - torch.log(torch.exp(t_marginal).mean() + 1e-8)

            # Maximize lower bound = minimize negative
            optimizer.zero_grad()
            (-mi_lower_bound).backward()
            optimizer.step()

            mi_estimates.append(mi_lower_bound.item())

        # Return moving average of last 10% of estimates
        n_last = max(len(mi_estimates) // 10, 1)
        return sum(mi_estimates[-n_last:]) / n_last


class InfoNCEEstimator:
    """InfoNCE: lower bound on MI via noise-contrastive estimation.

    van den Oord et al., "Representation Learning with Contrastive
    Predictive Coding", 2018.

    MI(X; Y) >= log(N) - log(sum_j exp(f(x, y_j)) / exp(f(x, y+)))

    Simpler and more stable than MINE for large batch sizes.
    """

    @staticmethod
    @torch.no_grad()
    def compute(x, y, temperature=0.1):
        """Compute InfoNCE estimate of MI(X; Y).

        Args:
            x: (N, dx) representations
            y: (N, dy) paired features
            temperature: softmax temperature

        Returns:
            float: InfoNCE lower bound on MI in nats

        """
        N = x.size(0)
        if N < 2:
            return 0.0

        # Normalize
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)

        # Similarity matrix
        sim = x_norm @ y_norm.T / temperature  # (N, N)

        # InfoNCE: for each x_i, y_i is positive, rest are negatives
        labels = torch.arange(N, device=x.device)
        loss = F.cross_entropy(sim, labels)

        # MI >= log(N) - loss
        mi = math.log(N) - loss.item()
        return max(mi, 0.0)


class ConditionalMIEstimator:
    """Conditional mutual information: MI(X; Y | Z).

    For our hypothesis: MI(h; abstract | surface) should be
    HIGHER for JEPA than MLM, meaning JEPA encodes more abstract
    information BEYOND what surface features provide.

    Estimation via: MI(X; Y | Z) = MI(X; Y, Z) - MI(X; Z)
    """

    @staticmethod
    def compute(
        representations,
        target_features,
        conditioning_features,
        method="infonce",
        n_steps=100,
    ):
        """Estimate MI(h; target | condition).

        Args:
            representations: (N, D) model representations
            target_features: (N, K) target features (e.g., POS tags)
            conditioning_features: (N, M) features to condition on (e.g., surface)
            method: 'infonce' or 'mine'
            n_steps: training steps for MINE

        Returns:
            dict with MI estimates

        """
        _N, D = representations.shape
        K = target_features.shape[1]
        M = conditioning_features.shape[1]

        # MI(h; target, condition) using InfoNCE
        combined = torch.cat([target_features, conditioning_features], dim=-1)
        if method == "infonce":
            mi_joint = InfoNCEEstimator.compute(representations, combined)
            mi_condition = InfoNCEEstimator.compute(representations, conditioning_features)
        else:
            mine_joint = MINEEstimator(D, K + M)
            mine_cond = MINEEstimator(D, M)
            mi_joint = mine_joint.compute_mi(representations, combined, n_steps)
            mi_condition = mine_cond.compute_mi(representations, conditioning_features, n_steps)

        # MI(h; target | condition) = MI(h; target, condition) - MI(h; condition)
        mi_conditional = mi_joint - mi_condition

        # Also compute MI(h; target) without conditioning
        if method == "infonce":
            mi_target = InfoNCEEstimator.compute(representations, target_features)
        else:
            mine_target = MINEEstimator(D, K)
            mi_target = mine_target.compute_mi(representations, target_features, n_steps)

        return {
            "mi_target": mi_target,
            "mi_condition": mi_condition,
            "mi_joint": mi_joint,
            "mi_conditional": mi_conditional,  # THE KEY METRIC
            "information_gain": mi_conditional
            / max(mi_target, 1e-10),  # Fraction of abstract info beyond surface
        }


class RepresentationCompression:
    """Measure how much the representation compresses the input.

    JEPA should compress MORE than MLM (fewer bits needed to
    describe the representation) because it only encodes predictable
    information, discarding noise.

    Metrics:
    - Shannon entropy of the representation distribution
    - Rate-distortion: how much information is preserved at each compression level
    - Minimum description length (MDL) approximation
    """

    @staticmethod
    @torch.no_grad()
    def entropy_estimate(representations, n_bins=30):
        """Shannon entropy of the representation distribution.

        Lower entropy = more compressed representation.
        JEPA should have LOWER entropy than MLM (more compressed,
        less noise).

        Uses histogram-based estimation per dimension,
        then averages.

        Args:
            representations: (N, D)
            n_bins: number of histogram bins per dimension

        Returns:
            float: average entropy in nats

        """
        if representations.dim() == 3:
            flat = representations.reshape(-1, representations.size(-1))
        else:
            flat = representations

        N, D = flat.shape
        if N < 2:
            return 0.0

        # Per-dimension entropy
        entropies = []
        for d in range(D):
            vals = flat[:, d]
            # Histogram
            try:
                counts = torch.histc(vals, bins=n_bins)
                probs = counts.float() / counts.sum()
                probs = probs[probs > 0]
                h = -(probs * torch.log(probs)).sum().item()
                entropies.append(h)
            except Exception:
                continue

        if not entropies:
            return 0.0
        return sum(entropies) / len(entropies)

    @staticmethod
    @torch.no_grad()
    def total_correlation(representations):
        """Total correlation: measures dependency between dimensions.

        TC = sum H(X_i) - H(X_1, ..., X_D)
        Higher TC = more dependency between dimensions = less disentangled.

        JEPA should have LOWER TC (more independent features).
        MLM should have HIGHER TC (entangled features).

        Approximation: TC ≈ sum H(X_i) - H(X_joint)
        We estimate H(X_joint) via the log-det of the covariance matrix.

        Args:
            representations: (N, D)

        Returns:
            float: total correlation in nats

        """
        if representations.dim() == 3:
            flat = representations.reshape(-1, representations.size(-1))
        else:
            flat = representations.float()

        N, D = flat.shape
        if N <= D:
            return 0.0

        # Marginal entropies
        marginal_entropy = RepresentationCompression.entropy_estimate(flat)

        # Joint entropy via covariance matrix
        centered = flat - flat.mean(dim=0)
        cov = (centered.T @ centered) / (N - 1)

        try:
            # H(X_joint) ≈ 0.5 * D * (1 + log(2π)) + 0.5 * log|Σ|
            sign, logdet = torch.linalg.slogdet(cov)
            if sign <= 0:
                return 0.0
            joint_entropy = 0.5 * D * (1 + math.log(2 * math.pi)) + 0.5 * logdet.item()
        except Exception:
            return 0.0

        tc = D * marginal_entropy - joint_entropy
        return max(tc, 0.0)

    @staticmethod
    @torch.no_grad()
    def compression_ratio(representations, original_dim):
        """Compression ratio: how much the representation compresses input.

        ratio = H(representation) / H(random_baseline)

        Lower = more compressed. JEPA should compress more.

        Args:
            representations: (N, D)
            original_dim: dimension of the original input space

        Returns:
            float: compression ratio in [0, inf)

        """
        rep_entropy = RepresentationCompression.entropy_estimate(representations)
        # Random baseline: entropy of isotropic Gaussian in D dims
        # H(N(0,I)) = 0.5 * D * (1 + log(2π))
        representations.size(-1) if representations.dim() <= 2 else representations.size(-1)
        baseline = 0.5 * original_dim * (1 + math.log(2 * math.pi))

        if baseline == 0:
            return 0.0
        return rep_entropy / baseline


class InformationPlane:
    """Information Plane analysis: I(h; X) vs I(h; Y).

    Shwartz-Ziv & Tishby (2017): neural networks compress then fit.
    JEPA should have LOWER I(h; X) (compressed) and HIGHER I(h; Y)
    (preserving task-relevant info) compared to MLM.

    The "information bottleneck gap" = I(h; Y) - I(h; X)
    Higher gap = better representation (more task info, less input info).
    """

    @staticmethod
    def compute(representations, input_features, task_labels, method="infonce"):
        """Compute information plane coordinates.

        Args:
            representations: (N, D) model representations
            input_features: (N, M) input features (token IDs, surface features)
            task_labels: (N, K) task-relevant features (POS, NER, etc.)
            method: 'infonce' or 'mine'

        Returns:
            dict with I(h; X), I(h; Y), and IB gap

        """
        if method == "infonce":
            mi_input = InfoNCEEstimator.compute(representations, input_features.float())
            mi_task = InfoNCEEstimator.compute(representations, task_labels.float())
        else:
            D = representations.size(-1)
            mine_x = MINEEstimator(D, input_features.size(-1))
            mine_y = MINEEstimator(D, task_labels.size(-1))
            mi_input = mine_x.compute_mi(representations, input_features.float(), n_steps=200)
            mi_task = mine_y.compute_mi(representations, task_labels.float(), n_steps=200)

        return {
            "mi_input": mi_input,  # I(h; X) — should be LOW for JEPA
            "mi_task": mi_task,  # I(h; Y) — should be HIGH for JEPA
            "ib_gap": mi_task - mi_input,  # Information bottleneck gap — should be HIGH for JEPA
            "compression_efficiency": mi_task / max(mi_input, 1e-10),  # MI per bit of input info
        }

    @staticmethod
    def compare(jepa_reps, baseline_reps, input_features, task_labels, method="infonce"):
        """Compare information plane between JEPA and baseline.

        THE KEY COMPARISON: if JEPA has higher IB gap, it means JEPA
        preserves more task info per bit of input info.
        """
        jepa_ip = InformationPlane.compute(jepa_reps, input_features, task_labels, method)
        baseline_ip = InformationPlane.compute(baseline_reps, input_features, task_labels, method)

        return {
            "jepa": jepa_ip,
            "baseline": baseline_ip,
            "jepa_more_compressed": jepa_ip["mi_input"] < baseline_ip["mi_input"],
            "jepa_preserves_more_task_info": jepa_ip["mi_task"] > baseline_ip["mi_task"],
            "jepa_higher_ib_gap": jepa_ip["ib_gap"] > baseline_ip["ib_gap"],
            "ib_gap_diff": jepa_ip["ib_gap"] - baseline_ip["ib_gap"],
        }
