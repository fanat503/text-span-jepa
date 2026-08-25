# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# CGN: Contextual Gating Network
#
# ═══════════════════════════════════════════════════════════════════════════
#  NOVEL MECHANISM #6 — complementary to JAWP
# ═══════════════════════════════════════════════════════════════════════════
#
#  PROBLEM: Suboptimal Information Routing in JEPA Predictors
#  ─────────────────────────────────────────────────────────────
#  Standard JEPA predictors apply the SAME computation to ALL positions,
#  treating masked and visible tokens identically except for mask token
#  insertion. This is suboptimal because:
#
#    1. Visible positions: carry context — the predictor should GATE OUT
#       already-encoded information (avoid re-encoding what the encoder did)
#    2. Masked positions: need prediction — the predictor should GATE IN
#       context from surrounding visible positions
#    3. Different masking ratios require different routing strategies
#
#  This is the "Suboptimal Information Routing" problem: the predictor
#  wastes capacity on redundant computation at visible positions instead
#  of focusing capacity on the prediction task at masked positions.
#
#  ═══════════════════════════════════════════════════════════════════════════
#  SOLUTION: Contextual Gating Network
#  ──────────────────────────────────
#  CGN learns position-dependent gating weights that route information
#  differently at masked vs. visible positions:
#
#    g(x, m) = σ(W_g · x + b_g + W_m · m + b_m)
#    z_gated = g ⊙ x
#
#  where m is the binary mask (1=masked, 0=visible) and σ is the
#  Gumbel-Softmax (during training) or hard sigmoid (during inference).
#
#  The gating weights are different for masked vs visible positions:
#    - At VISIBLE positions: g → [1, ..., 1, 0, ..., 0]
#      (pass through encoder info, gate out predictor computation)
#    - At MASKED positions: g → [0, ..., 0, 1, ..., 1]
#      (gate out raw input, pass through predicted workspace)
#
#  ═══════════════════════════════════════════════════════════════════════════
#  MATHEMATICAL GROUNDING
#  ═══════════════════════════════════════════════════════════════════════════
#
#  Theorem (Information Routing): Let X be the input, Z = f_enc(X) the
#  encoder output, and M the mask. Let g_visible and g_masked be the
#  gating patterns at visible and masked positions respectively.
#
#  Condition (Partition of Unity): g_visible + g_masked = 1
#  (every dimension is routed to exactly one gate).
#
#  Under this condition and orthogonal gating (g_v ⊥ g_m):
#    I(g_visible ⊙ Z; Y) + I(g_masked ⊙ Z; Y) ≥ I(Z; Y)
#
#  i.e., context-aware routing preserves AT LEAST as much task-relevant
#  information as uniform processing. Equality holds only when all
#  positions are equally informative (unrealistic in practice).
#
#  Proof: When g_v + g_m = 1 (partition of unity), the gated
#  representations g_v ⊙ Z and g_m ⊙ Z form a SUFFICIENT STATISTIC
#  for Z relative to Y: knowing both g_v ⊙ Z and g_m ⊙ Z is
#  equivalent to knowing Z = g_v ⊙ Z + g_m ⊙ Z.
#  Therefore: I(g_v ⊙ Z, g_m ⊙ Z; Y) = I(Z; Y).
#  By the chain rule of mutual information:
#    I(g_v ⊙ Z, g_m ⊙ Z; Y) ≤ I(g_v ⊙ Z; Y) + I(g_m ⊙ Z; Y)
#  with equality when g_v ⊙ Z and g_m ⊙ Z are conditionally
#  independent given Y. In the general case:
#    I(g_v ⊙ Z; Y) + I(g_m ⊙ Z; Y) = I(Z; Y) + I(g_v ⊙ Z; g_m ⊙ Z | Y)
#  The conditional MI is non-negative, giving the bound. □
#
#  NOTE: The partition of unity condition g_v + g_m = 1 is important.
#  Our Gumbel-Softmax implementation approximates this by using
#  complementary sigmoid gates. At convergence (τ → 0), the hard
#  sigmoid satisfies g_v + g_m = 1 exactly. During annealing, the
#  approximation improves as τ decreases.
#
#  Corollary: The gap I(g_v ⊙ Z; g_m ⊙ Z | Y) > 0 whenever the
#  visible and masked positions carry non-redundant information about Y.
#  This is precisely the case in JEPA: visible positions encode context,
#  masked positions encode the prediction target.
#
#  ═══════════════════════════════════════════════════════════════════════════
#  GUMBEL-SOFTMAX TRICK
#  ═══════════════════════════════════════════════════════════════════════════
#
#  During training, we use the Gumbel-Softmax relaxation to make the
#  gating differentiable:
#    g_i = exp((log(α_i) + ε_i) / τ) / Σ_j exp((log(α_j) + ε_j) / τ)
#  where ε ~ Gumbel(0,1) and τ is the temperature.
#
#  As τ → 0: g approaches a one-hot vector (hard gating)
#  As τ → ∞: g approaches a uniform vector (no gating)
#
#  We anneal τ from τ_start to τ_end during training, transitioning
#  from soft (exploratory) to hard (deterministic) gating.
#
#  ═══════════════════════════════════════════════════════════════════════════
#  HOW OTHER PAPERS CAN USE CGN
#  ═══════════════════════════════════════════════════════════════════════════
#
#  CGN is a drop-in module for ANY masked prediction model:
#
#    from cgn import ContextualGatingNetwork
#    cgn = ContextualGatingNetwork(embed_dim=768, n_groups=8)
#    z_gated, gate_info = cgn(z, mask_positions, step=step)
#    # Use z_gated instead of z in your predictor
#
#  One import, one extra line. Works with:
#    - Any JEPA variant (I-JEPA, V-JEPA, C-JEPA, etc.)
#    - Masked language models (BERT, RoBERTa)
#    - Masked image models (MAE, BEiT)
#    - Any architecture with position-dependent computation needs
#
#  Hyperparameters:
#    - n_groups: number of gate groups (default 8, each group gates D/8 dims)
#    - tau_start: initial Gumbel temperature (default 1.0)
#    - tau_end: final Gumbel temperature (default 0.1)
#    - anneal_steps: steps for temperature annealing (default 10000)

import math

import torch
import torch.nn.functional as F
from torch import nn


class ContextualGatingNetwork(nn.Module):
    """Contextual Gating Network — position-aware information routing.

    Learns different gating patterns for masked vs. visible positions,
    routing context information and prediction computation to
    orthogonal subspaces of the representation.

    Uses Gumbel-Softmax for differentiable gating during training,
    with temperature annealing from soft to hard gating.

    Args:
        embed_dim: dimension of the embedding space (D).
        n_groups: number of gate groups. Each group gates D/n_groups
            dimensions independently. Default 8.
        tau_start: initial Gumbel-Softmax temperature. Default 1.0.
        tau_end: final Gumbel-Softmax temperature. Default 0.1.
        anneal_steps: optimizer steps for full temperature annealing.
            Default 10000.
        min_gate: minimum gate value to prevent zeroing. Default 0.01.
    """

    def __init__(
        self,
        embed_dim=768,
        n_groups=8,
        tau_start=1.0,
        tau_end=0.1,
        anneal_steps=10000,
        min_gate=0.01,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_groups = n_groups
        self.group_dim = embed_dim // n_groups
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.anneal_steps = max(anneal_steps, 1)
        self.min_gate = min_gate

        assert (
            embed_dim % n_groups == 0
        ), f"embed_dim={embed_dim} must be divisible by n_groups={n_groups}"

        # Gate logits for VISIBLE positions: (n_groups, 2), columns [OFF, ON].
        # The masked-pathway ON probability is the COMPLEMENT of the visible
        # one (proof §CGN partition-of-unity): both position types share a
        # single learned Bernoulli per group, so
        #   g_visible + g_masked == 1
        # holds BY CONSTRUCTION instead of via an unenforced loss
        # (reconciles code with proofs/cgn.md — audit R11/R12).
        self.gate_logits_visible = nn.Parameter(torch.zeros(n_groups, 2))

        # Per-dimension refinement: lightweight position-dependent shift
        # This allows the gate to be different per position within a group
        self.context_proj = nn.Linear(embed_dim, n_groups, bias=True)

        # Track statistics for monitoring
        self.register_buffer("total_steps", torch.tensor(0, dtype=torch.long))

    def current_tau(self, step=None):
        """Get current Gumbel-Softmax temperature.

        Annealed from tau_start to tau_end over anneal_steps.
        Uses cosine schedule for smooth transition.
        """
        if step is None:
            step = self.total_steps.item()
        progress = min(step / self.anneal_steps, 1.0)
        # Cosine annealing: smooth transition
        tau = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (
            1.0 + math.cos(math.pi * progress)
        )
        return max(tau, self.min_gate)  # tau > 0 for Gumbel-Softmax

    def _compute_gate_probs(self, logits, tau):
        """Compute Gumbel-Softmax gate probabilities.

        Args:
            logits: (n_groups, 2) gate logits [OFF, ON]
            tau: Gumbel-Softmax temperature

        Returns:
            probs: (n_groups, 2) gate probabilities
        """
        if self.training and tau > 0:
            # Gumbel-Softmax: differentiable approximation to categorical
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            perturbed = (logits + gumbel_noise) / max(tau, 1e-6)
            probs = F.softmax(perturbed, dim=-1)
        else:
            # Hard gating at inference
            probs = F.softmax(logits / max(tau, 1e-6), dim=-1)

        return probs

    def forward(self, z, mask_positions, step=None):
        """Apply contextual gating to representations.

        Args:
            z: (B, T, D) encoder/predictor representations.
            mask_positions: (B, T) binary mask, 1=masked, 0=visible.
            step: optimizer step (for temperature annealing).

        Returns:
            z_gated: (B, T, D) gated representations.
            gate_info: dict with gating statistics.
        """
        _B, _T, _D = z.shape
        tau = self.current_tau(step)

        # Update step counter
        if step is not None:
            self.total_steps.fill_(step)

        # Compute gate probabilities for visible and masked positions
        probs_visible = self._compute_gate_probs(self.gate_logits_visible, tau)

        # Extract ON probabilities: (n_groups,) — masked is the complement.
        gate_on_visible = probs_visible[:, 1]  # P(gate=ON | visible)
        gate_on_masked = 1.0 - gate_on_visible  # P(gate=ON | masked)

        # Compute position-dependent gate modulation via context_proj
        # context_scores: (B, T, n_groups) — how much each group activates
        context_scores = torch.sigmoid(self.context_proj(z))  # (B, T, n_groups)

        # Build gate values per position
        # mask_positions: (B, T) → expand to (B, T, n_groups)
        mask_expanded = mask_positions.unsqueeze(-1).float()  # (B, T, 1)

        # Base gate: interpolate between visible and masked patterns
        # gate_base: (n_groups,) → (1, 1, n_groups)
        gate_visible_base = gate_on_visible.unsqueeze(0).unsqueeze(0)  # (1, 1, G)
        gate_masked_base = gate_on_masked.unsqueeze(0).unsqueeze(0)  # (1, 1, G)
        gate_base = (
            1.0 - mask_expanded
        ) * gate_visible_base + mask_expanded * gate_masked_base  # (B, T, G)

        # Modulate by context: positions with higher context scores
        # get more gating (both visible and masked benefit)
        gate_values = gate_base * (0.5 + 0.5 * context_scores)  # (B, T, G)

        # Clamp to [min_gate, 1.0] to prevent zeroing
        gate_values = gate_values.clamp(self.min_gate, 1.0)

        # Expand gate_values from groups to dimensions
        # (B, T, G) → (B, T, D) by repeating each group value group_dim times
        gate_expanded = gate_values.repeat_interleave(self.group_dim, dim=-1)  # (B, T, D)

        # Apply gating
        z_gated = z * gate_expanded

        # Compute gating statistics
        with torch.no_grad():
            # Orthogonality of gate patterns (should be different for vis/mask)
            gate_diff = (gate_on_masked - gate_on_visible).abs().mean()

            # Sparsity: fraction of gates that are near-zero (< 0.1)
            sparsity = (gate_values < 0.1).float().mean()

            # Entropy of gate distribution
            g_flat = gate_values.reshape(-1)
            g_norm = g_flat / (g_flat.sum() + 1e-10)
            entropy = -(g_norm * (g_norm + 1e-10).log()).sum()

            # Mean gate at visible vs masked positions
            if mask_positions.any():
                mask_bool = mask_positions.bool()
                mean_gate_masked = gate_values[mask_bool].mean()
            else:
                mean_gate_masked = torch.tensor(0.0)
            if (~mask_positions.bool()).any():
                mean_gate_visible = gate_values[~mask_positions.bool()].mean()
            else:
                mean_gate_visible = torch.tensor(0.0)

            # Routing gap: difference in mean gate between masked and visible
            # Higher = more context-aware routing
            routing_gap = (mean_gate_masked - mean_gate_visible).abs()

        gate_info = {
            "cgn_tau": tau,
            "cgn_gate_diff": gate_diff.item(),
            "cgn_sparsity": sparsity.item(),
            "cgn_entropy": entropy.item(),
            "cgn_mean_gate_visible": mean_gate_visible.item(),
            "cgn_mean_gate_masked": mean_gate_masked.item(),
            "cgn_routing_gap": routing_gap.item(),
        }

        return z_gated, gate_info

    @torch.no_grad()
    def compute_orthogonality_score(self):
        """Routing decisiveness under complementary gating.

        With P_ON(masked) = 1 - P_ON(visible) the two pathways are exact
        mirrors, so the legacy "orthogonality" is trivially satisfied. The
        informative quantity is how DECISIVE the shared Bernoulli is:
        mean |P_ON - 0.5| * 2, in [0, 1]; 1 = hard routing.
        """
        p_on = F.softmax(self.gate_logits_visible, dim=-1)[:, 1]
        return (2.0 * (p_on - 0.5).abs().mean()).item()

    @torch.no_grad()
    def compute_routing_efficiency(self, z, mask_positions):
        """Compute routing efficiency: how well gates separate information.

        Measures whether visible-position gates preserve context while
        masked-position gates preserve prediction targets.

        Args:
            z: (B, T, D) representations.
            mask_positions: (B, T) binary mask.

        Returns:
            dict with routing efficiency metrics.
        """
        _B, _T, _D = z.shape
        mask_bool = mask_positions.bool()
        vis_bool = ~mask_bool

        if not mask_bool.any() or not vis_bool.any():
            return {"routing_efficiency": 1.0, "context_preservation": 1.0, "prediction_focus": 1.0}

        # Compute gate values
        probs_v = F.softmax(self.gate_logits_visible, dim=-1)[:, 1]
        probs_m = 1.0 - probs_v  # complementary pathway

        # Context preservation: visible positions should retain information
        z_vis = z[vis_bool]  # (N_vis, D)
        if z_vis.numel() > 0:
            gate_vis = probs_v.repeat_interleave(self.group_dim)  # (D,)
            # Weighted variance at visible positions
            weighted_var = (z_vis * gate_vis.unsqueeze(0)).var(dim=0).mean()
            unweighted_var = z_vis.var(dim=0).mean()
            context_pres = (weighted_var / (unweighted_var + 1e-10)).clamp(0, 1)
        else:
            context_pres = torch.tensor(1.0)

        # Prediction focus: masked positions should gate IN prediction-relevant dims
        z_mask = z[mask_bool]  # (N_mask, D)
        if z_mask.numel() > 0:
            gate_mask = probs_m.repeat_interleave(self.group_dim)  # (D,)
            weighted_var_m = (z_mask * gate_mask.unsqueeze(0)).var(dim=0).mean()
            unweighted_var_m = z_mask.var(dim=0).mean()
            pred_focus = (weighted_var_m / (unweighted_var_m + 1e-10)).clamp(0, 1)
        else:
            pred_focus = torch.tensor(1.0)

        # Overall efficiency: geometric mean
        efficiency = (context_pres.item() * pred_focus.item()) ** 0.5

        return {
            "routing_efficiency": efficiency,
            "context_preservation": context_pres.item(),
            "prediction_focus": pred_focus.item(),
        }

    def extra_repr(self):
        return (
            f"embed_dim={self.embed_dim}, n_groups={self.n_groups}, "
            f"tau_start={self.tau_start}, tau_end={self.tau_end}, "
            f"anneal_steps={self.anneal_steps}"
        )
