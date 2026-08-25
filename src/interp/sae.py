# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Sparse Autoencoder for mechanistic interpretability of JEPA representations
# Architecture from Bricken et al. (2023) "Towards Monosemanticity"
# TopK activation from Gao et al. (2024) "Scaling and Evaluating Sparse Autoencoders"
# Dead feature resampling from Anthropic's dictionary learning

import torch
import torch.nn.functional as F
from torch import nn
from src.utils.torchio import safe_torch_load


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with TopK activation and dead feature resampling.

    Decomposes representations into sparse, interpretable features.
    From Bricken et al. (2023): SAE finds monosemantic directions
    in the superposition of polysemantic neurons.

    TopK sparsity (Gao et al., 2024): keep only top-k activations,
    setting the rest to zero. Simpler and more effective than L1 penalty.

    Dead feature resampling: periodically reset features that haven't
    fired recently, reinitializing them from encoder weights of active
    features. Prevents feature death during training.
    """

    def __init__(
        self,
        input_dim=768,
        latent_dim=4096,
        k=64,
        dead_feature_threshold=1e-6,
        resample_interval=1000,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k = k
        self.dead_feature_threshold = dead_feature_threshold
        self.resample_interval = resample_interval

        # Encoder: input → latent
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        # Decoder: latent → input (tied bias from encoder)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)

        # Track feature activation counts for dead feature resampling
        self.register_buffer("feature_act_count", torch.zeros(latent_dim))
        self.register_buffer("total_samples", torch.tensor(0, dtype=torch.long))
        self._steps_since_resample = 0

        # Initialize: decoder columns on unit sphere (Anthropic pattern)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.encoder.weight)
            nn.init.zeros_(self.encoder.bias)
            # Normalize decoder rows to unit norm
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=1)
            nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        """Encode input to sparse latent representation (TopK)."""
        pre_acts = self.encoder(x)
        # TopK: keep only top-k activations
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        # Reconstruct sparse activation vector
        latent = torch.zeros_like(pre_acts)
        latent.scatter_(-1, topk_idx, F.relu(topk_vals))
        return latent, topk_idx, topk_vals

    def decode(self, latent):
        """Decode sparse latent representation back to input space."""
        return self.decoder(latent)

    def forward(self, x):
        """Full forward pass: encode → decode with reconstruction loss.

        Returns:
            recons: reconstructed input
            latent: sparse latent representation
            loss: total loss (MSE + sparsity penalty)
            info: dict with auxiliary information
        """
        latent, topk_idx, topk_vals = self.encode(x)
        recons = self.decode(latent)

        # Reconstruction loss
        recons_loss = F.mse_loss(recons, x)

        # Sparsity: L1 on latent (auxiliary, TopK already enforces sparsity)
        sparsity = latent.abs().mean()

        # Combined loss
        loss = recons_loss + 0.01 * sparsity

        # Track feature activations for dead feature resampling
        if self.training:
            self._track_activations(latent)

        info = {
            "recons_loss": recons_loss.item(),
            "sparsity": sparsity.item(),
            "frac_active": (latent > 0).float().mean().item(),
            "topk_idx": topk_idx,
            "topk_vals": topk_vals,
        }

        return recons, latent, loss, info

    def _track_activations(self, latent):
        """Track which features fired for dead feature resampling."""
        fired = (latent.abs() > self.dead_feature_threshold).any(dim=0).float()
        self.feature_act_count += fired
        self.total_samples += latent.size(0)  # true sample count, not step count
        self._steps_since_resample += 1

    @torch.no_grad()
    def resample_dead_features(self):
        """Resample dead features that haven't fired recently.

        From Anthropic's dictionary learning: dead features get
        reinitialized from encoder weights of the most active features.
        """
        if self._steps_since_resample < self.resample_interval:
            return

        self._steps_since_resample = 0

        if self.total_samples == 0:
            return

        # Find dead features: fired in < 1% of samples
        act_rate = self.feature_act_count / self.total_samples.float()
        dead_mask = act_rate < 0.01
        n_dead = dead_mask.sum().item()

        if n_dead == 0:
            # Reset counters
            self.feature_act_count.zero_()
            self.total_samples.zero_()
            return

        # Find most active features for resampling source
        alive_mask = ~dead_mask
        if alive_mask.sum() == 0:
            # All features dead — reinitialize from random
            nn.init.xavier_uniform_(self.encoder.weight[:, dead_mask])
            self.decoder.weight.data[dead_mask] = F.normalize(
                self.encoder.weight[:, dead_mask].T, dim=1
            )
        else:
            # Sample from alive features
            alive_idx = alive_mask.nonzero(as_tuple=True)[0]
            dead_idx = dead_mask.nonzero(as_tuple=True)[0]

            for d_idx in dead_idx:
                # Pick random alive feature
                src = alive_idx[torch.randint(len(alive_idx), (1,)).item()]
                # Copy encoder row with small perturbation
                # encoder.weight shape: (latent_dim, input_dim), rows = features
                self.encoder.weight.data[d_idx] = self.encoder.weight.data[
                    src.item()
                ] + 0.02 * torch.randn_like(self.encoder.weight.data[src.item()])
                self.encoder.bias.data[d_idx] = 0.0
                # Reset decoder column (decoder shape: latent_dim, input_dim)
                # Column d_idx of decoder corresponds to feature d_idx
                col = self.decoder.weight.data[:, d_idx]
                self.decoder.weight.data[:, d_idx] = F.normalize(col, dim=0)

        # Reset counters
        self.feature_act_count.zero_()
        self.total_samples.zero_()


class SAETrainer:
    """Training loop for Sparse Autoencoder.

    Handles: training, dead feature resampling, checkpointing,
    and metric logging (MSE, L0, explained variance).
    """

    def __init__(self, sae, lr=1e-3, weight_decay=1e-5, device="cpu"):
        self.sae = sae.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(sae.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100000)
        self.step_count = 0

    def train_step(self, x):
        """Single training step.

        Args:
            x: (B, D) input representations
        Returns:
            info dict with losses
        """
        self.sae.train()
        x = x.to(self.device)
        recons, latent, loss, info = self.sae(x)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (Anthropic pattern)
        torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Resample dead features periodically
        self.sae.resample_dead_features()

        # Post-step: normalize decoder rows to unit norm
        with torch.no_grad():
            self.sae.decoder.weight.data = F.normalize(self.sae.decoder.weight.data, dim=1)

        self.step_count += 1

        info["step"] = self.step_count
        info["lr"] = self.scheduler.get_last_lr()[0]
        info["l0"] = (latent > 0).sum(dim=-1).float().mean().item()
        info["explained_variance"] = self._explained_variance(x, recons)

        return info

    @staticmethod
    def _explained_variance(x, recons):
        """Fraction of variance explained by reconstruction."""
        total_var = x.var(dim=0).sum()
        if total_var == 0:
            return 0.0
        residual_var = (x - recons).var(dim=0).sum()
        return (1 - residual_var / total_var).item()

    @torch.no_grad()
    def evaluate(self, dataloader, max_batches=100):
        """Evaluate SAE on held-out data.

        Returns:
            dict with average metrics
        """
        self.sae.eval()
        total_recons = 0
        total_l0 = 0
        total_ev = 0
        n_batches = 0

        for batch in dataloader:
            if n_batches >= max_batches:
                break
            x = (
                batch.to(self.device)
                if isinstance(batch, torch.Tensor)
                else batch[0].to(self.device)
            )
            recons, latent, _, info = self.sae(x)
            total_recons += info["recons_loss"]
            total_l0 += (latent > 0).sum(dim=-1).float().mean().item()
            total_ev += self._explained_variance(x, recons)
            n_batches += 1

        if n_batches == 0:
            return {"recons_loss": float("inf"), "l0": 0, "explained_variance": 0}

        return {
            "recons_loss": total_recons / n_batches,
            "l0": total_l0 / n_batches,
            "explained_variance": total_ev / n_batches,
        }

    def save(self, path):
        """Save SAE checkpoint."""
        torch.save(
            {
                "sae_state": self.sae.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "step_count": self.step_count,
                # resample cadence must survive resume, else the interval resets
                "steps_since_resample": self.sae._steps_since_resample,
            },
            path,
        )

    def load(self, path):
        """Load SAE checkpoint."""
        ckpt = safe_torch_load(path, map_location=self.device)
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.step_count = ckpt["step_count"]
        self.sae._steps_since_resample = int(ckpt.get("steps_since_resample", 0))
