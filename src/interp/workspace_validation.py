# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
#
# Workspace Validation via Sparse Autoencoder (SAE)
#
# Validates the JAWP workspace claim: that learned Q spans the same
# subspace as the "true workspace" recovered by an SAE decomposition.
#
# Anthropic (Gurnee et al., 2026, arXiv:2607.15495): J-space is the
# subspace where SAE features with high downstream relevance concentrate.
#
# Procedure:
#   1. Train a TopK SAE on encoder representations (TopKSAE + train_topk_sae)
#   2. Identify "workspace features" — the SAE features whose held-out
#      probe accuracy beats chance
#   3. Compute subspace similarity between SAE workspace and JAWP Q
#   4. Bootstrap CI for the similarity at 3+ model sizes
#
# The claim is only reported for a *trained* SAE. An untrained SAE has a
# random decoder, so every angle computed from it is an angle between
# random vectors: `validate_workspace_claim` raises UntrainedSAEError
# rather than returning such a number (see that function).
#
# The gate is on the *lower end of the bootstrap CI* of the similarity,
# not on a point estimate, and the CI must also beat the shuffled-basis
# control (`decide_workspace_claim`).

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

from src.interp import rng

logger = logging.getLogger(__name__)

_EPS = 1e-10


class UntrainedSAEError(RuntimeError):
    """A verdict was requested from an SAE that was never trained.

    Raised instead of returning a similarity measured against a random
    decoder. A caller that genuinely wants the exploratory pass must say
    so with ``allow_untrained_sae=True``; that path returns a refusal
    record which contains no similarity, no angles and no verdict.
    """


# ═══════════════════════════════════════════════════════════════════
#  TopK Sparse Autoencoder
# ═══════════════════════════════════════════════════════════════════


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder for representation decomposition.

    From Bricken et al. (2024) "Toward Monosemanticity: Training a
    TopK SAE". Only the top-k features fire per token.

    The module counts the optimizer steps it has actually taken in the
    persistent buffer ``n_train_steps``. Callers that train a SAE with
    their own loop must say so with :meth:`mark_trained`; nothing else
    can make the module claim to be trained, which is what lets
    :func:`validate_workspace_claim` refuse to measure with a random
    decoder.

    Args:
        embed_dim: input dimension (D).
        n_features: SAE latent dimension (typically 16x-64x D).
        k: number of active features per token.
        seed: seeds a private ``torch.Generator`` for the encoder and decoder
            matrices, so two SAEs built with the same seed are bit-identical and
            building one leaves the process-global RNG untouched.

    """

    def __init__(
        self, embed_dim: int, n_features: int = 8192, k: int = 32, seed: int | None = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_features = n_features
        self.k = k
        self.seed = seed
        if n_features <= 0 or embed_dim <= 0:
            raise ValueError(
                f"n_features and embed_dim must be positive, got {n_features}, {embed_dim}"
            )
        if not 1 <= k <= n_features:
            raise ValueError(f"k must be in [1, n_features={n_features}], got {k}")

        # Private generator: these two matrices are the module's only
        # randomness, and they previously came from the process-global stream,
        # so two SAEs built in one process differed by whatever else had drawn
        # in between. Same seed -> bit-identical decoder.
        gen = rng.generator_for(seed, "workspace_validation.topk_sae")

        # Encoder: x -> features
        self.W_enc = nn.Parameter(
            torch.randn(embed_dim, n_features, generator=gen) * (1.0 / embed_dim)
        )
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        # Decoder: features -> x_hat
        self.W_dec = nn.Parameter(
            torch.randn(n_features, embed_dim, generator=gen) * (1.0 / n_features)
        )
        self.b_dec = nn.Parameter(torch.zeros(embed_dim))

        # How many optimizer steps this SAE has actually taken. Persistent, so
        # a checkpoint cannot claim to be trained unless it was.
        self.register_buffer("n_train_steps", torch.zeros((), dtype=torch.long))

    @property
    def is_trained(self) -> bool:
        """True iff at least one optimizer step was taken (or declared)."""
        return int(self.n_train_steps) > 0

    def mark_trained(self, steps: int = 1) -> None:
        """Declare this SAE trained by an external training loop.

        Call after (not during) optimisation; a forward pass does not
        count, because a forward pass on random weights is exactly the
        case this module exists to refuse.
        """
        steps = int(steps)
        if steps < 1:
            raise ValueError(f"mark_trained needs a positive step count, got {steps}")
        self.n_train_steps.fill_(steps)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode with TopK activation.

        Returns:
            features: (N, n_features) sparse feature activations.
            indices: (N, k) indices of top-k features.

        """
        pre_acts = x @ self.W_enc + self.b_enc  # (N, n_features)
        topk_vals, topk_indices = torch.topk(pre_acts, self.k, dim=-1)

        features = torch.zeros_like(pre_acts)
        features.scatter_(-1, topk_indices, F.relu(topk_vals))
        return features, topk_indices

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to input space."""
        return features @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward pass with reconstruction loss.

        Returns:
            dict with loss, x_hat, features, indices, sparsity.

        """
        features, indices = self.encode(x)
        x_hat = self.decode(features)

        # Reconstruction loss
        loss_recon = F.mse_loss(x_hat, x)

        # Sparsity: average L0 per token
        sparsity = (features > 0).float().sum(dim=-1).mean()

        # Dead features: features never activated in this batch
        all_indices = indices.reshape(-1)
        active_features = torch.unique(all_indices)
        dead_fraction = 1.0 - active_features.numel() / self.n_features

        return {
            "loss": loss_recon,
            "x_hat": x_hat,
            "features": features,
            "indices": indices,
            "sparsity": sparsity,
            "dead_fraction": dead_fraction,
        }

    def extra_repr(self):
        return f"embed_dim={self.embed_dim}, n_features={self.n_features}, " f"k={self.k}"


def train_topk_sae(
    representations: torch.Tensor,
    n_features: int = 8192,
    k: int = 32,
    n_steps: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 20260824,
) -> TopKSAE:
    """Train a :class:`TopKSAE` on representations and return it trained.

    This is the only place in the module that is allowed to make an SAE
    trainable, so an SAE handed to :func:`validate_workspace_claim` from
    elsewhere either really was optimised or was declared so explicitly.

    The objective is plain reconstruction: TopK already enforces sparsity,
    so there is no auxiliary penalty whose weight could be tuned into a
    silent no-op.

    Args:
        representations: (N, D) encoder representations.
        n_features: SAE latent dimension.
        k: TopK sparsity.
        n_steps: optimizer steps. Must be >= 1 — an SAE with zero steps
            is the untrained object this module refuses to measure with.
        lr: Adam learning rate.
        batch_size: samples per step (clipped to N).
        seed: seed for the private batch generator. A private generator is
            used so training an SAE cannot perturb the caller's global RNG.

    Returns:
        A TopKSAE in eval mode with ``is_trained`` True.

    """
    if representations.dim() != 2:
        raise ValueError(f"representations must be (N, D), got {tuple(representations.shape)}")
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("train_topk_sae needs n_steps >= 1; n_steps=0 is an untrained SAE")

    N, D = representations.shape
    sae = TopKSAE(embed_dim=D, n_features=int(n_features), k=min(int(k), int(n_features)))
    opt = torch.optim.Adam(sae.parameters(), lr=float(lr))
    bsz = max(1, min(int(batch_size), N))

    gen = torch.Generator().manual_seed(int(seed))  # private: leaves global RNG alone
    was_training = sae.training
    sae.train()
    try:
        for _ in range(n_steps):
            idx_cpu = torch.randint(0, N, (bsz,), generator=gen)
            batch = representations[idx_cpu.to(representations.device)]
            loss = sae(batch)["loss"]
            if not torch.isfinite(loss):
                raise ValueError("SAE reconstruction loss went non-finite during training")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sae.n_train_steps += 1
    finally:
        opt.zero_grad(set_to_none=True)
        sae.train(was_training)

    sae.eval()
    if not sae.is_trained:
        # Unreachable while n_steps >= 1; kept as an explicit guard so the
        # "trained" claim can never be silently false.
        raise RuntimeError("internal: SAE reports itself untrained after training")
    return sae


# ═══════════════════════════════════════════════════════════════════
#  Workspace feature identification
# ═══════════════════════════════════════════════════════════════════


def _encode_features(sae: TopKSAE, representations: torch.Tensor) -> torch.Tensor:
    """Encode representations with the SAE, restoring its mode afterwards.

    The caller's ``sae.train()`` / ``sae.eval()`` choice is restored even
    if encoding raises, so an analysis call cannot silently leave an SAE
    that is about to be trained in eval mode (or one that is being
    evaluated in train mode).

    """
    if representations.dim() != 2:
        raise ValueError(f"representations must be (N, D), got {tuple(representations.shape)}")
    if not bool(torch.isfinite(representations).all()):
        raise ValueError(
            "representations contain non-finite values; refusing to fit probes on them"
        )
    w_dec = getattr(sae, "W_dec", None)
    if w_dec is None:
        raise TypeError(f"expected a TopKSAE with W_dec, got {type(sae).__name__}")
    if int(w_dec.shape[1]) != int(representations.shape[1]):
        raise ValueError(
            f"SAE embed_dim={int(w_dec.shape[1])} does not match representations "
            f"D={int(representations.shape[1])}"
        )

    was_training = sae.training
    sae.eval()
    try:
        with torch.no_grad():
            features, _ = sae.encode(representations)
    finally:
        sae.train(was_training)
    return features.detach()


def _resolve_split(
    n: int,
    holdout_fraction: float,
    seed: int,
    train_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (train_mask, test_mask) as bool tensors of length ``n``."""
    if train_mask is not None:
        mask = torch.as_tensor(train_mask).to(torch.bool).reshape(-1)
        if int(mask.numel()) != n:
            raise ValueError(f"train_mask has {int(mask.numel())} entries, expected {n}")
        n_train, n_test = int(mask.sum()), int((~mask).sum())
        if n_train < 2 or n_test < 2:
            raise ValueError(
                f"held-out probe evaluation needs >= 2 train and >= 2 held-out samples, "
                f"got {n_train} / {n_test}"
            )
        return mask, ~mask

    frac = float(holdout_fraction)
    if not 0.0 < frac < 1.0:
        raise ValueError(f"holdout_fraction must lie strictly in (0, 1), got {frac!r}")
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=gen)
    n_train = max(2, min(n - 2, round(n * (1.0 - frac))))
    mask = torch.zeros(n, dtype=torch.bool)
    mask[perm[:n_train]] = True
    return mask, ~mask


def _heldout_probe_scores(
    features: torch.Tensor,
    probe_labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Per-feature held-out balanced accuracy for a one-threshold probe.

    The probe is a single threshold per feature (the train mean) with the
    orientation given by the train-split correlation. It is fitted on the
    train split only and scored on the held-out split only, so the number
    reported is out-of-sample. Constant features score exactly chance.

    Returns:
        (scores (F,), number of tasks that were usable).

    """
    n_feat = int(features.shape[1])
    scores = torch.zeros(n_feat, dtype=torch.float64, device=features.device)
    n_tasks = 0

    x = features.to(torch.float64)
    x_tr = x[train_mask]
    x_te = x[test_mask]
    x_mean_tr = x_tr.mean(dim=0)
    sx = (x_tr - x_mean_tr).pow(2).mean(dim=0).sqrt()  # population std, train only
    live = sx >= _EPS

    y_all = probe_labels.detach().to(device=features.device, dtype=torch.float64)
    if y_all.dim() == 1:
        y_all = y_all.reshape(-1, 1)
    for task in range(y_all.shape[1]):
        y = y_all[:, task]
        uniq = torch.unique(y)
        if int(uniq.numel()) != 2:
            continue  # not a binary task; recorded by the caller via n_tasks
        pos = uniq[1]
        y_tr = (y[train_mask] == pos).to(torch.float64)
        y_te = y[test_mask] == pos
        n_pos = int(y_te.sum())
        n_neg = int(y_te.numel() - n_pos)
        if n_pos == 0 or n_neg == 0:
            continue

        y_tr_c = y_tr - y_tr.mean()
        sy = y_tr_c.pow(2).mean().sqrt()
        if sy < _EPS:
            continue
        corr = (x_tr - x_mean_tr).T @ y_tr_c / (float(x_tr.shape[0]) * sx * sy)
        sign = torch.where(corr >= 0, 1.0, -1.0)

        pred = (sign.unsqueeze(0) * (x_te - x_mean_tr.unsqueeze(0))) > 0
        tpr = (pred & y_te.unsqueeze(1)).to(torch.float64).sum(dim=0) / n_pos
        tnr = ((~pred) & (~y_te).unsqueeze(1)).to(torch.float64).sum(dim=0) / n_neg
        acc = 0.5 * (tpr + tnr)
        scores += torch.where(live, acc, torch.full_like(acc, 0.5))  # chance for dead features
        n_tasks += 1

    if n_tasks:
        scores = scores / float(n_tasks)
    return scores, n_tasks


def _identify_from_features(
    features: torch.Tensor,
    probe_labels: torch.Tensor,
    n_probes: int,
    top_fraction: float,
    holdout_fraction: float,
    seed: int,
    train_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Workspace-feature selection from precomputed SAE features."""
    n, n_feat = int(features.shape[0]), int(features.shape[1])
    labels = probe_labels.reshape(-1, 1) if probe_labels.dim() == 1 else probe_labels
    if int(labels.shape[0]) != n:
        raise ValueError(
            f"probe_labels has {int(labels.shape[0])} rows but representations have {n}"
        )
    n_probes = max(1, min(int(n_probes), int(labels.shape[1])))
    labels = labels[:, :n_probes]

    frac = float(top_fraction)
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"top_fraction must lie in (0, 1], got {frac!r}")

    tr_mask, te_mask = _resolve_split(n, holdout_fraction, seed, train_mask)
    scores, n_tasks = _heldout_probe_scores(features, labels, tr_mask, te_mask)

    n_workspace = max(1, min(n_feat, round(n_feat * frac)))
    order = torch.topk(scores, n_workspace).indices
    top_score = float(scores[order[0]])
    chance = 0.5
    info = {
        "n_workspace_features": n_workspace,
        "total_features": n_feat,
        "top_score": top_score,
        "mean_score": float(scores.mean()),
        # The number the docstring promises: held-out probe accuracy of the
        # best features, next to the chance level it has to beat.
        "probe_accuracy_heldout": top_score,
        "probe_accuracy_chance": chance,
        "features_above_chance": float((scores > chance + _EPS).sum()),
        "n_tasks_used": float(n_tasks),
        "n_probes_requested": float(n_probes),
        "n_train": float(int(tr_mask.sum())),
        "n_holdout": float(int(te_mask.sum())),
        "workspace_features_identified": float(top_score > chance + _EPS and n_tasks > 0),
    }
    return order, info


def identify_workspace_features(
    sae: TopKSAE,
    representations: torch.Tensor,
    probe_labels: torch.Tensor,
    n_probes: int = 5,
    top_fraction: float = 0.1,
    holdout_fraction: float = 0.5,
    seed: int = 20260824,
    train_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Identify SAE features that are most predictive of downstream tasks.

    For each SAE feature a one-threshold probe is fitted on the train split
    (threshold = train mean, orientation = sign of the train-split
    point-biserial correlation) and scored by **balanced accuracy on the
    held-out split**. Features whose held-out accuracy beats chance are
    workspace features; the top ``top_fraction`` of them are returned.
    The accuracy reported in ``info`` is out-of-sample, so it is a
    property of the probe rather than of the data it was fitted on.

    The SAE's train/eval mode is restored on every path, including the
    error path.

    Args:
        sae: SAE model.
        representations: (N, D) encoder representations.
        probe_labels: (N,) or (N, n_probes) binary downstream labels.
        n_probes: number of downstream probe tasks.
        top_fraction: fraction of features to classify as workspace.
        holdout_fraction: fraction of samples reserved for scoring. Ignored
            when ``train_mask`` is given.
        seed: seed for the split and for nothing else (the probe is
            deterministic given the split).
        train_mask: optional (N,) bool mask. Pass a caller-owned split to
            keep the held-out claim auditable against a known partition.

    Returns:
        workspace_indices: 1-D tensor of feature indices.
        info: dict with diagnostics, including
            ``probe_accuracy_heldout``, ``probe_accuracy_chance``,
            ``workspace_features_identified`` (1.0/0.0), ``n_train``,
            ``n_holdout``, ``n_tasks_used``.

    """
    features = _encode_features(sae, representations)
    return _identify_from_features(
        features,
        probe_labels,
        n_probes=n_probes,
        top_fraction=top_fraction,
        holdout_fraction=holdout_fraction,
        seed=seed,
        train_mask=train_mask,
    )


# ═══════════════════════════════════════════════════════════════════
#  Subspace similarity: JAWP Q vs SAE workspace
# ═══════════════════════════════════════════════════════════════════


def _require_orthonormal(Q: torch.Tensor, tol: float = 1e-3) -> None:
    """Reject a Q that the angle computation is not defined for."""
    if Q.dim() != 2:
        raise ValueError(f"Q must be (D, k), got {tuple(Q.shape)}")
    d, k = int(Q.shape[0]), int(Q.shape[1])
    if d == 0 or k == 0 or k > d:
        raise ValueError(f"Q must have shape (D, k) with 0 < k <= D, got {(d, k)}")
    if not bool(torch.isfinite(Q).all()):
        raise ValueError("Q contains non-finite values; refusing to compute angles from it")
    q64 = Q.detach().to(torch.float64)
    gram = q64.T @ q64
    err = float((gram - torch.eye(k, dtype=torch.float64)).abs().max())
    if err > tol:
        raise ValueError(
            f"Q is not orthonormal (max |QᵀQ - I| = {err:.3g} > {tol:.3g}). Principal "
            "angles and the similarity below are only defined for an orthonormal basis; "
            "orthonormalise Q (e.g. torch.linalg.qr) instead of rescaling the result."
        )


def compute_workspace_similarity(
    Q: torch.Tensor,
    sae: TopKSAE,
    workspace_feature_indices: torch.Tensor,
) -> dict[str, float]:
    """Compute subspace similarity between JAWP Q and SAE workspace.

    The SAE workspace subspace is spanned by the decoder vectors ``P`` of the
    identified workspace features. ``P`` is first reduced to an orthonormal
    basis ``Y`` of its column space (QR, truncated at the numerical rank), so
    that a long decoder does not buy a longer subspace. The principal angles
    are then the ``k`` angles ``arccos(sqrt(λᵢ))`` for the eigenvalues ``λᵢ``
    of the **k x k** matrix ``G = Qᵀ Y Yᵀ Q``, and the similarity is
    ``mean(λ)`` over those ``k`` values.

    Both halves of that matter. Forming the Gram from un-orthonormalised rows
    makes ``G`` scale with ``‖P‖²``, so its eigenvalues can be far above 1 and
    clamping them would report 1.0 for any two unrelated subspaces. And taking
    the singular values of the ``k x r`` cross-matrix ``QᵀY`` instead is only
    the same object when ``k == r``: when the SAE workspace is wider, the
    cross-matrix has more singular values than angles, its mean divides the
    same total by a larger number, and when it is narrower the mean is over
    only ``min(k, r)`` of them. Both are reported through ``G``, where
    ``subspace_similarity`` lies in [0, 1] by construction and equals 1
    exactly when the JAWP span is contained in the SAE span.

    This function measures geometry only; it says nothing about whether the
    SAE is trained. Callers that turn it into a verdict must check that
    (see :func:`validate_workspace_claim`).

    Args:
        Q: (D, k) JAWP workspace basis. Must be orthonormal.
        sae: SAE model.
        workspace_feature_indices: indices of workspace features, non-empty.

    Returns:
        dict with ``subspace_similarity`` in [0, 1], ``principal_angles``
        in degrees (ascending, padded with 90° when the SAE workspace is
        narrower than k), ``mean_angle``, ``workspace_dim_sae``,
        ``workspace_dim_jawp``, ``n_angles_defined``.

    """
    _require_orthonormal(Q)
    w_dec = getattr(sae, "W_dec", None)
    if w_dec is None:
        raise TypeError(f"expected a TopKSAE with W_dec, got {type(sae).__name__}")

    rows = torch.as_tensor(workspace_feature_indices).reshape(-1).to(torch.long)
    if int(rows.numel()) == 0:
        raise ValueError(
            "empty workspace feature set: similarity against an empty subspace is not a "
            "measurement (it is 0 by definition), so it will not be returned"
        )
    n_sae_features = int(w_dec.shape[0])
    if int(rows.min()) < 0 or int(rows.max()) >= n_sae_features:
        raise IndexError(
            f"workspace feature indices must lie in [0, {n_sae_features}), got "
            f"[{int(rows.min())}, {int(rows.max())}]"
        )

    with torch.no_grad():
        w_dec64 = w_dec.detach().to(torch.float64)
        if not bool(torch.isfinite(w_dec64).all()):
            raise ValueError(
                "SAE decoder contains non-finite values; refusing to compute principal "
                "angles from it"
            )
        q64 = Q.detach().to(torch.float64)
        P = w_dec64[rows.to(w_dec64.device)].T  # (D, m), un-orthonormal
        # An orthonormal basis of span(P), truncated at the numerical rank.
        #
        # QR is not usable here: when the leading block of P is rank-deficient
        # (duplicated decoder rows, which is the normal case for a workspace of
        # m >> k features) LAPACK sets tau=0 and skips the reflector, so the
        # later columns of Q are an arbitrary orthonormal completion rather than
        # vectors in span(P). Truncating them to the rank yields the wrong
        # subspace -- measured on a 3-D span at D=16 the similarity came out at
        # 0.67 instead of 1.0. The SVD's left singular vectors span the column
        # space by construction.
        U, S, _V = torch.linalg.svd(P, full_matrices=False)
        tol = float(S.max()) * max(P.shape) * float(torch.finfo(torch.float64).eps)
        rank = int((S > tol).sum())
        if rank == 0:
            raise ValueError(
                "the selected SAE decoder rows span a zero-dimensional space; there is no "
                "subspace to compare Q against"
            )
        Y = U[:, :rank]  # (D, rank), orthonormal columns spanning span(P)
        M = q64.T @ Y  # (k, rank)
        gram = M @ M.T  # (k, k) == Qᵀ Y Yᵀ Q == Qᵀ PP⁺Q
        eig = torch.linalg.eigvalsh(gram)  # ascending == squared angle cosines
        # The Gram is PSD with ||QᵀYYᵀQ|| <= 1 by construction; clamp only for
        # round-off, never to hide a real violation.
        if float(eig.max()) > 1.0 + 1e-6:
            raise ValueError(
                f"internal: squared angle cosines reached {float(eig.max())}, which is "
                "impossible; refusing to clamp a broken decomposition into a verdict"
            )
        eig = eig.clamp(0.0, 1.0)

    k = int(Q.shape[1])
    r = rank
    cosines = eig.flip(0).sqrt()  # ascending angle order
    angles_deg = torch.arccos(cosines) * (180.0 / math.pi)
    similarity = float(eig.mean())

    return {
        "subspace_similarity": similarity,
        "principal_angles": angles_deg.tolist(),
        "mean_angle": float(angles_deg.mean()),
        "workspace_dim_sae": r,
        "workspace_dim_jawp": k,
        # When r < k only min(k, r) principal angles exist; the remaining
        # entries of the list above are the padding to 90°.
        "n_angles_defined": min(k, r),
    }


# ═══════════════════════════════════════════════════════════════════
#  Bootstrap confidence interval
# ═══════════════════════════════════════════════════════════════════


def bootstrap_ci(
    values: torch.Tensor,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Args:
        values: 1-D tensor of sample values (any device).
        n_bootstrap: number of bootstrap resamples.
        confidence: confidence level (e.g., 0.95 for 95% CI).
        seed: seed for a private generator, so the CI cannot be perturbed
            by (or perturb) the caller's global RNG.

    Returns:
        (mean, ci_lower, ci_upper).

    Raises:
        ValueError: for fewer than two samples, non-finite samples, or a
            degenerate ``n_bootstrap`` / ``confidence``. A zero-width
            interval on a single sample looks like a tight measurement and
            is not one.

    """
    v = torch.as_tensor(values).detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    n = int(v.numel())
    if n < 2:
        raise ValueError(
            f"a bootstrap CI needs at least 2 samples, got {n}; a zero-width interval on "
            "one sample would look like a measurement"
        )
    if not bool(torch.isfinite(v).all()):
        raise ValueError("cannot bootstrap a CI from non-finite values")
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError(f"confidence must lie strictly in (0, 1), got {confidence!r}")

    gen = torch.Generator().manual_seed(int(seed))
    boot_means = []
    for _ in range(n_bootstrap):
        idx = torch.randint(0, n, (n,), generator=gen)
        boot_means.append(float(v[idx].mean()))
    boot_means.sort()

    alpha = 1.0 - float(confidence)
    lo_idx = min(max(int(n_bootstrap * alpha / 2.0), 0), n_bootstrap - 1)
    hi_idx = min(max(math.ceil(n_bootstrap * (1.0 - alpha / 2.0)) - 1, 0), n_bootstrap - 1)

    return float(v.mean()), boot_means[lo_idx], boot_means[hi_idx]


def bootstrap_similarity_ci(
    Q: torch.Tensor,
    sae: TopKSAE,
    representations: torch.Tensor,
    probe_labels: torch.Tensor,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    n_probes: int = 5,
    top_fraction: float = 0.1,
    holdout_fraction: float = 0.5,
    seed: int = 20260824,
) -> dict[str, float]:
    """Bootstrap CI for ``subspace_similarity`` — the module's headline number.

    Resamples the N representations (with their probe labels), re-runs the
    workspace-feature identification on the resample and recomputes the
    similarity, so the interval accounts for both sources of noise: which
    samples the SAE was fitted on and which features the probes picked.

    The SAE encodes once, outside the loop: resampling rows of the feature
    matrix is the same operation as resampling rows of the
    representations, and keeps this affordable.

    Args:
        Q: (D, k) orthonormal JAWP basis.
        sae: SAE model.
        representations: (N, D) encoder representations.
        probe_labels: (N,) or (N, n_probes) binary downstream labels.
        n_bootstrap: resamples.
        confidence: confidence level.
        n_probes, top_fraction, holdout_fraction, seed: forwarded to the
            identification step, unchanged.

    Returns:
        dict with ``similarity_mean``, ``similarity_ci_lower``,
        ``similarity_ci_upper``, ``similarity_ci_width``, ``n_bootstrap``.

    """
    features = _encode_features(sae, representations)
    n = int(features.shape[0])
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 2:
        raise ValueError(f"n_bootstrap must be >= 2 for an interval, got {n_bootstrap}")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError(f"confidence must lie strictly in (0, 1), got {confidence!r}")

    gen = torch.Generator().manual_seed(int(seed) + 1)  # private generator
    sims: list[float] = []
    for _ in range(n_bootstrap):
        idx = torch.randint(0, n, (n,), generator=gen)
        ws_idx, _info = _identify_from_features(
            features[idx],
            probe_labels.to(features.device)[idx.to(probe_labels.device)],
            n_probes=n_probes,
            top_fraction=top_fraction,
            holdout_fraction=holdout_fraction,
            seed=seed,
            train_mask=None,
        )
        sims.append(compute_workspace_similarity(Q, sae, ws_idx)["subspace_similarity"])

    sims_sorted = sorted(sims)
    alpha = 1.0 - float(confidence)
    lo = sims_sorted[min(max(int(n_bootstrap * alpha / 2.0), 0), n_bootstrap - 1)]
    hi = sims_sorted[min(max(math.ceil(n_bootstrap * (1.0 - alpha / 2.0)) - 1, 0), n_bootstrap - 1)]
    mean = sum(sims) / len(sims)
    return {
        "similarity_mean": mean,
        "similarity_ci_lower": float(lo),
        "similarity_ci_upper": float(hi),
        "similarity_ci_width": float(hi - lo),
        "n_bootstrap": float(n_bootstrap),
    }


# ═══════════════════════════════════════════════════════════════════
#  The verdict
# ═══════════════════════════════════════════════════════════════════


def decide_workspace_claim(
    subspace_similarity: float,
    similarity_ci_lower: float,
    placebo_similarity_max: float,
    features_identified: bool,
    gate: float = 0.8,
) -> bool:
    """The preregistered rule, isolated so it can be read and tested.

    The claim counts as validated only if all three hold:

      1. the workspace features are real — at least one feature beats chance
         on held-out data (``features_identified``),
      2. the **lower** end of the similarity's bootstrap CI clears ``gate``,
         not merely the point estimate,
      3. that lower end beats the worst of the shuffled-basis controls.

    A point estimate above 0.8 whose CI dips below it is noise that happened
    to land high; returning ``True`` for it is the failure this function
    exists to prevent.
    """
    if not bool(features_identified):
        return False
    if not float(similarity_ci_lower) > float(gate):
        return False
    return bool(float(similarity_ci_lower) > float(placebo_similarity_max))


# ═══════════════════════════════════════════════════════════════════
#  Full validation pipeline
# ═══════════════════════════════════════════════════════════════════


def validate_workspace_claim(
    Q: torch.Tensor,
    representations: torch.Tensor,
    probe_labels: torch.Tensor,
    sae: TopKSAE | None = None,
    n_sae_features: int = 8192,
    sae_k: int = 32,
    n_probes: int = 5,
    top_fraction: float = 0.1,
    n_bootstrap: int = 200,
    n_placebo: int = 8,
    similarity_gate: float = 0.8,
    holdout_fraction: float = 0.5,
    seed: int = 20260824,
    n_sae_train_steps: int = 0,
    allow_untrained_sae: bool = False,
) -> dict[str, object]:
    """Full workspace validation pipeline.

    1. Obtain a trained SAE — train one here with ``n_sae_train_steps`` steps,
       or pass a trained one. There is no path that measures with a random
       decoder (see Raises).
    2. Identify workspace features via **held-out** downstream probes.
    3. Compute subspace similarity with JAWP Q, plus shuffled-basis controls.
    4. Bootstrap CI for the similarity, and for per-sample utilisation.

    Args:
        Q: (D, k) orthonormal JAWP workspace basis.
        representations: (N, D) encoder representations.
        probe_labels: (N,) or (N, n_probes) binary downstream labels.
        sae: trained SAE. If None and ``n_sae_train_steps > 0``, one is
            trained here on ``representations``.
        n_sae_features: SAE latent dimension, used only by the
            train-here path.
        sae_k: SAE TopK sparsity, used only by the train-here path.
        n_probes: number of probe tasks.
        top_fraction: fraction of SAE features for workspace.
        n_bootstrap: bootstrap resamples, for both the similarity CI and
            the utilisation CI.
        n_placebo: shuffled-basis controls to draw.
        similarity_gate: the fixed similarity threshold, recorded in the
            result so the number cannot be read without it.
        holdout_fraction: fraction of samples held out from the probes.
        seed: seeds the split, the controls and the bootstraps (private
            generators; the caller's global RNG is untouched).
        n_sae_train_steps: optimizer steps for the SAE trained here.
            0 means "no SAE will be trained", which is refused.
        allow_untrained_sae: explicit opt-in to the diagnostic path with an
            untrained SAE. Returns a refusal record with **no** similarity,
            **no** angles and ``workspace_claim_valid = None``.

    Returns:
        dict with all validation results, or a refusal record
        (``verdict_status == "refused_untrained_sae"``) if the untrained
        diagnostic path was explicitly requested.

    Raises:
        UntrainedSAEError: if the SAE is untrained — either because none was
            given and no training steps were requested, or because the given
            SAE has taken no optimizer steps. A random decoder measures
            angles between random vectors; returning that as a verdict with
            a caveat attached would still be reporting a number this module
            did not earn, so it is refused instead.
        ValueError: for a non-orthonormal Q, mismatched SAE embed_dim,
            degenerate probe labels, or fewer samples than a held-out
            evaluation needs.

    """
    _require_orthonormal(Q)
    if representations.dim() != 2:
        raise ValueError(f"representations must be (N, D), got {tuple(representations.shape)}")
    if not bool(torch.isfinite(representations).all()):
        raise ValueError("representations contain non-finite values")
    D, k = int(Q.shape[0]), int(Q.shape[1])
    if int(representations.shape[1]) != D:
        raise ValueError(
            f"representations have D={int(representations.shape[1])} but Q has {D} rows"
        )
    N = int(representations.shape[0])
    n_placebo = int(n_placebo)
    if n_placebo < 1:
        raise ValueError(f"n_placebo must be >= 1 (a control is required), got {n_placebo}")
    if not 0.0 < float(similarity_gate) <= 1.0:
        raise ValueError(f"similarity_gate must lie in (0, 1], got {similarity_gate!r}")
    if not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError(f"top_fraction must lie in (0, 1], got {top_fraction!r}")

    # ── Obtain a *trained* SAE, or refuse ──────────────────────────────────
    if sae is None:
        if int(n_sae_train_steps) > 0:
            sae = train_topk_sae(
                representations,
                n_features=n_sae_features,
                k=sae_k,
                n_steps=int(n_sae_train_steps),
                seed=seed,
            )
            logger.info(
                "trained a %d-feature TopK SAE for %d steps",
                int(n_sae_features),
                int(n_sae_train_steps),
            )
        else:
            raise UntrainedSAEError(
                "refusing to report a workspace verdict from an untrained SAE. Pass a "
                "trained TopKSAE as sae=..., or set n_sae_train_steps>0 to train one here. "
                "Angles between a random decoder and Q measure the RNG, not the workspace."
            )
    elif not bool(getattr(sae, "is_trained", False)):
        message = (
            f"the supplied SAE reports {int(getattr(sae, 'n_train_steps', 0))} optimizer "
            "steps and is therefore untrained; every principal angle computed from its "
            "decoder would be an angle between random vectors"
        )
        if not allow_untrained_sae:
            raise UntrainedSAEError("refusing to report a workspace verdict: " + message)
        logger.error("workspace claim REFUSED: %s", message)
        return {
            "verdict_status": "refused_untrained_sae",
            "workspace_claim_valid": None,
            "refusal_reason": message,
            "n_samples": N,
            "embed_dim": D,
            "workspace_dim_jawp": k,
        }

    # ── 2. Workspace features, selected on held-out probe accuracy ─────────
    ws_indices, ws_info = identify_workspace_features(
        sae,
        representations,
        probe_labels,
        n_probes=n_probes,
        top_fraction=top_fraction,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )

    # ── 3. Similarity, plus shuffled-basis controls ───────────────────────
    similarity_result = compute_workspace_similarity(Q, sae, ws_indices)
    placebo_sims = []
    for i in range(n_placebo):
        gen = torch.Generator(device=Q.device).manual_seed(int(seed) + 1000 + i)
        q_rand = torch.randn(D, k, generator=gen, device=Q.device, dtype=torch.float64)
        Q_rand, _ = torch.linalg.qr(q_rand)
        placebo_sims.append(
            compute_workspace_similarity(Q_rand, sae, ws_indices)["subspace_similarity"]
        )
    placebo_mean = sum(placebo_sims) / len(placebo_sims)
    placebo_max = max(placebo_sims)

    # ── 4. CIs: for the similarity (the claim) and for utilisation ─────────
    sim_ci = bootstrap_similarity_ci(
        Q,
        sae,
        representations,
        probe_labels,
        n_bootstrap=n_bootstrap,
        confidence=0.95,
        n_probes=n_probes,
        top_fraction=top_fraction,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )

    with torch.no_grad():
        ws_projection = representations @ Q @ Q.T  # (N, D)
        ws_util = (ws_projection**2).sum(dim=-1) / (representations**2).sum(dim=-1).clamp(min=1e-10)

    mean_util, ci_lo, ci_hi = bootstrap_ci(ws_util, n_bootstrap=n_bootstrap)

    claim_valid = decide_workspace_claim(
        subspace_similarity=similarity_result["subspace_similarity"],
        similarity_ci_lower=sim_ci["similarity_ci_lower"],
        placebo_similarity_max=placebo_max,
        features_identified=bool(ws_info["workspace_features_identified"] > 0.5),
        gate=float(similarity_gate),
    )

    result = {
        **similarity_result,
        **ws_info,
        **sim_ci,
        "ws_utilization_mean": mean_util,
        "ws_utilization_ci_lower": ci_lo,
        "ws_utilization_ci_upper": ci_hi,
        "n_samples": N,
        "embed_dim": D,
        "workspace_dim_jawp": k,
        "sae_train_steps": int(getattr(sae, "n_train_steps", 0)),
        "similarity_gate": float(similarity_gate),
        "subspace_similarity_placebo": placebo_mean,
        "subspace_similarity_placebo_max": placebo_max,
        "n_placebo": n_placebo,
        "claim_margin_above_placebo": (similarity_result["subspace_similarity"] - placebo_mean),
        # Preregistered rule (decide_workspace_claim): the held-out CI lower
        # end must clear both the fixed gate and the shuffled-basis controls.
        "verdict_status": "evaluated",
        "workspace_claim_valid": claim_valid,
    }

    return result
