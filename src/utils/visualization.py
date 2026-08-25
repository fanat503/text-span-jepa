# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Visualization utilities for NeurIPS paper figures and training diagnostics.
# All plots use matplotlib with a clean academic style — no emojis, no clutter.
# Designed for inline notebook use AND standalone PNG/PDF export.

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import — matplotlib is optional (headless servers)
_mpl = None
_plt = None
_ScalarMappable = None


def _ensure_mpl():
    global _mpl, _plt, _ScalarMappable
    if _mpl is not None:
        return True
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless safe
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable

        _mpl = matplotlib
        _plt = plt
        _ScalarMappable = ScalarMappable
        return True
    except ImportError:
        logger.warning("matplotlib not available — visualization disabled")
        return False


def _cmap(name, n):
    """Version-safe discrete colormap: mpl >= 3.9 removed plt.get_cmap(name, lut)."""
    try:
        return _plt.get_cmap(name, n)
    except (TypeError, ValueError):
        return _mpl.colormaps[name].resampled(n)


# ═══════════════════════════════════════════════════════════════════
#  Academic plot style
# ═══════════════════════════════════════════════════════════════════


def setup_style():
    """Apply clean academic style — NeurIPS camera-ready compatible."""
    if not _ensure_mpl():
        return
    _plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
        }
    )


setup_style()


# ═══════════════════════════════════════════════════════════════════
#  Eigenvalue spectrum
# ═══════════════════════════════════════════════════════════════════


def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    title: str = "Eigenvalue Spectrum",
    highlight_k: int | None = None,
    save_path: str | None = None,
    ax=None,
):
    """Plot eigenvalue spectrum with optional workspace dimension highlight.

    Args:
        eigenvalues: 1-D array of eigenvalues in descending order.
        title: plot title.
        highlight_k: if set, shade the first k eigenvalues (workspace).
        save_path: if set, save figure to this path.
        ax: existing axes to draw on.
    """
    if not _ensure_mpl():
        return
    if ax is None:
        fig, ax = _plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.figure

    x = np.arange(len(eigenvalues))
    ax.semilogy(x, eigenvalues, color="#2166ac", marker="o", markersize=2, label="eigenvalues")

    if highlight_k is not None and 0 < highlight_k < len(eigenvalues):
        ax.axvline(
            highlight_k - 0.5,
            color="#b2182b",
            linestyle="--",
            linewidth=1,
            label=f"k={highlight_k} (workspace)",
        )
        ax.fill_between(
            x[:highlight_k], 1e-10, eigenvalues[:highlight_k], alpha=0.15, color="#b2182b"
        )

    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=max(eigenvalues.min() * 0.1, 1e-10))

    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
#  CKA heatmap
# ═══════════════════════════════════════════════════════════════════


def plot_cka_heatmap(
    cka_matrix: np.ndarray,
    layer_names: list[str] | None = None,
    title: str = "CKA Similarity",
    save_path: str | None = None,
    ax=None,
):
    """Plot CKA (Centered Kernel Alignment) heatmap between layers.

    Args:
        cka_matrix: (L, L) symmetric matrix of CKA scores.
        layer_names: labels for axes.
        title: plot title.
        save_path: save path.
        ax: existing axes.
    """
    if not _ensure_mpl():
        return
    if ax is None:
        fig, ax = _plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.figure

    im = ax.imshow(cka_matrix, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if layer_names is not None:
        n = min(len(layer_names), cka_matrix.shape[0])
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(layer_names[:n], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(layer_names[:n], fontsize=7)

    ax.set_title(title)
    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
#  SVCCA curve
# ═══════════════════════════════════════════════════════════════════


def plot_svcca_curve(
    svcca_scores: np.ndarray,
    title: str = "SVCCA",
    threshold: float = 0.99,
    save_path: str | None = None,
    ax=None,
):
    """Plot SVCCA singular values with cumulative threshold.

    Args:
        svcca_scores: 1-D array of SVCCA singular values (0 to 1).
        title: plot title.
        threshold: cumulative threshold line.
        save_path: save path.
        ax: existing axes.
    """
    if not _ensure_mpl():
        return
    if ax is None:
        fig, ax = _plt.subplots(figsize=(5, 3))
    else:
        fig = ax.figure

    x = np.arange(len(svcca_scores))
    ax.bar(x, svcca_scores, color="#4393c3", alpha=0.7, label="SVCCA")

    _denom = svcca_scores.sum()
    cumsum = np.cumsum(svcca_scores) / _denom if _denom > 0 else np.zeros_like(svcca_scores)
    ax2 = ax.twinx()
    ax2.plot(x, cumsum, color="#d6604d", linewidth=1.5, label="cumulative")
    ax2.axhline(threshold, color="#d6604d", linestyle=":", alpha=0.5)
    ax2.set_ylabel("Cumulative fraction")
    ax2.set_ylim(0, 1.05)

    ax.set_xlabel("Singular value index")
    ax.set_ylabel("SVCCA score")
    ax.set_title(title)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
#  PCA / t-SNE
# ═══════════════════════════════════════════════════════════════════


def plot_representation_2d(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    method: str = "pca",
    title: str | None = None,
    save_path: str | None = None,
    ax=None,
    n_components: int = 2,
    perplexity: int = 30,
):
    """2-D projection of embeddings via PCA or t-SNE.

    Args:
        embeddings: (N, D) array.
        labels: (N,) optional integer labels for coloring.
        method: 'pca' or 'tsne'.
        title: override title.
        save_path: save path.
        ax: existing axes.
        n_components: dimensions for projection.
        perplexity: t-SNE perplexity.
    """
    if not _ensure_mpl():
        return

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            projected = reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("sklearn not available, falling back to PCA")
            method = "pca"

    if method == "pca":
        centered = embeddings - embeddings.mean(axis=0)
        U, S, _Vt = np.linalg.svd(centered, full_matrices=False)
        projected = U[:, :n_components] * S[:n_components]

    if ax is None:
        fig, ax = _plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    if labels is not None:
        unique_labels = np.unique(labels)
        cmap = _cmap("tab10", len(unique_labels))
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            ax.scatter(
                projected[mask, 0], projected[mask, 1], c=[cmap(i)], s=8, alpha=0.6, label=str(lbl)
            )
        ax.legend(markerscale=2, fontsize=7, loc="best")
    else:
        ax.scatter(projected[:, 0], projected[:, 1], s=5, alpha=0.4, c="#2166ac")

    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(title or f"{method.upper()} Projection")
    ax.set_aspect("equal", "datalim")

    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
#  Stacked loss components
# ═══════════════════════════════════════════════════════════════════


def plot_stacked_losses(
    loss_history: dict[str, list[float]],
    title: str = "Training Loss Components",
    save_path: str | None = None,
    ax=None,
):
    """Stacked area plot of loss components over training steps.

    Args:
        loss_history: dict mapping loss name to list of values per step.
            Must include at least one key.
        title: plot title.
        save_path: save path.
        ax: existing axes.
    """
    if not _ensure_mpl():
        return
    if ax is None:
        fig, ax = _plt.subplots(figsize=(7, 3.5))
    else:
        fig = ax.figure

    keys = sorted(loss_history.keys())
    if not keys:
        return fig, ax

    steps = np.arange(len(loss_history[keys[0]]))
    values = np.array([loss_history[k] for k in keys])
    # Clamp negatives for stacking
    values = np.maximum(values, 0)

    colors = _cmap("Set2", len(keys))
    ax.stackplot(
        steps, values, labels=keys, colors=[colors(i) for i in range(len(keys))], alpha=0.8
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)

    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
#  Workspace utilization over training
# ═══════════════════════════════════════════════════════════════════


def plot_workspace_evolution(
    steps: Sequence[int],
    workspace_util: Sequence[float],
    target_ws_fraction: Sequence[float],
    k_values: Sequence[int],
    title: str = "JAWP Workspace Evolution",
    save_path: str | None = None,
):
    """Plot workspace metrics over training — workspace utilization, target
    fraction, and active dimension k.

    Three subplots stacked vertically.
    """
    if not _ensure_mpl():
        return
    fig, axes = _plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    axes[0].plot(steps, workspace_util, color="#2166ac")
    axes[0].set_ylabel("Workspace utilization")
    axes[0].set_ylim(0, 1)

    axes[1].plot(steps, target_ws_fraction, color="#b2182b")
    axes[1].set_ylabel("Target ws fraction")
    axes[1].set_ylim(0, 1)

    axes[2].plot(steps, k_values, color="#4393c3")
    axes[2].set_ylabel("Active k")
    axes[2].set_xlabel("Step")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig, axes


# ═══════════════════════════════════════════════════════════════════
#  Collapse diagnostics timeline
# ═══════════════════════════════════════════════════════════════════


def plot_collapse_timeline(
    steps: Sequence[int],
    effective_rank: Sequence[float],
    collapsed_dim_ratio: Sequence[float],
    embedding_std: Sequence[float],
    title: str = "Collapse Diagnostics",
    save_path: str | None = None,
):
    """Plot collapse prevention metrics over training.

    Three subplots: effective rank, collapsed dim ratio, embedding std.
    """
    if not _ensure_mpl():
        return
    fig, axes = _plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    axes[0].plot(steps, effective_rank, color="#2166ac")
    axes[0].set_ylabel("Effective rank")

    axes[1].plot(steps, collapsed_dim_ratio, color="#b2182b")
    axes[1].set_ylabel("Collapsed dim ratio")
    axes[1].set_ylim(0, 1)

    axes[2].plot(steps, embedding_std, color="#4393c3")
    axes[2].set_ylabel("Embedding std/dim")
    axes[2].set_xlabel("Step")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig, axes


# ═══════════════════════════════════════════════════════════════════
#  Scaling curve
# ═══════════════════════════════════════════════════════════════════


def plot_scaling_curve(
    param_counts: Sequence[float],
    metrics: dict[str, Sequence[float]],
    title: str = "Scaling Behavior",
    x_label: str = "Parameters (M)",
    save_path: str | None = None,
    ax=None,
):
    """Log-log scaling plot — metric vs parameter count.

    Args:
        param_counts: parameter counts (in millions).
        metrics: dict of metric_name -> values.
        title: plot title.
        x_label: x-axis label.
        save_path: save path.
        ax: existing axes.
    """
    if not _ensure_mpl():
        return
    if ax is None:
        fig, ax = _plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    colors = ["#2166ac", "#b2182b", "#4393c3", "#d6604d", "#762a83"]
    for i, (name, values) in enumerate(metrics.items()):
        ax.loglog(param_counts, values, marker="o", color=colors[i % len(colors)], label=name)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
#  JAWP risk vs PCA risk
# ═══════════════════════════════════════════════════════════════════


def plot_jawp_vs_pca(
    steps: Sequence[int],
    jawp_risk: Sequence[float],
    pca_risk: Sequence[float],
    title: str = "JAWP Risk vs PCA Risk",
    save_path: str | None = None,
    ax=None,
):
    """Plot JAWP workspace prediction risk vs PCA baseline risk over time.

    The Corollary guarantees jawp_risk <= pca_risk.
    """
    if not _ensure_mpl():
        return
    if ax is None:
        fig, ax = _plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.figure

    ax.plot(steps, jawp_risk, color="#2166ac", label="JAWP risk")
    ax.plot(steps, pca_risk, color="#b2182b", label="PCA risk", linestyle="--")
    ax.fill_between(
        steps,
        jawp_risk,
        pca_risk,
        where=[j <= p for j, p in zip(jawp_risk, pca_risk)],
        alpha=0.15,
        color="#2166ac",
        label="Corollary gap",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Prediction risk")
    ax.set_title(title)
    ax.legend(fontsize=8)

    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
#  CGN gating pattern
# ═══════════════════════════════════════════════════════════════════


def plot_gating_pattern(
    gate_values_visible: np.ndarray,
    gate_values_masked: np.ndarray,
    group_names: list[str] | None = None,
    title: str = "CGN Gating Pattern",
    save_path: str | None = None,
    ax=None,
):
    """Plot CGN gate values for visible vs masked positions.

    Shows how the Contextual Gating Network routes information
    differently at masked and visible positions — the key claim
    of the Information Routing theorem.

    Args:
        gate_values_visible: 1-D array of gate values at visible positions (n_groups,).
        gate_values_masked: 1-D array of gate values at masked positions (n_groups,).
        group_names: optional labels for gate groups.
        title: plot title.
        save_path: save path.
        ax: existing axes.
    """
    if not _ensure_mpl():
        return
    if ax is None:
        fig, ax = _plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.figure

    n = len(gate_values_visible)
    x = np.arange(n)
    width = 0.35

    ax.bar(x - width / 2, gate_values_visible, width, color="#2166ac", label="Visible", alpha=0.8)
    ax.bar(x + width / 2, gate_values_masked, width, color="#b2182b", label="Masked", alpha=0.8)

    if group_names is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, fontsize=7, rotation=45, ha="right")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([f"G{i}" for i in range(n)], fontsize=8)

    ax.set_ylabel("Gate value")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    # Add orthogonality indicator
    cos_sim = np.dot(gate_values_visible, gate_values_masked) / (
        np.linalg.norm(gate_values_visible) * np.linalg.norm(gate_values_masked) + 1e-10
    )
    ortho = 1.0 - abs(cos_sim)
    ax.text(
        0.02,
        0.95,
        f"Orthogonality: {ortho:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
#  Predictive rank utilization
# ═══════════════════════════════════════════════════════════════════


def plot_rank_utilization(
    steps: Sequence[int],
    effective_rank: Sequence[float],
    max_rank: int,
    title: str = "Workspace Rank Utilization",
    save_path: str | None = None,
    ax=None,
):
    """Plot effective rank of workspace prediction over training.

    Shows whether the predictor is using all k workspace dimensions
    or suffering from rank collapse.

    Args:
        steps: training step numbers.
        effective_rank: effective rank values.
        max_rank: maximum possible rank (k, workspace dimension).
        title: plot title.
        save_path: save path.
        ax: existing axes.
    """
    if not _ensure_mpl():
        return
    if ax is None:
        fig, ax = _plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.figure

    ax.plot(steps, effective_rank, color="#2166ac", label="Effective rank")
    ax.axhline(
        max_rank, color="#b2182b", linestyle="--", linewidth=1, label=f"Max rank (k={max_rank})"
    )
    ax.axhline(0.8 * max_rank, color="#4393c3", linestyle=":", linewidth=1, label="80% utilization")

    ax.set_xlabel("Step")
    ax.set_ylabel("Effective rank")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_ylim(0, max_rank * 1.1)

    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
#  Multi-panel dashboard
# ═══════════════════════════════════════════════════════════════════


def create_training_dashboard(
    log_dir: str,
    output_path: str | None = None,
):
    """Create a multi-panel training dashboard from CSV logs.

    Reads the CSV log file produced by CSVLogger and generates
    a 2x3 grid of key plots.

    Args:
        log_dir: directory containing log.csv.
        output_path: if set, save dashboard to this path.
    """
    if not _ensure_mpl():
        return

    csv_path = os.path.join(log_dir, "log.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"No log.csv found in {log_dir}")
        return

    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
    except Exception as e:
        logger.warning(f"Could not read CSV: {e}")
        return

    fig, axes = _plt.subplots(2, 3, figsize=(14, 8))

    names = list(data.dtype.names)

    # Panel 1: total loss
    if "loss" in names:
        axes[0, 0].plot(data["loss"], color="#2166ac")
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Step")

    # Panel 2: loss components
    comp_names = [n for n in names if n.startswith("loss_")]
    if comp_names:
        for n in comp_names:
            axes[0, 1].plot(data[n], label=n.replace("loss_", ""), alpha=0.7)
        axes[0, 1].set_title("Loss Components")
        axes[0, 1].legend(fontsize=6)
        axes[0, 1].set_xlabel("Step")

    # Panel 3: effective rank
    rank_names = [n for n in names if "rank" in n]
    if rank_names:
        for n in rank_names:
            axes[0, 2].plot(data[n], label=n)
        axes[0, 2].set_title("Effective Rank")
        axes[0, 2].legend(fontsize=6)
        axes[0, 2].set_xlabel("Step")

    # Panel 4: collapsed dim ratio
    coll_names = [n for n in names if "collapsed" in n]
    if coll_names:
        for n in coll_names:
            axes[1, 0].plot(data[n], label=n)
        axes[1, 0].set_title("Collapsed Dim Ratio")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].legend(fontsize=6)
        axes[1, 0].set_xlabel("Step")

    # Panel 5: learning rate
    if "lr" in names:
        axes[1, 1].plot(data["lr"], color="#b2182b")
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Step")

    # Panel 6: decoder accuracy
    if "dec_acc" in names:
        axes[1, 2].plot(data["dec_acc"], color="#4393c3")
        axes[1, 2].set_title("Decoder Accuracy")
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_xlabel("Step")

    fig.suptitle("Text-Span JEPA Training Dashboard")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
    return fig, axes


# ═══════════════════════════════════════════════════════════════════
#  Save helper
# ═══════════════════════════════════════════════════════════════════


def save_figure(fig, path: str, formats: tuple[str, ...] = ("png", "pdf")):
    """Save figure in multiple formats for NeurIPS submission.

    NeurIPS requires PDF for camera-ready. PNG for preview.
    """
    base, _ext = os.path.splitext(path)
    for fmt in formats:
        out_path = f"{base}.{fmt}"
        fig.savefig(out_path, format=fmt, bbox_inches="tight", dpi=300)
        logger.info(f"Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════
#  Spectral waterfall plot
# ═══════════════════════════════════════════════════════════════════


def plot_spectral_waterfall(
    eigenvalues_by_step: list[np.ndarray],
    steps: Sequence[int],
    highlight_k: int | None = None,
    title: str = "Spectral Waterfall",
    save_path: str | None = None,
):
    """Waterfall plot showing eigenvalue spectrum evolution over training.

    Each row is the eigenvalue spectrum at a different training step,
    stacked vertically to show how the spectrum evolves.

    Args:
        eigenvalues_by_step: list of 1-D arrays, one per step.
            Each array is eigenvalues in descending order.
        steps: corresponding step numbers.
        highlight_k: optional workspace dimension to mark.
        title: plot title.
        save_path: save path.
    """
    if not _ensure_mpl():
        return
    fig, ax = _plt.subplots(figsize=(8, 5))

    n_steps = len(eigenvalues_by_step)
    cmap = _cmap("viridis", n_steps)

    for i, (eigs, step) in enumerate(zip(eigenvalues_by_step, steps)):
        x = np.arange(len(eigs))
        color = cmap(i / max(n_steps - 1, 1))
        alpha = 0.3 + 0.7 * (i / max(n_steps - 1, 1))
        ax.semilogy(x, eigs, color=color, alpha=alpha, linewidth=0.8)

    if highlight_k is not None and len(eigenvalues_by_step) > 0:
        D = len(eigenvalues_by_step[-1])
        if 0 < highlight_k < D:
            ax.axvline(
                highlight_k - 0.5,
                color="#b2182b",
                linestyle="--",
                linewidth=1,
                label=f"k={highlight_k}",
            )

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title(title)
    if highlight_k is not None:
        ax.legend(fontsize=8)

    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
#  Information flow diagram
# ═══════════════════════════════════════════════════════════════════


def plot_information_flow(
    level_information: dict[str, float],
    title: str = "PCR Information Flow",
    save_path: str | None = None,
):
    """Bar chart showing information flow through PCR cascade levels.

    Visualizes the Cascade Capacity theorem: each level adds
    complementary information through orthogonal subspaces.

    Args:
        level_information: dict mapping level name to information
            value (in nats or bits). Keys like 'level_0', 'level_1', etc.
        title: plot title.
        save_path: save path.
    """
    if not _ensure_mpl():
        return
    fig, ax = _plt.subplots(figsize=(6, 3.5))

    keys = sorted(level_information.keys())
    values = [level_information[k] for k in keys]
    colors = ["#2166ac", "#4393c3", "#762a83", "#d6604d", "#b2182b"]

    ax.bar(
        range(len(keys)),
        values,
        color=[colors[i % len(colors)] for i in range(len(keys))],
        alpha=0.8,
    )

    # Cumulative line
    cumsum = np.cumsum(values)
    ax2 = ax.twinx()
    ax2.plot(
        range(len(keys)), cumsum, color="#b2182b", marker="o", linewidth=1.5, label="Cumulative"
    )
    ax2.set_ylabel("Cumulative information (nats)")

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Information per level (nats)")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
#  SWIP spectral shaping plot
# ═══════════════════════════════════════════════════════════════════


def plot_swip_spectral_shaping(
    eigenvalues_before: np.ndarray,
    eigenvalues_after: np.ndarray,
    k_workspace: int,
    target_variance: float = 1.0,
    title: str = "SWIP Spectral Shaping",
    save_path: str | None = None,
):
    """Plot eigenvalue spectrum before and after SWIP.

    Shows how SWIP whitens background while preserving workspace.

    Args:
        eigenvalues_before: eigenvalues before SWIP (descending).
        eigenvalues_after: eigenvalues after SWIP (descending).
        k_workspace: workspace dimension.
        target_variance: target background variance.
        title: plot title.
        save_path: save path.
    """
    if not _ensure_mpl():
        return
    fig, ax = _plt.subplots(figsize=(7, 3.5))

    x = np.arange(len(eigenvalues_before))
    ax.semilogy(
        x, eigenvalues_before, color="#2166ac", alpha=0.6, label="Before SWIP", linewidth=1.5
    )
    ax.semilogy(x, eigenvalues_after, color="#b2182b", label="After SWIP", linewidth=1.5)

    # Workspace boundary
    ax.axvline(
        k_workspace - 0.5, color="#762a83", linestyle="--", linewidth=1, label=f"k={k_workspace}"
    )
    # Target variance line
    ax.axhline(
        target_variance,
        color="#4393c3",
        linestyle=":",
        linewidth=1,
        label=f"target σ²={target_variance}",
    )

    # Shade workspace region
    ax.fill_between(
        x[:k_workspace],
        1e-10,
        np.maximum(eigenvalues_after[:k_workspace], 1e-10),
        alpha=0.1,
        color="#b2182b",
    )

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title(title)
    ax.legend(fontsize=8)

    if save_path:
        fig.savefig(save_path)
    return fig, ax


def plot_spc_band_analysis(
    band_weights: np.ndarray,
    band_residuals: np.ndarray,
    band_predictabilities: np.ndarray | None = None,
    band_snrs: np.ndarray | None = None,
    title: str = "SPC Band Analysis",
    save_path: str | None = None,
    ax=None,
):
    """Plot SPC band analysis: weights, residuals, predictability, SNR.

    Args:
        band_weights: (B,) band weights.
        band_residuals: (B,) per-band residual variance.
        band_predictabilities: (B,) per-band predictability (optional).
        band_snrs: (B,) per-band SNR (optional).
        title: plot title.
        save_path: if set, save to this path.
        ax: matplotlib axes to plot on.

    Returns:
        (fig, ax) tuple.
    """
    if not _ensure_mpl():
        return None, None
    setup_style()

    n_bands = len(band_weights)
    x = np.arange(n_bands)

    if ax is None:
        fig, axes = _plt.subplots(2, 2, figsize=(8, 6))
    else:
        fig = ax.get_figure()
        axes = ax

    # Band weights
    ax1 = axes[0, 0] if ax is None else axes[0]
    ax1.bar(x, band_weights, color="#4393c3", alpha=0.8)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    ax1.set_xlabel("Band index")
    ax1.set_ylabel("Weight")
    ax1.set_title("Band Weights")

    # Band residuals
    ax2 = axes[0, 1] if ax is None else axes[1]
    ax2.bar(x, band_residuals, color="#b2182b", alpha=0.8)
    ax2.set_xlabel("Band index")
    ax2.set_ylabel("Residual variance")
    ax2.set_title("Per-Band Residual")

    # Predictability
    ax3 = axes[1, 0] if ax is None else axes[2]
    if band_predictabilities is not None:
        ax3.bar(x, band_predictabilities, color="#1b7837", alpha=0.8)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel("Predictability (R²)")
    else:
        ax3.text(0.5, 0.5, "N/A", transform=ax3.transAxes, ha="center")
    ax3.set_xlabel("Band index")
    ax3.set_title("Per-Band Predictability")

    # SNR
    ax4 = axes[1, 1] if ax is None else axes[3]
    if band_snrs is not None:
        ax4.bar(x, np.log10(np.array(band_snrs) + 1e-10), color="#762a83", alpha=0.8)
        ax4.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax4.set_ylabel("SNR (log₁₀)")
    else:
        ax4.text(0.5, 0.5, "N/A", transform=ax4.transAxes, ha="center")
    ax4.set_xlabel("Band index")
    ax4.set_title("Per-Band SNR")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig, axes


def plot_wsd_drift(
    steps,
    drift_values,
    running_drift=None,
    jepa_loss=None,
    lambda_wsd=0.01,
    title="WSD: Workspace-Target Drift",
    ax=None,
    save_path=None,
):
    """Plot WSD drift over training steps.

    Args:
        steps: list of training step indices.
        drift_values: list of per-step drift values.
        running_drift: list of running average drift.
        jepa_loss: list of JEPA loss values (for comparison).
        lambda_wsd: WSD penalty weight.
        title: plot title.
        ax: optional matplotlib axes.
        save_path: optional save path.

    Returns:
        fig, ax
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.get_figure()

    steps = np.array(steps)
    drift_values = np.array(drift_values)

    ax.semilogy(
        steps, drift_values + 1e-10, color="#d53e4f", alpha=0.3, label="Instantaneous drift"
    )
    if running_drift is not None:
        ax.semilogy(
            steps,
            np.array(running_drift) + 1e-10,
            color="#d53e4f",
            linewidth=2,
            label="Running avg drift",
        )
    if jepa_loss is not None:
        ax.semilogy(
            steps,
            np.array(jepa_loss) + 1e-10,
            color="#3288bd",
            alpha=0.5,
            linewidth=1,
            label="JEPA loss",
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Grassmann drift (log scale)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
    return fig, ax


def plot_cmc_consistency(
    steps,
    cmc_losses,
    overlap_ratios=None,
    probe_norm=1.0,
    title="CMC: Cross-Mask Consistency",
    ax=None,
    save_path=None,
):
    """Plot CMC consistency loss and downstream stability bound.

    Args:
        steps: list of training step indices.
        cmc_losses: list of CMC loss values.
        overlap_ratios: list of overlap ratios.
        probe_norm: ||w|| for downstream stability bound.
        title: plot title.
        ax: optional matplotlib axes.
        save_path: optional save path.

    Returns:
        fig, axes
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig = ax.get_figure()
        axes = [ax, ax]

    steps = np.array(steps)
    cmc_losses = np.array(cmc_losses)

    # Left: CMC loss
    axes[0].semilogy(steps, cmc_losses + 1e-10, color="#66c2a5", linewidth=1.5)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("CMC loss (log scale)")
    axes[0].set_title("Cross-Mask Consistency")
    axes[0].grid(True, alpha=0.3)

    # Right: downstream stability bound ||w|| * sqrt(epsilon)
    stability_bound = probe_norm * np.sqrt(np.maximum(cmc_losses, 0))
    axes[1].plot(steps, stability_bound, color="#fc8d62", linewidth=1.5)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel(f"Downstream stability (||w||={probe_norm:.1f})")
    axes[1].set_title("Stability Bound: |f(z₁) - f(z₂)| ≤ ||w||√ε")
    axes[1].grid(True, alpha=0.3)

    if overlap_ratios is not None:
        ax2 = axes[0].twinx()
        ax2.plot(steps, overlap_ratios, color="#8da0cb", alpha=0.4, linewidth=0.8)
        ax2.set_ylabel("Overlap ratio", color="#8da0cb", fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig, axes


# ═══════════════════════════════════════════════════════════════════
#  STA spectral transport alignment plot
# ═══════════════════════════════════════════════════════════════════


def plot_sta_spectral_alignment(
    steps,
    w1_values,
    spectral_gaps=None,
    davis_kahan_bounds=None,
    running_w1=None,
    title="STA: Spectral Transport Alignment",
    ax=None,
    save_path=None,
):
    """Plot STA spectral alignment metrics over training.

    Shows W1 distance, spectral gap, and Davis-Kahan bound.
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.get_figure()

    steps = np.array(steps)
    w1_values = np.array(w1_values)

    ax.plot(steps, w1_values, color="#e41a1c", alpha=0.3, label="Instantaneous W1")
    if running_w1 is not None:
        ax.plot(steps, np.array(running_w1), color="#e41a1c", linewidth=2, label="Running avg W1")

    if davis_kahan_bounds is not None:
        ax2 = ax.twinx()
        ax2.plot(
            steps,
            np.array(davis_kahan_bounds),
            color="#377eb8",
            linewidth=1.5,
            linestyle="--",
            label="Davis-Kahan bound",
        )
        ax2.set_ylabel("Workspace drift bound (d_Gr)", color="#377eb8")
        ax2.tick_params(axis="y", labelcolor="#377eb8")

    ax.set_xlabel("Training step")
    ax.set_ylabel("W1 distance (spectral drift)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
    return fig, ax


def plot_gac_starved_fraction(
    steps,
    starved_fractions,
    running_starved=None,
    mean_grad_norms=None,
    title="GAC: Gradient Starvation Monitor",
    ax=None,
    save_path=None,
):
    """Plot GAC starved fraction over training."""
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.get_figure()

    steps = np.array(steps)
    starved_fractions = np.array(starved_fractions)

    ax.plot(steps, starved_fractions, color="#ff7f00", alpha=0.3, label="Starved fraction")
    if running_starved is not None:
        ax.plot(steps, np.array(running_starved), color="#ff7f00", linewidth=2, label="Running avg")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5, label="50% threshold")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Starved fraction")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
#  PUC: Overconfidence timeline
# ═══════════════════════════════════════════════════════════════════


def plot_puc_overconfidence_timeline(
    steps: Sequence,
    overconfidence_scores: Sequence,
    entropy_values: Sequence | None = None,
    target_entropy: float | None = None,
    title: str = "PUC Overconfidence Timeline",
    save_path: str | None = None,
    ax=None,
):
    """Plot PUC overconfidence and entropy over training.

    Args:
        steps: training step indices.
        overconfidence_scores: PUC overconfidence metric per step.
        entropy_values: estimated entropy per step (optional).
        target_entropy: target entropy threshold (horizontal line).
        title: plot title.
        save_path: path to save figure.
        ax: existing axes.
    """
    if not _ensure_mpl():
        return
    if ax is None:
        fig, ax = _plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.get_figure()

    import numpy as np

    steps = np.array(steps)
    overconfidence_scores = np.array(overconfidence_scores)

    ax.plot(steps, overconfidence_scores, color="#e41a1c", alpha=0.5, label="Overconfidence")
    # Running average
    if len(overconfidence_scores) > 10:
        kernel = np.ones(10) / 10
        running = np.convolve(overconfidence_scores, kernel, mode="same")
        ax.plot(steps, running, color="#e41a1c", linewidth=2, label="Running avg")
    ax.axhline(0.0, color="green", linestyle=":", linewidth=1, label="Calibrated (0)")

    if entropy_values is not None and target_entropy is not None:
        ax2 = ax.twinx()
        entropy_arr = np.array(entropy_values)
        ax2.plot(steps, entropy_arr, color="#377eb8", alpha=0.5, label="Entropy")
        ax2.axhline(target_entropy, color="#377eb8", linestyle="--", linewidth=1, label="H_target")
        ax2.set_ylabel("Entropy", color="#377eb8")
        ax2.tick_params(axis="y", labelcolor="#377eb8")
        ax2.legend(fontsize=8, loc="upper right")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Overconfidence score")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
#  RDC: Orthogonal drift ratio timeline
# ═══════════════════════════════════════════════════════════════════


def plot_rdc_drift_ratio(
    steps: Sequence,
    drift_ratios: Sequence,
    ortho_drift_norms: Sequence | None = None,
    workspace_drift_norms: Sequence | None = None,
    title: str = "RDC Orthogonal Drift Ratio",
    save_path: str | None = None,
    ax=None,
):
    """Plot RDC drift ratio and drift norms over training.

    Args:
        steps: training step indices.
        drift_ratios: ||Δz_⊥||/||Δz|| per step (should decrease with RDC).
        ortho_drift_norms: ||Δz_⊥|| per step (optional).
        workspace_drift_norms: ||Δz_∥|| per step (optional).
        title: plot title.
        save_path: path to save figure.
        ax: existing axes.
    """
    if not _ensure_mpl():
        return
    if ax is None:
        fig, ax = _plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.get_figure()

    import numpy as np

    steps = np.array(steps)
    drift_ratios = np.array(drift_ratios)

    ax.plot(steps, drift_ratios, color="#984ea3", alpha=0.4, label="Drift ratio")
    if len(drift_ratios) > 10:
        kernel = np.ones(10) / 10
        running = np.convolve(drift_ratios, kernel, mode="same")
        ax.plot(steps, running, color="#984ea3", linewidth=2, label="Running avg")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5, label="50% threshold")

    if ortho_drift_norms is not None and workspace_drift_norms is not None:
        ax2 = ax.twinx()
        ax2.plot(steps, np.array(ortho_drift_norms), color="#e41a1c", alpha=0.4, label="||Δz_⊥||")
        ax2.plot(
            steps, np.array(workspace_drift_norms), color="#4daf4a", alpha=0.4, label="||Δz_∥||"
        )
        ax2.set_ylabel("Drift norm")
        ax2.legend(fontsize=8, loc="upper right")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Orthogonal drift ratio")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
    return fig, ax


def plot_wsr_sharpness(
    steps,
    sharpness,
    spectral_sharpness=None,
    directional_sharpness=None,
    title="WSR Workspace Sharpness",
    save_path=None,
):
    """Plot Workspace Sharpness Regularization diagnostics.

    Shows total sharpness and its decomposition into spectral and
    directional components (Grassmann Sharpness Decomposition theorem).

    Args:
        steps: array of training steps.
        sharpness: array of total sharpness ρ_Q values.
        spectral_sharpness: array of spectral sharpness values (optional).
        directional_sharpness: array of directional sharpness values (optional).
        title: plot title.
        save_path: path to save figure.
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(steps, np.array(sharpness), color="#377eb8", linewidth=2, label="ρ_Q (total)")

    if spectral_sharpness is not None:
        ax.plot(
            steps,
            np.array(spectral_sharpness),
            color="#ff7f00",
            linewidth=1.5,
            alpha=0.8,
            label="ρ_spectral (STA-bound)",
        )

    if directional_sharpness is not None:
        ax.plot(
            steps,
            np.array(directional_sharpness),
            color="#4daf4a",
            linewidth=1.5,
            alpha=0.8,
            label="ρ_directional (WSR-bound)",
        )

    # Add fill between decomposition components
    if spectral_sharpness is not None and directional_sharpness is not None:
        ax.fill_between(steps, 0, np.array(spectral_sharpness), alpha=0.1, color="#ff7f00")
        ax.fill_between(
            steps,
            np.array(spectral_sharpness),
            np.array(spectral_sharpness) + np.array(directional_sharpness),
            alpha=0.1,
            color="#4daf4a",
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Sharpness")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
#  Additional per-mechanism plots (GWP framework)
# ═══════════════════════════════════════════════════════════════════


def plot_jawp_workspace_evolution(
    steps, active_k, workspace_loss, title="JAWP Workspace Evolution", save_path=None
):
    """Plot JAWP workspace dimension curriculum and loss."""
    _ensure_mpl()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(steps, active_k, color="#e41a1c", linewidth=1.5)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Active k (workspace dim)")
    ax1.set_title('"Workspace Dimension Curriculum')
    ax1.grid(True, alpha=0.3)
    ax2.plot(steps, workspace_loss, color="#377eb8", linewidth=1.5)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("JAWP Loss")
    ax2.set_yscale("log")
    ax2.set_title("Workspace Prediction Loss")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig, (ax1, ax2)


def plot_cgn_gate_distribution(
    steps, gate_visible, gate_masked, title="CGN Gate Distribution", save_path=None
):
    """Plot CGN visible vs masked gate probabilities over training."""
    _ensure_mpl()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, gate_visible, color="#2ca02c", linewidth=1.5, label="g_visible", alpha=0.8)
    ax.plot(steps, gate_masked, color="#d62728", linewidth=1.5, label="g_masked", alpha=0.8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Equal split")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gate probability")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path)
    return fig, ax


def plot_pcr_cascade_capacity(
    steps, level_losses, level_dims=None, title="PCR Cascade Capacity", save_path=None
):
    """Plot PCR per-level prediction losses and capacity bounds."""
    _ensure_mpl()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    if not level_losses or not level_losses[0]:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax
    n_levels = len(level_losses[0])
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_levels, 1)))
    for l in range(n_levels):
        # Pair every level value with its own step: level arrays are ragged and
        # steps[:len(vals)] silently misaligned earliest-steps with later values.
        pts = [(s, step[l]) for s, step in zip(steps, level_losses) if l < len(step)]
        if not pts:
            continue
        s_arr, v_arr = zip(*pts)
        ax.plot(s_arr, v_arr, color=colors[l], linewidth=1.5, label=f"Level {l}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Level loss")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path)
    return fig, ax


def plot_gwp_mechanism_summary(
    mechanism_names, mechanism_losses, title="GWP Mechanism Loss Summary", save_path=None
):
    """Plot summary of all active GWP mechanism losses."""
    _ensure_mpl()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(mechanism_names))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(mechanism_names), 1)))
    ax.bar(x, mechanism_losses, color=colors)
    ax.set_xticks(list(x))
    ax.set_xticklabels(mechanism_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Loss contribution")
    ax.set_title(title)
    if all(v > 0 for v in mechanism_losses):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    if save_path:
        fig.savefig(save_path)
    return fig, ax


def plot_ema_tau_schedule(
    steps, tau_values, schedule_type="cosine", title="EMA Tau Schedule", save_path=None
):
    """Plot EMA tau schedule over training."""
    _ensure_mpl()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, tau_values, color="#1f77b4", linewidth=2, label=f"{schedule_type} schedule")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="tau=1.0 (frozen -- BAD)")
    ax.axhline(
        y=0.9999, color="green", linestyle="--", alpha=0.5, label="tau=0.9999 (near-frozen -- GOOD)"
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("EMA tau")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_ylim(0.990, 1.0001)
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path)
    return fig, ax


def plot_workspace_quality_components(
    steps, components_dict, title="Workspace Quality Components", save_path=None
):
    """Plot each component of the 10-component workspace_quality metric."""
    _ensure_mpl()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, values in components_dict.items():
        ax.plot(steps[: len(values)], values, linewidth=1.2, label=name, alpha=0.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Component value")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path)
    return fig, ax
