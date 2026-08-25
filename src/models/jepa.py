# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Main JEPA model: glues encoder, predictor, decoder, collapse prevention
# Training loop patterns from I-JEPA (Assran et al., CVPR 2023)
# Target centering + layer norm from data2vec (Baevski et al., ICML 2022)
# VICReg collapse prevention from C-JEPA (NeurIPS 2024) / VICReg (ICLR 2022)

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .cgn import ContextualGatingNetwork
from .cmc import CrossMaskConsistency
from .collapse import (
    CollapseDiagnostics,
    CovarianceRegularization,
    TargetCentering,
    VarianceRegularization,
)
from .decoder import TiedTokenDecoder
from .encoder import TextSpanJEPAEncoder
from .gac import GradientAllocatedCapacity
from .jawp import JAWPModule
from .jspace import JSpaceMetrics
from .pcr import PredictiveCascadeRefinement
from .predictor import TextSpanJPAPredictor
from .puc import PredictionUncertaintyCalibration
from .rdc import RepresentationDriftCompensation
from .sigreg import SIGReg
from .spc import SpectralPredictiveCoding
from .sta import SpectralTransportAlignment
from .swip import SWIPModule
from .wsd import WorkspaceSyncDrift
from .wsr import WorkspaceSharpnessRegularization


class TextSpanJEPAConfig:
    """Configuration for Text-Span JEPA model."""

    def __init__(self, **kwargs):
        # Encoder
        self.vocab_size = kwargs.get("vocab_size", 50304)
        self.max_seq_len = kwargs.get("max_seq_len", 512)
        self.embed_dim = kwargs.get("embed_dim", 768)
        self.encoder_depth = kwargs.get("encoder_depth", 12)
        self.num_heads = kwargs.get("num_heads", 12)
        self.mlp_ratio = kwargs.get("mlp_ratio", 4.0)
        self.qkv_bias = kwargs.get("qkv_bias", True)
        self.drop_rate = kwargs.get("drop_rate", 0.0)
        self.attn_drop_rate = kwargs.get("attn_drop_rate", 0.0)
        self.drop_path_rate = kwargs.get("drop_path_rate", 0.1)
        self.gradient_checkpointing = kwargs.get("gradient_checkpointing", False)

        # Predictor
        self.predictor_embed_dim = kwargs.get("predictor_embed_dim", 384)
        self.predictor_depth = kwargs.get("predictor_depth", 6)
        self.future_offsets = kwargs.get("future_offsets", (1, 4, 16))
        self.num_refine_steps = kwargs.get("num_refine_steps", 3)
        self.refine_step_size = kwargs.get("refine_step_size", 0.1)

        # Decoder
        self.decoder_bias = kwargs.get("decoder_bias", False)

        # Collapse prevention
        self.variance_margin = kwargs.get("variance_margin", 1.0)
        self.centering_momentum = kwargs.get("centering_momentum", 0.9)

        # EMA target encoder
        self.ema_tau_start = kwargs.get("ema_tau_start", 0.996)
        self.ema_tau_end = kwargs.get("ema_tau_end", 0.9999)
        self.ema_schedule = kwargs.get("ema_schedule", "cosine")

        # Loss weights
        self.lambda_span = kwargs.get("lambda_span", 1.0)
        self.lambda_future = kwargs.get("lambda_future", 0.5)
        self.lambda_decoder = kwargs.get("lambda_decoder", 0.1)
        self.lambda_variance = kwargs.get("lambda_variance", 0.1)
        self.lambda_covariance = kwargs.get("lambda_covariance", 0.04)

        # Mask curriculum
        self.mask_ratio_start = kwargs.get("mask_ratio_start", 0.15)
        self.mask_ratio_end = kwargs.get("mask_ratio_end", 0.35)

        # Future loss warmup
        self.future_warmup_steps = kwargs.get("future_warmup_steps", 0)

        # SIGReg
        self.lambda_sigreg = kwargs.get("lambda_sigreg", 0.0)
        self.sigreg_n_sketches = kwargs.get("sigreg_n_sketches", 64)
        self.sigreg_n_integration_points = kwargs.get("sigreg_n_integration_points", 17)
        self.sigreg_sigma = kwargs.get("sigreg_sigma", 1.0)

        # J-Space
        self.jspace_variance_threshold = kwargs.get("jspace_variance_threshold", 0.10)
        self.jspace_k_workspace = kwargs.get("jspace_k_workspace", 25)

        # JAWP
        self.use_jawp = kwargs.get("use_jawp", True)
        self.jawk_k_start = kwargs.get("jawk_k_start", 1)
        self.jawk_k_end = kwargs.get("jawk_k_end", None)
        self.jawk_curriculum_steps = kwargs.get("jawk_curriculum_steps", 10000)
        self.jawk_alpha = kwargs.get("jawk_alpha", 0.1)
        self.jawk_init = kwargs.get("jawk_init", "identity")

        # Predictive Rank Regularization (from JAWP module)
        self.lambda_predictive_rank = kwargs.get("lambda_predictive_rank", 0.0)

        # CGN: Contextual Gating Network (novel mechanism #6)
        self.use_cgn = kwargs.get("use_cgn", False)
        self.cgn_n_groups = kwargs.get("cgn_n_groups", 8)
        self.cgn_tau_start = kwargs.get("cgn_tau_start", 1.0)
        self.cgn_tau_end = kwargs.get("cgn_tau_end", 0.1)
        self.cgn_anneal_steps = kwargs.get("cgn_anneal_steps", 10000)
        self.lambda_cgn_ortho = kwargs.get("lambda_cgn_ortho", 0.0)

        # SWIP: Selective Whitening with Information Preservation (novel mechanism #7)
        self.use_swip = kwargs.get("use_swip", False)
        self.swip_k_workspace = kwargs.get("swip_k_workspace", None)
        self.swip_target_variance = kwargs.get("swip_target_variance", 1.0)
        self.lambda_swip = kwargs.get("lambda_swip", 0.0)

        # PCR: Predictive Cascade Refinement (novel mechanism #8)
        self.use_pcr = kwargs.get("use_pcr", False)
        self.pcr_n_levels = kwargs.get("pcr_n_levels", 3)
        self.pcr_level_dims = kwargs.get("pcr_level_dims", None)
        self.pcr_warmup_steps = kwargs.get("pcr_warmup_steps", 1000)

        # SPC: Spectral Predictive Coding (novel mechanism #9)
        self.use_spc = kwargs.get("use_spc", False)
        self.spc_n_bands = kwargs.get("spc_n_bands", 8)
        self.spc_init = kwargs.get("spc_init", "dct")
        self.lambda_spc = kwargs.get("lambda_spc", 0.0)

        # WSD: Workspace-Target Synchronization Drift (novel mechanism #10)
        self.use_wsd = kwargs.get("use_wsd", False)
        self.wsd_k = kwargs.get("wsd_k", None)
        self.wsd_sync_interval = kwargs.get("wsd_sync_interval", 100)
        self.wsd_ema_beta = kwargs.get("wsd_ema_beta", 0.99)
        self.lambda_wsd = kwargs.get("lambda_wsd", 0.0)

        # CMC: Cross-Mask Consistency Regularization (novel mechanism #11)
        self.use_cmc = kwargs.get("use_cmc", False)
        self.cmc_second_mask_ratio = kwargs.get("cmc_second_mask_ratio", None)
        self.cmc_min_overlap_ratio = kwargs.get("cmc_min_overlap_ratio", 0.2)
        self.cmc_mode = kwargs.get("cmc_mode", "interval")
        self.cmc_interval = kwargs.get("cmc_interval", 10)
        self.lambda_cmc = kwargs.get("lambda_cmc", 0.0)

        # GAC: Gradient-Allocated Capacity (novel mechanism #12)
        self.use_gac = kwargs.get("use_gac", False)
        self.gac_gamma = kwargs.get("gac_gamma", 0.01)
        self.gac_tau_grad = kwargs.get("gac_tau_grad", 1e-4)
        self.gac_warmup_steps = kwargs.get("gac_warmup_steps", 1000)
        self.lambda_gac = kwargs.get("lambda_gac", 0.0)

        # STA: Spectral Transport Alignment (novel mechanism #13)
        self.use_sta = kwargs.get("use_sta", False)
        self.sta_eta = kwargs.get("sta_eta", 0.01)
        self.sta_ema_beta = kwargs.get("sta_ema_beta", 0.999)
        self.sta_warmup_steps = kwargs.get("sta_warmup_steps", 500)
        self.sta_update_interval = kwargs.get("sta_update_interval", 10)

        # PUC: Prediction Uncertainty Calibration (novel mechanism #14)
        self.use_puc = kwargs.get("use_puc", False)
        self.puc_eta = kwargs.get("puc_eta", 0.01)
        self.puc_ema_beta = kwargs.get("puc_ema_beta", 0.999)
        self.puc_warmup_steps = kwargs.get("puc_warmup_steps", 500)
        self.lambda_sta = kwargs.get("lambda_sta", 0.0)
        self.lambda_puc = kwargs.get("lambda_puc", 0.0)

        # RDC: Representation Drift Compensation (novel mechanism #15)
        self.use_rdc = kwargs.get("use_rdc", False)
        self.rdc_eta = kwargs.get("rdc_eta", 0.01)
        self.rdc_ema_beta = kwargs.get("rdc_ema_beta", 0.999)
        self.rdc_warmup_steps = kwargs.get("rdc_warmup_steps", 500)
        self.rdc_k_workspace = kwargs.get("rdc_k_workspace", None)
        self.lambda_rdc = kwargs.get("lambda_rdc", 0.0)

        # WSR: Workspace Sharpness Regularization (novel mechanism #16)
        self.use_wsr = kwargs.get("use_wsr", False)
        self.wsr_rho = kwargs.get("wsr_rho", 0.05)
        self.wsr_eta = kwargs.get("wsr_eta", 0.01)
        self.wsr_ema_beta = kwargs.get("wsr_ema_beta", 0.999)
        self.wsr_warmup_steps = kwargs.get("wsr_warmup_steps", 500)
        self.wsr_mode = kwargs.get("wsr_mode", "gradient")
        self.lambda_wsr = kwargs.get("lambda_wsr", 0.0)

    def validate(self):
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by num_heads={self.num_heads}"
            )
        if self.predictor_embed_dim % self.num_heads != 0:
            raise ValueError(
                f"predictor_embed_dim={self.predictor_embed_dim} must be divisible by num_heads={self.num_heads}"
            )
        if self.encoder_depth < 1:
            raise ValueError(f"encoder_depth must be >= 1, got {self.encoder_depth}")
        if self.predictor_depth < 1:
            raise ValueError(f"predictor_depth must be >= 1, got {self.predictor_depth}")
        if self.ema_schedule not in ("cosine", "linear"):
            raise ValueError(
                f"ema_schedule must be 'cosine' or 'linear', got '{self.ema_schedule}'"
            )
        if self.lambda_span < 0:
            raise ValueError(f"lambda_span must be >= 0, got {self.lambda_span}")
        if self.lambda_future < 0:
            raise ValueError(f"lambda_future must be >= 0, got {self.lambda_future}")
        if self.variance_margin <= 0:
            raise ValueError(f"variance_margin must be > 0, got {self.variance_margin}")
        if not 0 < self.centering_momentum < 1:
            raise ValueError(f"centering_momentum must be in (0,1), got {self.centering_momentum}")
        if self.use_jawp:
            if self.jawk_k_start < 1:
                raise ValueError(f"jawk_k_start must be >= 1, got {self.jawk_k_start}")
            if self.jawk_k_end is not None and self.jawk_k_end > self.embed_dim:
                raise ValueError(
                    f"jawk_k_end={self.jawk_k_end} cannot exceed embed_dim={self.embed_dim}"
                )
            if self.jawk_k_end is not None and self.jawk_k_start > self.jawk_k_end:
                raise ValueError(f"jawk_k_start={self.jawk_k_start} > jawk_k_end={self.jawk_k_end}")
            if self.jawk_alpha < 0:
                raise ValueError(f"jawk_alpha must be >= 0, got {self.jawk_alpha}")
            if self.jawk_init not in ("identity", "random", "pca"):
                raise ValueError(
                    f"jawk_init must be 'identity', 'random', or 'pca', got '{self.jawk_init}'"
                )
        if self.lambda_sigreg < 0:
            raise ValueError(f"lambda_sigreg must be >= 0, got {self.lambda_sigreg}")
        if self.lambda_sigreg > 0 and self.sigreg_sigma <= 0:
            raise ValueError(
                f"sigreg_sigma must be > 0 when SIGReg is active, got {self.sigreg_sigma}"
            )
        if self.future_warmup_steps < 0:
            raise ValueError(f"future_warmup_steps must be >= 0, got {self.future_warmup_steps}")
        if self.lambda_predictive_rank < 0:
            raise ValueError(
                f"lambda_predictive_rank must be >= 0, got {self.lambda_predictive_rank}"
            )
        if self.use_cgn:
            if self.embed_dim % self.cgn_n_groups != 0:
                raise ValueError(
                    f"embed_dim={self.embed_dim} must be divisible by cgn_n_groups={self.cgn_n_groups}"
                )
            if self.cgn_n_groups < 1:
                raise ValueError(f"cgn_n_groups must be >= 1, got {self.cgn_n_groups}")
            if self.cgn_tau_start <= 0 or self.cgn_tau_end <= 0:
                raise ValueError(
                    f"cgn temperatures must be > 0, got start={self.cgn_tau_start}, end={self.cgn_tau_end}"
                )
            if self.lambda_cgn_ortho < 0:
                raise ValueError(f"lambda_cgn_ortho must be >= 0, got {self.lambda_cgn_ortho}")
        if self.use_swip:
            if self.swip_k_workspace is not None and self.swip_k_workspace > self.embed_dim:
                raise ValueError(
                    f"swip_k_workspace={self.swip_k_workspace} cannot exceed embed_dim={self.embed_dim}"
                )
            if self.swip_target_variance <= 0:
                raise ValueError(
                    f"swip_target_variance must be > 0, got {self.swip_target_variance}"
                )
            if self.lambda_swip < 0:
                raise ValueError(f"lambda_swip must be >= 0, got {self.lambda_swip}")
        if self.use_pcr:
            if self.pcr_n_levels < 1:
                raise ValueError(f"pcr_n_levels must be >= 1, got {self.pcr_n_levels}")
            if self.pcr_level_dims is not None:
                total = sum(self.pcr_level_dims)
                if total > self.embed_dim:
                    raise ValueError(
                        f"sum(pcr_level_dims)={total} cannot exceed embed_dim={self.embed_dim}"
                    )
            if self.pcr_warmup_steps < 0:
                raise ValueError(f"pcr_warmup_steps must be >= 0, got {self.pcr_warmup_steps}")
        if self.use_spc:
            if self.embed_dim % self.spc_n_bands != 0:
                raise ValueError(
                    f"embed_dim={self.embed_dim} must be divisible by spc_n_bands={self.spc_n_bands}"
                )
            if self.spc_n_bands < 1:
                raise ValueError(f"spc_n_bands must be >= 1, got {self.spc_n_bands}")
            if self.lambda_spc < 0:
                raise ValueError(f"lambda_spc must be >= 0, got {self.lambda_spc}")
            if self.spc_init not in ("dct", "random"):
                raise ValueError(f"spc_init must be 'dct' or 'random', got '{self.spc_init}'")
        if self.use_wsd:
            if self.lambda_wsd < 0:
                raise ValueError(f"lambda_wsd must be >= 0, got {self.lambda_wsd}")
            if self.wsd_sync_interval < 1:
                raise ValueError(f"wsd_sync_interval must be >= 1, got {self.wsd_sync_interval}")
        if self.use_cmc:
            if self.lambda_cmc < 0:
                raise ValueError(f"lambda_cmc must be >= 0, got {self.lambda_cmc}")
            if self.cmc_mode not in ("always", "interval", "reuse_encoder"):
                raise ValueError(
                    f"cmc_mode must be always/interval/reuse_encoder, got '{self.cmc_mode}'"
                )
            if self.cmc_min_overlap_ratio < 0 or self.cmc_min_overlap_ratio > 1:
                raise ValueError(
                    f"cmc_min_overlap_ratio must be in [0,1], got {self.cmc_min_overlap_ratio}"
                )
        if self.use_gac:
            if self.lambda_gac < 0:
                raise ValueError(f"lambda_gac must be >= 0, got {self.lambda_gac}")
            if self.gac_gamma < 0:
                raise ValueError(f"gac_gamma must be >= 0, got {self.gac_gamma}")
            if self.gac_tau_grad < 0:
                raise ValueError(f"gac_tau_grad must be >= 0, got {self.gac_tau_grad}")
        if self.use_sta:
            if self.lambda_sta < 0:
                raise ValueError(f"lambda_sta must be >= 0, got {self.lambda_sta}")
            if self.sta_eta < 0:
                raise ValueError(f"sta_eta must be >= 0, got {self.sta_eta}")
            if not (0 < self.sta_ema_beta < 1):
                raise ValueError(f"sta_ema_beta must be in (0,1), got {self.sta_ema_beta}")
            if self.sta_warmup_steps < 0:
                raise ValueError(f"sta_warmup_steps must be >= 0, got {self.sta_warmup_steps}")
        if self.use_puc:
            if self.lambda_puc < 0:
                raise ValueError(f"lambda_puc must be >= 0, got {self.lambda_puc}")
            if self.puc_eta < 0:
                raise ValueError(f"puc_eta must be >= 0, got {self.puc_eta}")
            if not (0 < self.puc_ema_beta < 1):
                raise ValueError(f"puc_ema_beta must be in (0,1), got {self.puc_ema_beta}")
            if self.puc_warmup_steps < 0:
                raise ValueError(f"puc_warmup_steps must be >= 0, got {self.puc_warmup_steps}")
        if self.use_rdc:
            if self.lambda_rdc < 0:
                raise ValueError(f"lambda_rdc must be >= 0, got {self.lambda_rdc}")
            if self.rdc_eta < 0:
                raise ValueError(f"rdc_eta must be >= 0, got {self.rdc_eta}")
            if not (0 < self.rdc_ema_beta < 1):
                raise ValueError(f"rdc_ema_beta must be in (0,1), got {self.rdc_ema_beta}")
            if self.rdc_warmup_steps < 0:
                raise ValueError(f"rdc_warmup_steps must be >= 0, got {self.rdc_warmup_steps}")
        if self.use_wsr:
            if self.lambda_wsr < 0:
                raise ValueError(f"lambda_wsr must be >= 0, got {self.lambda_wsr}")
            if self.wsr_rho <= 0:
                raise ValueError(f"wsr_rho must be > 0, got {self.wsr_rho}")
            if self.wsr_eta < 0:
                raise ValueError(f"wsr_eta must be >= 0, got {self.wsr_eta}")
            if not (0 < self.wsr_ema_beta < 1):
                raise ValueError(f"wsr_ema_beta must be in (0,1), got {self.wsr_ema_beta}")
            if self.wsr_warmup_steps < 0:
                raise ValueError(f"wsr_warmup_steps must be >= 0, got {self.wsr_warmup_steps}")
            if self.wsr_mode not in ("gradient", "hessian", "param"):
                raise ValueError(f"wsr_mode must be gradient/hessian/param, got '{self.wsr_mode}'")
        # EMA tau range check
        if not (0 < self.ema_tau_start <= self.ema_tau_end < 1.0):
            raise ValueError(
                f"ema_tau_start={self.ema_tau_start} must be in (0, ema_tau_end) "
                f"and ema_tau_end={self.ema_tau_end} must be in (ema_tau_start, 1.0)"
            )
        return True


class TextSpanJEPA(nn.Module):
    """Text-Span JEPA: Latent Predictive Learning for Language Representations."""

    def __init__(self, config: TextSpanJEPAConfig):
        super().__init__()
        self.config = config

        self.encoder = TextSpanJEPAEncoder(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
            gradient_checkpointing=config.gradient_checkpointing,
        )

        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = TextSpanJPAPredictor(
            embed_dim=config.embed_dim,
            predictor_embed_dim=config.predictor_embed_dim,
            depth=config.predictor_depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            max_seq_len=config.max_seq_len,
            future_offsets=config.future_offsets,
            num_refine_steps=config.num_refine_steps,
            refine_step_size=config.refine_step_size,
        )

        self.decoder = TiedTokenDecoder(
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size,
            bias=config.decoder_bias,
        )

        self.variance_reg = VarianceRegularization(margin=config.variance_margin)
        self.covariance_reg = CovarianceRegularization()
        self.target_centering = TargetCentering(
            dim=config.embed_dim, momentum=config.centering_momentum
        )
        self.diagnostics = CollapseDiagnostics()

        self.sigreg = SIGReg(
            embed_dim=config.embed_dim,
            n_sketches=config.sigreg_n_sketches,
            n_integration_points=config.sigreg_n_integration_points,
            sigma=config.sigreg_sigma,
        )

        self.jspace_metrics = JSpaceMetrics(
            variance_threshold=config.jspace_variance_threshold,
            k_workspace=config.jspace_k_workspace,
        )

        # JAWP — novel mechanism: Jacobian-Aligned Workspace Prediction
        if config.use_jawp:
            self.jawp = JAWPModule(
                embed_dim=config.embed_dim,
                k_start=config.jawk_k_start,
                k_end=config.jawk_k_end,
                curriculum_steps=config.jawk_curriculum_steps,
                alpha=config.jawk_alpha,
                init=config.jawk_init,
            )
        else:
            self.jawp = None

        # CGN — novel mechanism #6: Contextual Gating Network
        if config.use_cgn:
            self.cgn = ContextualGatingNetwork(
                embed_dim=config.embed_dim,
                n_groups=config.cgn_n_groups,
                tau_start=config.cgn_tau_start,
                tau_end=config.cgn_tau_end,
                anneal_steps=config.cgn_anneal_steps,
            )
        else:
            self.cgn = None

        # SWIP — novel mechanism #7: Selective Whitening with Information Preservation
        if config.use_swip:
            self.swip = SWIPModule(
                embed_dim=config.embed_dim,
                k_workspace=config.swip_k_workspace,
                target_variance=config.swip_target_variance,
                use_jawp_workspace=config.use_jawp,
            )
        else:
            self.swip = None

        # PCR — novel mechanism #8: Predictive Cascade Refinement
        if config.use_pcr:
            self.pcr = PredictiveCascadeRefinement(
                embed_dim=config.embed_dim,
                n_levels=config.pcr_n_levels,
                level_dims=config.pcr_level_dims,
            )
            self.pcr.warmup_steps = config.pcr_warmup_steps
        else:
            self.pcr = None

        # SPC — novel mechanism #9: Spectral Predictive Coding
        if config.use_spc:
            self.spc = SpectralPredictiveCoding(
                embed_dim=config.embed_dim,
                n_bands=config.spc_n_bands,
                init=config.spc_init,
            )
        else:
            self.spc = None

        # WSD — novel mechanism #10: Workspace-Target Synchronization Drift
        if config.use_wsd and config.use_jawp:
            jawp_k = config.jawk_k_end or max(config.embed_dim // 10, 1)
            self.wsd = WorkspaceSyncDrift(
                embed_dim=config.embed_dim,
                k=jawp_k,
                sync_interval=config.wsd_sync_interval,
                ema_beta=config.wsd_ema_beta,
            )
        else:
            self.wsd = None

        # CMC — novel mechanism #11: Cross-Mask Consistency Regularization
        if config.use_cmc:
            self.cmc = CrossMaskConsistency(
                embed_dim=config.embed_dim,
                second_mask_ratio=config.cmc_second_mask_ratio,
                min_overlap_ratio=config.cmc_min_overlap_ratio,
                mode=config.cmc_mode,
                interval=config.cmc_interval,
            )
        else:
            self.cmc = None

        # GAC — novel mechanism #12: Gradient-Allocated Capacity
        if config.use_gac:
            self.gac = GradientAllocatedCapacity(
                embed_dim=config.embed_dim,
                gamma=config.gac_gamma,
                tau_grad=config.gac_tau_grad,
                warmup_steps=config.gac_warmup_steps,
            )
        else:
            self.gac = None

        # STA — novel mechanism #13: Spectral Transport Alignment
        if config.use_sta:
            self.sta = SpectralTransportAlignment(
                embed_dim=config.embed_dim,
                eta=config.sta_eta,
                ema_beta=config.sta_ema_beta,
                warmup_steps=config.sta_warmup_steps,
                update_interval=config.sta_update_interval,
            )
        else:
            self.sta = None

        # PUC — novel mechanism #14: Prediction Uncertainty Calibration
        if config.use_puc:
            self.puc = PredictionUncertaintyCalibration(
                embed_dim=config.embed_dim,
                eta=config.puc_eta,
                ema_beta=config.puc_ema_beta,
                warmup_steps=config.puc_warmup_steps,
            )
        else:
            self.puc = None

        # RDC — novel mechanism #15: Representation Drift Compensation
        if config.use_rdc:
            self.rdc = RepresentationDriftCompensation(
                embed_dim=config.embed_dim,
                eta=config.rdc_eta,
                ema_beta=config.rdc_ema_beta,
                warmup_steps=config.rdc_warmup_steps,
                k_workspace=config.rdc_k_workspace,
            )
        else:
            self.rdc = None

        # Mechanism 16: WSR — Workspace Sharpness Regularization
        if getattr(config, "use_wsr", False):
            self.wsr = WorkspaceSharpnessRegularization(
                embed_dim=config.embed_dim,
                rho=getattr(config, "wsr_rho", 0.05),
                eta=getattr(config, "wsr_eta", 0.01),
                ema_beta=getattr(config, "wsr_ema_beta", 0.999),
                warmup_steps=getattr(config, "wsr_warmup_steps", 500),
                mode=getattr(config, "wsr_mode", "gradient"),
            )
        else:
            self.wsr = None

        # Wiring state for externally-orchestrated mechanisms (round-2 audit):
        self._gac_z = None  # live slot predictions, set when lambda_gac>0
        self._cmc_pass = None  # per-pass CMC bridge state (slots/idx/valid/mask)

    def update_target_encoder(self, tau):
        """EMA update: param_k <- tau * param_k + (1 - tau) * param_q.

        Micro-opts:
          1. precompute (1 - tau) outside loop
          2. @torch.no_grad to skip autograd tracking
          3. in-place mul_ + add_ (no intermediate allocation)
        I-JEPA pattern: called once per training step.
        """
        with torch.no_grad():
            one_minus_tau = 1.0 - tau
            for param_q, param_k in zip(
                self.encoder.parameters(), self.target_encoder.parameters()
            ):
                param_k.data.mul_(tau).add_(param_q.data, alpha=one_minus_tau)

    def _future_loss_weight(self, current_step):
        if self.config.future_warmup_steps <= 0:
            return self.config.lambda_future
        if current_step >= self.config.future_warmup_steps:
            return self.config.lambda_future
        progress = current_step / self.config.future_warmup_steps
        return self.config.lambda_future * progress

    def compute_loss_with_targets(
        self, masked_input_ids, original_input_ids, mask_positions, current_step=0, total_steps=1
    ):
        if masked_input_ids.size(0) == 0:
            zero = torch.tensor(0.0, device=masked_input_ids.device)
            return (
                zero,
                {
                    "loss": 0.0,
                    "loss_span": 0.0,
                    "loss_future": 0.0,
                    "loss_decoder": 0.0,
                    "loss_variance": 0.0,
                    "loss_covariance": 0.0,
                    "decoder_accuracy": 0.0,
                    "future_weight": 0.0,
                },
                {},
            )

        h_online, token_embeds_online = self.encoder(masked_input_ids)

        with torch.no_grad():
            self._prev_target_h = getattr(self, "_prev_target_h", None)
            h_target, _ = self.target_encoder(original_input_ids)
            h_target = self.target_centering(h_target)
            h_target = F.layer_norm(h_target, (h_target.size(-1),))

        # CGN: apply contextual gating before predictor
        # Routes information differently at masked vs visible positions
        cgn_info = {}
        if self.cgn is not None:
            h_online, cgn_info = self.cgn(h_online, mask_positions, step=current_step)

        span_preds, _num_masked, valid_mask, future_losses, _future_preds = self.predictor(
            h_online, mask_positions, token_embeds_online, h_target.detach()
        )

        # GAC: expose live slot predictions so the training loop can read
        # per-dimension gradient norms after backward (only when weighted).
        self._gac_z = None
        if (
            self.gac is not None
            and self.training
            and torch.is_grad_enabled()
            and self.config.lambda_gac > 0
        ):
            span_preds.retain_grad()
            self._gac_z = span_preds

        # PCR: Predictive Cascade Refinement — refine predictions in orthogonal subspaces
        # Bypasses information bottleneck of single-pass prediction (Theorem: Cascade Capacity)
        pcr_info = {}
        if self.pcr is not None and valid_mask.any():
            target_gathered_pcr, _, target_valid_pcr = TextSpanJPAPredictor._gather_masked(
                h_target.detach(), mask_positions
            )
            min_cols_pcr = min(valid_mask.size(1), target_valid_pcr.size(1))
            combined_valid_pcr = valid_mask[:, :min_cols_pcr] & target_valid_pcr[:, :min_cols_pcr]
            if combined_valid_pcr.any():
                span_preds_valid_pcr = span_preds[:, :min_cols_pcr][combined_valid_pcr]
                target_valid_pcr_flat = target_gathered_pcr[:, :min_cols_pcr][combined_valid_pcr]
                span_preds_refined, pcr_info = self.pcr(
                    span_preds_valid_pcr, target_valid_pcr_flat, step=current_step
                )
                # Write refined predictions back
                span_preds[:, :min_cols_pcr][combined_valid_pcr] = span_preds_refined

        # Zero-loss helper: avoids creating unnecessary computation graph nodes.
        # h_online.sum() * 0.0 still requires grad through h_online.
        # Instead, create a proper zero loss that participates in the graph.
        _zero_loss = h_online.new_tensor(0.0, requires_grad=True)
        if valid_mask.any():
            target_gathered, _, target_valid = TextSpanJPAPredictor._gather_masked(
                h_target.detach(), mask_positions
            )
            min_cols = min(valid_mask.size(1), target_valid.size(1))
            combined_valid = valid_mask[:, :min_cols] & target_valid[:, :min_cols]
            if combined_valid.any():
                if self.jawp is not None:
                    span_preds_valid = span_preds[:, :min_cols][combined_valid]
                    target_gathered_valid = target_gathered[:, :min_cols][combined_valid]
                    loss_span, jawp_info = self.jawp.compute_loss(
                        span_preds_valid, target_gathered_valid, step=current_step
                    )
                else:
                    loss_span = F.smooth_l1_loss(
                        span_preds[:, :min_cols][combined_valid],
                        target_gathered[:, :min_cols][combined_valid],
                    )
                    jawp_info = {}
            else:
                loss_span = _zero_loss
                jawp_info = {}
        else:
            loss_span = _zero_loss
            jawp_info = {}

        if len(future_losses) > 0:
            loss_future = sum(future_losses.values()) / len(future_losses)
        else:
            loss_future = torch.tensor(0.0, device=h_online.device, requires_grad=True)
        future_weight = self._future_loss_weight(current_step)

        loss_decoder = _zero_loss
        decoder_acc = torch.tensor(0.0, device=h_online.device)
        if mask_positions.any():
            target_tokens = original_input_ids[mask_positions.bool()]
            h_at_masked = h_online[mask_positions.bool()]
            if h_at_masked.size(0) > 0:
                logits = self.decoder(h_at_masked, self.encoder.token_embedding.weight)
                loss_decoder = F.cross_entropy(logits, target_tokens)
                decoder_acc = (logits.argmax(dim=-1) == target_tokens).float().mean()

        loss_variance = self.variance_reg(h_online)
        loss_covariance = self.covariance_reg(h_online)

        loss_sigreg = self._sigreg_loss(h_online)

        # Predictive Rank Regularization (prevents rank collapse in workspace)
        loss_pred_rank = self._pred_rank_loss(span_preds, valid_mask)

        # CGN orthogonality loss: encourage visible/masked gates to differ
        loss_cgn_ortho = self._cgn_ortho_loss(h_online)

        # SWIP: selective whitening that still lets SWIP shape the workspace Q
        loss_swip, swip_info = self._swip_loss(h_online)

        # SPC: frequency-band-aware prediction loss
        loss_spc, spc_info = self._spc_loss(
            span_preds, target_gathered, combined_valid, valid_mask, min_cols, h_target
        )

        # WSD: penalizes desynchronization between JAWP Q and target workspace
        loss_wsd, wsd_info = self._wsd_loss(h_online, h_target, current_step)

        # CMC: Cross-Mask Consistency Regularization.
        # The training loop runs a second forward with a different mask and calls
        # compute_cmc_between_passes(); stash what that bridge needs. Primary
        # slots are detached (module default stop_grad_primary=True keeps them
        # out of the gradient path anyway); validity uses predictor-side slots.
        self._cmc_pass = None
        if self.cmc is not None:
            _idx = self._slot_indices(mask_positions, span_preds.size(1))
            _valid_slots = torch.zeros_like(_idx, dtype=torch.bool)
            _mc = min(_idx.size(1), valid_mask.size(1))
            _valid_slots[:, :_mc] = valid_mask[:, :_mc]
            self._cmc_pass = {
                "slots": span_preds,  # live — secondary pass keeps its graph
                "slots_det": span_preds.detach(),  # primary side consumes this
                "idx": _idx,
                "valid": _valid_slots,
                "mask": mask_positions.detach(),
                "T": int(mask_positions.size(1)),
                "B": int(mask_positions.size(0)),
            }
        loss_cmc = _zero_loss
        cmc_info = {}

        # GAC: Gradient-Allocated Capacity
        # NOTE: GAC requires per-dim gradient norms from the main loss.
        # This is computed externally in the training loop after .backward().
        # Here we set loss_gac = 0 by default; the training loop adds it.
        loss_gac = _zero_loss
        gac_info = {}

        # STA: spectral transport alignment of h_online spectrum
        loss_sta, sta_info = self._sta_loss(h_online, current_step)

        # PUC: prediction uncertainty calibration on predictor output
        loss_puc, puc_info = self._puc_loss(
            h_online, span_preds, h_target, valid_mask, current_step
        )

        # RDC: representation drift compensation (running z_previous buffer)
        loss_rdc, rdc_info = self._rdc_loss(h_online, current_step)

        # WSR: sharpness penalty on the live workspace slice
        loss_wsr, wsr_info = self._wsr_loss(h_online, current_step)

        total_loss = (
            self.config.lambda_span * loss_span
            + future_weight * loss_future
            + self.config.lambda_decoder * loss_decoder
            + self.config.lambda_variance * loss_variance
            + self.config.lambda_covariance * loss_covariance
            + self.config.lambda_sigreg * loss_sigreg
            + self.config.lambda_predictive_rank * loss_pred_rank
            + self.config.lambda_cgn_ortho * loss_cgn_ortho
            + self.config.lambda_swip * loss_swip
            + self.config.lambda_spc * loss_spc
            + self.config.lambda_wsd * loss_wsd
            + self.config.lambda_cmc * loss_cmc
            + self.config.lambda_gac * loss_gac
            + self.config.lambda_sta * loss_sta
            + self.config.lambda_puc * loss_puc
            + self.config.lambda_rdc * loss_rdc
            + self.config.lambda_wsr * loss_wsr
        )

        loss_dict = {
            "loss": total_loss.item(),
            "loss_span": loss_span.item(),
            "loss_future": loss_future.item(),
            "loss_decoder": loss_decoder.item(),
            "loss_variance": loss_variance.item(),
            "loss_covariance": loss_covariance.item(),
            "loss_sigreg": loss_sigreg.item(),
            "loss_predictive_rank": loss_pred_rank.item(),
            "loss_cgn_ortho": loss_cgn_ortho.item(),
            "loss_swip": loss_swip.item(),
            "loss_spc": loss_spc.item(),
            "loss_wsd": loss_wsd.item(),
            "loss_cmc": loss_cmc.item(),
            "loss_gac": loss_gac.item(),
            "loss_sta": loss_sta.item(),
            "loss_puc": loss_puc.item(),
            "loss_rdc": loss_rdc.item(),
            "loss_wsr": loss_wsr.item(),
            "decoder_accuracy": decoder_acc.item(),
            "future_weight": future_weight,
        }
        for d, l in future_losses.items():
            loss_dict[f"loss_future_d{d}"] = l.item()
        if jawp_info:
            loss_dict.update({f"jawk_{k}": v for k, v in jawp_info.items()})
        if cgn_info:
            loss_dict.update({f"cgn_{k}": v for k, v in cgn_info.items()})
        if swip_info:
            loss_dict.update({f"swip_{k}": v for k, v in swip_info.items()})
        if spc_info:
            loss_dict.update({f"spc_{k}": v for k, v in spc_info.items()})
        if wsd_info:
            loss_dict.update({f"wsd_{k}": v for k, v in wsd_info.items()})
        if cmc_info:
            loss_dict.update({f"cmc_{k}": v for k, v in cmc_info.items()})
        if gac_info:
            loss_dict.update({f"gac_{k}": v for k, v in gac_info.items()})
        if sta_info:
            loss_dict.update({f"sta_{k}": v for k, v in sta_info.items()})
        if puc_info:
            loss_dict.update({f"puc_{k}": v for k, v in puc_info.items()})
        if rdc_info:
            loss_dict.update({f"rdc_{k}": v for k, v in rdc_info.items()})
        if wsr_info:
            loss_dict.update({f"wsr_{k}": v for k, v in wsr_info.items()})
        if pcr_info:
            loss_dict.update({f"pcr_{k}": v for k, v in pcr_info.items()})

        diag_dict = self.diagnostics.compute(
            h_online.detach(), h_target.detach(), prev_target_h=self._prev_target_h
        )
        diag_dict["target_center_norm"] = self.target_centering.center.norm().item()
        diag_dict["mask_fraction"] = mask_positions.float().mean().item()

        jspace_dict = self.jspace_metrics.compute(
            h_online.detach(), h_target.detach(), predictor_h=None
        )
        diag_dict.update(jspace_dict)

        if h_online.size(0) * h_online.size(1) >= 2:
            diag_dict["embedding_std_per_dim"] = h_online.std(dim=(0, 1)).mean().item()
        else:
            diag_dict["embedding_std_per_dim"] = 0.0

        # workspace_quality composite metric — single scalar health score
        diag_dict["workspace_quality"] = CollapseDiagnostics.workspace_quality(diag_dict)

        self._prev_target_h = h_target.detach().clone()
        return total_loss, loss_dict, diag_dict

    def compute_cmc_loss(self, z_pred_primary, z_pred_secondary, overlap_mask):
        """Compute CMC loss between predictions from two different masks.

        Call this from the training loop after running a second forward pass
        with a different mask pattern. The CMC loss enforces that predictions
        at overlapping masked positions agree across different masks.

        Args:
            z_pred_primary: (B, T, D) predictions from primary mask m₁.
            z_pred_secondary: (B, T, D) predictions from secondary mask m₂.
            overlap_mask: (B, T) binary — 1 if masked in BOTH m₁ and m₂.

        Returns:
            loss_cmc: scalar tensor.
            cmc_info: dict with diagnostics.
        """
        if self.cmc is None:
            zero = torch.tensor(0.0, device=z_pred_primary.device)
            return zero, {"cmc_loss": 0.0, "cmc_skipped": True}
        return self.cmc(z_pred_primary, z_pred_secondary, overlap_mask)

    @staticmethod
    def _slot_indices(mask_positions, num_slots):
        """(B,T) binary mask -> (B,num_slots) absolute masked positions, -1 padded."""
        B = mask_positions.size(0)
        idx = torch.full((B, num_slots), -1, dtype=torch.long, device=mask_positions.device)
        for b in range(B):
            mi = mask_positions[b].nonzero(as_tuple=True)[0][:num_slots]
            idx[b, : mi.numel()] = mi
        return idx

    def compute_cmc_between_passes(self, primary_pass, secondary_pass):
        """Bridge two compute_loss_with_targets passes into the CMC loss.

        Both passes store COMPACT slot predictions (B,S,D) ordered by masked
        position. This scatters them back to full sequence space via advanced
        indexing (never scatter_: -1 padding would wrap to the last token),
        restricts the overlap to positions valid in BOTH passes, and delegates
        to compute_cmc_loss (stop_grad_primary applies there).
        """
        B, T = secondary_pass["B"], secondary_pass["T"]
        D = secondary_pass["slots"].size(-1)

        def _scatter(pass_state, live):
            key = "slots" if live else "slots_det"
            z = pass_state[key].float()
            idx, valid = pass_state["idx"], pass_state["valid"]
            full = z.new_zeros(pass_state["B"], pass_state["T"], D)
            b, s = valid.nonzero(as_tuple=True)
            pos = idx[b, s]
            ok = pos >= 0
            full[b[ok], pos[ok]] = z[b[ok], s[ok]]
            return full

        def _position_valid(pass_state):
            pv = torch.zeros(pass_state["B"], pass_state["T"], dtype=torch.bool, device=z1.device)
            b, s = pass_state["valid"].nonzero(as_tuple=True)
            pos = pass_state["idx"][b, s]
            ok = pos >= 0
            pv[b[ok], pos[ok]] = True
            return pv

        z1 = _scatter(primary_pass, live=False)  # detached snapshot
        z2 = _scatter(secondary_pass, live=True)  # live: gradient flows to pass 2
        overlap = self.cmc.compute_overlap_mask(primary_pass["mask"], secondary_pass["mask"])
        overlap = overlap.bool() & _position_valid(primary_pass) & _position_valid(secondary_pass)
        if int(overlap.sum()) == 0:
            # Disjoint masks or all-invalid rows: skip instead of letting the
            # module's masked-mean see an empty reduction.
            zero = torch.tensor(0.0, device=z2.device)
            return zero, {"cmc_loss": 0.0, "cmc_skipped": True}
        return self.compute_cmc_loss(z1, z2, overlap)

    # ═══ Per-mechanism loss collectors (round-6 decomposition) ═══
    # Each returns (loss, info) — or a bare loss where no diagnostics exist.
    # Disabled mechanisms yield a graph-attached zero so the reducer shape
    # and autograd participation stay identical to the inline originals.

    def _zero(self, like):
        return like.new_tensor(0.0, requires_grad=True)

    def _sigreg_loss(self, h_online):
        if self.config.lambda_sigreg <= 0:
            return self._zero(h_online)
        return self.sigreg(h_online)

    def _pred_rank_loss(self, span_preds, valid_mask):
        if (
            self.config.lambda_predictive_rank <= 0
            or self.jawp is None
            or not valid_mask.any()
            or span_preds.size(0) <= 1
        ):
            return self._zero(span_preds)
        return self.jawp.predictive_rank_loss(span_preds)

    def _cgn_ortho_loss(self, h_online):
        if self.config.lambda_cgn_ortho <= 0 or self.cgn is None:
            return self._zero(h_online)
        probs_v = F.softmax(self.cgn.gate_logits_visible, dim=-1)[:, 1]
        probs_m = F.softmax(self.cgn.gate_logits_masked, dim=-1)[:, 1]
        cos_sim = F.cosine_similarity(probs_v.unsqueeze(0), probs_m.unsqueeze(0))
        return cos_sim.pow(2)  # minimize → gates orthogonal

    def _swip_loss(self, h_online):
        if self.config.lambda_swip <= 0 or self.swip is None:
            return self._zero(h_online), {}
        ws_Q = None
        if self.jawp is not None:
            k_active = int(self.jawp.active_k.item())
            ws_Q = self.jawp.workspace_Q[:, :k_active]  # live view: SWIP may shape Q
        return self.swip(h_online, workspace_Q=ws_Q)

    def _spc_loss(
        self, span_preds, target_gathered, combined_valid, valid_mask, min_cols, h_target
    ):
        if self.config.lambda_spc <= 0 or self.spc is None or not valid_mask.any():
            return self._zero(span_preds), {}
        spc_z_pred = (
            span_preds[:, :min_cols][combined_valid]
            if combined_valid.any()
            else span_preds.reshape(-1, span_preds.size(-1))
        )
        spc_z_target = (
            target_gathered[:, :min_cols][combined_valid]
            if combined_valid.any()
            else h_target.detach().reshape(-1, h_target.size(-1))
        )
        return self.spc(spc_z_pred, spc_z_target)

    def _wsd_loss(self, h_online, h_target, current_step):
        if self.wsd is None or self.jawp is None:
            return self._zero(h_online), {}
        k_active = int(self.jawp.active_k.item())
        Q_ws = self.jawp.workspace_Q[:, :k_active]
        return self.wsd.compute_drift(Q_ws, h_target=h_target.detach(), step=current_step)

    def _sta_loss(self, h_online, current_step):
        if self.sta is None:
            return self._zero(h_online), {}
        return self.sta(h_online, step=current_step)

    def _puc_loss(self, h_online, span_preds, h_target, valid_mask, current_step):
        if self.puc is None:
            return self._zero(h_online), {}
        if valid_mask.any() and span_preds.size(0) > 0 and span_preds.size(1) > 0:
            return self.puc(span_preds, z_target=h_target, step=current_step)
        return self.puc(h_online, z_target=h_target, step=current_step)

    def _rdc_loss(self, h_online, current_step):
        if self.rdc is None:
            return self._zero(h_online), {}
        ws_Q_rdc = None
        if self.jawp is not None:
            k_active_rdc = int(self.jawp.active_k.item())
            ws_Q_rdc = self.jawp.workspace_Q[:, :k_active_rdc]  # live view
        return self.rdc(h_online, workspace_Q=ws_Q_rdc, step=current_step)

    def _wsr_loss(self, h_online, current_step):
        if self.wsr is None or self.jawp is None:
            return self._zero(h_online), {}
        k_active_wsr = int(self.jawp.active_k.item())
        ws_Q_wsr = self.jawp.workspace_Q[:, :k_active_wsr]  # live view: WSR shapes Q
        return self.wsr(ws_Q_wsr, step=current_step)

    def get_num_params(self, non_embedding=True):
        enc = self.encoder.get_num_params(non_embedding)
        pred = self.predictor.get_num_params()
        dec = sum(p.numel() for p in self.decoder.parameters())
        return enc + pred + dec

    def get_num_params_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
