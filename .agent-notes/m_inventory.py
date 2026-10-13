"""The graded inventory, as data. Counts are DERIVED from this list, not typed.

Run:  python .agent-notes/m_inventory.py
Emits the per-mechanism table and the grade tallies used in docs/METRICS.md.
"""

from __future__ import annotations

MEANINGFUL = "MEANINGFUL"
DEGENERATE = "DEGENERATE"
MIS = "MIS-SCALED / TAUTOLOGICAL"
UNTESTED = "UNTESTED"

# (mech, metric, grade, sweep-that-was-run, verdict)
INV = [
    # ── JAWP ────────────────────────────────────────────────────────────────
    ("jawp", "loss_workspace", MEANINGFUL, "residual scale 0..100 -> 0 .. 1.02e4",
     "the Courant-Fischer objective itself; monotone in prediction error"),
    ("jawp", "loss_predictor_focus", MEANINGFUL, "residual scale 0..100 -> 0.889 .. 8978",
     "tracks the energy left outside span(Q)"),
    ("jawp", "k (active workspace width)", DEGENERATE, "curriculum step 0..5000 -> 1,1,12,23,24,24,24",
     "it moves, but it is a pure function of `step` and the k_start/k_end curriculum; "
     "it reads no data, so it is a schedule, not evidence. Log it once, not per step"),
    ("jawp", "workspace_utilization", MEANINGFUL, "z_pred energy placed 0..100% in span(Q) -> 0.0 .. 1.0",
     "exactly the quantity its name states"),
    ("jawp", "target_ws_fraction", MEANINGFUL, "z_target energy placed 0..100% in span(Q) -> 0.0 .. 1.0",
     "exactly the quantity its name states"),
    ("jawp", "workspace_cosine", MEANINGFUL, "z_pred = -z_target -> 1.0 .. -1.0",
     "spans the full cosine range under exact anti-correlation"),
    ("jawp", "ortho_score", MEANINGFUL, "off-diagonal mass injected 0..0.5 -> 1.0 .. 0.660",
     "moves; CONSTANT at 1.0 under a uniform ||Q|| rescale, which is correct "
     "(Q^TQ stays diagonal)"),
    ("jawp", "predictive_relevance", MEANINGFUL, "residual share w in span(Q) 0..1 -> 1.000 .. 0.000",
     "the D/k mis-scaling is fixed this wave and the fix holds under measurement"),
    ("jawp", "pca_alignment", MEANINGFUL, "Q = top-k PCA vs bottom-k PCA of z_target -> 1.0 / 0.0",
     "exactly 0 and exactly 1 at the two constructed endpoints"),
    ("jawp", "wip_score", MEANINGFUL, "Q spans top-k / orthogonal to it / random -> 1.0 / 0.0 / 0.066; "
     "explicit features rotated out of span(Q) -> 1.0 .. 0.0",
     "responds to the feature's actual alignment with span(Q)"),
    ("jawp", "bg_fraction", MIS, "|bg_fraction - (1 - wip_score)| <= 6.0e-08 over 8 random (Q,z) pairs",
     "an exact restatement of wip_score under a second name"),
    ("jawp", "min_preservation", MEANINGFUL, "Q spans top-k / orthogonal -> 1.0 / 0.0",
     "the per-feature minimum does move"),
    ("jawp", "max_preservation", MEANINGFUL, "Q spans top-k / orthogonal -> 1.0 / 0.130",
     "the per-feature maximum does move"),
    ("jawp", "k_star (spectral gap)", MIS, "true rank 1..24 residual -> 1,1,1,1,1,1; "
     "MP-favourable spectrum amp 0.1 -> 1,2,3,6; amp 0.5 -> all 1",
     "moves only when the signal is below ~0.29x the noise; pinned at 1 otherwise"),
    ("jawp", "k_gap", MEANINGFUL, "MP-favourable true rank 1..24 -> 1,2,3,6,12,24 exactly",
     "recovers the constructed k with zero error"),
    ("jawp", "k_mp", MIS, "signal/noise amplitude 1.0..0.4 -> k_mp = 0 at every level",
     "the Marchenko-Pastur branch is 0 for every plausible amplitude and only "
     "fires below 0.29; c_est is hard-coded at 0.5"),
    ("jawp", "max_gap_ratio", MIS, "true rank 1..24 -> 1.0175..1.0451, peak at rank 6 not at the true rank",
     "the documented 'min_gap_ratio' criterion is never applied; this is an "
     "unfalsifiable maximum"),
    ("jawp", "mp_lower_bound", MIS, "true rank 1..24 -> 0.0887..0.136",
     "moves, but with c_est pinned at 0.5 it is a rescaled median, not an MP bound"),
    ("jawp", "sigma2_est", MEANINGFUL, "true rank 1..24 -> 1.100 .. 1.588", "moves"),
    ("jawp", "ws_residual", MEANINGFUL, "residual placed 0..100% inside span(Q) -> 0.0 .. 0.994", "moves"),
    ("jawp", "bg_residual", MEANINGFUL, "same sweep -> 0.9208 .. 0.0", "moves"),
    ("jawp", "bg_complexity_ratio", MEANINGFUL, "same sweep -> 8.339 .. 0.0",
     "moves, but reads exactly 1.0 at zero ws_residual via a sentinel, and 1.0 is "
     "also a value the ratio genuinely takes near frac=0.03"),
    ("jawp", "effective_rank", MEANINGFUL, "workspace prediction of true rank 1..6 -> 1.000 .. 3.586", "moves"),
    ("jawp", "rank_utilization", MIS, "|rank_utilization - min(effective_rank/k,1)| = 0.0 at all 5 ranks",
     "an exact rescale of effective_rank by 1/k"),
    ("jawp", "min_singular", MEANINGFUL, "true rank 1..6 -> 1.0e-10,1.0e-10,1.0e-10,1.0e-10,7.34e-05",
     "moves, but is pinned at the 1e-10 clamp for every rank-deficient case"),
    ("jawp", "condition_number", MEANINGFUL, "true rank 1..6 -> 1.598e+05 .. 5.438e+10",
     "moves; when min_singular is clamped it degenerates to max_sv / 1e-10"),
    ("jawp", "predictive_rank_loss", MEANINGFUL, "z_pred scale 1e-3..100 -> 55.26 .. -55.27",
     "a signed -log det; moves monotonically"),
    ("jawp", "gauge_norm", MEANINGFUL, "gradient scale 0..100 -> 0.0 .. 562.7", "moves"),
    ("jawp", "max principal angle", MEANINGFUL, "one column rotated out of span(Q), 0..pi/2 -> 0.0 .. 1.5707963",
     "exact to 8 digits"),
    ("jawp", "subspace_distance", MEANINGFUL, "same -> 0.0 .. 1.0", "exact"),
    # ── CGN ────────────────────────────────────────────────────────────────
    ("cgn", "cgn_tau", DEGENERATE, "step 0..2000 -> 1.0, 0.978, 0.55, 0.100002, 0.1, 0.1",
     "a pure function of `step` and the annealing config; reads the data not at all"),
    ("cgn", "cgn_gate_diff", MEANINGFUL, "masked fraction 0..0.9 -> 0.848 .. 0.754; "
     "800 Adam steps -> 0.815 .. 0.801", "moves"),
    ("cgn", "cgn_sparsity", MEANINGFUL, "masked fraction 0..0.9 -> 0.500 .. 0.094", "moves"),
    ("cgn", "cgn_entropy", MEANINGFUL, "masked fraction 0..0.9 -> 5.487 .. 6.095", "moves"),
    ("cgn", "cgn_mean_gate_visible", MEANINGFUL, "masked fraction 0..0.9 -> 0.248 .. 0.106", "moves"),
    ("cgn", "cgn_mean_gate_masked", MEANINGFUL, "masked fraction 0..0.9 -> 0.000 .. 0.656", "moves"),
    ("cgn", "cgn_routing_gap", MEANINGFUL, "masked fraction 0..0.9 -> 0.248 .. 0.550", "moves"),
    ("cgn", "routing_efficiency", MIS, "masked fraction 0..1 -> 1.0, 0.25, 0.25, 0.25, 1.0; "
     "800 Adam steps -> 0.2489 .. 0.2530",
     "pinned at ~1/n_groups, and returns the sentinel 1.0 - the ideal value - "
     "whenever the mask is degenerate"),
    ("cgn", "context_preservation", MEANINGFUL, "masked fraction 0..1 -> 1.0, 0.25, 0.25, 0.25, 1.0",
     "moves; equals the mean squared gate"),
    ("cgn", "prediction_focus", MEANINGFUL, "same -> 1.0, 0.25, 0.25, 0.25, 1.0", "moves"),
    ("cgn", "compute_orthogonality_score", MIS, "ON-logit -20..20 -> 1.0, 0.987, 0.462, 0.0, 0.462, 0.987, 1.0",
     "measures decisiveness, not orthogonality; the code says so and the name does not"),
    # ── SWIP ───────────────────────────────────────────────────────────────
    ("swip", "anisotropy_ratio", MEANINGFUL, "five spectra -> 23.0, 1208.5, 1.2e5, 862.1, 1.9e9", "moves"),
    ("swip", "bg_uniformity", MEANINGFUL, "five spectra -> 0.688 .. 1.156", "moves"),
    ("swip", "ws_variance_fraction", MEANINGFUL, "five spectra -> 0.227 .. 1.000", "moves"),
    ("swip", "spectral_gap", MEANINGFUL, "five spectra -> 1.035 .. 2.279e6", "moves"),
    ("swip", "k_workspace", DEGENERATE, "five spectra -> 6,6,6,6,6", "constructor echo"),
    ("swip", "bg_dim", DEGENERATE, "five spectra -> 58,58,58,58,58", "a restatement of D - k_workspace"),
    ("swip", "swip_loss (background term)", MEANINGFUL, "three spectra -> 50.06, 1465.23, 948.66",
     "moves; and at MATCHED background mean it moves 46.8 -> 1197.9 across a 1x..8x "
     "anisotropy sweep, where the proof's own sum-of-squares form gives 21.9 -> 214.3"),
    ("swip", "bg_snr", MEANINGFUL, "three spectra -> 0.294, 969.08, 0.313", "moves"),
    ("swip", "power_law_alpha", MEANINGFUL, "three spectra -> 0.799, 2.811, 1.216", "moves"),
    ("swip", "condition_number (full diagnostics)", MIS,
     "|condition_number - anisotropy_ratio| = 0.0 at five spectra",
     "an exact duplicate of anisotropy_ratio under a second name"),
    ("swip", "effective_rank (full diagnostics)", MEANINGFUL,
     "three spectra -> 50.03, 5.873, 46.65", "moves"),
    ("swip", "bg_variance_fraction", MIS,
     "|bg_variance_fraction - (1 - ws_variance_fraction)| <= 1.7e-07 at five spectra",
     "an exact restatement of ws_variance_fraction"),
    ("swip", "spectral_gap_raw", MEANINGFUL, "three spectra -> 0.0323, 57.55, 4.393", "moves"),
    # ── PCR ────────────────────────────────────────────────────────────────
    ("pcr", "pcr_improvement", MEANINGFUL, "300 Adam steps on the heads -> 0.0, 0.0, 0.0160, 0.1917, 0.4987",
     "exactly 0 while the heads are at their zero init, and moves once they are trained - "
     "untrained, not degenerate"),
    ("pcr", "pcr_total_refinement_norm", MEANINGFUL, "same -> 0.0, 0.0, 2.045, 16.07, 39.58", "same caveat"),
    ("pcr", "pcr_initial_residual", MEANINGFUL, "prediction error 0..1 -> 0.0, 4.565, 18.258, 45.646, 91.291",
     "moves; it is a property of the data, not of the mechanism"),
    ("pcr", "pcr_final_residual", MEANINGFUL, "prediction error 0..1 -> 0.0, 4.565, 18.258, 45.646, 91.291; "
     "trained -> 27.15, 27.15, 26.72, 21.95, 13.61", "moves"),
    ("pcr", "pcr_ortho_score", MEANINGFUL, "off-diagonal mass injected 0..0.2 -> 1.0 .. 0.660",
     "moves, and CONSTANT at 1.0 under uniform ||Q|| rescale, which is correct"),
    ("pcr", "pcr_n_levels", DEGENERATE, "five inputs -> 3,3,3,3,3", "constructor echo"),
    ("pcr", "pcr_warmup_factor", DEGENERATE, "step 0..29 -> 1.0 at every step past warmup",
     "a pure function of `step`"),
    ("pcr", "pcr_level_{0,1,2}_correction_norm", MEANINGFUL,
     "300 Adam steps -> 0.0/0.0/0.769/6.690/15.92 (level 0)", "moves once trained"),
    ("pcr", "pcr_level_{0,1,2}_gate", MEANINGFUL,
     "300 Adam steps -> 0.500/0.500/0.504/0.532/0.570 (level 0)",
     "moves, but only within 0.5..0.57 because sigmoid(alpha_l) starts at 0"),
    ("pcr", "pcr_level_{0,1,2}_subspace_fraction", MEANINGFUL,
     "300 Adam steps -> 0.112, 0.112, 0.120, 0.180, 0.307 (level 1)", "moves once trained"),
    ("pcr", "pcr_level_{0,1,2}_dim", DEGENERATE, "five inputs -> 16,16,16,16,16 / 8,8,8,8,8",
     "a restatement of the constructor's level_dims"),
    ("pcr", "compute_cascade_capacity_bound", DEGENERATE,
     "prediction error 0, 0.1, 0.5, 1.0 -> 0.0, 10.7384, 10.7391, 10.7392",
     "a 10x change in the quantity it names moves it by a relative 7.5e-05 - "
     "a constant wearing a number's clothes"),
    # ── SPC ────────────────────────────────────────────────────────────────
    ("spc", "spc_total_loss", MEANINGFUL, "n_bands 1..16 -> SPC/MSE = 1.000, 2.000, 4.000, 8.000, 16.000",
     "moves; the Parseval scaling is B * MSE, not MSE/B"),
    ("spc", "spc_uniform_loss", MIS,
     "|spc_uniform_loss - mean(spc_band_residuals)| = 0.0 at n_bands = 1,2,4,8,16",
     "an exact function of the band-residual vector it is derived from"),
    ("spc", "spc_weight_entropy", MEANINGFUL,
     "band-structured residual, 0/5/20/500 adapt_weights calls -> 2.079433, 2.078956, 2.074306, 0.974160",
     "pinned at log(n_bands) under an ISOTROPIC residual (the correct response to an "
     "isotropic residual) and moves to 0.974 once the residual is band-structured"),
    ("spc", "spc_n_significant_bands", MEANINGFUL,
     "same -> 8, 8, 8, 2",
     "uniform band weights are exactly 1.0 so every band clears the >0.5 test; it "
     "only becomes informative once the weights actually move"),
    ("spc", "spc_spectral_tilt", MEANINGFUL,
     "same -> -9.5e-07, 0.024382, 0.078834, -2.326443",
     "moves only with band-structured residuals, like the weights it summarises"),
    ("spc", "spc_ortho_error", MEANINGFUL,
     "off-diagonal injected into freq_basis, eps 0..0.1 -> 4.34e-06 .. 1.067",
     "reads ~4e-06 on a fresh DCT basis and tracks real loss of orthonormality"),
    ("spc", "spc_band_weights", MEANINGFUL,
     "500 adapt calls on a band-structured residual -> 1.0x8 -> 6.13, 0.52, 0.42, 0.22, 0.21, 0.17, 0.17, 0.16",
     "the weight vector itself"),
    ("spc", "spc_band_predictability", MEANINGFUL,
     "band-structured residual -> per band 7e-05, 0.00927, ..., 0.00990",
     "genuinely per-band; the near-uniform values under an isotropic residual are "
     "the correct response, not a bug"),
    ("spc", "spc_band_residuals", MEANINGFUL, "n_bands 1..16", "moves"),
    ("spc", "spc_band_losses", MIS, "|spc_band_loss - spc_band_residual| = 0.0 on all 31 band entries",
     "an exact duplicate of spc_band_residuals"),
    # ── WSD ────────────────────────────────────────────────────────────────
    ("wsd", "wsd_drift", MEANINGFUL, "Q rotated 0..1.5 rad from the target -> 0.0 .. 1.411",
     "monotone in the rotation"),
    ("wsd", "wsd_drift_sq", MEANINGFUL, "same -> 0.0 .. 1.990", "moves"),
    ("wsd", "wsd_running_drift", MEANINGFUL, "same -> 0.0 .. 0.0556", "moves"),
    ("wsd", "wsd_overlap", MEANINGFUL, "same -> 1.000 .. 0.833", "moves"),
    ("wsd", "wsd_spectral_alignment", MEANINGFUL, "same -> 1.000 .. 0.845", "moves"),
    ("wsd", "wsd_max_principal_angle", MEANINGFUL, "same -> 0.0 .. 1.503", "moves"),
    ("wsd", "wsd_mean_principal_angle", MEANINGFUL, "same -> 0.0 .. 0.288", "moves"),
    ("wsd", "wsd_k", DEGENERATE, "six inputs -> 6,6,6,6,6,6", "constructor echo"),
    ("wsd", "wsd_steady_state_error", MIS,
     "|wsd_steady_state_error - wsd_running_drift| <= 1.4e-17 over 60 (interval, step) pairs",
     "nu/lambda with nu = rd/T and lambda = 1/T; T cancels and the key is a "
     "second name for the running-drift EMA"),
    ("wsd", "wsd_nu_max_estimate", MIS, "rotation sweep -> 0.0 .. 0.0111; interval 1..1000 -> 0 .. 1",
     "it is running_drift / sync_interval, i.e. a misalignment divided by a time, "
     "not a rate of change"),
    ("wsd", "wsd_convergence_time", DEGENERATE, "six inputs -> 5.0,5.0,5.0,5.0,5.0,5.0 (sync_interval=5)",
     "it is 1/lambda_estimate = sync_interval, a restatement of the config"),
    ("wsd", "wsd_lambda_estimate", MIS, "sync_interval 1,2,5,10,100,500 -> 1.0, 0.5, 0.2, 0.1, 0.01, 0.002",
     "it is 1/sync_interval. It equals the true lambda_wsd only because the shipped "
     "defaults happen to make them equal"),
    # ── CMC ────────────────────────────────────────────────────────────────
    ("cmc", "cmc_loss", MEANINGFUL, "prediction disagreement 0..5 -> 0.0, 0.620, 15.49, 61.97, 1549.27", "moves"),
    ("cmc", "cmc_per_position_loss", MIS, "|cmc_loss - cmc_per_position_loss| = 0.0 at all 5 levels",
     "cmc.py:476 and cmc.py:481 are the same expression"),
    ("cmc", "cmc_max_inconsistency", MEANINGFUL, "same -> 0.0, 0.926, 23.15, 92.60, 2314.90", "moves"),
    ("cmc", "cmc_overlap_count", MEANINGFUL, "true both-masked positions 0..T -> 0, 12, 32, 64, 96, 128", "moves"),
    ("cmc", "cmc_overlap_ratio", MIS,
     "true both-masked positions 0..T -> 0.0, 0.094, 0.25, 0.50, 0.75, 1.00; "
     "repo span masks at the shipped 0.15/0.35 -> 0.047 .. 0.164",
     "it is |Omega| / (B*T), the fraction of ALL positions, not |Omega| / |masked| "
     "as cmc.md defines it; and at the shipped mask ratios the min_overlap_ratio=0.2 "
     "gate skips 6 of 6 combinations"),
    ("cmc", "cmc_running_consistency", MEANINGFUL, "disagreement 0..5 -> 0.0 .. 15.49", "moves"),
    ("cmc", "cmc_skipped", MEANINGFUL, "9 shipped-ratio span-mask pairs -> 6 True, 3 False",
     "the only discriminator between 'agreement' and 'did not run'; absent from the "
     "skip-path key set only in that cmc_running_consistency is missing there"),
    ("cmc", "compute_downstream_stability_bound", MEANINGFUL, "cmc_loss 0..100 -> 0.0, 0.01, 0.1, 1.0, 10.0",
     "moves; it is ||w||*sqrt(eps), and the pointwise-vs-averaged caveat is documented"),
    ("cmc", "compute_representation_variance_bound", MEANINGFUL,
     "cmc_loss 0..100 -> 1.0, 1.000000005, 1.00005, 1.005, 1.5, 51.0", "moves"),
    # ── GAC ────────────────────────────────────────────────────────────────
    ("gac", "gac_loss", MEANINGFUL,
     "starved fraction 0, .25, .5, .9, 1.0 -> 0.0, 1.60e-05, 3.20e-05, 5.73e-05, 6.39e-05; "
     "batch size 1..16 -> 6.315e-05 .. 6.309e-05",
     "moves with the starved set and is INVARIANT to batch size, which is the "
     "measured form of the documented 1/N energy term"),
    ("gac", "gac_n_starved", MEANINGFUL, "same -> 0, 16, 32, 57, 64", "moves"),
    ("gac", "gac_starved_fraction", MEANINGFUL, "same -> 0.0, 0.25, 0.50, 0.891, 1.0", "moves"),
    ("gac", "gac_warmup_factor", DEGENERATE, "step 0..2000 -> absent, 0.001, 0.5, 0.999, 1.0, 1.0",
     "a pure function of `step` and the warmup config"),
    ("gac", "gac_mean_grad_norm", MEANINGFUL, "starved sweep -> 1.0, 0.750, 0.500, 0.109, 1e-06", "moves"),
    ("gac", "gac_min_grad_norm", MEANINGFUL, "same -> 1.0, 1e-06, 1e-06, 1e-06, 1e-06", "moves"),
    ("gac", "gac_max_grad_norm", MEANINGFUL, "same -> 1.0, 1.0, 1.0, 1.0, 1e-06", "moves"),
    ("gac", "gac_running_starved_fraction", MIS, "same -> 0.0, 0.0025, 0.005, 0.0089, 0.01",
     "an EMA of gac_starved_fraction divided by (1 - 0.99); a smoothed second name"),
    ("gac", "gac_warmup", DEGENERATE, "step 1..2000 -> False at every step",
     "a boolean restatement of gac_warmup_factor < 1"),
    # ── STA ────────────────────────────────────────────────────────────────
    ("sta", "sta_w1", MEANINGFUL, "spectrum rescaled 1x..4.5x over 8 steps -> 4.6e-07 .. 23.08", "moves"),
    ("sta", "sta_loss", MEANINGFUL, "same -> 9.3e-12 .. 0.00369",
     "moves, but requires_grad == False: it scales a constant"),
    ("sta", "sta_warmup_factor", DEGENERATE, "step 0..2000 -> 0.0, 0.002, 1.0, 1.0, 1.0, 1.0, 1.0",
     "a pure function of `step`"),
    ("sta", "sta_warmup", DEGENERATE, "step 0..2000 -> True, True, False x5",
     "a boolean restatement of the warmup factor"),
    ("sta", "sta_spectral_gap", MEANINGFUL, "four constructed spectra -> 0.00308, 0.000183, 0.07397, 0.00348",
     "moves, and it is NOT the delta inside the bound: the min over significant pairs "
     "reads 0.00308 where the k/k+1 pair reads 0.000104 at the same call"),
    ("sta", "sta_running_w1", MEANINGFUL, "eight steps -> 4.6e-09 .. 0.757", "moves"),
    ("sta", "sta_running_spectral_gap", MEANINGFUL, "same -> 8.8e-05 .. 0.00311", "moves"),
    ("sta", "sta_max_eigenvalue", MEANINGFUL, "same -> 6.55 .. 68.46", "moves"),
    ("sta", "sta_min_eigenvalue", MEANINGFUL, "same -> 0.247 .. 2.886", "moves"),
    ("sta", "sta_condition_number", MEANINGFUL, "same -> 26.48 .. 23.71 (range 42.41)", "moves"),
    ("sta", "compute_davis_kahan_bound", MIS,
     "signal amplitude 0.01..50 -> 6.0e-08 .. 1.0e-05; k = 2 vs k = 6 -> 1.29e-08 vs 2.90e-08",
     "it tracks running_w1 almost exactly (bound/w1 = 6.7..13.9) and is floored at "
     "w1/eps whenever the gap is absent, so a large value is the ABSENCE of a gap"),
    ("sta", "compute_downstream_stability_bound", MEANINGFUL,
     "signal amplitude 0..100 -> 1.426, 1.414, 1.378, 7.292, 1.425, 129.57", "moves"),
    # ── PUC ────────────────────────────────────────────────────────────────
    ("puc", "puc_entropy", MIS,
     "spectrum decay 0..8, true H = 80.3 .. -97.6, reported 98.3 .. 81.1, error 18.0 .. 178.7",
     "softplus floors every small eigenvalue at log 2, so the reported entropy is "
     "dominated by that floor; the error grows with the decay it is meant to measure"),
    ("puc", "puc_entropy_deficit", MEANINGFUL, "decay 0..8 -> 0.0, 0.0, 0.434, 4.323, 7.782, 9.700",
     "moves, but one-sided: it saturates at 0 and inherits the entropy bias"),
    ("puc", "puc_overconfidence", MIS,
     "|puc_overconfidence - puc_entropy_deficit / puc_target_entropy| = 0.0 at decay 0,1,2,4,8",
     "an exact rescale of the deficit by a constant"),
    ("puc", "puc_target_entropy", DEGENERATE, "four inputs -> 90.81206612509905 x4",
     "a restatement of D/2 * log(2*pi*e)"),
    ("puc", "puc_loss", MIS,
     "decay 0..8 -> 0.0, 0.0, 0.10014, 0.10135, 0.10243, 0.10303",
     "a true entropy range of 80 -> -98 nats produces a 2.9% change in the loss"),
    ("puc", "puc_log_det", MEANINGFUL, "five spectra -> 14.70, 25.41, 39.70, 59.00", "moves"),
    ("puc", "puc_min_eigenvalue", MEANINGFUL,
     "spectrum amplitude 0.01..1000 -> 0.693168 .. 0.934423",
     "moves, but its floor is softplus(0) = log 2 = 0.693: no eigenvalue below that "
     "is ever reported, however small it is"),
    ("puc", "puc_max_eigenvalue", MEANINGFUL, "four spectra -> 2.80 .. 3541.85", "moves"),
    ("puc", "puc_n_components", DEGENERATE, "embed_dim 32/64/128/256 -> 32, 64, 64, 64",
     "min(D, 64) - a restatement of the config, and it does not scale with D"),
    ("puc", "puc_warmup_factor", DEGENERATE, "step 0..2000 -> 0.0, 0.002, 0.5, 0.998, 1.0, 1.0, 1.0",
     "a pure function of `step`"),
    ("puc", "puc_carries_grad", MEANINGFUL,
     "differentiable path -> True, inert EMA path -> False",
     "a valid discriminator, but False whenever the loss is 0 because the gate is "
     "closed, so it conflates 'no gradient needed' with 'inert path'"),
    # ── RDC ────────────────────────────────────────────────────────────────
    ("rdc", "rdc_ortho_drift_norm", MEANINGFUL, "drift placed 0..100% inside span(Q) -> 1525.9 .. 1.94e-13", "moves"),
    ("rdc", "rdc_workspace_drift_norm", MEANINGFUL, "same -> 2.05e-13 .. 487.6", "moves"),
    ("rdc", "rdc_total_drift_norm", MEANINGFUL, "same -> 1525.9 .. 487.6", "moves"),
    ("rdc", "rdc_drift_ratio", MEANINGFUL, "same -> 1.0 .. 3.98e-16", "moves"),
    ("rdc", "rdc_loss", MEANINGFUL, "same -> 23284.4 .. 3.77e-28", "moves"),
    ("rdc", "rdc_transient_bound", MIS, "step 10..50000 -> 1.77, 431.6, 411.8, 12.12, 2.2e-03, 1.14e-15, "
     "6.93e-37, 3.49e-36",
     "at step 10000 it reads 6.9e-37 while the orthogonal drift on the same call "
     "reads 7626 - a 34-order-of-magnitude understatement that looks like a proof "
     "of vanishing drift; it also stops moving past the 10000-step T_eff cap"),
    ("rdc", "rdc_stationary_bound", MIS, "step 10..50000 -> 19.4 .. 1.54e6",
     "it is eps_ema * (1-eta)/(eta*sqrt(k)) with eps_ema a running EMA, so it is a "
     "second name for the drift EMA times a constant"),
    ("rdc", "rdc_theoretical_bound", MIS, "identical series to rdc_transient_bound",
     "min(transient, stationary); it inherits the transient bound's underflow verbatim"),
    ("rdc", "rdc_warmup_factor", DEGENERATE, "step 0..600 -> 0.002, 0.2, 0.998, 1.0, 1.0",
     "a pure function of `step`"),
    ("rdc", "rdc_k_workspace", DEGENERATE, "five inputs -> 6,6,6,6,6", "constructor echo"),
    ("rdc", "rdc_eta", DEGENERATE, "five inputs -> 0.01,0.01,0.01,0.01,0.01", "constructor echo"),
    # ── WSR ────────────────────────────────────────────────────────────────
    ("wsr", "wsr_sharpness", MEANINGFUL, "lagged gradient 0..100 -> 0.0, 3.6e-09, 3.6e-05, 0.00363, 0.363, 36.30",
     "moves; requires_grad == False, so the loss that consumes it cannot train"),
    ("wsr", "wsr_spectral_sharpness", MIS, "|wsr_sharpness - wsr_spectral_sharpness| = 0.0 at all 6 levels",
     "character-identical to wsr_sharpness; Theorem 3's spectral term is never computed"),
    ("wsr", "wsr_directional_sharpness", MEANINGFUL,
     "same -> 0.0, 1.33e-09, 1.33e-05, 0.00133, 0.1332, 13.32",
     "moves, and is a genuine second component - it is never reported as part of a sum"),
    ("wsr", "wsr_grad_norm", MEANINGFUL, "same -> 0.0 .. 1778.3", "moves"),
    ("wsr", "wsr_loss", MEANINGFUL, "same -> 0.0, 1.45e-13, 1.45e-09, 1.45e-07, 1.45e-05, 1.45e-03",
     "moves; requires_grad == False in both modes"),
    ("wsr", "wsr_sharpness (sam mode)", MEANINGFUL,
     "on a genuine Q-dependent loss_fn -> 0.0, 0.0015095, 0.0015095, 0.0015095",
     "the perturbation is normalised, so it responds to the landscape, not the "
     "gradient norm; at |grad| = 0 it is exactly 0"),
    ("wsr", "wsr_loss_current (sam mode)", MEANINGFUL, "the same calls", "reads L(Q)"),
    ("wsr", "wsr_loss_perturbed (sam mode)", MEANINGFUL, "the same calls -> 1.8925, 1.8940, 1.8940, 1.8940",
     "reads L(Q + Delta)"),
    ("wsr", "wsr_rho", DEGENERATE, "five inputs -> 0.05,0.05,0.05,0.05,0.05", "constructor echo"),
    ("wsr", "wsr_warmup_factor", DEGENERATE, "step 0..900 -> 0.0, 0.2, 0.998, 1.0, 1.0",
     "a pure function of `step`"),
    ("wsr", "wsr_warmup", DEGENERATE, "five inputs -> False x5", "a boolean restatement of the warmup factor"),
    ("wsr", "wsr_grad_source", MEANINGFUL, "lagged -> 'lagged', none supplied -> 'unavailable'",
     "the one provenance flag that is genuinely load-bearing and correct"),
    ("wsr", "wsr_gradient_substituted", MEANINGFUL, "same -> False / True",
     "the boolean form of wsr_grad_source"),
]

GRADES = [MEANINGFUL, DEGENERATE, MIS, UNTESTED]


def _tests_blob():
    import glob

    out = ""
    for f in sorted(glob.glob(r"C:\dev\wt-metrics\tests\*.py")):
        with open(f, encoding="utf-8", errors="replace") as fh:
            out += fh.read()
    return out.lower()


def main():
    by_mech = {}
    for mech, name, grade, _sweep, _why in INV:
        by_mech.setdefault(mech, []).append((name, grade))

    tally = {g: 0 for g in GRADES}
    for _m, _n, grade, _s, _w in INV:
        tally[grade] += 1

    total = len(INV)
    blob = _tests_blob()

    def named(name):
        stem = name.split("(")[0].split("{")[0].strip().lower()
        stem = stem.replace("compute_", "")
        return stem in blob

    print("=" * 96)
    print("GRADED TALLY")
    print("=" * 96)
    for g in GRADES:
        print(f"  {g:28s} {tally[g]:4d}   {100.0 * tally[g] / total:5.1f}%")
    print(f"  {'TOTAL':28s} {total:4d}")

    untested = [(m, n, g) for m, n, g, _s, _w in INV if not named(n)]
    untested_meaningful = [x for x in untested if x[2] == MEANINGFUL]
    print()
    print("=" * 96)
    print("UNTESTED (orthogonal to the grade: the metric's NAME never appears in tests/)")
    print("=" * 96)
    print(f"  untested names, all grades        {len(untested):4d}")
    print(f"  untested names graded MEANINGFUL   {len(untested_meaningful):4d}")
    for m, n, g in untested:
        print(f"    {m:6s} {n:44s} {g}")
    print()
    print("=" * 96)
    print(f"{'MECH':6s} {'METRIC':44s} GRADE")
    print("=" * 96)
    for mech in ["jawp", "cgn", "swip", "pcr", "spc", "wsd", "cmc", "gac", "sta", "puc", "rdc", "wsr"]:
        for name, grade in by_mech.get(mech, []):
            print(f"{mech:6s} {name:44s} {grade}")
        print(f"{'':6s} {'-' * 44} ({len(by_mech.get(mech, []))} metrics, "
              f"{sum(1 for _n, g in by_mech.get(mech, []) if g == MEANINGFUL)} meaningful)")
        print()


if __name__ == "__main__":
    main()