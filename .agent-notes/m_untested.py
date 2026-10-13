"""Which metric names does the test suite actually assert on?

UNTESTED = no occurrence of the metric's name anywhere under tests/. That is a
weaker condition than "nothing asserts it can change", so it is an upper bound
on the untested set: a name that IS mentioned may still only be pinned to a
constant. Both facts are reported; the grade in docs/METRICS.md uses the
measurement, not this count.
"""

from __future__ import annotations

import glob
import os
import re

ROOT = r"C:\dev\wt-metrics"
TESTDIR = os.path.join(ROOT, "tests")

blob = ""
files = sorted(glob.glob(os.path.join(TESTDIR, "*.py")))
for f in files:
    with open(f, encoding="utf-8", errors="replace") as fh:
        blob += fh.read()
blob_l = blob.lower()

NAMES = [
    # JAWP
    "loss_workspace", "loss_predictor_focus", "workspace_utilization", "target_ws_fraction",
    "workspace_cosine", "ortho_score", "predictive_relevance", "pca_alignment",
    "wip_score", "bg_fraction", "min_preservation", "max_preservation",
    "k_star", "k_gap", "k_mp", "max_gap_ratio", "mp_lower_bound", "sigma2_est",
    "ws_residual", "bg_residual", "bg_complexity_ratio",
    "effective_rank", "rank_utilization", "min_singular", "condition_number",
    "predictive_rank_loss", "gauge_norm", "principal_angles", "subspace_distance",
    # CGN
    "cgn_tau", "cgn_gate_diff", "cgn_sparsity", "cgn_entropy",
    "cgn_mean_gate_visible", "cgn_mean_gate_masked", "cgn_routing_gap",
    "routing_efficiency", "context_preservation", "prediction_focus",
    "orthogonality_score",
    # SWIP
    "anisotropy_ratio", "bg_uniformity", "ws_variance_fraction", "spectral_gap",
    "bg_snr", "power_law_alpha", "effective_rank",
    # PCR
    "pcr_improvement", "pcr_total_refinement_norm", "pcr_initial_residual",
    "pcr_final_residual", "pcr_ortho_score", "pcr_warmup_factor", "correction_norm",
    "subspace_fraction", "cascade_capacity_bound",
    # SPC
    "spc_weight_entropy", "spc_n_significant_bands", "spc_spectral_tilt",
    "spc_ortho_error", "spc_band_weights", "spc_band_predictability",
    "spc_band_residuals", "spc_band_losses", "spc_uniform_loss",
    # WSD
    "wsd_drift", "wsd_running_drift", "wsd_overlap", "wsd_spectral_alignment",
    "wsd_max_principal_angle", "wsd_mean_principal_angle",
    "wsd_steady_state_error", "wsd_nu_max_estimate", "wsd_convergence_time",
    "wsd_lambda_estimate",
    # CMC
    "cmc_overlap_ratio", "cmc_overlap_count", "cmc_per_position_loss",
    "cmc_max_inconsistency", "cmc_running_consistency", "cmc_skipped",
    # GAC
    "gac_n_starved", "gac_starved_fraction", "gac_mean_grad_norm",
    "gac_min_grad_norm", "gac_max_grad_norm", "gac_running_starved_fraction",
    # STA
    "sta_w1", "sta_spectral_gap", "sta_running_w1", "sta_running_spectral_gap",
    "sta_max_eigenvalue", "sta_min_eigenvalue", "sta_condition_number",
    "davis_kahan",
    # PUC
    "puc_entropy", "puc_target_entropy", "puc_entropy_deficit", "puc_overconfidence",
    "puc_min_eigenvalue", "puc_max_eigenvalue", "puc_log_det", "puc_n_components",
    "puc_carries_grad",
    # RDC
    "rdc_ortho_drift_norm", "rdc_workspace_drift_norm", "rdc_total_drift_norm",
    "rdc_drift_ratio", "rdc_theoretical_bound", "rdc_transient_bound",
    "rdc_stationary_bound",
    # WSR
    "wsr_sharpness", "wsr_spectral_sharpness", "wsr_directional_sharpness",
    "wsr_grad_norm", "wsr_grad_source", "wsr_gradient_substituted",
]

present, absent = [], []
print(f"scanned {len(files)} test files under tests/\n")
for n in sorted(set(NAMES)):
    hits = len(re.findall(re.escape(n.lower()), blob_l))
    if hits:
        present.append((n, hits))
    else:
        absent.append(n)

print(f"MENTIONED under tests/ ({len(present)}):")
for n, h in present:
    print(f"   {n:36s} {h:4d} occurrence(s)")
print(f"\nNOT MENTIONED under tests/ ({len(absent)}):")
for n in absent:
    print(f"   {n}")

# How many of the mentioned ones are only ever PINS (assert == constant)?
pin_only = []
for n, h in present:
    pat = re.compile(r"assert[^\n]*" + re.escape(n.lower()) + r"[^\n]*==[^\n]*", re.I)
    pin_hits = len(pat.findall(blob))
    nonpin = []
    for line in blob.splitlines():
        if n.lower() in line.lower() and "assert" in line and "==" not in line:
            nonpin.append(line.strip())
    if pin_hits and not nonpin:
        pin_only.append(n)
print(f"\nasserted only by equality against a constant, never by an inequality: {len(pin_only)}")
for n in pin_only:
    print(f"   {n}")