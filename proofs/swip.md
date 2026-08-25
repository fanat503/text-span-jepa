# SWIP: Selective Whitening with Information Preservation

> **IMPLEMENTATION STATUS (audited/reconciled 2026-08-24)** — see
> `proofs/IMPLEMENTATION_STATUS.md`. Term 2 is IMPLEMENTED as of R15;
> the v1 scale-invariance property is RETRACTED (fixed σ² target).

## Statement (matches `src/models/swip.py` as of R15)

Let λ₁ ≥ λ₂ ≥ … be the workspace eigenvalues (descending) and σᵢ the
background singular values. The implemented loss is:

$$\mathcal{L}_\mathrm{SWIP} = \underbrace{\sum_{i>k} \left(\log \sigma_i - \log \sigma_\mathrm{target}\right)^2}_{\text{background whitening}} + w_h \underbrace{\sum_{i=1}^{k-1} \operatorname{ReLU}(\lambda_{i+1} - \lambda_i + \delta)}_{\text{workspace hierarchy (R15)}}$$

with constructor parameters `hierarchy_margin = δ` (default 0.0) and
`hierarchy_weight = w_h` (default 1.0), applied identically on both code
paths (JAWP-projected spectrum and top-k PCA spectrum).

## Properties that hold

1. Non-negativity: sum of squared-log deviations and ReLU terms.
2. Background isotropy at optimum: λ-background term vanishes iff every
   background log-variance equals log σ².
3. Ordering enforcement: hierarchy term vanishes iff
   λ_{i+1} ≤ λ_i − δ for all consecutive pairs.
4. Operational orthonormality of F ∈ O(D): per-step SVD retraction wired
   in the training loop; `spc_ortho_error`-style diagnostic `ortho_err`
   exported every forward.

## RETRACTED v1 claims

1. ~~Scale-invariance: L(αZ) = L(Z)~~ — false: σ_target is fixed while a
   global rescale α multiplies every λᵢ and σᵢ, changing the background
   term (audit R11). Coordinate-wise convexity of the log-deviation term
   does hold; the earlier "unique minimizer" statement survives only
   per-coordinate.
2. ~~Unconditional "guaranteed orthonormality"~~ — holds operationally
   (per-step retraction); the pre-R14 fallback path was broken
   (`torch.linalg.q3r` typo) and could silently drop F off O(D).

## Notes

- On the JAWP path the background term uses a trace-based Jensen
  approximation (documented inline); diagnostics slice by the EFFECTIVE
  workspace width k_eff (R11 fix).
- Hierarchy margin δ > 0 enforces a minimum spectral gap inside the
  workspace; δ = 0 enforces plain descending order.
