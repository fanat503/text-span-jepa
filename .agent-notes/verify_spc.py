# SPC: does uniform weighting reproduce MSE, MSE/B, or B*MSE?
# proofs/spc.md:46-47 claims "uniform weights reproduce standard MSE/B".
# Read-only, provenance asserted.

import os
import sys

import torch

sys.path.insert(0, os.path.abspath("."))
import src  # noqa: E402

assert src.__file__.startswith(os.path.abspath(".")), f"wrong src: {src.__file__}"
from src.models.spc import SpectralPredictiveCoding  # noqa: E402

print("PROVENANCE src =", src.__file__)
D, N = 64, 256
torch.manual_seed(0)
zp = torch.randn(N, D)
zt = torch.randn(N, D)
mse = torch.nn.functional.mse_loss(zp, zt).item()
print(f"plain MSE = {mse:.6f}\n")

print(
    f"{'n_bands':>8}{'SPC(uniform w)':>16}{'SPC/MSE':>12}{'SPC/(B*MSE)':>13}"
    f"{'doc says MSE/B':>16}"
)
for B in (1, 2, 4, 8):
    m = SpectralPredictiveCoding(embed_dim=D, n_bands=B)
    with torch.no_grad():
        m.log_band_weights.fill_(0.0)  # uniform logits -> w_b = 1 after renorm
    loss, info = m(zp, zt)
    w = info.get("spc_band_weights")
    wsum = float(torch.as_tensor(w).sum()) if w is not None else float("nan")
    print(
        f"{B:>8}{loss.item():>16.6f}{loss.item() / mse:>12.4f}{loss.item() / (B * mse):>13.4f}"
        f"{B * mse:>16.6f}"
    )
    if w is not None:
        print(
            f"         band weights at uniform logits: {[round(float(x), 4) for x in w]}"
            f"  sum={wsum:.4f}  (proof property 1 says sum == B = {B})"
        )
        print(
            f"         min weight = {min(float(x) for x in w):.4f}"
            f"  (min_weight floor = {m.min_weight})"
        )
print()
print("doc claim  : uniform weights reproduce MSE/B  -> SPC/MSE should be 1/B")
print("test claim : 0.8 < SPC/(n_bands*MSE) < 1.2    -> SPC/MSE should be n_bands")
print("measured   : SPC/MSE tracks n_bands (column 3), not 1/n_bands.")
print("=> proofs/spc.md:46-47 has the scaling inverted by a factor of B^2;")
print("   the property as WRITTEN is false, the test it cites agrees with the code.")
