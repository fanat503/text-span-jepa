"""Host-key inventory — metrics inventory driver 6 of 5.

What key does a reviewer actually read in a training log? `MechanismBundle.forward`
and `TextSpanJEPA.forward` both do `info.update({f"{mech}_{k}": v ...})`, so a
module whose info dict already self-prefixes its keys produces a DOUBLE prefix.
Measured here through the real host path, not inferred.
"""

from __future__ import annotations

from mh import guard, record, save  # noqa: E402

guard()

import torch  # noqa: E402

from src.models.mechanisms import MechanismBundle  # noqa: E402

B, T, D = 4, 32, 64

torch.manual_seed(1234)
z = torch.randn(B, T, D)
zt = torch.randn(B, T, D)
mask = torch.zeros(B, T, dtype=torch.bool)
mask[:, 4:12] = True

bundle = MechanismBundle(
    embed_dim=D,
    use_jawp=True,
    jawk_k_start=6,
    jawk_k_end=6,
    use_cgn=True,
    cgn_n_groups=4,
    use_swip=True,
    swip_k_workspace=6,
    use_pcr=True,
    pcr_n_levels=3,
    use_spc=True,
    spc_n_bands=4,
    use_wsd=True,
    wsd_k=6,
    wsd_sync_interval=5,
    use_gac=True,
    use_sta=True,
    use_puc=True,
)

out, info = bundle(z, zt, mask, step=3000)
print(f"\nMechanismBundle.forward produced {len(info)} keys:\n")
DOUBLE = []
for k in sorted(info):
    parts = k.split("_")
    dup = len(parts) > 1 and parts[0] == parts[1]
    flag = "  <-- DOUBLE PREFIX" if dup else ""
    if dup:
        DOUBLE.append(k)
    print(f"   {k:44s}{flag}")

print(f"\n{len(DOUBLE)} of {len(info)} keys are double-prefixed: {len(DOUBLE)}")
record("host", "double-prefixed info keys", [float(len(DOUBLE))],
       "MechanismBundle.forward with all 11 forward-path mechanisms on")
print(f"\nDouble-prefixed keys ({len(DOUBLE)}):")
for k in DOUBLE:
    print(f"   {k}   (a reviewer looking for `{k}` has to know the module self-prefixes)")

save(r".agent-notes\res_host.json")