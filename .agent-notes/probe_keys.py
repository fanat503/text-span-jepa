# Probe: does a mechanism module's early-return path return a SHORTER info
# key set than its normal path?  Drives every one of the 12 modules at step 0
# (inside warmup) and at a later step (past warmup) and diffs the key sets.
#
# No training.  No optimiser step.  Pure forward calls.

from __future__ import annotations

import os
import sys

ROOT = r"C:\dev\wt-keys"
# An editable install on this machine points at a DIFFERENT clone; force the
# worktree to win before `import src`.
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402

import src  # noqa: E402

assert src.__file__.startswith(ROOT), src.__file__
print("src.__file__ =", src.__file__)

from src.models.cgn import ContextualGatingNetwork  # noqa: E402
from src.models.cmc import CrossMaskConsistency  # noqa: E402
from src.models.gac import GradientAllocatedCapacity  # noqa: E402
from src.models.jawp import JAWPModule  # noqa: E402
from src.models.pcr import PredictiveCascadeRefinement  # noqa: E402
from src.models.puc import PredictionUncertaintyCalibration  # noqa: E402
from src.models.rdc import RepresentationDriftCompensation  # noqa: E402
from src.models.spc import SpectralPredictiveCoding  # noqa: E402
from src.models.sta import SpectralTransportAlignment  # noqa: E402
from src.models.swip import SWIPModule  # noqa: E402
from src.models.wsd import WorkspaceSyncDrift  # noqa: E402
from src.models.wsr import WorkspaceSharpnessRegularization  # noqa: E402

torch.manual_seed(0)

D = 16
B, T = 4, 8
LATE = 5000  # a step far past every shipped warmup_steps


def zt(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g)


def run(name, build, drive):
    """Drive `build()`'s module at step 0 and step LATE, diff the info keys."""
    mod = build()
    mod.train()
    i0 = drive(mod, 0)
    iL = drive(mod, LATE)
    k0, kL = set(i0), set(iL)
    missing = kL - k0  # keys the warmup path does not carry
    extra = k0 - kL
    status = "SHRINKS" if missing else ("ok" if not extra else "GROWS")
    print(f"{name:5s} {status:7s} warmup={len(k0):2d} late={len(kL):2d}")
    if missing:
        print(f"      missing on early return: {sorted(missing)}")
    if extra:
        print(f"      only on early return  : {sorted(extra)}")
    return name, missing


results = []

# ── jawp ────────────────────────────────────────────────────────────────────
def _jawp_drive(m, step):
    a, b = zt(1 + step), zt(2 + step)
    loss, info = m.compute_loss(a, b, step=step)
    return info


results.append(run("jawp", lambda: JAWPModule(embed_dim=D, k_start=2, k_end=4, curriculum_steps=0), _jawp_drive))

# ── cgn ─────────────────────────────────────────────────────────────────────
def _cgn_drive(m, step):
    g = torch.Generator().manual_seed(7)
    mask = (torch.rand(B, T, generator=g) < 0.4).long()
    _, info = m(zt(3 + step), mask, step=step)
    return info


results.append(run("cgn", lambda: ContextualGatingNetwork(embed_dim=D, n_groups=4), _cgn_drive))

# ── swip ────────────────────────────────────────────────────────────────────
def _swip_drive(m, step):
    Q = torch.linalg.qr(torch.randn(D, 4, generator=torch.Generator().manual_seed(11)))[0]
    _, info = m(zt(4 + step), workspace_Q=Q)
    return info


results.append(run("swip", lambda: SWIPModule(embed_dim=D, k_workspace=4), _swip_drive))

# ── pcr ─────────────────────────────────────────────────────────────────────
def _pcr_drive(m, step):
    _, info = m(zt(5 + step), zt(6 + step), step=step)
    return info


results.append(run("pcr", lambda: PredictiveCascadeRefinement(embed_dim=D, n_levels=2), _pcr_drive))

# ── spc ─────────────────────────────────────────────────────────────────────
def _spc_drive(m, step):
    _, info = m(zt(8 + step), zt(9 + step))
    return info


results.append(run("spc", lambda: SpectralPredictiveCoding(embed_dim=D, n_bands=4), _spc_drive))

# ── wsd ─────────────────────────────────────────────────────────────────────
def _wsd_drive(m, step):
    Q = torch.linalg.qr(torch.randn(D, 4, generator=torch.Generator().manual_seed(13)))[0]
    _, info = m.compute_drift(Q, zt(10 + step), step=step)
    return info


results.append(run("wsd", lambda: WorkspaceSyncDrift(embed_dim=D, k=4, sync_interval=100), _wsd_drive))

# ── cmc ─────────────────────────────────────────────────────────────────────
def _cmc_drive(m, step):
    z1, z2 = zt(12 + step), zt(13 + step)
    ov = torch.zeros(B, T, dtype=torch.long)
    ov[:, 1:4] = 1
    _, info = m(z1, z2, ov)
    return info


results.append(run("cmc", lambda: CrossMaskConsistency(embed_dim=D), _cmc_drive))

# ── gac ─────────────────────────────────────────────────────────────────────
def _gac_drive(m, step):
    g = torch.Generator().manual_seed(17)
    grad_norms = torch.rand(D, generator=g) * 1e-2
    _, info = m(zt(14 + step), grad_norms, step=step)
    return info


results.append(run("gac", lambda: GradientAllocatedCapacity(embed_dim=D, warmup_steps=1000), _gac_drive))

# ── sta ─────────────────────────────────────────────────────────────────────
def _sta_drive(m, step):
    _, info = m(zt(15 + step), step=step)
    return info


results.append(run("sta", lambda: SpectralTransportAlignment(embed_dim=D, warmup_steps=500), _sta_drive))

# ── puc ─────────────────────────────────────────────────────────────────────
def _puc_drive(m, step):
    _, info = m(zt(16 + step), step=step)
    return info


results.append(run("puc", lambda: PredictionUncertaintyCalibration(embed_dim=D, warmup_steps=500), _puc_drive))

# ── rdc ─────────────────────────────────────────────────────────────────────
def _rdc_drive(m, step):
    Q = torch.eye(D)[:, :4]
    _, info = m(zt(18 + step), z_previous=zt(19), workspace_Q=Q, step=step)
    return info


results.append(run("rdc", lambda: RepresentationDriftCompensation(embed_dim=D, warmup_steps=500), _rdc_drive))

# ── wsr ─────────────────────────────────────────────────────────────────────
def _wsr_drive(m, step):
    Q = torch.linalg.qr(torch.randn(D, 4, generator=torch.Generator().manual_seed(23)))[0]
    if step > 0:
        m.set_lagged_gradient(torch.randn(D, 4, generator=torch.Generator().manual_seed(24)))
    _, info = m(Q, step=step)
    return info


results.append(run("wsr", lambda: WorkspaceSharpnessRegularization(embed_dim=D, warmup_steps=500), _wsr_drive))

bad = [n for n, miss in results if miss]
print()
print("MODULES WHOSE EARLY RETURN DROPS KEYS:", bad or "(none)")
sys.exit(1 if bad else 0)
