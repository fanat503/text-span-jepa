# Differential: the key-set fix must not change any VALUE the five modules
# report on the post-warmup path, and must not change any TRAINING STATE they
# advance. Runs the same driver against the pre-fix sources (git show main:...)
# and the post-fix worktree in two separate processes, then diffs the JSON.

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(r"C:\dev\wt-keys")
PY = r"C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe"
FIVE = ["gac", "sta", "puc", "rdc", "wsr"]

DRIVER = r'''
import json, os, sys
sys.path.insert(0, sys.argv[1])
import torch, src
assert src.__file__.startswith(sys.argv[1]), src.__file__
from src.models.gac import GradientAllocatedCapacity
from src.models.sta import SpectralTransportAlignment
from src.models.puc import PredictionUncertaintyCalibration
from src.models.rdc import RepresentationDriftCompensation
from src.models.wsr import WorkspaceSharpnessRegularization

D = 16
def zt(s): return torch.randn(4, 8, D, generator=torch.Generator().manual_seed(s))
def Q(c, s):
    return torch.linalg.qr(torch.randn(D, c, generator=torch.Generator().manual_seed(s)))[0]

out = {}

def grab(t):
    """State buffers + params of a module, as plain floats."""
    d = {}
    for n, b in list(t.named_buffers()) + list(t.named_parameters()):
        d[n] = [round(float(x), 12) for x in b.detach().reshape(-1)]
    return d

def fresh(fn):
    """Seeded construction: PUC random-initialises proj_vectors, so two
    processes would otherwise start from different states."""
    torch.manual_seed(1234)
    return fn()

# --- GAC: drive 3 steps past warmup, then one inside warmup on a fresh module
g = fresh(lambda: GradientAllocatedCapacity(embed_dim=D, warmup_steps=10)); g.train()
rec = []
for s in (0, 5, 12, 20):
    gn = torch.rand(D, generator=torch.Generator().manual_seed(17)) * 1e-2
    l, i = g(zt(14 + s), gn, step=s)
    rec.append({"step": s, "loss": float(l), "info": {k: (round(float(v),12) if isinstance(v,(int,float)) and not isinstance(v,bool) else v) for k,v in i.items()}})
out["gac"] = {"steps": rec, "state": grab(g)}

# --- STA
s_ = fresh(lambda: SpectralTransportAlignment(embed_dim=D, warmup_steps=10)); s_.train()
rec = []
for s in (0, 5, 12, 20, 33):
    l, i = s_(zt(15 + s), step=s)
    rec.append({"step": s, "loss": float(l), "info": {k: (round(float(v),12) if isinstance(v,(int,float)) and not isinstance(v,bool) else v) for k,v in i.items()}})
out["sta"] = {"steps": rec, "state": grab(s_)}

# --- PUC
p = fresh(lambda: PredictionUncertaintyCalibration(embed_dim=D, warmup_steps=10)); p.train()
rec = []
for s in (0, 5, 12, 20, 33):
    l, i = p(zt(16 + s), step=s)
    rec.append({"step": s, "loss": float(l), "info": {k: (round(float(v),12) if isinstance(v,(int,float)) and not isinstance(v,bool) else v) for k,v in i.items()}})
out["puc"] = {"steps": rec, "state": grab(p)}

# --- RDC
r = fresh(lambda: RepresentationDriftCompensation(embed_dim=D, warmup_steps=10)); r.train()
rec = []
for s in (0, 5, 12, 20, 33):
    l, i = r(zt(18 + s), z_previous=zt(19), workspace_Q=torch.eye(D)[:, :4], step=s)
    rec.append({"step": s, "loss": float(l), "info": {k: (round(float(v),12) if isinstance(v,(int,float)) and not isinstance(v,bool) else v) for k,v in i.items()}})
out["rdc"] = {"steps": rec, "state": grab(r)}

# --- WSR both modes
for mode in ("gradient", "sam"):
    w = fresh(lambda: WorkspaceSharpnessRegularization(embed_dim=D, warmup_steps=10, mode=mode)); w.train()
    rec = []
    for s in (0, 5, 12, 20, 33):
        w.set_lagged_gradient(Q(4, 24))
        l, i = w(Q(4, 23), loss_fn=lambda a,bb,c: ((bb-c)**2).mean(), z_pred=zt(25), z_target=zt(26), step=s)
        rec.append({"step": s, "loss": float(l), "info": {k: (round(float(v),12) if isinstance(v,(int,float)) and not isinstance(v,bool) else v) for k,v in i.items()}})
    out["wsr_" + mode] = {"steps": rec, "state": grab(w)}

print(json.dumps(out))
'''

tmp = Path(tempfile.mkdtemp(prefix="prefix-src-"))
# A full copy of the worktree's src/ tree, so package-relative imports work,
# with the five module files replaced by their pre-fix versions from main.
subprocess.run(
    ["robocopy", str(ROOT / "src"), str(tmp / "src"), "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NP"],
    capture_output=True,
)
sub = tmp / "src"
for name in ["gac", "sta", "puc", "rdc", "wsr"]:
    blob = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"main:src/models/{name}.py"],
        capture_output=True, check=True,
    ).stdout
    (sub / "models" / f"{name}.py").write_bytes(blob)

drv = tmp / "driver.py"
drv.write_text(DRIVER, encoding="utf-8")


def run(root):
    r = subprocess.run([PY, str(drv), str(root)], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"driver failed for {root}")
    return json.loads(r.stdout.strip().splitlines()[-1])


before = run(tmp)
after = run(ROOT)

bad = False
for mech in [m for m in before if m != "__none"]:
    for phase in ("steps", "state"):
        b, a = before[mech][phase], after[mech][phase]
        if phase == "state":
            # Compare only the buffers/params that EXISTED pre-fix; the fix
            # adds no state, so any difference here is a regression.
            if b != a:
                bad = True
                print(f"DIFF {mech}.state")
                for k in b:
                    if b[k] != a.get(k):
                        print(f"   {k}: {b[k][:4]} -> {str(a.get(k))[:4]}")
            else:
                print(f"ok   {mech}.state ({len(b)} tensors identical)")
            continue
        for rb, ra in zip(b, a):
            if rb["step"] != ra["step"]:
                bad = True
                continue
            # Post-warmup steps only (step >= warmup_steps=10) must be IDENTICAL
            # in loss AND in every pre-existing info key.
            if rb["step"] < 10:
                continue
            if rb["loss"] != ra["loss"]:
                bad = True
                print(f"DIFF {mech} step={rb['step']} loss {rb['loss']!r} -> {ra['loss']!r}")
            for k, v in rb["info"].items():
                if k not in ra["info"]:
                    bad = True
                    print(f"DIFF {mech} step={rb['step']} key {k} REMOVED")
                elif ra["info"][k] != v:
                    bad = True
                    print(f"DIFF {mech} step={rb['step']} {k}: {v!r} -> {ra['info'][k]!r}")
            added = sorted(set(ra["info"]) - set(rb["info"]))
            if added:
                print(f"     {mech} step={rb['step']}: added keys {added}")
print()
print("POST-WARMUP VALUES AND STATE IDENTICAL:", not bad)
sys.exit(1 if bad else 0)
