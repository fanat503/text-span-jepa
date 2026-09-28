"""Bounded test runner.

Every agent in this repo runs tests through this script. It exists because
prompt-level instructions were demonstrably not enough: an agent ran the full
686-test suite (221s, all 6 cores) on a box whose owner was gaming, and a later
agent burned 92s on two heavy files because "any single test file" was allowed.

Enforced in code, not by request:

1.  OMP / MKL / OPENBLAS threads = 1.  Verified: torch.get_num_threads()
    reports 1 instead of 6, so no test can saturate the machine.

2.  A test path is MANDATORY.  A bare `pytest` (the whole suite) is refused.
    The full suite belongs in GitHub Actions, which is free and off-machine.

3.  A CUMULATIVE wall-clock budget across every file in the invocation, not a
    per-invocation timeout.  Passing twenty cheap files must not buy twenty
    times the allowance.  Default 90s.  Overrunning kills the run and reports
    failure, so a hang cannot silently eat the machine.

4.  Known-expensive files are REFUSED unless `--slow` is passed, because they
    are what a 92-second "single run" actually consisted of.  The flag is a
    deliberate, greppable statement that the cost was accepted.

5.  No training.  `src.train` and the distributed launchers are never invoked.

Usage
-----
    python tools/rt.py tests/test_sta.py
    python tools/rt.py tests/test_sta.py -k workspace
    python tools/rt.py tests/test_a.py tests/test_b.py       # still one budget
    python tools/rt.py --slow tests/test_model.py            # explicit opt-in

Exit codes: 0 pass, 1 test failure, 2 refused, 3 budget exhausted.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Must be set before numpy/torch import in the child.
_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

# One budget for the whole invocation. Deliberately small: the box is shared.
BUDGET_SECONDS = int(os.environ.get("RT_BUDGET_SECONDS", "90"))

# Files that cost tens of seconds each. These are the ones that turn a
# "single file" run into minutes. Require --slow.
SLOW_FILES = {
    "tests/test_model.py",
    "tests/test_training_e2e.py",
    "tests/test_v025_integration.py",
    "tests/test_config_system.py",
}

FORBIDDEN = (
    "src.train",
    "torchrun",
    "torch.distributed",
    "accelerate",
    "deepspeed",
    "pytorch_xla",
)


def die(msg: str, code: int = 2) -> int:
    print(f"rt.py: REFUSED: {msg}", file=sys.stderr)
    return code


def classify(args: list[str], allow_slow: bool) -> tuple[list[str], str | None]:
    """Return (pytest args, refusal reason or None)."""
    for a in args:
        for bad in FORBIDDEN:
            if bad in a:
                return [], f"refusing to invoke a training/distributed entry point: {a!r}"

    paths = [a for a in args if not a.startswith("-") and ("test_" in a or a.endswith(".py"))]
    if not paths:
        return [], (
            "no test file in the arguments. A whole-suite run belongs in GitHub "
            "Actions, not on this CPU. Name a file, e.g. tests/test_sta.py"
        )

    normalised = []
    for p in paths:
        q = str(Path(p)).replace("\\", "/")
        normalised.append(q)
        if not (REPO / p).exists() and not Path(p).exists():
            return [], f"test file not found: {p}"

    if not allow_slow:
        heavy = sorted(q for q in normalised if q in SLOW_FILES)
        if heavy:
            return [], (
                f"these files cost tens of seconds each and are what turned a "
                f"'single file' run into 92s: {heavy}. Re-run with --slow if you "
                f"have accepted that cost."
            )
    return args, None


def main(argv: list[str]) -> int:
    allow_slow = False
    if "--slow" in argv:
        allow_slow = True
        argv = [a for a in argv if a != "--slow"]

    args = argv[1:]
    if not args:
        return die(
            "no test file given. The full suite runs in GitHub Actions, not on "
            "this CPU. Pass one or more paths, e.g. tests/test_sta.py"
        )

    args, reason = classify(args, allow_slow)
    if reason:
        return die(reason)

    env = {**os.environ, **_ENV}
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", *args]
    budget = int(os.environ.get("RT_BUDGET_SECONDS", BUDGET_SECONDS))

    print("rt.py: " + " ".join(cmd), flush=True)
    print(f"rt.py: threads=1  total_budget={budget}s  slow_ok={allow_slow}", flush=True)

    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env)
    except Exception as exc:  # pragma: no cover
        return die(f"could not start pytest: {exc}")

    try:
        proc.wait(timeout=budget)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print(
            f"rt.py: BUDGET EXHAUSTED after {time.monotonic() - t0:.0f}s "
            f"(limit {budget}s for ALL files together). Killed so it could not "
            f"keep using the CPU. Run fewer files per invocation, or raise "
            f"RT_BUDGET_SECONDS if you are certain the work is cheap.",
            file=sys.stderr,
        )
        return 3

    return proc.returncode


if __name__ == "__main__":
    for _k, _v in _ENV.items():
        os.environ[_k] = _v
    sys.exit(main(sys.argv))
