#!/usr/bin/env bash
# Campaign gate. exit 0 = green, non-zero = red.
#
# DESIGN NOTE — why the default is not the full suite.
# The owner of this machine uses it for gaming, and the full suite is 1541
# tests that saturate all 6 cores for ~82s. So the default gate is the cheap
# deterministic part (lint, format, and the fast test files on ONE thread via
# tools/rt.py), and the authoritative whole-suite signal is the required
# GitHub Actions check `pytest`, which runs free and off-machine.
#
#   bash .agent-notes/gate.sh          # fast gate, ~1 thread, the default
#   bash .agent-notes/gate.sh --full   # every test, expects --slow to be needed
#
# Works under Git-Bash on Windows. Deliberately avoids bash-isms that break
# there, and avoids `set -e` so every stage reports rather than aborting early.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 2

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

# Pin the interpreter. `python` on PATH is a broken shim on this host.
PY="C:/Users/Илья/AppData/Local/Programs/Python/Python310/python.exe"
if [ ! -x "$PY" ]; then
  if command -v python3 >/dev/null 2>&1; then PY="python3"
  elif command -v python  >/dev/null 2>&1; then PY="python"
  else echo "gate: no usable python interpreter found" >&2; exit 2
  fi
fi

FAILED=0
stage() { printf '\n=== %s ===\n' "$1"; }

stage "new regression files (must not run only in CI)"
# The tick-1 verifier found this hole: four test files shipped by workers were
# never in the gate, so they ran only under the required CI check. A gate that
# does not run a test is a comment.
"$PY" tools/rt.py \
  tests/test_cmc_resume.py \
  tests/test_run_comparison.py \
  tests/test_determinism.py \
  tests/test_feature_composition.py \
  tests/test_ablation_module.py || FAILED=1

stage "merge damage (a conflicted tree still passes every other stage)"
# Found the hard way: wave 7 merged two workers that both owned proofs/jawp.md
# and proofs/wsd.md, left conflict markers in both, and the gate stayed GREEN -
# no stage reads markdown, so `<<<<<<<` is invisible to all nine of them.
# A gate that cannot see an unfinished merge will also not see a half-applied
# patch that happens to be syntactically valid.
if grep -rlE '^(<<<<<<< |>>>>>>> )' --include='*.md' --include='*.py' \
     --include='*.yaml' --include='*.toml' . 2>/dev/null | grep -qv '^\.git/'; then
  echo "  FAILED: conflict markers in the tree:" >&2
  grep -rlE '^(<<<<<<< |>>>>>>> )' --include='*.md' --include='*.py' \
       --include='*.yaml' --include='*.toml' . 2>/dev/null | grep -v '^\.git/' >&2
  FAILED=1
fi

stage "agent safety policy (the guard must itself be guarded)"
# policy.js is what stands between a confused agent and `shutil.rmtree` on the
# repo. A policy that silently stops matching is worse than none, because the
# owner believes it is protected. 61 cases: the named blind spots, the
# interpreter escape hatch, Windows shells, git history, publishing, and the
# ordinary work that must stay frictionless.
# Routed through python, not `node "C:/Users/..."`: Git-Bash mangles the
# non-ASCII path segment and node then cannot resolve the module. The wrapper
# also fails loud if the plugin or its test is missing, rather than skipping.
"$PY" tools/check_policy.py || FAILED=1

stage "opencode config syntax (brace balance + JSONC parse)"
"$PY" tools/check_config_syntax.py || FAILED=1

stage "ruff"
"$PY" -m ruff check . --output-format concise || FAILED=1

stage "black"
"$PY" -m black --check . 2>&1 | tail -n 2 || FAILED=1

stage "config grammar (every shipped config deep-merges and validates)"
# 556 assertions, no model instantiation, no training. Declared --slow
# because it is the single largest fast test file in the campaign.
"$PY" tools/rt.py --slow tests/test_config_system.py || FAILED=1

stage "security gates (checkpoint loading cannot execute a payload)"
"$PY" tools/rt.py tests/test_torchio.py || FAILED=1

stage "checkpoint fidelity (resume must match a continuous run)"
"$PY" tools/rt.py tests/test_checkpoint_fidelity.py || FAILED=1

stage "CMC mask resume safety (a fresh interpreter must continue the sequence)"
"$PY" tools/rt.py tests/test_cmc_resume.py || FAILED=1

stage "training-state guards (eval must not mutate training state)"
"$PY" tools/rt.py tests/test_training_state_guards.py || FAILED=1

stage "mechanism regression (all twelve, plus sterility)"
"$PY" tools/rt.py \
  tests/test_wsd.py tests/test_sta.py tests/test_rdc.py tests/test_puc.py \
  tests/test_gac.py tests/test_spc.py tests/test_cgn.py tests/test_wsr.py \
  tests/test_cmc.py tests/test_jawp.py tests/test_pcr.py tests/test_swip.py \
  tests/test_sterility.py tests/test_mechanism_wiring.py || FAILED=1

stage "interpretability contracts"
"$PY" tools/rt.py \
  tests/test_probes_split.py tests/test_index_and_cka.py \
  tests/test_info_and_disentangle.py tests/test_causal_intervention.py || FAILED=1

if [ "$FULL" -eq 1 ]; then
  stage "FULL suite (saturates the CPU; the CI check is the free equivalent)"
  "$PY" -m pytest tests/ -q --no-header -p no:cacheprovider || FAILED=1
else
  printf '\n=== FULL suite SKIPPED (pass --full to run it) ===\n'
  printf '    authoritative signal is the required CI check: pytest\n'
fi

printf '\n========================================\n'
if [ "$FAILED" -eq 0 ]; then
  printf 'GATE: GREEN\n'
  exit 0
fi
printf 'GATE: RED\n'
exit 1
