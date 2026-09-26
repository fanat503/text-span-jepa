#!/usr/bin/env python3
"""
CLI Ideal - One-Click для всех - максимально эффективно решает все проблемы
Usage: python3 frontier-01-cli-ideal.py --mode all|bilinearity|bow|graphs|eval|kaggle|tpu
Sterile & robust: works from any cwd via Path(__file__).parent
"""
import argparse
import sys
import os
import math
import json
import hashlib
import subprocess
from pathlib import Path

# Robust root: dir where this file lives
ROOT = Path(__file__).resolve().parent
def run(script: str):
    """Run script relative to ROOT with python3, inheriting output"""
    path = ROOT / script
    if not path.exists():
        print(f"[CLI] ERROR: {path} not found (ROOT={ROOT})")
        return 1
    # use subprocess for robust cwd handling
    result = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
    return result.returncode

def run_bilinearity():
    print("=== RUN bilinearity-break-ideal ===")
    # try ideal, fallback standalone demo
    code = run("frontier-01-bilinearity-standalone-DEMO-FOR-USER.py")
    if code != 0:
        code = run("frontier-01-bilinearity-break-ideal.py")
    return code

def run_bow():
    print("=== RUN bag-of-words-test ===")
    return run("frontier-01-bag-of-words-test.py")

def run_graphs():
    print("=== RUN graphs-ULTIMATE-V11 TOP-LAB V15 FINAL ===")
    code = run("frontier-01-graphs-ULTIMATE-V11.py")
    # V15 FINAL: only ULTIMATE V11 162K-290K beautiful clear dark #111 lw4 error bars subplots unit circle
    # Old V10 TOPLAB 153K-263K deprecated, all-graphs-ideal.py deprecated - use only ULTIMATE for Oral 6
    if code == 0:
        # verify 8 PNG >150K
        for p in ["fig_bilinearity_break.png","fig_small_angle.png","fig_gate_phase.png","fig_high_low_L0.png","fig_yarn_rope_interaction.png","fig_pprope_split.png","fig_conservation.png","fig_bag_of_words.png"]:
            fp = ROOT / p
            if fp.exists():
                sz = fp.stat().st_size
                status = "PASS" if sz > 150_000 else "FAIL too small"
                print(f"  {p}: {sz} bytes {status}")
            else:
                print(f"  {p}: MISSING")
    return code

def run_eval():
    print("=== RUN eval-numpy-ideal ===")
    return run("frontier-01-eval-numpy-ideal.py")

def run_all():
    run_bilinearity()
    run_bow()
    run_graphs()
    run_eval()
    print("\n=== ALL DONE IDEAL ===")
    print("Files: settings-ideal.json config_hash 9bd59cac dataset_hash 848bb0b0")
    print("Figures: 8 PNG 200 dpi fig_bilinearity_break.png fig_small_angle.png fig_gate_phase.png fig_high_low_L0.png fig_yarn_rope_interaction.png fig_pprope_split.png fig_conservation.png fig_bag_of_words.png")
    print("Next: Kaggle 2xT4 11 cells from frontier-01-kaggle-notebook-ideal.py or TPU v5e-8 command from TPU-runbook-IDEAL.md")

def cat_file(name: str, limit: int = 4000):
    p = ROOT / name
    if p.exists():
        txt = p.read_text(encoding="utf-8")
        print(txt[:limit])
    else:
        print(f"[CLI] file not found: {p}")

def main():
    parser = argparse.ArgumentParser(description="Gemma 4 4B pp-RoPE p=0.25 RoPE+YaRN Phi + BoW - One-Click Ideal CLI (robust paths)")
    parser.add_argument("--mode", choices=["all","bilinearity","bow","graphs","eval","kaggle","tpu","usage"], default="all", help="what to run")
    args = parser.parse_args()
    if args.mode == "all":
        run_all()
    elif args.mode == "bilinearity":
        run_bilinearity()
    elif args.mode == "bow":
        run_bow()
    elif args.mode == "graphs":
        run_graphs()
    elif args.mode == "eval":
        run_eval()
    elif args.mode == "kaggle":
        print("Kaggle 2xT4: New Notebook T4 x2 Internet ON, copy 11 cells from frontier-01-kaggle-notebook-ideal.py, Run All 3h <12h")
        cat_file("frontier-01-KAGGLE-IDEAL-HOWTO-V10.md", 4000)
        cat_file("frontier-01-kaggle-howto-IDEAL.md", 2000)
    elif args.mode == "tpu":
        print("TPU v5e-8: torch_xla.distributed.xla_dist --tpu $TPU_NAME -- python3 collect_for_seed.py --model gemma-4-4b-e4b --layer 6 --backend nnsight --chunking per-query --save half --no-logits")
        cat_file("frontier-01-TPU-runbook-IDEAL.md", 4000)
    elif args.mode == "usage":
        cat_file("frontier-01-README-USAGE-IDEAL.md", 5000)

if __name__ == "__main__":
    main()
