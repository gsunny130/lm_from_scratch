# sweep.py
# Hyperparameter sweep over LoRA r, lora_alpha, and lora_dropout.
#
# For each config, runs train.py as a subprocess, reads the saved metrics JSON,
# and logs everything to sweep_results.csv. Prints a results table at the end.
#
# Sweep grid:
#   r:            [4, 8, 16]
#   lora_alpha:   always 2 * r  (standard heuristic: alpha/r = 2 keeps the
#                               effective update scale constant across ranks)
#   lora_dropout: [0.05, 0.1]
#
# Total: 6 runs — ~10 minutes each on an A10 → ~1 hour total.
#
# Usage:
#   ~/env/bin/python sweep.py

import csv
import json
import os
import subprocess
import sys
from itertools import product

# --------------------------------------------------------------------------
# 1. Define sweep configs
# --------------------------------------------------------------------------
# product([4,8,16], [0.05,0.1]) generates all 6 (r, dropout) combinations.
# lora_alpha = 2*r means the effective per-layer update magnitude (alpha/r)
# stays fixed at 2 regardless of rank — this isolates the effect of rank
# from scaling effects.

CONFIGS = [
    {
        "r":            r,
        "lora_alpha":   2 * r,
        "lora_dropout": dropout,
    }
    for r, dropout in product([4, 8, 16], [0.05, 0.1])
]

RESULTS_CSV = "sweep_results.csv"


# --------------------------------------------------------------------------
# 2. Run one training config
# --------------------------------------------------------------------------

def run_config(cfg, run_idx, total):
    """
    Run train.py with the given config. Returns the metrics dict from the
    saved JSON file.

    Each run gets its own output directory so adapters don't overwrite each
    other. The naming scheme encodes the hyperparameters so directories are
    self-documenting.
    """
    tag        = f"r{cfg['r']}_alpha{cfg['lora_alpha']}_drop{cfg['lora_dropout']}"
    output_dir = f"./sweep-{tag}"
    metrics_file = os.path.join(output_dir, "metrics.json")

    print(f"\n{'='*60}")
    print(f"Run {run_idx}/{total}: {tag}")
    print(f"  r={cfg['r']}  lora_alpha={cfg['lora_alpha']}  lora_dropout={cfg['lora_dropout']}")
    print(f"  output_dir: {output_dir}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "train.py",
        "--r",            str(cfg["r"]),
        "--lora-alpha",   str(cfg["lora_alpha"]),
        "--lora-dropout", str(cfg["lora_dropout"]),
        "--output-dir",   output_dir,
        "--metrics-file", metrics_file,
    ]

    # check=True raises CalledProcessError if train.py exits non-zero.
    # This stops the sweep immediately on any training failure so you can
    # diagnose the error rather than getting a silent partial results file.
    subprocess.run(cmd, check=True)

    with open(metrics_file) as f:
        metrics = json.load(f)

    return metrics


# --------------------------------------------------------------------------
# 3. Print results table
# --------------------------------------------------------------------------

def print_results_table(results):
    """Print a formatted comparison table of all sweep results."""
    header = (
        f"  {'r':>4}  {'alpha':>5}  {'dropout':>7}  "
        f"{'train_loss':>10}  {'eval_loss':>9}  {'test_ppl':>8}"
    )
    separator = "  " + "-" * (len(header) - 2)

    print("\n" + "=" * len(header))
    print("SWEEP RESULTS")
    print("=" * len(header))
    print(header)
    print(separator)

    # Sort by test perplexity (ascending = best first)
    for row in sorted(results, key=lambda x: x.get("test_perplexity") or float("inf")):
        tl = row.get("train_loss")
        el = row.get("eval_loss")
        tp = row.get("test_perplexity")
        print(
            f"  {row['r']:>4}  "
            f"{row['lora_alpha']:>5}  "
            f"{row['lora_dropout']:>7.2f}  "
            f"{tl if tl is not None else 'N/A':>10}  "
            f"{el if el is not None else 'N/A':>9}  "
            f"{tp if tp is not None else 'N/A':>8}"
        )

    print("=" * len(header))
    best = min(results, key=lambda x: x.get("test_perplexity") or float("inf"))
    print(
        f"\nBest config: r={best['r']}, lora_alpha={best['lora_alpha']}, "
        f"lora_dropout={best['lora_dropout']}  "
        f"(test_ppl={best.get('test_perplexity')})"
    )


# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------

def main():
    total   = len(CONFIGS)
    results = []

    for i, cfg in enumerate(CONFIGS, start=1):
        metrics = run_config(cfg, i, total)
        results.append(metrics)
        print(f"\nRun {i}/{total} done. test_perplexity={metrics.get('test_perplexity')}")

    # Save CSV so you can load results into pandas/Excel later
    fieldnames = ["r", "lora_alpha", "lora_dropout", "train_loss", "eval_loss", "test_perplexity"]
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {RESULTS_CSV}")
    print_results_table(results)


if __name__ == "__main__":
    main()
