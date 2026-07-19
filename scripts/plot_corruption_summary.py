#!/usr/bin/env python3
"""
Plot a per-corruption summary from report_by_corruption.py output.

Shows baseline ArcFace accuracy vs Trust-gated retained accuracy (at a chosen
reject rate) for each corruption type, making the rain failure case obvious.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _bootstrap import ROOT  # noqa: F401

from src.utils import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--table-csv",
        type=str,
        default="results/metrics/by_corruption_table.csv",
        help="Output of scripts/report_by_corruption.py",
    )
    parser.add_argument(
        "--reject-rate-label",
        type=str,
        default="20%",
        help="Which fixed reject rate column to plot (e.g. 10% / 20% / 30%)",
    )
    parser.add_argument("--prefix", type=str, default="corruption_summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    table_path = resolve_path(args.table_csv)
    results_dir = resolve_path(cfg.get("evaluation", {}).get("results_dir", "results"))
    fig_dir = results_dir / "figures"
    met_dir = results_dir / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    if not table_path.exists():
        print(f"[error] table missing: {table_path}", file=sys.stderr)
        print("Run first: python scripts/report_by_corruption.py", file=sys.stderr)
        return 1

    df = pd.read_csv(table_path)
    sub = df[df["reject_rate_label"] == args.reject_rate_label].copy()
    if sub.empty:
        avail = sorted(df["reject_rate_label"].unique().tolist())
        print(
            f"[error] no rows for reject_rate_label={args.reject_rate_label}; available: {avail}",
            file=sys.stderr,
        )
        return 1

    # Keep only true corruption types (drop aggregate 'pristine' duplicate if 'none' present)
    sub = sub.sort_values("baseline_accuracy", ascending=False).reset_index(drop=True)

    labels = sub["corruption"].tolist()
    baseline = sub["baseline_accuracy"].to_numpy()
    gated = sub["accuracy"].to_numpy()
    recovery = gated - baseline

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
    ax.bar(x - width / 2, baseline, width, label="Baseline ArcFace", color="#b0413e")
    ax.bar(
        x + width / 2,
        gated,
        width,
        label=f"Trust-gated (retained @ {args.reject_rate_label} reject)",
        color="#1f4e79",
    )
    for i, r in enumerate(recovery):
        ax.text(x[i], max(baseline[i], gated[i]) + 0.02, f"+{r:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Identification accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Per-corruption accuracy: baseline vs Trust gating @ {args.reject_rate_label} reject")
    ax.legend(loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_png = fig_dir / f"{args.prefix}.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    summary = sub[["corruption", "n", "baseline_accuracy", "accuracy", "error_rate"]].copy()
    summary = summary.rename(columns={"accuracy": "gated_accuracy"})
    summary["recovery"] = summary["gated_accuracy"] - summary["baseline_accuracy"]
    out_csv = met_dir / f"{args.prefix}.csv"
    summary.to_csv(out_csv, index=False)

    print(f"Saved figure: {out_png}")
    print(f"Saved table:  {out_csv}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
