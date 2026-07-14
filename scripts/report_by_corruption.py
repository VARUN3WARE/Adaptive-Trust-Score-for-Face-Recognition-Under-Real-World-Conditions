#!/usr/bin/env python3
"""
Break down Trust Score ERC metrics by corruption type and pristine/corrupted split.

Joins ``corruption_manifest.jsonl`` onto the labeled features table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT  # noqa: F401

from src.evaluation import error_vs_reject_curve, metrics_at_reject_rates
from src.feature_extraction import QUALITY_FEATURE_NAMES
from src.trust_predictor import TrustPredictor
from src.utils import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--features-csv",
        type=str,
        default="data/features/trust_training_table.csv",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/corrupted/corruption_manifest.jsonl",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/trust_predictor.joblib",
    )
    parser.add_argument("--prefix", type=str, default="by_corruption")
    parser.add_argument(
        "--reject-rates",
        type=float,
        nargs="+",
        default=None,
    )
    return parser.parse_args()


def load_manifest(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    man = pd.DataFrame(rows)
    # Normalize paths for joins
    man["output_norm"] = man["output"].map(lambda p: str(Path(p).resolve()))
    man["source_norm"] = man["source"].map(lambda p: str(Path(p).resolve()))
    return man


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    csv_path = resolve_path(args.features_csv)
    manifest_path = resolve_path(args.manifest)
    model_path = resolve_path(args.model)
    results_dir = resolve_path(cfg.get("evaluation", {}).get("results_dir", "results"))
    reject_rates = args.reject_rates or cfg.get("evaluation", {}).get(
        "reject_rates", [0.10, 0.20, 0.30]
    )
    n_steps = int(cfg.get("evaluation", {}).get("risk_coverage_steps", 50))

    for p, label in [(csv_path, "features CSV"), (model_path, "trust model")]:
        if not p.exists():
            print(f"[error] {label} missing: {p}", file=sys.stderr)
            return 1

    df = pd.read_csv(csv_path)
    model = TrustPredictor.load(model_path)
    feature_names = list(model.feature_names) or list(QUALITY_FEATURE_NAMES)
    if "face_found" in feature_names:
        if "face_found" not in df.columns:
            df["face_found"] = 1.0
        df["face_found"] = df["face_found"].astype(float)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        print(f"[error] missing feature columns: {missing}", file=sys.stderr)
        return 1

    df = df.copy()
    df["predicted_identity_raw"] = df["predicted_identity"]
    df["image_path_norm"] = df["image_path"].map(lambda p: str(Path(p).resolve()))
    df["trust_score"] = model.predict_trust(
        df[feature_names].fillna(0.0).to_numpy(dtype="float32")
    )

    # Attach corruption type
    df["corruption"] = "unknown"
    if "split" in df.columns:
        df.loc[df["split"] == "pristine", "corruption"] = "pristine"

    if manifest_path.exists():
        man = load_manifest(manifest_path)
        # Prefer joining corrupted probes on output path
        merged = df.merge(
            man[["output_norm", "corruption"]].rename(
                columns={"output_norm": "image_path_norm", "corruption": "corruption_m"}
            ),
            on="image_path_norm",
            how="left",
        )
        mask = merged["corruption_m"].notna()
        merged.loc[mask, "corruption"] = merged.loc[mask, "corruption_m"]
        df = merged.drop(columns=["corruption_m"], errors="ignore")
    else:
        print(f"[warn] manifest missing ({manifest_path}); using split labels only")

    met_dir = results_dir / "metrics"
    met_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"n_samples": int(len(df)), "by_corruption": {}, "by_split": {}}

    # Per split
    if "split" in df.columns:
        for split_name, sub in df.groupby("split"):
            tmp = sub.copy()
            summary["by_split"][str(split_name)] = {
                "n": int(len(tmp)),
                "baseline_accuracy": float(
                    (
                        tmp["predicted_identity_raw"].fillna("__none__").astype(str)
                        == tmp["ground_truth"].astype(str)
                    ).mean()
                ),
                "at_reject_rate": metrics_at_reject_rates(
                    tmp, score_col="trust_score", reject_rates=reject_rates
                ),
            }

    # Per corruption type
    rows_table = []
    for corr, sub in df.groupby("corruption"):
        tmp = sub.copy()
        baseline = float(
            (
                tmp["predicted_identity_raw"].fillna("__none__").astype(str)
                == tmp["ground_truth"].astype(str)
            ).mean()
        )
        at_rr = metrics_at_reject_rates(tmp, score_col="trust_score", reject_rates=reject_rates)
        erc = error_vs_reject_curve(tmp, score_col="trust_score", n_steps=min(n_steps, 30))
        erc.to_csv(met_dir / f"{args.prefix}_{corr}_erc.csv", index=False)
        summary["by_corruption"][str(corr)] = {
            "n": int(len(tmp)),
            "baseline_accuracy": baseline,
            "at_reject_rate": at_rr,
        }
        for rr_key, vals in at_rr.items():
            rows_table.append(
                {
                    "corruption": corr,
                    "n": len(tmp),
                    "baseline_accuracy": baseline,
                    "reject_rate_label": rr_key,
                    **vals,
                }
            )

    table = pd.DataFrame(rows_table)
    table_path = met_dir / f"{args.prefix}_table.csv"
    table.to_csv(table_path, index=False)
    summary["table_csv"] = str(table_path)

    out_json = met_dir / f"{args.prefix}_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
