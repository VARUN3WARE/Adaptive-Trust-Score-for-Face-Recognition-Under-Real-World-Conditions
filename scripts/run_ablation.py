#!/usr/bin/env python3
"""
One-command trust-feature ablation for the results section.

Trains three identity-disjoint models and compares them on ERC:
  - quality      (handcrafted quality features only)
  - sim_only     (ArcFace similarity only — classifier form)
  - quality_sim  (quality + similarity + face_found)
Also reports the raw similarity score rejector for reference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from _bootstrap import ROOT  # noqa: F401

from src.evaluation import (
    error_vs_reject_curve,
    metrics_at_reject_rates,
    plot_error_vs_reject,
)
from src.trust_predictor import TrustPredictor
from src.utils import load_config, resolve_path

VARIANTS = (
    ("quality", "models/ablation_quality.joblib"),
    ("sim_only", "models/ablation_sim_only.joblib"),
    ("quality_sim", "models/ablation_quality_sim.joblib"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--features-csv",
        type=str,
        default="data/features/trust_training_table.csv",
    )
    parser.add_argument("--model-type", type=str, default="xgboost")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--prefix", type=str, default="ablation")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse existing ablation_*.joblib models",
    )
    return parser.parse_args()


def _train_variant(
    features: str,
    model_out: Path,
    csv_path: Path,
    model_type: str,
    seed: int,
    test_size: float,
    config: Optional[str],
) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_trust_module.py"),
        "--features-csv",
        str(csv_path),
        "--model-out",
        str(model_out),
        "--features",
        features,
        "--model-type",
        model_type,
        "--seed",
        str(seed),
        "--test-size",
        str(test_size),
        "--split",
        "identity",
    ]
    if config:
        cmd.extend(["--config", config])
    print(f"[ablation] training features={features} → {model_out}")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"Training failed for {features}")
    # Last JSON object in stdout
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        return {"features": features, "model_path": str(model_out)}
    return json.loads(lines[-1])


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    csv_path = resolve_path(args.features_csv)
    results_dir = resolve_path(cfg.get("evaluation", {}).get("results_dir", "results"))
    reject_rates = cfg.get("evaluation", {}).get("reject_rates", [0.10, 0.20, 0.30, 0.40])
    n_steps = int(cfg.get("evaluation", {}).get("risk_coverage_steps", 50))

    if not csv_path.exists():
        print(f"[error] features CSV missing: {csv_path}", file=sys.stderr)
        return 1

    train_metrics: dict[str, dict] = {}
    model_paths: dict[str, Path] = {}
    for features, rel_out in VARIANTS:
        out = resolve_path(rel_out)
        model_paths[features] = out
        if args.skip_train and out.exists():
            train_metrics[features] = {"skipped_train": True, "model_path": str(out)}
            continue
        train_metrics[features] = _train_variant(
            features=features,
            model_out=out,
            csv_path=csv_path,
            model_type=args.model_type,
            seed=args.seed,
            test_size=args.test_size,
            config=args.config,
        )

    df = pd.read_csv(csv_path)
    df = df.copy()
    df["predicted_identity_raw"] = df["predicted_identity"]
    if "face_found" in df.columns:
        df["face_found"] = df["face_found"].astype(float)
    else:
        df["face_found"] = 1.0

    erc_map: dict[str, pd.DataFrame] = {}
    summary: dict = {
        "n_samples": int(len(df)),
        "train": train_metrics,
        "variants": {},
    }

    # Raw similarity rejector (no trained model)
    df["score_similarity_raw"] = df["similarity"].astype(float).fillna(0.0)
    tmp = df.copy()
    tmp["trust_score"] = tmp["score_similarity_raw"]
    erc_map["similarity_raw"] = error_vs_reject_curve(tmp, score_col="trust_score", n_steps=n_steps)
    summary["variants"]["similarity_raw"] = {
        "at_reject_rate": metrics_at_reject_rates(
            tmp, score_col="trust_score", reject_rates=reject_rates
        ),
        "note": "ArcFace match score used directly as reject ranking",
    }

    for name, path in model_paths.items():
        if not path.exists():
            print(f"[error] missing model: {path}", file=sys.stderr)
            return 1
        model = TrustPredictor.load(path)
        feats = list(model.feature_names)
        missing = [c for c in feats if c not in df.columns]
        if missing:
            print(f"[error] {name} missing columns: {missing}", file=sys.stderr)
            return 1
        scores = model.predict_trust(df[feats].fillna(0.0).to_numpy(dtype="float32"))
        tmp = df.copy()
        tmp["trust_score"] = scores
        erc_map[name] = error_vs_reject_curve(tmp, score_col="trust_score", n_steps=n_steps)
        summary["variants"][name] = {
            "feature_names": feats,
            "model_path": str(path),
            "at_reject_rate": metrics_at_reject_rates(
                tmp, score_col="trust_score", reject_rates=reject_rates
            ),
            "test_roc_auc": train_metrics.get(name, {}).get("test_roc_auc"),
        }

    fig_dir = results_dir / "figures"
    met_dir = results_dir / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    primary = erc_map["quality_sim"]
    extras = {k: v for k, v in erc_map.items() if k != "quality_sim"}
    for name, erc in erc_map.items():
        erc.to_csv(met_dir / f"{args.prefix}_{name}_erc.csv", index=False)

    erc_path = plot_error_vs_reject(
        primary,
        fig_dir / f"{args.prefix}_erc.png",
        title="Ablation — Error vs Reject",
        extra_curves=extras,
    )
    summary["erc_plot"] = str(erc_path)

    # Compact paper table at 10/20/30% reject
    table_rows = []
    for name, payload in summary["variants"].items():
        row = {"variant": name, "test_roc_auc": payload.get("test_roc_auc")}
        for rr in ("10%", "20%", "30%"):
            vals = payload.get("at_reject_rate", {}).get(rr, {})
            row[f"acc@{rr}"] = vals.get("accuracy")
            row[f"err@{rr}"] = vals.get("error_rate")
        table_rows.append(row)
    table = pd.DataFrame(table_rows)
    table_path = met_dir / f"{args.prefix}_table.csv"
    table.to_csv(table_path, index=False)
    summary["table_csv"] = str(table_path)

    out_json = met_dir / f"{args.prefix}_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
