#!/usr/bin/env python3
"""
Compare reject strategies on Error-vs-Reject curves:

1. similarity-only  — reject by ArcFace match score
2. quality-trust    — trained Trust Score (quality features)
3. quality+similarity — Trust Score trained on quality + similarity
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from _bootstrap import ROOT  # noqa: F401

from src.evaluation import (
    error_vs_reject_curve,
    metrics_at_reject_rates,
    plot_error_vs_reject,
)
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
        "--quality-model",
        type=str,
        default="models/trust_predictor.joblib",
        help="Pretrained quality-only trust model",
    )
    parser.add_argument(
        "--fused-model-out",
        type=str,
        default="models/trust_predictor_quality_sim.joblib",
    )
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="baseline_compare")
    return parser.parse_args()


def _correctness(df: pd.DataFrame) -> pd.Series:
    pred = df.get("predicted_identity_raw", df["predicted_identity"])
    return (
        pred.fillna("__none__").astype(str) == df["ground_truth"].astype(str)
    ).astype(int)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    csv_path = resolve_path(args.features_csv)
    quality_model_path = resolve_path(args.quality_model)
    fused_out = resolve_path(args.fused_model_out)
    results_dir = resolve_path(cfg.get("evaluation", {}).get("results_dir", "results"))
    model_type = args.model_type or cfg.get("trust", {}).get("model_type", "xgboost")
    quality_names = list(cfg.get("trust", {}).get("features", QUALITY_FEATURE_NAMES))
    # Strip sim/face_found if present so quality-only stays pure
    quality_names = [n for n in quality_names if n not in ("similarity", "face_found")]
    fused_names = quality_names + ["similarity"]
    n_steps = int(cfg.get("evaluation", {}).get("risk_coverage_steps", 50))

    if not csv_path.exists():
        print(f"[error] features CSV missing: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    if "similarity" not in df.columns:
        print("[error] CSV missing 'similarity' column", file=sys.stderr)
        return 1

    df = df.copy()
    df["predicted_identity_raw"] = df["predicted_identity"]
    df["label_correct"] = _correctness(df)

    # --- 1) similarity-only scores ---
    df["score_similarity"] = df["similarity"].astype(float).fillna(0.0)

    # --- 2) quality-trust scores ---
    if quality_model_path.exists():
        q_model = TrustPredictor.load(quality_model_path)
        Xq = df[list(q_model.feature_names)].fillna(0.0).to_numpy(dtype="float32")
        df["score_quality_trust"] = q_model.predict_trust(Xq)
    else:
        print(f"[warn] quality model missing ({quality_model_path}); training one in-script")
        train_df = df[df.get("face_found", True) == True].copy() if "face_found" in df.columns else df  # noqa: E712
        X = train_df[quality_names].fillna(0.0).to_numpy(dtype="float32")
        y = train_df["label_correct"].astype(int).to_numpy()
        Xtr, _, ytr, _ = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y if len(set(y)) > 1 else None
        )
        q_model = TrustPredictor(model_type=model_type, feature_names=quality_names, random_state=args.seed)
        q_model.fit(Xtr, ytr)
        q_model.save(quality_model_path)
        df["score_quality_trust"] = q_model.predict_trust(
            df[quality_names].fillna(0.0).to_numpy(dtype="float32")
        )

    # --- 3) quality + similarity trust model ---
    if "face_found" in df.columns:
        face_df = df[df["face_found"] == True].copy()  # noqa: E712
    else:
        face_df = df
    Xf = face_df[fused_names].fillna(0.0).to_numpy(dtype="float32")
    yf = face_df["label_correct"].astype(int).to_numpy()
    idx = np.arange(len(face_df))
    itr, ite = train_test_split(
        idx,
        test_size=0.2,
        random_state=args.seed,
        stratify=yf if len(set(yf)) > 1 else None,
    )
    fused = TrustPredictor(model_type=model_type, feature_names=fused_names, random_state=args.seed)
    fused.fit(Xf[itr], yf[itr])
    fused.save(fused_out)
    df["score_quality_sim"] = fused.predict_trust(
        df[fused_names].fillna(0.0).to_numpy(dtype="float32")
    )

    score_cols = {
        "similarity_only": "score_similarity",
        "quality_trust": "score_quality_trust",
        "quality_plus_similarity": "score_quality_sim",
    }

    fig_dir = results_dir / "figures"
    met_dir = results_dir / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    erc_map: dict[str, pd.DataFrame] = {}
    summary: dict[str, object] = {"n_samples": int(len(df)), "baselines": {}}
    primary_name = "quality_trust"
    for name, col in score_cols.items():
        tmp = df.copy()
        tmp["trust_score"] = tmp[col]
        erc = error_vs_reject_curve(tmp, score_col="trust_score", n_steps=n_steps)
        erc_map[name] = erc
        erc.to_csv(met_dir / f"{args.prefix}_{name}_erc.csv", index=False)
        summary["baselines"][name] = {
            "at_reject_rate": metrics_at_reject_rates(tmp, score_col="trust_score"),
            "score_col": col,
        }

    primary = erc_map[primary_name]
    extras = {k: v for k, v in erc_map.items() if k != primary_name}
    erc_path = plot_error_vs_reject(
        primary,
        fig_dir / f"{args.prefix}_erc.png",
        title="Error vs Reject — baseline comparison",
        extra_curves=extras,
    )
    summary["erc_plot"] = str(erc_path)
    summary["fused_model_path"] = str(fused_out)

    out_json = met_dir / f"{args.prefix}_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
