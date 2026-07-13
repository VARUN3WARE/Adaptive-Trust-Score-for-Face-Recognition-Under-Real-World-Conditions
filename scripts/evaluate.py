#!/usr/bin/env python3
"""
Evaluate ArcFace baseline vs ArcFace + Trust gating (biometrics metrics).

Produces Risk-Coverage / FAR-FRR plots under results/.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from _bootstrap import ROOT  # noqa: F401

from src.evaluation import evaluate_recognition_df
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
        help="Labeled table from build_trust_labels.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/trust_predictor.joblib",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--prefix", type=str, default="trust_eval")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    csv_path = resolve_path(args.features_csv)
    model_path = resolve_path(args.model)
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(cfg.get("trust", {}).get("reject_threshold", 0.5))
    )
    results_dir = resolve_path(cfg.get("evaluation", {}).get("results_dir", "results"))
    n_steps = int(cfg.get("evaluation", {}).get("risk_coverage_steps", 50))

    if not csv_path.exists():
        print(f"[error] features CSV missing: {csv_path}", file=sys.stderr)
        return 1
    if not model_path.exists():
        print(f"[error] trust model missing: {model_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    model = TrustPredictor.load(model_path)
    feature_names = list(model.feature_names) or list(QUALITY_FEATURE_NAMES)

    # Raw ArcFace prediction kept for baseline accuracy
    df["predicted_identity_raw"] = df["predicted_identity"]
    X = df[feature_names].fillna(0.0).to_numpy(dtype="float32")
    df["trust_score"] = model.predict_trust(X)
    df["accepted"] = df["trust_score"] >= threshold
    # Gated prediction: reject → null identity
    df["predicted_identity"] = df.apply(
        lambda r: r["predicted_identity_raw"] if r["accepted"] else None,
        axis=1,
    )

    metrics = evaluate_recognition_df(
        df,
        results_dir=results_dir,
        n_steps=n_steps,
        prefix=args.prefix,
    )
    metrics["trust_threshold"] = threshold
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
