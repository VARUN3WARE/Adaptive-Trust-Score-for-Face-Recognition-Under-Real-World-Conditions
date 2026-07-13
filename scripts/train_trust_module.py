#!/usr/bin/env python3
"""Train the Trust Score classifier from a labeled quality-feature table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from _bootstrap import ROOT  # noqa: F401

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
        "--model-out",
        type=str,
        default="models/trust_predictor.joblib",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["random_forest", "xgboost", "mlp"],
        default=None,
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    csv_path = resolve_path(args.features_csv)
    model_out = resolve_path(args.model_out)
    model_type = args.model_type or cfg.get("trust", {}).get("model_type", "xgboost")
    feature_names = list(cfg.get("trust", {}).get("features", QUALITY_FEATURE_NAMES))

    df = pd.read_csv(csv_path)
    # Drop rows with no face — quality features are undefined / zeros
    if "face_found" in df.columns:
        df = df[df["face_found"] == True].copy()  # noqa: E712

    missing = [c for c in feature_names + ["label_correct"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {csv_path}: {missing}")

    X = df[feature_names].to_numpy(dtype="float32")
    y = df["label_correct"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y if len(set(y)) > 1 else None,
    )

    model = TrustPredictor(model_type=model_type, feature_names=feature_names, random_state=args.seed)
    model.fit(X_train, y_train)
    model.save(model_out)

    proba = model.predict_trust(X_test)
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "model_type": model_type,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "model_path": str(model_out),
    }
    try:
        metrics["test_roc_auc"] = float(roc_auc_score(y_test, proba))
    except ValueError:
        metrics["test_roc_auc"] = None

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
