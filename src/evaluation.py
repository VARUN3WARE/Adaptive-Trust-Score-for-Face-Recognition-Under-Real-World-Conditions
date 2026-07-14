"""
Biometrics-style evaluation for ArcFace ± Trust Score gating.

Reports accuracy, Risk-Coverage curves, Error-vs-Reject (ERC), FAR / FRR, and EER.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import PathLike, PROJECT_ROOT


def results_to_dataframe(results: Sequence[Any]) -> pd.DataFrame:
    """Convert ``RecognitionResult`` objects (or dict rows) to a DataFrame."""
    rows = []
    for r in results:
        if hasattr(r, "to_row"):
            rows.append(r.to_row())
        elif isinstance(r, dict):
            rows.append(r)
        else:
            raise TypeError(f"Unsupported result type: {type(r)}")
    return pd.DataFrame(rows)


def identification_accuracy(
    df: pd.DataFrame,
    prediction_col: str = "predicted_identity",
    gt_col: str = "ground_truth",
    only_accepted: bool = False,
    accepted_col: str = "accepted",
) -> float:
    """
    Closed-set style accuracy: fraction where prediction == ground truth.

    When ``only_accepted`` is True, accuracy is computed on the accepted
    subset only (coverage-aware). Rejected samples are excluded, not counted wrong.
    """
    data = df.dropna(subset=[gt_col]).copy()
    if only_accepted and accepted_col in data.columns:
        data = data[data[accepted_col] == True]  # noqa: E712
    if len(data) == 0:
        return float("nan")
    correct = data[prediction_col].fillna("__none__").astype(str) == data[gt_col].astype(str)
    return float(correct.mean())


def risk_coverage_curve(
    df: pd.DataFrame,
    score_col: str = "trust_score",
    prediction_col: str = "predicted_identity",
    gt_col: str = "ground_truth",
    n_steps: int = 50,
) -> pd.DataFrame:
    """
    Risk-Coverage curve via selective prediction on Trust Score.

    Coverage = fraction of samples kept (score >= threshold).
    Accuracy = accuracy on the kept subset.
    Risk = 1 - accuracy on the kept subset.
    """
    data = df.dropna(subset=[gt_col, score_col]).copy()
    if len(data) == 0:
        return pd.DataFrame(columns=["threshold", "coverage", "accuracy", "risk", "n_kept"])

    scores = data[score_col].astype(float).to_numpy()
    thresholds = np.quantile(scores, np.linspace(0.0, 1.0, n_steps))
    # Also include endpoints
    thresholds = np.unique(np.concatenate([[scores.min() - 1e-9], thresholds, [scores.max() + 1e-9]]))

    rows = []
    n = len(data)
    preds = data[prediction_col].fillna("__none__").astype(str).to_numpy()
    gts = data[gt_col].astype(str).to_numpy()
    correct = preds == gts

    for thr in thresholds:
        mask = scores >= thr
        kept = int(mask.sum())
        coverage = kept / n
        if kept == 0:
            acc = float("nan")
            risk = float("nan")
        else:
            acc = float(correct[mask].mean())
            risk = 1.0 - acc
        rows.append(
            {
                "threshold": float(thr),
                "coverage": float(coverage),
                "accuracy": acc,
                "risk": risk,
                "n_kept": kept,
            }
        )
    return pd.DataFrame(rows)


def binary_far_frr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    FAR / FRR over score thresholds for a binary verification framing.

    Here ``y_true=1`` means the ArcFace prediction was correct, and a positive
    trust decision means "accept the prediction". Then:

    - FAR: accept when incorrect  (false accept of a bad recognition)
    - FRR: reject when correct    (false reject of a good recognition)
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()
    if thresholds is None:
        thresholds = np.unique(y_score)
        thresholds = np.concatenate([[y_score.min() - 1e-9], thresholds, [y_score.max() + 1e-9]])

    n_pos = max(int((y_true == 1).sum()), 1)
    n_neg = max(int((y_true == 0).sum()), 1)
    rows = []
    for thr in thresholds:
        accept = y_score >= thr
        # False accept: accepted & incorrect
        far = float(((accept) & (y_true == 0)).sum() / n_neg)
        # False reject: rejected & correct
        frr = float(((~accept) & (y_true == 1)).sum() / n_pos)
        rows.append({"threshold": float(thr), "FAR": far, "FRR": frr})
    return pd.DataFrame(rows)


def equal_error_rate(far_frr: pd.DataFrame) -> dict[str, float]:
    """Estimate EER as the operating point where |FAR - FRR| is minimized."""
    if far_frr.empty:
        return {"EER": float("nan"), "threshold": float("nan"), "FAR": float("nan"), "FRR": float("nan")}
    diff = np.abs(far_frr["FAR"].to_numpy() - far_frr["FRR"].to_numpy())
    idx = int(np.argmin(diff))
    row = far_frr.iloc[idx]
    eer = float(0.5 * (row["FAR"] + row["FRR"]))
    return {
        "EER": eer,
        "threshold": float(row["threshold"]),
        "FAR": float(row["FAR"]),
        "FRR": float(row["FRR"]),
    }


def error_vs_reject_curve(
    df: pd.DataFrame,
    score_col: str = "trust_score",
    prediction_col: str = "predicted_identity_raw",
    gt_col: str = "ground_truth",
    n_steps: int = 50,
    reject_rates: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Error-vs-Reject Curve (ERC) used in FIQA / NIST FRVT-QA style reporting.

    Images are ranked by ``score_col`` (higher = more trusted). The lowest-
    scoring fraction is rejected; error rate is computed on the remainder:

    - reject_rate = fraction discarded
    - error_rate  = 1 - accuracy on retained images
    - fnmr_proxy  = same as error_rate for closed-set ID (fraction wrong among kept)
    """
    pred_col = prediction_col if prediction_col in df.columns else "predicted_identity"
    data = df.dropna(subset=[gt_col, score_col]).copy()
    if len(data) == 0:
        return pd.DataFrame(
            columns=["reject_rate", "coverage", "error_rate", "accuracy", "n_kept", "threshold"]
        )

    if reject_rates is None:
        reject_rates = np.linspace(0.0, 0.9, n_steps)
    else:
        reject_rates = np.asarray(reject_rates, dtype=float)

    scores = data[score_col].astype(float).to_numpy()
    preds = data[pred_col].fillna("__none__").astype(str).to_numpy()
    gts = data[gt_col].astype(str).to_numpy()
    correct = preds == gts
    n = len(data)
    order = np.argsort(scores)  # lowest trust first → rejected first

    rows = []
    for rr in reject_rates:
        rr = float(np.clip(rr, 0.0, 1.0))
        n_reject = int(np.floor(rr * n))
        if n_reject >= n:
            rows.append(
                {
                    "reject_rate": rr,
                    "coverage": 0.0,
                    "error_rate": float("nan"),
                    "accuracy": float("nan"),
                    "n_kept": 0,
                    "threshold": float(scores[order[-1]]) if n else float("nan"),
                }
            )
            continue
        keep_idx = order[n_reject:]
        acc = float(correct[keep_idx].mean())
        thr = float(scores[order[n_reject]]) if n_reject < n else float(scores.max())
        rows.append(
            {
                "reject_rate": rr,
                "coverage": float(len(keep_idx) / n),
                "error_rate": 1.0 - acc,
                "accuracy": acc,
                "n_kept": int(len(keep_idx)),
                "threshold": thr,
            }
        )
    return pd.DataFrame(rows)


def metrics_at_reject_rates(
    df: pd.DataFrame,
    score_col: str = "trust_score",
    prediction_col: str = "predicted_identity_raw",
    gt_col: str = "ground_truth",
    reject_rates: Sequence[float] = (0.10, 0.20, 0.30, 0.40),
) -> dict[str, dict[str, float]]:
    """Accuracy / error at fixed reject rates for paper tables."""
    erc = error_vs_reject_curve(
        df,
        score_col=score_col,
        prediction_col=prediction_col,
        gt_col=gt_col,
        reject_rates=reject_rates,
    )
    out: dict[str, dict[str, float]] = {}
    for _, row in erc.iterrows():
        key = f"{int(round(row['reject_rate'] * 100))}%"
        out[key] = {
            "reject_rate": float(row["reject_rate"]),
            "coverage": float(row["coverage"]),
            "accuracy": float(row["accuracy"]) if pd.notna(row["accuracy"]) else float("nan"),
            "error_rate": float(row["error_rate"]) if pd.notna(row["error_rate"]) else float("nan"),
            "n_kept": float(row["n_kept"]),
            "threshold": float(row["threshold"]) if pd.notna(row["threshold"]) else float("nan"),
        }
    return out


def plot_risk_coverage(
    curve: pd.DataFrame,
    output_path: PathLike,
    title: str = "Risk-Coverage Curve",
) -> Path:
    """Save a Risk-Coverage plot (accuracy vs coverage)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ordered = curve.sort_values("coverage")
    ax.plot(ordered["coverage"], ordered["accuracy"], lw=2, color="#1f4e79")
    ax.set_xlabel("Coverage (fraction of images accepted)")
    ax.set_ylabel("Accuracy on accepted set")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_error_vs_reject(
    erc: pd.DataFrame,
    output_path: PathLike,
    title: str = "Error vs Reject Curve (ERC)",
    extra_curves: Optional[dict[str, pd.DataFrame]] = None,
) -> Path:
    """Save an Error-vs-Reject plot (FIQA / NIST style)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    curves = {"Trust Score": erc}
    if extra_curves:
        curves.update(extra_curves)
    for label, curve in curves.items():
        ordered = curve.sort_values("reject_rate")
        ax.plot(ordered["reject_rate"], ordered["error_rate"], lw=2, label=label)
    ax.set_xlabel("Reject rate (fraction of images discarded)")
    ax.set_ylabel("Error rate on retained images")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(0.05, float(erc["error_rate"].dropna().max()) * 1.15 if len(erc) else 1.0))
    ax.grid(True, alpha=0.3)
    if len(curves) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_far_frr(
    far_frr: pd.DataFrame,
    eer: dict[str, float],
    output_path: PathLike,
    title: str = "FAR / FRR vs Trust Threshold",
) -> Path:
    """Save FAR/FRR curves with EER marker."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(far_frr["threshold"], far_frr["FAR"], label="FAR", lw=2)
    ax.plot(far_frr["threshold"], far_frr["FRR"], label="FRR", lw=2)
    if not np.isnan(eer.get("threshold", np.nan)):
        ax.axvline(eer["threshold"], color="gray", ls="--", label=f"EER≈{eer['EER']:.3f}")
    ax.set_xlabel("Trust Score threshold")
    ax.set_ylabel("Error rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def evaluate_recognition_df(
    df: pd.DataFrame,
    results_dir: PathLike = "results",
    n_steps: int = 50,
    prefix: str = "run",
) -> dict[str, Any]:
    """
    Full evaluation suite for a recognition results table.

    Expects columns: predicted_identity, ground_truth, trust_score, and
    optionally a baseline ``correct_raw`` column for ungated ArcFace labels.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    fig_dir = results_dir / "figures"
    met_dir = results_dir / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    # Baseline accuracy: treat empty prediction as wrong; ignore trust gate
    baseline_df = df.copy()
    if "similarity" in baseline_df.columns and "predicted_identity" in baseline_df.columns:
        # If trust already nullified predictions, prefer a raw column when present
        pred_col = "predicted_identity_raw" if "predicted_identity_raw" in baseline_df.columns else "predicted_identity"
    else:
        pred_col = "predicted_identity"

    baseline_acc = identification_accuracy(baseline_df, prediction_col=pred_col)

    # Gated accuracy at default accept if column present
    gated_acc = float("nan")
    if "accepted" in df.columns:
        gated_acc = identification_accuracy(df, only_accepted=True)

    # Correctness labels for FAR/FRR: prefer explicit binary label
    if "label_correct" in df.columns:
        y_true = df["label_correct"].astype(int).to_numpy()
    else:
        raw_pred = df.get("predicted_identity_raw", df["predicted_identity"])
        y_true = (
            raw_pred.fillna("__none__").astype(str) == df["ground_truth"].astype(str)
        ).astype(int).to_numpy()

    if "trust_score" not in df.columns or df["trust_score"].isna().all():
        metrics = {
            "baseline_accuracy": baseline_acc,
            "gated_accuracy_accepted_only": gated_acc,
            "EER": None,
            "n_samples": int(len(df)),
        }
        out_json = met_dir / f"{prefix}_metrics.json"
        out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    score_col = "trust_score"
    pred_for_erc = (
        "predicted_identity_raw" if "predicted_identity_raw" in df.columns else pred_col
    )

    curve = risk_coverage_curve(df, score_col=score_col, n_steps=n_steps)
    erc = error_vs_reject_curve(
        df,
        score_col=score_col,
        prediction_col=pred_for_erc,
        n_steps=n_steps,
    )
    far_frr = binary_far_frr(y_true, df[score_col].astype(float).fillna(0.0).to_numpy())
    eer = equal_error_rate(far_frr)

    rc_path = plot_risk_coverage(curve, fig_dir / f"{prefix}_risk_coverage.png")
    erc_path = plot_error_vs_reject(erc, fig_dir / f"{prefix}_erc.png")
    far_path = plot_far_frr(far_frr, eer, fig_dir / f"{prefix}_far_frr.png")

    curve.to_csv(met_dir / f"{prefix}_risk_coverage.csv", index=False)
    erc.to_csv(met_dir / f"{prefix}_erc.csv", index=False)
    far_frr.to_csv(met_dir / f"{prefix}_far_frr.csv", index=False)

    metrics: dict[str, Any] = {
        "baseline_accuracy": baseline_acc,
        "gated_accuracy_accepted_only": gated_acc,
        "EER": eer,
        "n_samples": int(len(df)),
        "risk_coverage_plot": str(rc_path),
        "erc_plot": str(erc_path),
        "far_frr_plot": str(far_path),
    }
    out_json = met_dir / f"{prefix}_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
