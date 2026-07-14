"""
Trust Score predictor: maps quality features → P(correct recognition).

Supports Random Forest, XGBoost, and a small scikit-learn MLP. Models are
serialized with joblib for reproducible evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_extraction import QUALITY_FEATURE_NAMES
from .utils import PathLike

ModelType = Literal["random_forest", "xgboost", "mlp"]


def _make_estimator(
    model_type: ModelType,
    random_state: int = 42,
    scale_pos_weight: Optional[float] = None,
) -> Any:
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state,
        )
    if model_type == "mlp":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-4,
                        max_iter=400,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for model_type='xgboost'. "
                "Install with: pip install xgboost"
            ) from exc
        kwargs: dict[str, Any] = dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
            tree_method="hist",
        )
        if scale_pos_weight is not None:
            kwargs["scale_pos_weight"] = float(scale_pos_weight)
        return XGBClassifier(**kwargs)
    raise ValueError(f"Unknown model_type: {model_type}")


@dataclass
class TrustPredictor:
    """Lightweight wrapper around a binary classifier for Trust Scores."""

    model_type: ModelType = "xgboost"
    feature_names: Sequence[str] = QUALITY_FEATURE_NAMES
    random_state: int = 42
    estimator: Any = None
    balance_classes: bool = True
    scale_pos_weight: Optional[float] = None

    def __post_init__(self) -> None:
        self.feature_names = tuple(self.feature_names)
        if self.estimator is None:
            self.estimator = _make_estimator(
                self.model_type,
                self.random_state,
                scale_pos_weight=self.scale_pos_weight,
            )

    def _as_matrix(self, X: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        expected = len(self.feature_names)
        if arr.shape[1] != expected:
            raise ValueError(f"Expected {expected} features, got {arr.shape[1]}")
        return arr

    def fit(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]]],
        y: Union[np.ndarray, Sequence[int]],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "TrustPredictor":
        """Fit on quality features ``X`` and correctness labels ``y`` ∈ {0, 1}."""
        X_mat = self._as_matrix(X)
        y_arr = np.asarray(y).astype(int).ravel()
        if set(np.unique(y_arr)) - {0, 1}:
            raise ValueError("Labels must be binary {0, 1}")

        # Rebuild XGBoost with scale_pos_weight from the observed label ratio
        if self.balance_classes and self.model_type == "xgboost" and self.estimator is not None:
            n_pos = max(int((y_arr == 1).sum()), 1)
            n_neg = max(int((y_arr == 0).sum()), 1)
            # Incorrect (0) is minority → weight positives down relative to negatives
            # scale_pos_weight = n_neg / n_pos weights the positive class; for rare
            # failures we want to emphasize class 0, so use sample weights instead.
            sw = np.ones(len(y_arr), dtype=np.float64)
            # Weight each class inversely to frequency
            sw[y_arr == 0] = n_pos / (2.0 * n_neg)
            sw[y_arr == 1] = n_neg / (2.0 * n_pos)
            if sample_weight is not None:
                sw = sw * np.asarray(sample_weight, dtype=np.float64)
            sample_weight = sw
            self.scale_pos_weight = float(n_neg / n_pos)

        if self.balance_classes and self.model_type == "mlp" and sample_weight is None:
            n_pos = max(int((y_arr == 1).sum()), 1)
            n_neg = max(int((y_arr == 0).sum()), 1)
            sw = np.ones(len(y_arr), dtype=np.float64)
            sw[y_arr == 0] = n_pos / (2.0 * n_neg)
            sw[y_arr == 1] = n_neg / (2.0 * n_pos)
            sample_weight = sw

        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            if isinstance(self.estimator, Pipeline):
                fit_kwargs["clf__sample_weight"] = sample_weight
            else:
                fit_kwargs["sample_weight"] = sample_weight
        try:
            self.estimator.fit(X_mat, y_arr, **fit_kwargs)
        except TypeError:
            self.estimator.fit(X_mat, y_arr)
        return self

    def predict_trust(self, X: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        """Return P(correct) for each row in ``X``."""
        X_mat = self._as_matrix(X)
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X_mat)
            classes = list(getattr(self.estimator, "classes_", [0, 1]))
            # Handle Pipeline: classes_ lives on the final estimator
            if isinstance(self.estimator, Pipeline):
                classes = list(self.estimator.named_steps["clf"].classes_)
            if 1 in classes:
                idx = classes.index(1)
                return np.asarray(proba[:, idx], dtype=np.float64)
            return np.asarray(proba[:, -1], dtype=np.float64)
        # Fallback: treat decision_function / predict as score
        if hasattr(self.estimator, "decision_function"):
            scores = np.asarray(self.estimator.decision_function(X_mat), dtype=np.float64)
            return 1.0 / (1.0 + np.exp(-scores))
        return np.asarray(self.estimator.predict(X_mat), dtype=np.float64)

    def predict(self, X: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        """Hard 0/1 prediction at 0.5 trust threshold."""
        return (self.predict_trust(X) >= 0.5).astype(int)

    def predict_proba(self, X: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        """sklearn-compatible ``(n, 2)`` probabilities for [P(0), P(1)]."""
        p1 = self.predict_trust(X)
        return np.column_stack([1.0 - p1, p1])

    def save(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_type": self.model_type,
                "feature_names": list(self.feature_names),
                "random_state": self.random_state,
                "estimator": self.estimator,
                "balance_classes": self.balance_classes,
                "scale_pos_weight": self.scale_pos_weight,
            },
            path,
        )

    @classmethod
    def load(cls, path: PathLike) -> "TrustPredictor":
        payload = joblib.load(path)
        obj = cls(
            model_type=payload["model_type"],
            feature_names=payload["feature_names"],
            random_state=payload.get("random_state", 42),
            estimator=payload["estimator"],
            balance_classes=payload.get("balance_classes", True),
            scale_pos_weight=payload.get("scale_pos_weight"),
        )
        return obj


def train_trust_predictor(
    X: np.ndarray,
    y: np.ndarray,
    model_type: ModelType = "xgboost",
    random_state: int = 42,
    balance_classes: bool = True,
) -> TrustPredictor:
    """Convenience: construct, fit, and return a TrustPredictor."""
    model = TrustPredictor(
        model_type=model_type,
        random_state=random_state,
        balance_classes=balance_classes,
    )
    model.fit(X, y)
    return model
