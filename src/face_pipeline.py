"""
End-to-end face recognition pipeline with optional Trust Score gating.

Flow: image → RetinaFace detect → ArcFace embed → gallery match →
quality features → Trust Score → accept / reject.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from .feature_extraction import (
    QUALITY_FEATURE_NAMES,
    FaceEncoder,
    FaceObservation,
    build_gallery,
    match_embedding,
)
from .utils import PathLike, identity_from_path, l2_normalize

ImageArray = np.ndarray


@dataclass
class RecognitionResult:
    """Single-image recognition output for evaluation / logging."""

    image_path: Optional[str]
    predicted_identity: Optional[str]
    similarity: float
    ground_truth: Optional[str] = None
    correct: Optional[bool] = None
    trust_score: Optional[float] = None
    accepted: Optional[bool] = None
    det_score: float = 0.0
    quality: dict[str, float] = field(default_factory=dict)
    face_found: bool = False
    embedding: Optional[np.ndarray] = None

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "image_path": self.image_path,
            "predicted_identity": self.predicted_identity,
            "similarity": float(self.similarity),
            "ground_truth": self.ground_truth,
            "correct": self.correct,
            "trust_score": self.trust_score,
            "accepted": self.accepted,
            "det_score": float(self.det_score),
            "face_found": bool(self.face_found),
        }
        for name in QUALITY_FEATURE_NAMES:
            row[name] = float(self.quality.get(name, 0.0))
        return row


class FaceRecognitionPipeline:
    """
    Recognition against an in-memory gallery, with optional trust gating.

    ``trust_model`` must expose ``predict_proba(X) -> (n, 2)`` or
    ``predict_trust(X) -> (n,)`` returning P(correct).
    """

    def __init__(
        self,
        encoder: Optional[FaceEncoder] = None,
        gallery: Optional[dict[str, np.ndarray]] = None,
        similarity_threshold: float = 0.35,
        trust_model: Any = None,
        trust_threshold: float = 0.5,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = 0,
    ) -> None:
        self.encoder = encoder or FaceEncoder(
            model_name=model_name,
            det_size=det_size,
            ctx_id=ctx_id,
        )
        self.gallery: dict[str, np.ndarray] = gallery or {}
        self.similarity_threshold = float(similarity_threshold)
        self.trust_model = trust_model
        self.trust_threshold = float(trust_threshold)

    def set_gallery(self, gallery: dict[str, np.ndarray]) -> None:
        self.gallery = {k: l2_normalize(v) for k, v in gallery.items()}

    def enroll_from_directory(
        self,
        root: PathLike,
        images_per_id: Optional[int] = None,
        identity_depth: int = 1,
    ) -> dict[str, np.ndarray]:
        """
        Build a gallery from an LFW-style directory tree.

        Each immediate subfolder name is treated as an identity. Optionally
        limit how many enrollment images are used per identity.
        """
        root = Path(root)
        buckets: dict[str, list[np.ndarray]] = {}
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in exts):
            identity = identity_from_path(path, depth=identity_depth)
            if images_per_id is not None and len(buckets.get(identity, [])) >= images_per_id:
                continue
            obs = self.encoder.primary_face(cv2.imread(str(path), cv2.IMREAD_COLOR))
            if obs is None:
                continue
            buckets.setdefault(identity, []).append(obs.embedding)

        gallery = build_gallery(buckets)
        self.set_gallery(gallery)
        return gallery

    def _predict_trust(self, quality: dict[str, float]) -> Optional[float]:
        if self.trust_model is None:
            return None
        x = np.asarray(
            [[float(quality.get(n, 0.0)) for n in QUALITY_FEATURE_NAMES]],
            dtype=np.float32,
        )
        if hasattr(self.trust_model, "predict_trust"):
            score = self.trust_model.predict_trust(x)
            return float(np.asarray(score).ravel()[0])
        if hasattr(self.trust_model, "predict_proba"):
            proba = self.trust_model.predict_proba(x)
            return float(np.asarray(proba)[0, 1])
        if hasattr(self.trust_model, "predict"):
            return float(np.asarray(self.trust_model.predict(x)).ravel()[0])
        raise TypeError("trust_model must implement predict_trust, predict_proba, or predict")

    def recognize_image(
        self,
        image: ImageArray,
        image_path: Optional[PathLike] = None,
        ground_truth: Optional[str] = None,
        apply_trust_gate: bool = True,
    ) -> RecognitionResult:
        """Run detection, matching, and optional trust rejection on one image."""
        path_str = str(image_path) if image_path is not None else None
        if image is None:
            return RecognitionResult(
                image_path=path_str,
                predicted_identity=None,
                similarity=0.0,
                ground_truth=ground_truth,
                correct=False if ground_truth is not None else None,
                face_found=False,
            )

        obs = self.encoder.primary_face(image)
        if obs is None:
            result = RecognitionResult(
                image_path=path_str,
                predicted_identity=None,
                similarity=0.0,
                ground_truth=ground_truth,
                correct=False if ground_truth is not None else None,
                face_found=False,
            )
            return result

        pred_id, sim = match_embedding(
            obs.embedding,
            self.gallery,
            threshold=self.similarity_threshold,
        )
        trust = self._predict_trust(obs.quality)
        accepted: Optional[bool] = None
        final_pred = pred_id
        if apply_trust_gate and trust is not None:
            accepted = bool(trust >= self.trust_threshold)
            if not accepted:
                final_pred = None

        correct: Optional[bool] = None
        if ground_truth is not None:
            correct = final_pred is not None and final_pred == ground_truth

        return RecognitionResult(
            image_path=path_str,
            predicted_identity=final_pred,
            similarity=sim,
            ground_truth=ground_truth,
            correct=correct,
            trust_score=trust,
            accepted=accepted,
            det_score=obs.det_score,
            quality=dict(obs.quality),
            face_found=True,
            embedding=obs.embedding,
        )

    def recognize_path(
        self,
        path: PathLike,
        ground_truth: Optional[str] = None,
        identity_depth: int = 1,
        apply_trust_gate: bool = True,
    ) -> RecognitionResult:
        """Load an image from disk and run ``recognize_image``."""
        path = Path(path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if ground_truth is None:
            try:
                ground_truth = identity_from_path(path, depth=identity_depth)
            except ValueError:
                ground_truth = None
        return self.recognize_image(
            image,
            image_path=path,
            ground_truth=ground_truth,
            apply_trust_gate=apply_trust_gate,
        )

    def recognize_directory(
        self,
        root: PathLike,
        identity_depth: int = 1,
        apply_trust_gate: bool = True,
        limit: Optional[int] = None,
    ) -> list[RecognitionResult]:
        """Score all images under ``root`` (sorted for determinism)."""
        root = Path(root)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
        if limit is not None:
            paths = paths[: int(limit)]

        results: list[RecognitionResult] = []
        for path in paths:
            results.append(
                self.recognize_path(
                    path,
                    identity_depth=identity_depth,
                    apply_trust_gate=apply_trust_gate,
                )
            )
        return results


def observations_to_feature_matrix(
    observations: Sequence[FaceObservation],
    names: Sequence[str] = QUALITY_FEATURE_NAMES,
) -> np.ndarray:
    """Stack quality vectors from FaceObservation objects into an (N, F) matrix."""
    if not observations:
        return np.zeros((0, len(names)), dtype=np.float32)
    return np.stack([obs.quality_vector(names) for obs in observations], axis=0)
