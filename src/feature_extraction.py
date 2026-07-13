"""
Quality feature extraction and ArcFace embedding helpers (InsightFace).

The Trust Score predictor consumes handcrafted quality cues — not just
embedding similarity — so failures under blur, pose, and lighting become
predictable before a high-confidence mistake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from .utils import PathLike, cosine_similarity, l2_normalize

ImageArray = np.ndarray

# Ordered feature names consumed by the Trust Score classifier.
QUALITY_FEATURE_NAMES: tuple[str, ...] = (
    "blur_laplacian_var",
    "brightness",
    "contrast",
    "yaw",
    "pitch",
    "roll",
    "face_size_ratio",
    "det_score",
    "entropy",
)


@dataclass
class FaceObservation:
    """Detection + embedding + quality features for one face in an image."""

    bbox: np.ndarray  # [x1, y1, x2, y2]
    det_score: float
    embedding: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    kps: Optional[np.ndarray] = None
    quality: dict[str, float] = field(default_factory=dict)

    def quality_vector(self, names: Sequence[str] = QUALITY_FEATURE_NAMES) -> np.ndarray:
        """Return quality features as a float32 vector in ``names`` order."""
        return np.asarray([float(self.quality.get(n, 0.0)) for n in names], dtype=np.float32)

    def to_dict(self, include_embedding: bool = False) -> dict[str, Any]:
        payload = {
            "bbox": self.bbox.tolist(),
            "det_score": float(self.det_score),
            "yaw": float(self.yaw),
            "pitch": float(self.pitch),
            "roll": float(self.roll),
            "quality": dict(self.quality),
        }
        if include_embedding:
            payload["embedding"] = self.embedding.astype(np.float32).tolist()
        return payload


# ---------------------------------------------------------------------------
# Handcrafted image / face quality features
# ---------------------------------------------------------------------------


def variance_of_laplacian(gray: ImageArray) -> float:
    """Blur score: higher = sharper. Classic focus measure."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(gray: ImageArray) -> float:
    """Mean pixel intensity in [0, 255]."""
    return float(np.mean(gray))


def contrast_score(gray: ImageArray) -> float:
    """Std-dev of pixel intensities (simple global contrast)."""
    return float(np.std(gray))


def image_entropy(gray: ImageArray, bins: int = 256) -> float:
    """Shannon entropy of the grayscale histogram (bits)."""
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel()
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def face_size_ratio(bbox: Sequence[float], image_shape: tuple[int, ...]) -> float:
    """Bounding-box area relative to full image area."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    denom = float(max(h * w, 1))
    return float(area / denom)


def crop_face(
    image: ImageArray,
    bbox: Sequence[float],
    margin: float = 0.15,
) -> ImageArray:
    """Crop face ROI with a relative margin, clipped to image bounds."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - margin * bw))
    y1 = max(0, int(y1 - margin * bh))
    x2 = min(w, int(x2 + margin * bw))
    y2 = min(h, int(y2 + margin * bh))
    if x2 <= x1 or y2 <= y1:
        return image.copy()
    return image[y1:y2, x1:x2].copy()


def extract_quality_features(
    image: ImageArray,
    bbox: Sequence[float],
    det_score: float = 0.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    use_face_crop: bool = True,
) -> dict[str, float]:
    """
    Compute the Trust Module quality feature dict for one detected face.

    Blur / brightness / contrast / entropy are measured on the face crop when
    possible; pose, size, and detection score come from the detector.
    """
    roi = crop_face(image, bbox) if use_face_crop else image
    if roi.size == 0:
        roi = image
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

    return {
        "blur_laplacian_var": variance_of_laplacian(gray),
        "brightness": brightness_score(gray),
        "contrast": contrast_score(gray),
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll),
        "face_size_ratio": face_size_ratio(bbox, image.shape),
        "det_score": float(det_score),
        "entropy": image_entropy(gray),
    }


def extract_quality_features_from_image(
    image: ImageArray,
    bbox: Optional[Sequence[float]] = None,
) -> dict[str, float]:
    """
    Quality features when no detector is available.

    If ``bbox`` is None, uses the full frame (det_score/pose default to 0).
    """
    if bbox is None:
        h, w = image.shape[:2]
        bbox = (0.0, 0.0, float(w), float(h))
    return extract_quality_features(image, bbox)


# ---------------------------------------------------------------------------
# InsightFace / ArcFace backend
# ---------------------------------------------------------------------------


class FaceEncoder:
    """
    Thin wrapper around InsightFace ``FaceAnalysis`` (RetinaFace + ArcFace).

    Lazy-loads the model pack on first use so importing this module stays cheap.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        ctx_id: int = 0,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_name = model_name
        self.det_size = tuple(det_size)
        self.ctx_id = ctx_id
        self.providers = list(providers) if providers else None
        self._app = None

    def _ensure_app(self):
        if self._app is not None:
            return self._app
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface is required for FaceEncoder. "
                "Install with: pip install insightface onnxruntime"
            ) from exc

        kwargs: dict[str, Any] = {"name": self.model_name}
        if self.providers is not None:
            kwargs["providers"] = self.providers
        app = FaceAnalysis(**kwargs)
        app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        self._app = app
        return app

    @property
    def app(self):
        return self._ensure_app()

    def detect_and_embed(
        self,
        image: ImageArray,
        max_faces: Optional[int] = None,
    ) -> list[FaceObservation]:
        """
        Run detection + ArcFace embedding and attach quality features.

        Faces are sorted by detection score (descending).
        """
        faces = self.app.get(image)
        faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
        if max_faces is not None:
            faces = faces[: max(0, int(max_faces))]

        observations: list[FaceObservation] = []
        for face in faces:
            bbox = np.asarray(face.bbox, dtype=np.float32)
            det_score = float(face.det_score)
            embedding = l2_normalize(np.asarray(face.embedding, dtype=np.float32))

            yaw = pitch = roll = 0.0
            pose = getattr(face, "pose", None)
            if pose is not None:
                pose_arr = np.asarray(pose, dtype=np.float32).ravel()
                if pose_arr.size >= 3:
                    # InsightFace pose is typically (pitch, yaw, roll)
                    pitch, yaw, roll = float(pose_arr[0]), float(pose_arr[1]), float(pose_arr[2])

            quality = extract_quality_features(
                image,
                bbox=bbox,
                det_score=det_score,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
            )
            kps = None
            if getattr(face, "kps", None) is not None:
                kps = np.asarray(face.kps, dtype=np.float32)

            observations.append(
                FaceObservation(
                    bbox=bbox,
                    det_score=det_score,
                    embedding=embedding,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    kps=kps,
                    quality=quality,
                )
            )
        return observations

    def primary_face(self, image: ImageArray) -> Optional[FaceObservation]:
        """Return the highest-confidence face, or None if no detection."""
        faces = self.detect_and_embed(image, max_faces=1)
        return faces[0] if faces else None

    def embed_path(self, path: PathLike, max_faces: int = 1) -> list[FaceObservation]:
        """Load an image from disk and run ``detect_and_embed``."""
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return self.detect_and_embed(image, max_faces=max_faces)


def match_embedding(
    query: np.ndarray,
    gallery: dict[str, np.ndarray],
    threshold: Optional[float] = None,
) -> tuple[Optional[str], float]:
    """
    1:N identification against an identity → embedding gallery.

    Returns ``(best_identity_or_None, best_similarity)``. If ``threshold`` is
    set and the best score is below it, identity is returned as None.
    """
    if not gallery:
        return None, 0.0

    best_id: Optional[str] = None
    best_sim = -1.0
    q = l2_normalize(query)
    for identity, emb in gallery.items():
        sim = cosine_similarity(q, emb)
        if sim > best_sim:
            best_sim = sim
            best_id = identity

    if threshold is not None and best_sim < threshold:
        return None, float(best_sim)
    return best_id, float(best_sim)


def build_gallery(
    identity_embeddings: dict[str, Sequence[np.ndarray]],
) -> dict[str, np.ndarray]:
    """
    Build a gallery by mean-pooling (then L2-normalizing) embeddings per ID.
    """
    gallery: dict[str, np.ndarray] = {}
    for identity, embs in identity_embeddings.items():
        if not embs:
            continue
        stacked = np.stack([l2_normalize(e) for e in embs], axis=0)
        mean_emb = l2_normalize(stacked.mean(axis=0))
        gallery[identity] = mean_emb
    return gallery


__all__ = [
    "QUALITY_FEATURE_NAMES",
    "FaceObservation",
    "variance_of_laplacian",
    "brightness_score",
    "contrast_score",
    "image_entropy",
    "face_size_ratio",
    "crop_face",
    "extract_quality_features",
    "extract_quality_features_from_image",
    "FaceEncoder",
    "match_embedding",
    "build_gallery",
]
