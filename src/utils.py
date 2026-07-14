"""Shared helpers: config loading, path utils, and small numeric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import yaml

PathLike = Union[str, Path]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Optional[PathLike] = None) -> dict[str, Any]:
    """Load a YAML experiment config (defaults to configs/default.yaml)."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    return data


def resolve_path(path: PathLike, root: Optional[PathLike] = None) -> Path:
    """Resolve a path relative to the project root unless already absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    base = Path(root) if root is not None else PROJECT_ROOT
    return (base / p).resolve()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D embedding vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def l2_normalize(embedding: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return an L2-normalized copy of ``embedding``."""
    vec = np.asarray(embedding, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec.copy()
    return vec / norm


def identity_from_path(path: PathLike, depth: int = 1) -> str:
    """
    Infer identity label from a dataset path.

    For LFW-style trees (.../identity_name/img_0001.jpg) ``depth=1`` returns
    the parent folder name.
    """
    parts = Path(path).resolve().parts
    if len(parts) <= depth:
        raise ValueError(f"Cannot infer identity from path: {path}")
    return parts[-(depth + 1)]


def identity_disjoint_split(
    identities: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split row indices so no identity appears in both train and test.

    ``identities`` is length-N (one label per row). Returns boolean masks
    ``(train_mask, test_mask)``.
    """
    from sklearn.model_selection import train_test_split

    identities = np.asarray(identities)
    n = len(identities)
    unique = np.unique(identities)
    if len(unique) < 2:
        raise ValueError("Need at least 2 identities for an identity-disjoint split")

    id_train, id_test = train_test_split(
        unique,
        test_size=test_size,
        random_state=random_state,
    )
    test_set = set(id_test.tolist())
    test_mask = np.array([i in test_set for i in identities], dtype=bool)
    train_mask = ~test_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Identity split produced an empty partition")
    return train_mask, test_mask
