#!/usr/bin/env python3
"""
Single-image Trust Score demo.

Loads a gallery from pristine LFW, runs ArcFace + Trust gating on one probe,
and prints identity / similarity / trust / accept-reject.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from _bootstrap import ROOT  # noqa: F401

from src.face_pipeline import FaceRecognitionPipeline
from src.feature_extraction import FaceEncoder
from src.trust_predictor import TrustPredictor
from src.utils import identity_from_path, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=str, help="Path to a face image")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--gallery-dir",
        type=str,
        default=None,
        help="Enrollment directory (default: config dataset.raw_dir)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/trust_predictor.joblib",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--gallery-images-per-id",
        type=int,
        default=1,
        help="Enrollment images per identity (default: 1)",
    )
    parser.add_argument("--ctx-id", type=int, default=-1, help="InsightFace ctx (-1=CPU)")
    parser.add_argument(
        "--gallery-cache",
        type=str,
        default="models/demo_gallery.npz",
        help="Optional cache path for gallery embeddings",
    )
    parser.add_argument(
        "--rebuild-gallery",
        action="store_true",
        help="Ignore gallery cache and rebuild from --gallery-dir",
    )
    return parser.parse_args()


def _load_or_build_gallery(
    pipeline: FaceRecognitionPipeline,
    gallery_dir: Path,
    cache_path: Path,
    images_per_id: int,
    rebuild: bool,
) -> dict:
    import numpy as np

    if cache_path.exists() and not rebuild:
        data = np.load(cache_path, allow_pickle=True)
        ids = data["identities"].tolist()
        embs = data["embeddings"]
        gallery = {i: embs[k] for k, i in enumerate(ids)}
        pipeline.set_gallery(gallery)
        print(f"[demo] loaded gallery cache ({len(gallery)} IDs): {cache_path}")
        return gallery

    print(f"[demo] building gallery from {gallery_dir} …")
    gallery = pipeline.enroll_from_directory(gallery_dir, images_per_id=images_per_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(gallery.keys())
    if ids:
        np.savez_compressed(
            cache_path,
            identities=np.array(ids, dtype=object),
            embeddings=np.stack([gallery[i] for i in ids], axis=0),
        )
        print(f"[demo] saved gallery cache ({len(ids)} IDs): {cache_path}")
    return gallery


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    image_path = resolve_path(args.image)
    gallery_dir = resolve_path(args.gallery_dir or cfg["dataset"]["raw_dir"])
    model_path = resolve_path(args.model)
    cache_path = resolve_path(args.gallery_cache)
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(cfg.get("trust", {}).get("reject_threshold", 0.5))
    )
    face_cfg = cfg.get("face", {})

    if not image_path.exists():
        print(f"[error] image not found: {image_path}", file=sys.stderr)
        return 1
    if not gallery_dir.exists():
        print(f"[error] gallery dir missing: {gallery_dir}", file=sys.stderr)
        return 1
    if not model_path.exists():
        print(f"[error] trust model missing: {model_path}", file=sys.stderr)
        print("Train first: python scripts/train_trust_module.py --features quality_sim", file=sys.stderr)
        return 1

    trust_model = TrustPredictor.load(model_path)
    encoder = FaceEncoder(
        model_name=face_cfg.get("model_name", "buffalo_l"),
        det_size=tuple(face_cfg.get("det_size", [640, 640])),
        ctx_id=args.ctx_id,
    )
    pipeline = FaceRecognitionPipeline(
        encoder=encoder,
        similarity_threshold=float(face_cfg.get("similarity_threshold", 0.35)),
        trust_model=trust_model,
        trust_threshold=threshold,
    )
    _load_or_build_gallery(
        pipeline,
        gallery_dir,
        cache_path,
        images_per_id=args.gallery_images_per_id,
        rebuild=args.rebuild_gallery,
    )

    try:
        gt = identity_from_path(image_path)
    except ValueError:
        gt = None

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    result = pipeline.recognize_image(
        image,
        image_path=image_path,
        ground_truth=gt,
        apply_trust_gate=True,
    )

    payload = {
        "image": str(image_path),
        "ground_truth": result.ground_truth,
        "predicted_identity": result.predicted_identity,
        "similarity": result.similarity,
        "trust_score": result.trust_score,
        "trust_threshold": threshold,
        "accepted": result.accepted,
        "face_found": result.face_found,
        "det_score": result.det_score,
        "correct": result.correct,
        "decision": (
            "ACCEPT"
            if result.accepted
            else ("REJECT" if result.accepted is False else "NO_TRUST_MODEL")
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
