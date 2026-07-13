#!/usr/bin/env python3
"""
Run ArcFace on pristine + corrupted images and build a Trust training table.

For each probe image:
  1. Detect face + extract ArcFace embedding and quality features
  2. Match against an enrollment gallery built from pristine data
  3. Label 1 if predicted identity == ground truth, else 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from _bootstrap import ROOT  # noqa: F401

from src.face_pipeline import FaceRecognitionPipeline
from src.feature_extraction import QUALITY_FEATURE_NAMES, FaceEncoder
from src.utils import identity_from_path, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--corrupted-dir", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="data/features/trust_training_table.csv",
        help="CSV with quality features + correctness labels",
    )
    parser.add_argument(
        "--gallery-images-per-id",
        type=int,
        default=1,
        help="Enrollment images per identity (from pristine set)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--ctx-id", type=int, default=0, help="InsightFace ctx_id (-1 = CPU)")
    return parser.parse_args()


def collect_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    raw_dir = resolve_path(args.raw_dir or cfg["dataset"]["raw_dir"])
    corrupted_dir = resolve_path(args.corrupted_dir or cfg["dataset"]["corrupted_dir"])
    out_path = resolve_path(args.output)

    if not raw_dir.exists():
        print(f"[error] raw dir missing: {raw_dir}", file=sys.stderr)
        return 1

    face_cfg = cfg.get("face", {})
    encoder = FaceEncoder(
        model_name=face_cfg.get("model_name", "buffalo_l"),
        det_size=tuple(face_cfg.get("det_size", [640, 640])),
        ctx_id=args.ctx_id,
    )
    pipeline = FaceRecognitionPipeline(
        encoder=encoder,
        similarity_threshold=float(face_cfg.get("similarity_threshold", 0.35)),
    )

    print(f"Building gallery from {raw_dir} ...")
    gallery = pipeline.enroll_from_directory(
        raw_dir,
        images_per_id=args.gallery_images_per_id,
    )
    print(f"Gallery size: {len(gallery)} identities")
    if not gallery:
        print("[error] empty gallery — check detections / dataset layout", file=sys.stderr)
        return 1

    # Probe sets: remaining pristine (optional) + all corrupted
    # Use all images under raw + corrupted as probes; gallery already used first N per ID.
    probe_roots = []
    if corrupted_dir.exists():
        probe_roots.append(("corrupted", corrupted_dir))
    else:
        print(f"[warn] corrupted dir missing ({corrupted_dir}); labeling pristine only")
    probe_roots.append(("pristine", raw_dir))

    rows = []
    seen = 0
    for split_name, root in probe_roots:
        paths = collect_images(root)
        if args.limit is not None:
            remaining = max(args.limit - seen, 0)
            paths = paths[:remaining]
        for path in tqdm(paths, desc=f"label:{split_name}"):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            gt = identity_from_path(path)
            # Bypass trust gate — we need raw ArcFace correctness labels
            result = pipeline.recognize_image(
                image,
                image_path=path,
                ground_truth=gt,
                apply_trust_gate=False,
            )
            label = int(bool(result.correct)) if result.correct is not None else 0
            row = {
                "image_path": str(path),
                "split": split_name,
                "ground_truth": gt,
                "predicted_identity": result.predicted_identity,
                "similarity": result.similarity,
                "face_found": result.face_found,
                "label_correct": label,
            }
            for name in QUALITY_FEATURE_NAMES:
                row[name] = float(result.quality.get(name, 0.0)) if result.face_found else 0.0
            rows.append(row)
            seen += 1
            if args.limit is not None and seen >= args.limit:
                break
        if args.limit is not None and seen >= args.limit:
            break

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    n = len(df)
    n_pos = int(df["label_correct"].sum()) if n else 0
    summary = {
        "n_samples": n,
        "n_correct": n_pos,
        "accuracy": float(n_pos / n) if n else None,
        "gallery_size": len(gallery),
        "output": str(out_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
