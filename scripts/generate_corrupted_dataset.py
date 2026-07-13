#!/usr/bin/env python3
"""Generate a corrupted twin of a face dataset for Trust Score experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401  # adds project root to sys.path

from src.data_corruption import CorruptionType, generate_corrupted_dataset
from src.utils import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--input-dir", type=str, default=None, help="Pristine images root")
    parser.add_argument("--output-dir", type=str, default=None, help="Corrupted output root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-pristine-ratio",
        type=float,
        default=None,
        help="Fraction of images left uncorrupted",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        help="Subset of corruption types (default: all)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/corrupted/corruption_manifest.jsonl",
        help="JSONL path for per-image corruption metadata",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    input_dir = resolve_path(args.input_dir or cfg["dataset"]["raw_dir"])
    output_dir = resolve_path(args.output_dir or cfg["dataset"]["corrupted_dir"])
    keep = (
        args.keep_pristine_ratio
        if args.keep_pristine_ratio is not None
        else float(cfg["corruption"].get("keep_pristine_ratio", 0.0))
    )
    types = args.types
    if types is None:
        types = [t.value for t in CorruptionType if t is not CorruptionType.NONE]

    if not input_dir.exists():
        print(f"[error] input dir not found: {input_dir}", file=sys.stderr)
        print("Place LFW (or CelebA subset) under data/raw/ first.", file=sys.stderr)
        return 1

    print(f"Corrupting images from {input_dir} → {output_dir}")
    records = generate_corrupted_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        corruption_types=types,
        seed=args.seed,
        keep_pristine_ratio=keep,
    )

    manifest_path = resolve_path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    counts: dict[str, int] = {}
    for rec in records:
        counts[rec["corruption"]] = counts.get(rec["corruption"], 0) + 1

    print(f"Wrote {len(records)} images. Manifest: {manifest_path}")
    for name, n in sorted(counts.items()):
        print(f"  {name}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
