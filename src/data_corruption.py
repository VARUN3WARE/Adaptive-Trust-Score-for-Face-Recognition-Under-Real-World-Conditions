"""
Image corruption utilities for trust-score experiments.

Applies controlled degradations that mimic real-world face capture failures
(blur, compression, weather, low light, sensor noise). Used to stress ArcFace
and label when recognition succeeds vs fails for Trust Score training.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

import cv2
import numpy as np

PathLike = Union[str, Path]
ImageArray = np.ndarray


class CorruptionType(str, Enum):
    """Supported degradation families."""

    GAUSSIAN_BLUR = "gaussian_blur"
    JPEG_COMPRESSION = "jpeg_compression"
    RAIN = "rain"
    FOG = "fog"
    LOW_LIGHT = "low_light"
    GAUSSIAN_NOISE = "gaussian_noise"
    NONE = "none"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_image(path: PathLike) -> ImageArray:
    """Load an image as BGR uint8. Raises FileNotFoundError if missing."""
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def save_image(path: PathLike, image: ImageArray) -> None:
    """Write a BGR image to disk, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write image: {path}")


def ensure_uint8(image: ImageArray) -> ImageArray:
    """Clip and cast an image to uint8 without changing channel layout."""
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Individual corruptions
# ---------------------------------------------------------------------------


def apply_gaussian_blur(
    image: ImageArray,
    kernel_size: int = 5,
    sigma: float = 0.0,
) -> ImageArray:
    """
    Apply Gaussian blur.

    Parameters
    ----------
    kernel_size:
        Odd positive integer (3, 5, 7, ...). Even values are bumped to next odd.
    sigma:
        Gaussian std-dev. ``0`` lets OpenCV derive it from kernel_size.
    """
    k = int(kernel_size)
    if k < 1:
        raise ValueError("kernel_size must be >= 1")
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma)


def apply_jpeg_compression(
    image: ImageArray,
    quality: int = 30,
) -> ImageArray:
    """
    Re-encode the image as JPEG at the given quality (1–100) and decode back.

    Lower quality → stronger blocking / ringing artifacts.
    """
    quality = int(np.clip(quality, 1, 100))
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded


def apply_gaussian_noise(
    image: ImageArray,
    std: float = 25.0,
    mean: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> ImageArray:
    """Add i.i.d. Gaussian noise (per channel) and clip to valid range."""
    if std < 0:
        raise ValueError("std must be >= 0")
    rng = rng or np.random.default_rng()
    noise = rng.normal(loc=mean, scale=std, size=image.shape)
    noisy = image.astype(np.float32) + noise.astype(np.float32)
    return ensure_uint8(noisy)


def apply_low_light(
    image: ImageArray,
    brightness_factor: float = 0.4,
    gamma: float = 1.0,
) -> ImageArray:
    """
    Simulate underexposure by scaling intensity and optional gamma darkening.

    Parameters
    ----------
    brightness_factor:
        Multiplicative scale in (0, 1]. Smaller → darker.
    gamma:
        Values > 1 further darken midtones after scaling.
    """
    if not (0.0 < brightness_factor <= 1.0):
        raise ValueError("brightness_factor must be in (0, 1]")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    darkened = image.astype(np.float32) * float(brightness_factor)
    if gamma != 1.0:
        normalized = np.clip(darkened / 255.0, 0.0, 1.0)
        darkened = (normalized ** gamma) * 255.0
    return ensure_uint8(darkened)


def apply_fog(
    image: ImageArray,
    fog_alpha: float = 0.5,
    fog_color: Sequence[int] = (200, 200, 200),
    center_bias: bool = True,
) -> ImageArray:
    """
    Blend the image toward a fog color with optional radial density falloff.

    ``fog_alpha`` in [0, 1] controls overall haze strength (1 = fully fogged).
    """
    if not (0.0 <= fog_alpha <= 1.0):
        raise ValueError("fog_alpha must be in [0, 1]")

    h, w = image.shape[:2]
    fog = np.full_like(image, fog_color, dtype=np.float32)

    if center_bias:
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        # Distance from center, normalized to [0, 1]
        dist = np.sqrt(((yy - cy) / max(cy, 1e-6)) ** 2 + ((xx - cx) / max(cx, 1e-6)) ** 2)
        dist = np.clip(dist, 0.0, 1.0)
        # Heavier fog toward edges / distance (simple atmospheric cue)
        weight = fog_alpha * (0.55 + 0.45 * dist)
        weight = weight[..., None]
    else:
        weight = fog_alpha

    blended = image.astype(np.float32) * (1.0 - weight) + fog * weight
    return ensure_uint8(blended)


def apply_rain(
    image: ImageArray,
    density: float = 0.02,
    drop_length: int = 15,
    drop_width: int = 1,
    drop_color: Sequence[int] = (200, 200, 200),
    angle: float = -30.0,
    blur_kernel: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> ImageArray:
    """
    Overlay synthetic rain streaks and lightly blur to soften edges.

    Parameters
    ----------
    density:
        Fraction of pixels that act as rain-drop seeds (approx.).
    angle:
        Streak direction in degrees (negative = typical left-to-right fall).
    """
    if density < 0:
        raise ValueError("density must be >= 0")
    rng = rng or np.random.default_rng()

    h, w = image.shape[:2]
    rain_layer = np.zeros_like(image, dtype=np.uint8)
    n_drops = int(h * w * density)

    # Direction vector for streaks
    rad = np.deg2rad(angle)
    dx = int(np.round(np.cos(rad) * drop_length))
    dy = int(np.round(np.sin(rad) * drop_length))

    xs = rng.integers(0, w, size=n_drops)
    ys = rng.integers(0, h, size=n_drops)

    for x, y in zip(xs, ys):
        x2, y2 = int(x + dx), int(y + dy)
        cv2.line(
            rain_layer,
            (int(x), int(y)),
            (x2, y2),
            tuple(int(c) for c in drop_color),
            thickness=max(1, int(drop_width)),
        )

    if blur_kernel and blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        rain_layer = cv2.blur(rain_layer, (k, k))

    # Screen-like composite: brighten where rain exists
    base = image.astype(np.float32)
    rain = rain_layer.astype(np.float32)
    out = base + rain * (1.0 - base / 255.0)
    return ensure_uint8(out)


# ---------------------------------------------------------------------------
# Dispatch / random corruption
# ---------------------------------------------------------------------------

_CORRUPTION_FN: dict[CorruptionType, Callable[..., ImageArray]] = {
    CorruptionType.GAUSSIAN_BLUR: apply_gaussian_blur,
    CorruptionType.JPEG_COMPRESSION: apply_jpeg_compression,
    CorruptionType.RAIN: apply_rain,
    CorruptionType.FOG: apply_fog,
    CorruptionType.LOW_LIGHT: apply_low_light,
    CorruptionType.GAUSSIAN_NOISE: apply_gaussian_noise,
}


def apply_corruption(
    image: ImageArray,
    corruption: Union[str, CorruptionType],
    **params,
) -> ImageArray:
    """
    Apply a named corruption. Pass ``corruption='none'`` to return a copy.

    Extra keyword arguments are forwarded to the underlying function.
    """
    ctype = CorruptionType(corruption)
    if ctype is CorruptionType.NONE:
        return image.copy()
    fn = _CORRUPTION_FN[ctype]
    return fn(image, **params)


def random_corruption_params(
    corruption: Union[str, CorruptionType],
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Sample severity parameters for a corruption type.

    Ranges are intentionally aggressive so ArcFace failure rates are useful
    for Trust Score supervision.
    """
    rng = rng or np.random.default_rng()
    ctype = CorruptionType(corruption)

    if ctype is CorruptionType.GAUSSIAN_BLUR:
        return {"kernel_size": int(rng.choice([3, 5, 7, 9, 11, 13]))}
    if ctype is CorruptionType.JPEG_COMPRESSION:
        return {"quality": int(rng.integers(8, 46))}
    if ctype is CorruptionType.GAUSSIAN_NOISE:
        return {"std": float(rng.uniform(8.0, 45.0))}
    if ctype is CorruptionType.LOW_LIGHT:
        return {
            "brightness_factor": float(rng.uniform(0.15, 0.55)),
            "gamma": float(rng.uniform(1.0, 1.8)),
        }
    if ctype is CorruptionType.FOG:
        return {"fog_alpha": float(rng.uniform(0.25, 0.75))}
    if ctype is CorruptionType.RAIN:
        return {
            "density": float(rng.uniform(0.008, 0.045)),
            "drop_length": int(rng.integers(10, 22)),
            "angle": float(rng.uniform(-40.0, -15.0)),
        }
    if ctype is CorruptionType.NONE:
        return {}
    raise ValueError(f"Unsupported corruption: {ctype}")


def apply_random_corruption(
    image: ImageArray,
    corruption_types: Optional[Sequence[Union[str, CorruptionType]]] = None,
    rng: Optional[np.random.Generator] = None,
    include_none: bool = False,
) -> tuple[ImageArray, CorruptionType, dict]:
    """
    Sample a corruption type + severity and apply it.

    Returns
    -------
    corrupted, chosen_type, params
    """
    rng = rng or np.random.default_rng()
    if corruption_types is None:
        types = [
            CorruptionType.GAUSSIAN_BLUR,
            CorruptionType.JPEG_COMPRESSION,
            CorruptionType.RAIN,
            CorruptionType.FOG,
            CorruptionType.LOW_LIGHT,
            CorruptionType.GAUSSIAN_NOISE,
        ]
    else:
        types = [CorruptionType(t) for t in corruption_types]

    if include_none:
        types = list(types) + [CorruptionType.NONE]

    chosen = CorruptionType(rng.choice(types))
    params = random_corruption_params(chosen, rng=rng)
    out = apply_corruption(image, chosen, **params)
    return out, chosen, params


def apply_corruption_chain(
    image: ImageArray,
    steps: Sequence[tuple[Union[str, CorruptionType], dict]],
) -> ImageArray:
    """Apply a sequence of ``(corruption_type, params)`` steps in order."""
    out = image
    for name, params in steps:
        out = apply_corruption(out, name, **params)
    return out


# ---------------------------------------------------------------------------
# Dataset-level helpers
# ---------------------------------------------------------------------------


def corrupt_image_file(
    src: PathLike,
    dst: PathLike,
    corruption: Union[str, CorruptionType],
    **params,
) -> dict:
    """
    Load ``src``, apply corruption, write ``dst``.

    Returns a metadata dict suitable for a CSV / JSONL index.
    """
    src, dst = Path(src), Path(dst)
    image = load_image(src)
    corrupted = apply_corruption(image, corruption, **params)
    save_image(dst, corrupted)
    return {
        "source": str(src),
        "output": str(dst),
        "corruption": CorruptionType(corruption).value,
        "params": params,
    }


def iter_image_paths(
    root: PathLike,
    extensions: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> list[Path]:
    """Recursively collect image paths under ``root`` (sorted for determinism)."""
    root = Path(root)
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)


def generate_corrupted_dataset(
    input_dir: PathLike,
    output_dir: PathLike,
    corruption_types: Optional[Sequence[Union[str, CorruptionType]]] = None,
    seed: int = 42,
    keep_pristine_ratio: float = 0.0,
    relative_to: Optional[PathLike] = None,
) -> list[dict]:
    """
    Walk ``input_dir``, write corrupted copies under ``output_dir``.

    Directory structure under ``output_dir`` mirrors the input relative layout
    (useful for LFW identity folders). Each file gets one sampled corruption
    (or is copied pristine with probability ``keep_pristine_ratio``).

    Returns a list of per-image metadata records.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    relative_to = Path(relative_to) if relative_to is not None else input_dir
    rng = np.random.default_rng(seed)

    records: list[dict] = []
    for src in iter_image_paths(input_dir):
        rel = src.relative_to(relative_to)
        dst = output_dir / rel

        if keep_pristine_ratio > 0 and rng.random() < keep_pristine_ratio:
            image = load_image(src)
            save_image(dst, image)
            records.append(
                {
                    "source": str(src),
                    "output": str(dst),
                    "corruption": CorruptionType.NONE.value,
                    "params": {},
                }
            )
            continue

        image = load_image(src)
        corrupted, ctype, params = apply_random_corruption(
            image,
            corruption_types=corruption_types,
            rng=rng,
            include_none=False,
        )
        save_image(dst, corrupted)
        records.append(
            {
                "source": str(src),
                "output": str(dst),
                "corruption": ctype.value,
                "params": params,
            }
        )

    return records


__all__ = [
    "CorruptionType",
    "load_image",
    "save_image",
    "ensure_uint8",
    "apply_gaussian_blur",
    "apply_jpeg_compression",
    "apply_gaussian_noise",
    "apply_low_light",
    "apply_fog",
    "apply_rain",
    "apply_corruption",
    "random_corruption_params",
    "apply_random_corruption",
    "apply_corruption_chain",
    "corrupt_image_file",
    "iter_image_paths",
    "generate_corrupted_dataset",
]
