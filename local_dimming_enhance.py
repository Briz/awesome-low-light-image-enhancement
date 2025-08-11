import argparse
import os
from typing import Tuple

import cv2
import numpy as np


def ensure_1080p_frame(image_bgr: np.ndarray) -> np.ndarray:
    """Resize input to 1920x1080 if needed using area interpolation (downscale-friendly)."""
    target_w, target_h = 1920, 1080
    h, w = image_bgr.shape[:2]
    if (w, h) == (target_w, target_h):
        return image_bgr
    resized = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized


def compute_adaptive_gamma_grid(
    y_channel_norm: np.ndarray,
    num_rows: int = 27,
    num_cols: int = 40,
    gamma_dark: float = 0.85,
    gamma_bright: float = 1.15,
) -> np.ndarray:
    """Compute a per-tile gamma grid based on tile mean luminance, then upsample to full-res.

    - y_channel_norm: 1080x1920 float32 in [0, 1]
    - Returns: gamma_map (1080x1920) float32
    """
    height, width = y_channel_norm.shape[:2]

    # Tiles are specified as 27 rows x 40 cols => 40px height x 48px width per tile
    tile_h = height // num_rows
    tile_w = width // num_cols

    # Safety check to ensure expected dimensions
    assert tile_h * num_rows == height and tile_w * num_cols == width, (
        f"Y channel shape {y_channel_norm.shape} not divisible by grid {num_rows}x{num_cols}"
    )

    # Compute tile mean luminance using reshape for speed
    tiles = y_channel_norm.reshape(num_rows, tile_h, num_cols, tile_w)
    tile_means = tiles.mean(axis=(1, 3))  # shape: (num_rows, num_cols)

    # Map tile mean [0,1] to gamma in [gamma_dark, gamma_bright] (linear interpolation)
    gamma_grid = gamma_dark + (gamma_bright - gamma_dark) * tile_means

    # Upsample gamma grid to per-pixel gamma map
    gamma_map = cv2.resize(
        gamma_grid.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
    )

    # Optional gentle smoothing to avoid seams
    gamma_map = cv2.GaussianBlur(gamma_map, ksize=(0, 0), sigmaX=1.0, sigmaY=1.0)

    return gamma_map


def apply_per_pixel_gamma(y_channel_norm: np.ndarray, gamma_map: np.ndarray) -> np.ndarray:
    """Apply Y' = (Y)^(1/gamma) elementwise. Inputs in [0,1], output in [0,1]."""
    # Avoid log(0) or pow(0, negative)
    y_safe = np.clip(y_channel_norm, 1e-6, 1.0)
    inv_gamma = np.clip(1.0 / np.clip(gamma_map, 1e-3, 10.0), 0.05, 20.0)
    y_gamma = np.power(y_safe, inv_gamma)
    return np.clip(y_gamma, 0.0, 1.0)


def apply_clahe(y_channel_uint8: np.ndarray, clip_limit: float = 2.0, grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE on 8-bit Y channel to improve local contrast and shadow details."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    y_clahe = clahe.apply(y_channel_uint8)
    return y_clahe


def percentile_contrast_stretch(y_channel_norm: np.ndarray, low_pct: float = 1.5, high_pct: float = 99.5) -> np.ndarray:
    """Stretch contrast by mapping percentiles to 0 and 1; deepens blacks and extends highlights."""
    low_val = float(np.percentile(y_channel_norm, low_pct))
    high_val = float(np.percentile(y_channel_norm, high_pct))
    if high_val <= low_val + 1e-6:
        return np.clip(y_channel_norm, 0.0, 1.0)
    y_stretched = (y_channel_norm - low_val) / (high_val - low_val)
    return np.clip(y_stretched, 0.0, 1.0)


def enhance_image_local_dimming(
    image_bgr: np.ndarray,
    gamma_dark: float = 0.85,
    gamma_bright: float = 1.15,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    low_pct: float = 1.5,
    high_pct: float = 99.5,
) -> np.ndarray:
    """Complete enhancement pipeline returning enhanced BGR image of size 1920x1080."""
    # Ensure expected size for grid partitioning
    image_bgr_1080p = ensure_1080p_frame(image_bgr)

    # Convert to YUV
    image_yuv = cv2.cvtColor(image_bgr_1080p, cv2.COLOR_BGR2YUV)
    y = image_yuv[:, :, 0].astype(np.float32) / 255.0

    # 1) Per-tile adaptive gamma correction on Y
    gamma_map = compute_adaptive_gamma_grid(
        y_channel_norm=y, gamma_dark=gamma_dark, gamma_bright=gamma_bright
    )
    y_gamma = apply_per_pixel_gamma(y, gamma_map)

    # 2) Local contrast enhancement (CLAHE) to improve local contrast and shadow details
    y_gamma_u8 = np.clip(y_gamma * 255.0 + 0.5, 0, 255).astype(np.uint8)
    y_clahe_u8 = apply_clahe(y_gamma_u8, clip_limit=clahe_clip, grid=clahe_grid)
    y_clahe = y_clahe_u8.astype(np.float32) / 255.0

    # 3) Percentile-based contrast stretch to deepen blacks and extend highlights (dynamic range)
    y_enh = percentile_contrast_stretch(y_clahe, low_pct=low_pct, high_pct=high_pct)

    # Re-assemble YUV and convert back to BGR
    y_out_u8 = np.clip(y_enh * 255.0 + 0.5, 0, 255).astype(np.uint8)
    image_yuv[:, :, 0] = y_out_u8
    enhanced_bgr = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    return enhanced_bgr


def make_side_by_side_1080p(original_bgr_1080p: np.ndarray, enhanced_bgr_1080p: np.ndarray) -> np.ndarray:
    """Create a 1920x1080 frame with original and enhanced images side-by-side (each 960x1080)."""
    target_h = 1080
    half_w = 960

    left = cv2.resize(original_bgr_1080p, (half_w, target_h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(enhanced_bgr_1080p, (half_w, target_h), interpolation=cv2.INTER_AREA)

    combined = np.hstack([left, right])
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1080p image local-dimming style enhancement and side-by-side output.")
    parser.add_argument("--input", "-i", required=True, help="Path to input image (expected 1920x1080 or will be resized).")
    parser.add_argument("--output", "-o", default="enhanced_side_by_side_1080p.png", help="Path to save 1920x1080 side-by-side result.")

    parser.add_argument("--gamma-dark", type=float, default=0.85, help="Gamma used for darkest tiles (lower brightens shadows).")
    parser.add_argument("--gamma-bright", type=float, default=1.15, help="Gamma used for brightest tiles (higher tames highlights).")

    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clipLimit.")
    parser.add_argument(
        "--clahe-grid",
        type=str,
        default="8x8",
        help="CLAHE tile grid size as WxH (e.g., 8x8).",
    )

    parser.add_argument("--black-pct-low", type=float, default=1.5, help="Low percentile for contrast stretch (deepens blacks).")
    parser.add_argument("--white-pct-high", type=float, default=99.5, help="High percentile for contrast stretch (extends highlights).")

    return parser.parse_args()


def parse_grid(arg: str) -> Tuple[int, int]:
    try:
        parts = arg.lower().split("x")
        if len(parts) != 2:
            raise ValueError
        w = int(parts[0])
        h = int(parts[1])
        if w <= 0 or h <= 0:
            raise ValueError
        return (w, h)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid grid spec '{arg}', expected formats like '8x8'") from exc


def main() -> None:
    args = parse_args()

    input_path = args.input
    output_path = args.output

    clahe_grid = parse_grid(args.clahe_grid)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {input_path}")

    bgr_1080p = ensure_1080p_frame(bgr)

    enhanced_bgr = enhance_image_local_dimming(
        bgr_1080p,
        gamma_dark=args.gamma_dark,
        gamma_bright=args.gamma_bright,
        clahe_clip=args.clahe_clip,
        clahe_grid=clahe_grid,
        low_pct=args.black_pct_low,
        high_pct=args.white_pct_high,
    )

    side_by_side = make_side_by_side_1080p(bgr_1080p, enhanced_bgr)

    ok = cv2.imwrite(output_path, side_by_side)
    if not ok:
        raise RuntimeError(f"Failed to write output to: {output_path}")

    print(f"Saved side-by-side 1920x1080 output to: {output_path}")


if __name__ == "__main__":
    main()