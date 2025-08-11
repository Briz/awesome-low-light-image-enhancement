#!/usr/bin/env python3
import argparse
import sys
from typing import Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    print("Error: OpenCV (cv2) is required. Install with: pip install opencv-python", file=sys.stderr)
    raise


BLOCK_ROWS = 27
BLOCK_COLS = 40
BLOCK_H = 40  # pixels
BLOCK_W = 48  # pixels
TARGET_W = 1920
TARGET_H = 1080
Y_MAX = 235.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="27x40 zone local dimming for 1080p images with chroma correction and side-by-side output"
    )
    parser.add_argument("input", help="Path to input image (expected 1920x1080; will be resized if not)")
    parser.add_argument("output", help="Path to output 1920x1080 side-by-side image")
    parser.add_argument(
        "--g-min", type=float, default=0.70,
        help="Minimum per-pixel gain after smoothing (default: 0.70)"
    )
    parser.add_argument(
        "--g-max", type=float, default=1.25,
        help="Maximum per-pixel gain after smoothing (default: 1.25)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.7,
        help="Exponent controlling block gain mapping strength (default: 0.7)"
    )
    parser.add_argument(
        "--guidance-radius", type=int, default=16,
        help="Guided/bilateral filter radius to smooth gain map (default: 16)"
    )
    parser.add_argument(
        "--highlight-rolloff", type=float, default=0.5,
        help="How strongly to reduce gain near highlights [0..1] (default: 0.5)"
    )
    parser.add_argument(
        "--ccm-limit", type=float, default=0.10,
        help="Limit for per-channel CCM gains in RGB (e.g., 0.10 => [0.9,1.1]) (default: 0.10)"
    )
    return parser.parse_args()


def ensure_size_1080p(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if (w, h) == (TARGET_W, TARGET_H):
        return img_bgr
    return cv2.resize(img_bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


def rgb_like_to_yuv(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Convert BGR to YUV (Y: [0,255], U/V: [0,255])
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)
    return y.astype(np.float32), u.astype(np.float32), v.astype(np.float32)


def compute_block_means(y: np.ndarray) -> np.ndarray:
    # y is HxW float32 (1080x1920 expected)
    assert y.shape[0] == TARGET_H and y.shape[1] == TARGET_W, "Input Y must be 1080x1920"
    y_blocks = y.reshape(BLOCK_ROWS, BLOCK_H, BLOCK_COLS, BLOCK_W)
    # Reorder to (rows, cols, block_h, block_w)
    y_blocks = np.transpose(y_blocks, (0, 2, 1, 3))
    block_means = y_blocks.mean(axis=(2, 3))  # (27, 40)
    return block_means


def compute_gain_from_blocks(
    block_means: np.ndarray,
    global_mean: float,
    g_min: float,
    g_max: float,
    beta: float,
) -> np.ndarray:
    # Gain >1 for bright areas, <1 for dark areas, with moderation via beta
    ratio = (block_means + 1e-6) / (global_mean + 1e-6)
    gain_block = np.clip(np.power(ratio, beta), g_min, g_max)
    return gain_block.astype(np.float32)


def upsample_and_smooth_gain(
    gain_block: np.ndarray,
    guide_y: np.ndarray,
    radius: int,
    g_min: float,
    g_max: float,
) -> np.ndarray:
    # Upsample from (27,40) to (1080,1920) using bicubic
    gain_full = cv2.resize(gain_block, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    # Edge-aware smoothing to reduce block boundaries and blooming
    guide = np.clip(guide_y / 255.0, 0.0, 1.0).astype(np.float32)
    src = gain_full.astype(np.float32)

    gain_smooth = None
    try:
        # Try guided filter if ximgproc is available
        xip = cv2.ximgproc  # type: ignore[attr-defined]
        gain_smooth = xip.guidedFilter(guide=guide, src=src, radius=radius, eps=1e-3)
    except Exception:
        # Fallback to bilateral filter on gain map
        # Bilateral expects ranges ~[0,1]; set sigmaColor relative to range
        gain_smooth = cv2.bilateralFilter(src, d=2 * radius + 1, sigmaColor=0.1, sigmaSpace=float(radius))

    # Light Gaussian blur to further suppress blocking without spreading highlights too far
    gain_smooth = cv2.GaussianBlur(gain_smooth, (0, 0), sigmaX=max(1.0, radius / 4.0), sigmaY=max(1.0, radius / 4.0))

    return np.clip(gain_smooth, g_min, g_max)


def apply_highlight_rolloff(gain_full: np.ndarray, y: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return gain_full
    y_norm = np.clip(y / 255.0, 0.0, 1.0)
    # Start rolloff near 0.8, fully applied by 1.0
    t = np.clip((y_norm - 0.8) / 0.2, 0.0, 1.0)
    rolloff = 1.0 - (strength * t)
    return np.clip(gain_full * rolloff, 0.0, None)


def enhance_luma(
    y: np.ndarray,
    g_min: float,
    g_max: float,
    beta: float,
    radius: int,
    highlight_rolloff: float,
) -> np.ndarray:
    block_means = compute_block_means(y)
    global_mean = float(y.mean())
    gain_block = compute_gain_from_blocks(block_means, global_mean, g_min, g_max, beta)
    gain_full = upsample_and_smooth_gain(gain_block, y, radius, g_min, g_max)
    gain_full = apply_highlight_rolloff(gain_full, y, highlight_rolloff)

    y_enh = y * gain_full
    # Clip to Y_MAX to avoid overexposure in bright regions
    y_enh = np.clip(y_enh, 0.0, Y_MAX)

    # Light bilateral filter to suppress banding and residual blocking
    y_enh_norm = (y_enh / Y_MAX).astype(np.float32)
    y_enh_sm = cv2.bilateralFilter(y_enh_norm, d=7, sigmaColor=0.08, sigmaSpace=7)
    y_enh = np.clip(y_enh_sm * Y_MAX, 0.0, Y_MAX)

    return y_enh.astype(np.float32)


def correct_uv_bias(u_proc: np.ndarray, v_proc: np.ndarray, u_orig: np.ndarray, v_orig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Detect chroma bias drift and re-center U/V means toward original to suppress color cast
    du = float(u_proc.mean() - u_orig.mean())
    dv = float(v_proc.mean() - v_orig.mean())
    u_corr = np.clip(u_proc - du, 0.0, 255.0)
    v_corr = np.clip(v_proc - dv, 0.0, 255.0)
    return u_corr, v_corr


def merge_yuv_and_to_bgr(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    yuv = cv2.merge([
        np.clip(y, 0.0, 255.0).astype(np.uint8),
        np.clip(u, 0.0, 255.0).astype(np.uint8),
        np.clip(v, 0.0, 255.0).astype(np.uint8),
    ])
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return bgr


def apply_rgb_ccm(bgr_proc: np.ndarray, bgr_ref: np.ndarray, ccm_limit: float) -> np.ndarray:
    # Diagonal CCM based on channel mean ratio; limited within [1-ccm_limit, 1+ccm_limit]
    proc_mean = bgr_proc.reshape(-1, 3).mean(axis=0)
    ref_mean = bgr_ref.reshape(-1, 3).mean(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        gains = np.where(proc_mean > 1e-6, ref_mean / proc_mean, 1.0)
    gains = np.clip(gains, 1.0 - ccm_limit, 1.0 + ccm_limit).astype(np.float32)

    proc_f = bgr_proc.astype(np.float32)
    proc_f[..., 0] *= gains[0]  # B
    proc_f[..., 1] *= gains[1]  # G
    proc_f[..., 2] *= gains[2]  # R

    return np.clip(proc_f, 0.0, 255.0).astype(np.uint8)


def side_by_side_1920x1080(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    left_resized = cv2.resize(left_bgr, (TARGET_W // 2, TARGET_H), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right_bgr, (TARGET_W // 2, TARGET_H), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    canvas[:, : TARGET_W // 2] = left_resized
    canvas[:, TARGET_W // 2 :] = right_resized
    return canvas


def process_image(
    img_bgr: np.ndarray,
    g_min: float,
    g_max: float,
    beta: float,
    radius: int,
    highlight_rolloff: float,
    ccm_limit: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # Resize to 1920x1080 for consistent zoning
    img_bgr_1080p = ensure_size_1080p(img_bgr)

    # Convert to YUV and keep originals for chroma reference
    y_orig, u_orig, v_orig = rgb_like_to_yuv(img_bgr_1080p)

    # Enhance luminance with zone-based gain
    y_enh = enhance_luma(y_orig, g_min, g_max, beta, radius, highlight_rolloff)

    # Combine Enhanced Y with original U/V, correct chroma bias drift
    u_corr, v_corr = correct_uv_bias(u_orig, v_orig, u_orig, v_orig)

    # Convert back to BGR
    bgr_proc = merge_yuv_and_to_bgr(y_enh, u_corr, v_corr)

    # Apply a restrained RGB diagonal CCM to suppress color cast further
    bgr_proc_ccm = apply_rgb_ccm(bgr_proc, img_bgr_1080p, ccm_limit)

    return img_bgr_1080p, bgr_proc_ccm


def main() -> None:
    args = parse_args()

    img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Error: failed to read image: {args.input}", file=sys.stderr)
        sys.exit(1)

    left_bgr, right_bgr = process_image(
        img_bgr,
        g_min=args.g_min,
        g_max=args.g_max,
        beta=args.beta,
        radius=args.guidance_radius,
        highlight_rolloff=args.highlight_rolloff,
        ccm_limit=args.ccm_limit,
    )

    out = side_by_side_1920x1080(left_bgr, right_bgr)

    if not cv2.imwrite(args.output, out):
        print(f"Error: failed to write output: {args.output}", file=sys.stderr)
        sys.exit(2)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()