#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np


# ------------------------------
# MSRCR: Multi-Scale Retinex with Color Restoration
# ------------------------------

def msrcr(
    bgr_image: np.ndarray,
    gaussian_sigmas: List[float] = (15.0, 80.0, 250.0),
    gaussian_weights: List[float] = None,
    alpha: float = 125.0,
    beta: float = 46.0,
    gain: float = 1.0,
    offset: float = 0.0,
    low_clip_percentile: float = 1.0,
    high_clip_percentile: float = 99.0,
) -> np.ndarray:
    if gaussian_weights is None:
        gaussian_weights = [1.0 / len(gaussian_sigmas)] * len(gaussian_sigmas)

    img = bgr_image.astype(np.float32)
    img = np.clip(img, 0.0, 255.0) + 1.0  # avoid log(0)

    # Retinex per channel
    retinex = np.zeros_like(img, dtype=np.float32)
    for sigma, weight in zip(gaussian_sigmas, gaussian_weights):
        blur = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
        retinex += weight * (np.log(img) - np.log(blur + 1e-6))

    # Color restoration term
    sum_channels = np.sum(img, axis=2, keepdims=True)
    color_restoration = beta * (np.log(alpha * img) - np.log(sum_channels + 1e-6))

    msrcr_img = gain * (retinex * color_restoration + offset)

    # Normalize using percentile clipping per channel
    out = np.zeros_like(msrcr_img, dtype=np.float32)
    for c in range(3):
        channel = msrcr_img[:, :, c]
        lo = np.percentile(channel, low_clip_percentile)
        hi = np.percentile(channel, high_clip_percentile)
        if hi <= lo:
            hi = lo + 1.0
        channel = np.clip(channel, lo, hi)
        channel = (channel - lo) / (hi - lo)
        out[:, :, c] = channel

    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    return out


# ------------------------------
# YUV local dimming over 27x40 grid with smoothing and anti-blooming
# ------------------------------

def compute_region_means(y_channel: np.ndarray, grid_rows: int, grid_cols: int) -> np.ndarray:
    h, w = y_channel.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    means = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for r in range(grid_rows):
        y0 = r * cell_h
        y1 = (r + 1) * cell_h if r < grid_rows - 1 else h
        for c in range(grid_cols):
            x0 = c * cell_w
            x1 = (c + 1) * cell_w if c < grid_cols - 1 else w
            region = y_channel[y0:y1, x0:x1]
            means[r, c] = float(np.mean(region))
    return means


def build_gain_map_from_means(
    y_channel: np.ndarray,
    region_means: np.ndarray,
    g_min: float = 0.6,
    g_max: float = 1.4,
    gamma: float = 0.85,
    smooth_sigma: float = 15.0,
) -> np.ndarray:
    h, w = y_channel.shape
    grid_rows, grid_cols = region_means.shape

    # Normalize region means to [0,1]
    m_min = np.percentile(region_means, 1.0)
    m_max = np.percentile(region_means, 99.0)
    if m_max <= m_min:
        m_max = m_min + 1.0
    m_norm = (region_means - m_min) / (m_max - m_min)
    m_norm = np.clip(m_norm, 0.0, 1.0)

    # Map to gain: darker regions -> lower gain (<1), brighter -> higher gain (>1)
    # Use gamma to bias and spread
    gain_grid = g_min + (g_max - g_min) * (m_norm ** gamma)

    # Upsample to full res
    gain_map = cv2.resize(gain_grid, (w, h), interpolation=cv2.INTER_CUBIC)

    # Smooth to reduce blockiness and blooming
    if smooth_sigma > 0:
        gain_map = cv2.GaussianBlur(gain_map, ksize=(0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma, borderType=cv2.BORDER_REFLECT)

    # Limit local contrast of gain map using a soft clamp
    gain_map = np.clip(gain_map, g_min, g_max)

    return gain_map.astype(np.float32)


def apply_local_dimming(
    y_channel: np.ndarray,
    grid_rows: int = 27,
    grid_cols: int = 40,
    g_min: float = 0.65,
    g_max: float = 1.35,
    gamma: float = 0.9,
    smooth_sigma: float = 13.0,
    max_y: int = 235,
    anti_bloom_radius: int = 21,
    anti_bloom_boost: float = 1.10,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = y_channel.shape

    # Compute region means
    means = compute_region_means(y_channel, grid_rows=grid_rows, grid_cols=grid_cols)

    # Build smoothed gain map
    gain_map = build_gain_map_from_means(
        y_channel,
        means,
        g_min=g_min,
        g_max=g_max,
        gamma=gamma,
        smooth_sigma=smooth_sigma,
    )

    y_float = y_channel.astype(np.float32)
    enhanced_y = y_float * gain_map

    # Anti-blooming: limit by local max of original Y
    if anti_bloom_radius > 0:
        kernel_size = max(3, anti_bloom_radius | 1)  # ensure odd
        local_max = cv2.dilate(y_float, np.ones((kernel_size, kernel_size), np.uint8))
        enhanced_y = np.minimum(enhanced_y, local_max * anti_bloom_boost)

    # Clip to threshold to avoid over-exposure in blocks
    enhanced_y = np.clip(enhanced_y, 0.0, float(max_y))

    return enhanced_y.astype(np.uint8), gain_map


# ------------------------------
# Chroma correction and conversions
# ------------------------------

def bgr_to_yuv(bgr: np.ndarray) -> np.ndarray:
    # OpenCV uses full-range by default for BGR<->YUV in this code path
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)


def yuv_to_bgr(yuv: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def diagonal_ccm_match_means(reference_bgr: np.ndarray, image_bgr: np.ndarray, clamp_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    ref_means = np.mean(reference_bgr.reshape(-1, 3), axis=0) + 1e-6
    img_means = np.mean(image_bgr.reshape(-1, 3), axis=0) + 1e-6
    gains = ref_means / img_means
    gains = np.clip(gains, clamp_range[0], clamp_range[1]).astype(np.float32)
    corrected = image_bgr.astype(np.float32) * gains.reshape(1, 1, 3)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


# ------------------------------
# Processing pipeline for a frame
# ------------------------------

def process_frame(
    bgr_frame: np.ndarray,
    assume_1080p: bool = True,
    side_by_side_1920x1080: bool = True,
) -> np.ndarray:
    src_h, src_w = bgr_frame.shape[:2]

    # Ensure 1920x1080 for processing grid 27x40 (48x40 cells)
    if assume_1080p and (src_w != 1920 or src_h != 1080):
        frame_bgr = cv2.resize(bgr_frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    else:
        frame_bgr = bgr_frame

    # 1) MSRCR
    msrcr_bgr = msrcr(frame_bgr)

    # 2) BGR -> YUV, split Y
    yuv = bgr_to_yuv(msrcr_bgr)
    Y = yuv[:, :, 0]

    # 3) Local dimming on Y with constraints
    enhanced_Y, gain_map = apply_local_dimming(Y, grid_rows=27, grid_cols=40)

    # 4) Merge enhanced Y, preserve U/V, convert back, then color-correct to reduce chroma shift
    yuv_enhanced = yuv.copy()
    yuv_enhanced[:, :, 0] = enhanced_Y
    enhanced_bgr = yuv_to_bgr(yuv_enhanced)

    # Diagonal CCM to match global color means with original MSRCR output
    enhanced_bgr = diagonal_ccm_match_means(msrcr_bgr, enhanced_bgr, clamp_range=(0.92, 1.08))

    # 5) Side-by-side into 1920x1080 canvas
    if side_by_side_1920x1080:
        canvas_w, canvas_h = 1920, 1080
        left = cv2.resize(frame_bgr, (canvas_w // 2, canvas_h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(enhanced_bgr, (canvas_w // 2, canvas_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, : canvas_w // 2] = left
        canvas[:, canvas_w // 2 :] = right
        return canvas

    return enhanced_bgr


# ------------------------------
# IO: image or video
# ------------------------------

def process_image(input_path: str, output_path: str, no_side_by_side: bool = False) -> None:
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {input_path}")
    out = process_frame(img, assume_1080p=True, side_by_side_1920x1080=not no_side_by_side)
    ok = cv2.imwrite(output_path, out)
    if not ok:
        raise RuntimeError(f"Failed to write image: {output_path}")


def process_video(input_path: str, output_path: str, no_side_by_side: bool = False, fourcc_str: str = "mp4v", fps: float = None) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {input_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = input_fps if input_fps and input_fps > 0 else 30.0

    out_w, out_h = 1920, 1080
    if no_side_by_side:
        # If not side-by-side, keep output at source size if available; otherwise default to 1920x1080
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if src_w > 0 and src_h > 0:
            out_w, out_h = src_w, src_h

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = process_frame(frame, assume_1080p=True, side_by_side_1920x1080=not no_side_by_side)
            if not no_side_by_side and (processed.shape[1] != out_w or processed.shape[0] != out_h):
                processed = cv2.resize(processed, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if no_side_by_side and (processed.shape[1] != out_w or processed.shape[0] != out_h):
                processed = cv2.resize(processed, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(processed)
    finally:
        cap.release()
        writer.release()


# ------------------------------
# CLI
# ------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="1080p projector local dimming with MSRCR, YUV grid-based dimming, anti-blooming, and color correction."
    )
    parser.add_argument("input", help="Path to input image or video")
    parser.add_argument("output", help="Path to output image or video")
    parser.add_argument("--no-side-by-side", action="store_true", help="Output processed frame only (no 1920x1080 side-by-side)")
    parser.add_argument("--video", action="store_true", help="Force video processing mode")
    parser.add_argument("--image", action="store_true", help="Force image processing mode")
    parser.add_argument("--fps", type=float, default=0.0, help="Output FPS for video; defaults to input FPS or 30")
    parser.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="FourCC for video writing (e.g., mp4v, XVID, avc1). Container must match extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_lower = args.input.lower()
    force_image = args.image
    force_video = args.video

    is_video_ext = any(input_lower.endswith(ext) for ext in [
        ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".webm"
    ])

    if force_image and force_video:
        raise ValueError("Cannot force both image and video modes.")

    if force_video or (not force_image and is_video_ext):
        process_video(args.input, args.output, no_side_by_side=args.no_side_by_side, fourcc_str=args.fourcc, fps=args.fps)
    else:
        process_image(args.input, args.output, no_side_by_side=args.no_side_by_side)


if __name__ == "__main__":
    main()