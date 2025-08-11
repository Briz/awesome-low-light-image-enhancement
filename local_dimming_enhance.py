#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local dimming style enhancement for 1080p video.

Pipeline per frame:
 1) Convert BGR -> YUV, take Y channel, split into 27x40 cells (40px tall x 48px wide),
    run per-cell gamma correction with gamma derived from local mean luminance.
 2) Enhance black level via adaptive black-point crush using low percentile.
 3) Improve local contrast using CLAHE on Y.
 4) Recombine YUV -> BGR, compose side-by-side with original into 1920x1080:
    left half = original (resized to 960x1080), right half = enhanced (resized to 960x1080).

Usage example:
  python /workspace/local_dimming_enhance.py \
    --input /path/to/input.mp4 \
    --output /path/to/output.mp4 \
    --fps 30

Notes:
- Input frames are resized to 1920x1080 for processing and composition.
- Parameters are exposed via CLI to tune strength.
"""

import argparse
import math
import os
from typing import Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False


GRID_ROWS = 27
GRID_COLS = 40
CELL_H = 40  # 1080 / 27
CELL_W = 48  # 1920 / 40
TARGET_W = 1920
TARGET_H = 1080
HALF_W = TARGET_W // 2  # 960


def compute_gamma_from_mean(mean_luma_01: float, gamma_min: float, gamma_max: float, pivot: float = 0.5) -> float:
    """Map local mean luminance in [0,1] to a gamma value in [gamma_min, gamma_max].

    - Darker regions (mean < pivot) get higher gamma (>1) to deepen blacks slightly.
    - Brighter regions (mean > pivot) get lower gamma (<1) to compress highlights a bit.
    """
    # Linear mapping around pivot, then clamp
    span = gamma_max - gamma_min
    # Map mean to a factor in [0,1]
    t = np.clip(1.0 - (mean_luma_01 - 0.0) / (1.0 - 0.0), 0.0, 1.0)
    # Blend towards higher gamma for darker regions
    gamma = gamma_min + t * span
    # Bias around pivot so mid-tones are closer to 1.0
    gamma = (gamma + 1.0 + (pivot - mean_luma_01) * 0.4) / 2.0
    return float(np.clip(gamma, gamma_min, gamma_max))


def apply_per_cell_gamma(y_uint8: np.ndarray, gamma_min: float = 0.9, gamma_max: float = 1.15) -> np.ndarray:
    """Apply per-cell gamma correction to Y channel (uint8, shape HxW) using 27x40 grid.
    Returns uint8 array of same shape.
    """
    h, w = y_uint8.shape
    assert h == TARGET_H and w == TARGET_W, "Y channel must be 1920x1080 (WxH flipped) before per-cell gamma."

    y_float = y_uint8.astype(np.float32) / 255.0

    for row in range(GRID_ROWS):
        y0 = row * CELL_H
        y1 = y0 + CELL_H
        for col in range(GRID_COLS):
            x0 = col * CELL_W
            x1 = x0 + CELL_W
            cell = y_float[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            mean_luma = float(cell.mean())
            gamma = compute_gamma_from_mean(mean_luma, gamma_min, gamma_max)
            # Avoid zero warnings
            cell_corrected = np.power(np.clip(cell, 0.0, 1.0), gamma)
            y_float[y0:y1, x0:x1] = cell_corrected

    y_out = np.clip(np.round(y_float * 255.0), 0, 255).astype(np.uint8)
    return y_out


essential_eps = 1e-6


def enhance_black_level(y_uint8: np.ndarray, percentile: float = 5.0, max_crush: float = 12.0) -> np.ndarray:
    """Crush blacks by remapping low-end luma to zero based on low percentile.

    - percentile: low percentile (0-10 recommended) used as black point in [0,255]
    - max_crush: cap for black point (in absolute luma units) to avoid over-crushing
    """
    y = y_uint8.astype(np.float32)
    p = float(np.percentile(y, percentile))
    p = float(np.clip(p, 0.0, max_crush))
    # Map [p, 255] -> [0, 255]
    y = (y - p) * (255.0 / max(255.0 - p, 1.0))
    y = np.clip(y, 0.0, 255.0)
    return y.astype(np.uint8)


def apply_clahe(y_uint8: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(y_uint8)


def process_frame(frame_bgr: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    """Process a single frame. Returns (original_960x1080_bgr, enhanced_960x1080_bgr)."""
    # Ensure 1080p canvas
    frame_bgr = cv2.resize(frame_bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    # Original to left half (resized to 960x1080)
    left_bgr = cv2.resize(frame_bgr, (HALF_W, TARGET_H), interpolation=cv2.INTER_AREA)

    # Convert to YUV and split
    yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)

    # 1) Per-cell gamma on Y
    y_gamma = apply_per_cell_gamma(y, gamma_min=args.gamma_min, gamma_max=args.gamma_max)

    # 2) Black level enhancement
    y_black = enhance_black_level(y_gamma, percentile=args.black_percentile, max_crush=args.black_max)

    # 3) Local contrast (CLAHE)
    y_enhanced = apply_clahe(y_black, clip_limit=args.clahe_clip, tile_grid_size=(args.clahe_grid, args.clahe_grid))

    # Merge and convert back
    yuv_enh = cv2.merge([y_enhanced, u, v])
    enhanced_bgr_full = cv2.cvtColor(yuv_enh, cv2.COLOR_YUV2BGR)

    # Right half: enhanced resized to 960x1080
    right_bgr = cv2.resize(enhanced_bgr_full, (HALF_W, TARGET_H), interpolation=cv2.INTER_AREA)

    return left_bgr, right_bgr


def compose_side_by_side(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    # Ensure both halves are 960x1080
    left_bgr = cv2.resize(left_bgr, (HALF_W, TARGET_H), interpolation=cv2.INTER_AREA)
    right_bgr = cv2.resize(right_bgr, (HALF_W, TARGET_H), interpolation=cv2.INTER_AREA)
    combined = np.hstack([left_bgr, right_bgr])
    return combined


def guess_fps(cap: cv2.VideoCapture, fallback: float) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3 or math.isnan(fps):
        return fallback
    return fps


def build_writer(path: str, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (TARGET_W, TARGET_H))
    if not writer.isOpened():
        # Fallback to MJPG if mp4v unavailable
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, fps, (TARGET_W, TARGET_H))
    return writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1080p local-dimming style enhancement with per-cell gamma, black level, and CLAHE.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path (mp4 recommended)")
    parser.add_argument("--fps", type=float, default=30.0, help="Output FPS if input FPS is unavailable")

    # Per-cell gamma params
    parser.add_argument("--gamma-min", type=float, default=0.90, dest="gamma_min", help="Minimum gamma (bright/highlights compression)")
    parser.add_argument("--gamma-max", type=float, default=1.15, dest="gamma_max", help="Maximum gamma (darkening to deepen blacks)")

    # Black level
    parser.add_argument("--black-percentile", type=float, default=5.0, help="Low percentile used as black point (0-10 recommended)")
    parser.add_argument("--black-max", type=float, default=12.0, help="Max absolute black crush in luma units (0-255)")

    # CLAHE
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tile grid size (NxN)")

    # Optional preview
    parser.add_argument("--preview", action="store_true", help="Show a preview window while processing")
    parser.add_argument("--preview-skip", type=int, default=0, help="Skip N frames between previews for speed")

    args = parser.parse_args()

    if args.gamma_min <= 0 or args.gamma_max <= 0:
        parser.error("gamma values must be > 0")
    if args.gamma_min > args.gamma_max:
        parser.error("--gamma-min must be <= --gamma-max")
    if args.clahe_grid <= 0:
        parser.error("--clahe-grid must be > 0")

    return args


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.input}")

    fps = guess_fps(cap, fallback=args.fps)
    writer = build_writer(args.output, fps=fps)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output writer: {args.output}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    progress_iter = range(total_frames) if total_frames > 0 else iter(int, 1)  # endless if unknown

    if TQDM_AVAILABLE and total_frames > 0:
        progress_iter = tqdm(range(total_frames), desc="Processing", unit="frame")

    frame_index = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            left_bgr, right_bgr = process_frame(frame_bgr, args)
            combined = compose_side_by_side(left_bgr, right_bgr)

            writer.write(combined)

            if args.preview:
                if args.preview_skip == 0 or (frame_index % (args.preview_skip + 1) == 0):
                    cv2.imshow("Local Dimming Enhancement (Left=Original, Right=Enhanced)", combined)
                    # Non-blocking small wait; press 'q' to quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

            if TQDM_AVAILABLE and total_frames > 0:
                pass  # tqdm advances automatically via loop index
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        if args.preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()