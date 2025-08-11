#!/usr/bin/env python3
import argparse
from typing import Tuple

import cv2
import numpy as np


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def compute_gamma_map(
    luminance_y: np.ndarray,
    num_tile_rows: int,
    num_tile_cols: int,
    gamma_for_dark_tiles: float,
    gamma_for_bright_tiles: float,
    min_gamma: float,
    max_gamma: float,
) -> np.ndarray:
    height, width = luminance_y.shape
    tile_height = height // num_tile_rows
    tile_width = width // num_tile_cols

    gamma_tiles = np.zeros((num_tile_rows, num_tile_cols), dtype=np.float32)

    for row_idx in range(num_tile_rows):
        row_start = row_idx * tile_height
        row_end = row_start + tile_height
        for col_idx in range(num_tile_cols):
            col_start = col_idx * tile_width
            col_end = col_start + tile_width

            tile = luminance_y[row_start:row_end, col_start:col_end]
            mean_luma = float(np.mean(tile)) / 255.0
            # Dark tiles use higher gamma (deepen blacks), bright tiles lower gamma (brighten highlights)
            linear_gamma = (
                gamma_for_dark_tiles
                + (gamma_for_bright_tiles - gamma_for_dark_tiles) * mean_luma
            )
            gamma_value = clamp(linear_gamma, min_gamma, max_gamma)
            gamma_tiles[row_idx, col_idx] = gamma_value

    gamma_map = cv2.resize(
        gamma_tiles,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    return gamma_map.astype(np.float32)


def apply_tile_gamma(luminance_y: np.ndarray, gamma_map: np.ndarray) -> np.ndarray:
    y_norm = np.clip(luminance_y.astype(np.float32) / 255.0, 0.0, 1.0)
    # Avoid raising 0 to a power > 1 causing underflow issues
    y_norm = np.maximum(y_norm, 1e-6)
    y_gamma = np.power(y_norm, gamma_map)
    y_out = np.clip(y_gamma * 255.0, 0, 255).astype(np.uint8)
    return y_out


def apply_black_toe(
    luminance_y: np.ndarray,
    toe_strength: float = 0.5,
    toe_threshold: float = 0.2,
) -> np.ndarray:
    # toe_strength in [0,1], toe_threshold in (0,1]
    y_norm = np.clip(luminance_y.astype(np.float32) / 255.0, 0.0, 1.0)
    threshold = float(np.clip(toe_threshold, 1e-3, 1.0))
    strength = float(np.clip(toe_strength, 0.0, 1.0))

    # Weight higher for low luminance, fade to 0 past threshold
    weight = np.clip((threshold - y_norm) / threshold, 0.0, 1.0)
    # Toe curve darkens shadows smoothly
    toe_curve = np.power(y_norm, 1.0 + 2.0 * strength)
    y_mix = (1.0 - weight) * y_norm + weight * toe_curve

    y_out = np.clip(y_mix * 255.0, 0, 255).astype(np.uint8)
    return y_out


def apply_local_contrast(
    luminance_y: np.ndarray,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_clip_limit),
        tileGridSize=tuple(clahe_tile_grid_size),
    )
    y_out = clahe.apply(luminance_y)
    return y_out


def ensure_1080p_bgr(image_bgr: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if (width, height) == (1920, 1080):
        return image_bgr
    return cv2.resize(image_bgr, (1920, 1080), interpolation=cv2.INTER_AREA)


def enhance_image(
    image_bgr: np.ndarray,
    num_tile_rows: int = 27,
    num_tile_cols: int = 40,
    gamma_for_dark_tiles: float = 2.2,
    gamma_for_bright_tiles: float = 0.8,
    min_gamma: float = 0.6,
    max_gamma: float = 2.4,
    toe_strength: float = 0.5,
    toe_threshold: float = 0.2,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: int = 8,
) -> np.ndarray:
    base_bgr = ensure_1080p_bgr(image_bgr)
    yuv = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2YUV)
    y_channel, u_channel, v_channel = cv2.split(yuv)

    gamma_map = compute_gamma_map(
        y_channel,
        num_tile_rows=num_tile_rows,
        num_tile_cols=num_tile_cols,
        gamma_for_dark_tiles=gamma_for_dark_tiles,
        gamma_for_bright_tiles=gamma_for_bright_tiles,
        min_gamma=min_gamma,
        max_gamma=max_gamma,
    )

    y_after_gamma = apply_tile_gamma(y_channel, gamma_map)
    y_after_toe = apply_black_toe(y_after_gamma, toe_strength=toe_strength, toe_threshold=toe_threshold)
    y_after_clahe = apply_local_contrast(
        y_after_toe,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=(clahe_tile_grid, clahe_tile_grid),
    )

    enhanced_yuv = cv2.merge((y_after_clahe, u_channel, v_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_yuv, cv2.COLOR_YUV2BGR)
    return enhanced_bgr


def compose_side_by_side_1920x1080(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    left_1080p = ensure_1080p_bgr(left_bgr)
    right_1080p = ensure_1080p_bgr(right_bgr)

    left_half = cv2.resize(left_1080p, (960, 1080), interpolation=cv2.INTER_AREA)
    right_half = cv2.resize(right_1080p, (960, 1080), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    canvas[:, :960, :] = left_half
    canvas[:, 960:, :] = right_half
    return canvas


def process_video(
    input_path: str,
    output_path: str,
    rows: int,
    cols: int,
    gamma_dark: float,
    gamma_bright: float,
    min_gamma: float,
    max_gamma: float,
    toe_strength: float,
    toe_threshold: float,
    clahe_clip: float,
    clahe_grid: int,
    fallback_fps: float = 30.0,
    codec: str = "mp4v",
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = src_fps if src_fps and src_fps > 1e-2 else fallback_fps

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    frame_index = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        base_bgr = ensure_1080p_bgr(frame_bgr)
        enhanced_bgr = enhance_image(
            base_bgr,
            num_tile_rows=rows,
            num_tile_cols=cols,
            gamma_for_dark_tiles=gamma_dark,
            gamma_for_bright_tiles=gamma_bright,
            min_gamma=min_gamma,
            max_gamma=max_gamma,
            toe_strength=toe_strength,
            toe_threshold=toe_threshold,
            clahe_clip_limit=clahe_clip,
            clahe_tile_grid=clahe_grid,
        )
        composite = compose_side_by_side_1920x1080(base_bgr, enhanced_bgr)

        writer.write(composite)
        frame_index += 1

    writer.release()
    cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply tiled gamma correction (27x40, 48x40 px), deepen blacks, "
            "enhance local contrast on 1080p input, and export a 1920x1080 side-by-side composite."
        )
    )
    parser.add_argument("input", type=str, help="Path to input image/video (1080p expected or will be resized)")
    parser.add_argument("output", type=str, help="Path to output image/video (side-by-side 1920x1080)")

    parser.add_argument("--rows", type=int, default=27, help="Number of tile rows (default: 27)")
    parser.add_argument("--cols", type=int, default=40, help="Number of tile cols (default: 40)")

    parser.add_argument("--gamma_dark", type=float, default=2.2, help="Gamma for dark tiles (default: 2.2)")
    parser.add_argument("--gamma_bright", type=float, default=0.8, help="Gamma for bright tiles (default: 0.8)")
    parser.add_argument("--min_gamma", type=float, default=0.6, help="Minimum gamma clamp (default: 0.6)")
    parser.add_argument("--max_gamma", type=float, default=2.4, help="Maximum gamma clamp (default: 2.4)")

    parser.add_argument("--toe_strength", type=float, default=0.5, help="Black toe strength in [0,1] (default: 0.5)")
    parser.add_argument("--toe_threshold", type=float, default=0.2, help="Black toe threshold in (0,1] (default: 0.2)")

    parser.add_argument("--clahe_clip", type=float, default=2.0, help="CLAHE clip limit (default: 2.0)")
    parser.add_argument("--clahe_grid", type=int, default=8, help="CLAHE tile grid size (NxN) (default: 8)")

    parser.add_argument("--video", action="store_true", help="Treat input as video and write video output")
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback FPS if source has no FPS (video mode)")
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec for video writing, e.g., mp4v, avc1, XVID (video mode)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.video:
        process_video(
            input_path=args.input,
            output_path=args.output,
            rows=args.rows,
            cols=args.cols,
            gamma_dark=args.gamma_dark,
            gamma_bright=args.gamma_bright,
            min_gamma=args.min_gamma,
            max_gamma=args.max_gamma,
            toe_strength=args.toe_strength,
            toe_threshold=args.toe_threshold,
            clahe_clip=args.clahe_clip,
            clahe_grid=args.clahe_grid,
            fallback_fps=args.fps,
            codec=args.codec,
        )
        return

    image_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read input image: {args.input}")

    base_bgr = ensure_1080p_bgr(image_bgr)

    enhanced_bgr = enhance_image(
        base_bgr,
        num_tile_rows=args.rows,
        num_tile_cols=args.cols,
        gamma_for_dark_tiles=args.gamma_dark,
        gamma_for_bright_tiles=args.gamma_bright,
        min_gamma=args.min_gamma,
        max_gamma=args.max_gamma,
        toe_strength=args.toe_strength,
        toe_threshold=args.toe_threshold,
        clahe_clip_limit=args.clahe_clip,
        clahe_tile_grid=args.clahe_grid,
    )

    composite = compose_side_by_side_1920x1080(base_bgr, enhanced_bgr)

    ok = cv2.imwrite(args.output, composite)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {args.output}")


if __name__ == "__main__":
    main()