#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分区调光与图像增强脚本（适用于 1920x1080 帧）

功能：
1) 将输入图像转到 YUV 空间，分离亮度通道 Y；将 Y 划分为 27x40 个子区域（每格 40x48 像素），对每个子区域执行自适应 gamma 校正；
2) 提升画面的黑位深度（toe 曲线）；
3) 提升图像的局部对比度（CLAHE）；
4) 改善图像暗部细节可见性（暗区更小的 gamma）；
5) 改善整体动态范围（百分位拉伸）；
6) 将原图与增强后的图并列放入 1920x1080 输出（每侧 960x1080）。

用法示例：
  处理图片：
    python zone_dimming_enhance.py --input input.jpg --output output.jpg

  处理视频：
    python zone_dimming_enhance.py --input input.mp4 --output output.mp4

可调参数见 --help
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import cv2
import numpy as np


# 常量：1080p 分区
NUM_ROWS = 27
NUM_COLS = 40
TILE_H = 40  # 27 * 40 = 1080
TILE_W = 48  # 40 * 48 = 1920
FRAME_W = 1920
FRAME_H = 1080


def apply_local_gamma(
    y_channel_u8: np.ndarray,
    gamma_min: float = 0.7,
    gamma_max: float = 1.3,
) -> np.ndarray:
    """对亮度通道按 27x40 分区执行自适应 gamma 校正。

    - 暗块使用更小的 gamma（提亮细节），亮块使用更大的 gamma（压低、增加黑位深度）。
    - 输入/输出：uint8 [0,255]
    """
    assert y_channel_u8.dtype == np.uint8
    h, w = y_channel_u8.shape
    if (h, w) != (FRAME_H, FRAME_W):
        raise ValueError(f"输入帧尺寸需为 {FRAME_W}x{FRAME_H}，当前为 {w}x{h}")

    y_out = y_channel_u8.astype(np.float32) / 255.0

    for r in range(NUM_ROWS):
        y0 = r * TILE_H
        y1 = y0 + TILE_H
        for c in range(NUM_COLS):
            x0 = c * TILE_W
            x1 = x0 + TILE_W
            tile = y_out[y0:y1, x0:x1]
            mean_luma = float(np.mean(tile))  # [0,1]
            # 亮块 -> 更大的 gamma（>1），暗块 -> 更小的 gamma（<1）
            gamma = gamma_min + (gamma_max - gamma_min) * mean_luma
            # 避免数值问题
            gamma = max(0.1, min(5.0, gamma))
            # 幂律变换
            y_out[y0:y1, x0:x1] = np.power(np.clip(tile, 0.0, 1.0), gamma)

    y_out = np.clip(y_out * 255.0, 0, 255).astype(np.uint8)
    return y_out


def apply_clahe_local_contrast(y_channel_u8: np.ndarray, clip_limit: float = 2.0,
                                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """对亮度通道 Y 应用 CLAHE 增强局部对比度。"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(y_channel_u8)


def percentile_contrast_stretch(y_channel_u8: np.ndarray,
                                low_percentile: float = 1.0,
                                high_percentile: float = 99.0) -> np.ndarray:
    """按百分位拉伸动态范围，抑制极暗/极亮异常值影响。"""
    y = y_channel_u8.astype(np.float32) / 255.0
    p_low = np.percentile(y, low_percentile)
    p_high = np.percentile(y, high_percentile)
    if p_high <= p_low + 1e-6:
        return y_channel_u8
    y = (y - p_low) / (p_high - p_low)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def apply_toe_curve(y_channel_u8: np.ndarray, toe_strength: float = 0.08) -> np.ndarray:
    """应用轻微 toe 曲线以加深黑位（不显著压暗中高亮）。

    toe_strength: 0~0.5 较合理，越大黑位越深。
    实现方式：y = y^(1 + toe_strength)
    """
    toe_strength = float(np.clip(toe_strength, 0.0, 0.5))
    y = y_channel_u8.astype(np.float32) / 255.0
    gamma = 1.0 + toe_strength
    y = np.power(y, gamma)
    return (np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)


def enhance_frame_bgr(
    frame_bgr: np.ndarray,
    gamma_min: float = 0.7,
    gamma_max: float = 1.3,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    p_low: float = 1.0,
    p_high: float = 99.0,
    toe_strength: float = 0.08,
    force_resize_1080p: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """对单帧进行增强，返回 (original_1080p_bgr, enhanced_1080p_bgr)。"""
    h, w = frame_bgr.shape[:2]
    if force_resize_1080p and (w != FRAME_W or h != FRAME_H):
        frame_bgr = cv2.resize(frame_bgr, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)

    # 转 YUV 并分离 Y
    yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)

    # 1) 分区自适应 gamma
    y = apply_local_gamma(y, gamma_min=gamma_min, gamma_max=gamma_max)

    # 3) 局部对比度增强（CLAHE）
    y = apply_clahe_local_contrast(y, clip_limit=clahe_clip, tile_grid_size=clahe_grid)

    # 5) 动态范围：百分位拉伸
    y = percentile_contrast_stretch(y, low_percentile=p_low, high_percentile=p_high)

    # 2) 黑位深度（toe 曲线）
    y = apply_toe_curve(y, toe_strength=toe_strength)

    # 合并回 BGR
    yuv_enhanced = cv2.merge((y, u, v))
    enhanced_bgr = cv2.cvtColor(yuv_enhanced, cv2.COLOR_YUV2BGR)

    # 轻微防伪影与色彩溢出保护
    enhanced_bgr = np.clip(enhanced_bgr, 0, 255).astype(np.uint8)

    return frame_bgr, enhanced_bgr


def side_by_side_1920x1080(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    """将两幅 1920x1080 帧缩放为各 960x1080 并左右拼接为 1920x1080。"""
    left = cv2.resize(left_bgr, (FRAME_W // 2, FRAME_H), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right_bgr, (FRAME_W // 2, FRAME_H), interpolation=cv2.INTER_AREA)
    return np.hstack([left, right])


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def is_video_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg"}


def process_image(input_path: str, output_path: str, args: argparse.Namespace) -> None:
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{input_path}")

    orig_bgr, enhanced_bgr = enhance_frame_bgr(
        img,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        clahe_clip=args.clahe_clip,
        clahe_grid=(args.clahe_grid, args.clahe_grid),
        p_low=args.p_low,
        p_high=args.p_high,
        toe_strength=args.toe,
        force_resize_1080p=True,
    )

    canvas = side_by_side_1920x1080(orig_bgr, enhanced_bgr)
    ok = cv2.imwrite(output_path, canvas)
    if not ok:
        raise RuntimeError(f"写入输出图片失败：{output_path}")


def process_video(input_path: str, output_path: str, args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (FRAME_W, FRAME_H))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频：{output_path}")

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            orig_bgr, enhanced_bgr = enhance_frame_bgr(
                frame,
                gamma_min=args.gamma_min,
                gamma_max=args.gamma_max,
                clahe_clip=args.clahe_clip,
                clahe_grid=(args.clahe_grid, args.clahe_grid),
                p_low=args.p_low,
                p_high=args.p_high,
                toe_strength=args.toe,
                force_resize_1080p=True,
            )

            canvas = side_by_side_1920x1080(orig_bgr, enhanced_bgr)
            writer.write(canvas)
            frame_index += 1
    finally:
        cap.release()
        writer.release()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="1080p 分区调光 + 画质增强 并排输出")
    p.add_argument("--input", required=True, help="输入图片或视频路径 (建议 1920x1080)")
    p.add_argument("--output", required=True, help="输出图片或视频路径 (与输入类型匹配)")

    # 分区 gamma 参数
    p.add_argument("--gamma-min", dest="gamma_min", type=float, default=0.7,
                   help="分区最小 gamma（用于暗块，<1 提亮，默认 0.7）")
    p.add_argument("--gamma-max", dest="gamma_max", type=float, default=1.3,
                   help="分区最大 gamma（用于亮块，>1 压低，默认 1.3）")

    # CLAHE 参数
    p.add_argument("--clahe-clip", dest="clahe_clip", type=float, default=2.0,
                   help="CLAHE clipLimit，默认 2.0")
    p.add_argument("--clahe-grid", dest="clahe_grid", type=int, default=8,
                   help="CLAHE tileGridSize (gxg)，默认 8")

    # 动态范围百分位
    p.add_argument("--p-low", dest="p_low", type=float, default=1.0,
                   help="低百分位（默认 1.0）")
    p.add_argument("--p-high", dest="p_high", type=float, default=99.0,
                   help="高百分位（默认 99.0）")

    # 黑位 toe 曲线
    p.add_argument("--toe", dest="toe", type=float, default=0.08,
                   help="黑位 toe 强度 0~0.5（默认 0.08）")

    return p


def main():
    args = build_arg_parser().parse_args()
    in_path = args.input
    out_path = args.output

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"输入文件不存在：{in_path}")

    if is_image_file(in_path):
        process_image(in_path, out_path, args)
    elif is_video_file(in_path):
        process_video(in_path, out_path, args)
    else:
        raise ValueError("无法识别的输入类型，请提供常见图片或视频格式。")


if __name__ == "__main__":
    main()