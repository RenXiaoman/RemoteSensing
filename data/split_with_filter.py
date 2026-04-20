#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遥感图像TIFF文件读取和预处理脚本（含标签过滤）
用于分析image.tif和label.tif文件的特征，并将其分割成适合训练的小块
只保留包含正样本（标签>0）的块
"""

import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
import os
from pathlib import Path
import argparse


def analyze_tif_file(filepath, name=""):
    """
    分析TIFF文件的基本属性
    """
    print(f"\n=== 分析 {name} 文件: {filepath} ===")

    with rasterio.open(filepath) as src:
        print(f"文件形状: {src.shape}")
        print(f"波段数量 (通道数): {src.count}")
        print(f"数据类型: {src.dtypes[0]}")
        print(f"坐标参考系统 (CRS): {src.crs}")
        print(f"仿射变换矩阵: \n{src.transform}")
        print(f"nodata值: {src.nodata}")

        # 如果文件较小，则直接读取全部数据统计
        total_pixels = src.width * src.height
        if total_pixels < 10000 * 10000:  # 小于1亿像素时才整体分析
            data = src.read()  # 读取所有波段
            for band_idx in range(src.count):
                band_data = data[band_idx]
                print(f"波段 {band_idx+1} - 最小值: {band_data.min()}, 最大值: {band_data.max()}, 平均值: {band_data.mean():.2f}")
        else:
            print("文件太大，无法整体分析，将进行分块采样分析")


def sample_analysis(filepath, samples=5):
    """
    对大文件进行采样分析
    """
    print(f"\n=== 对 {Path(filepath).name} 进行采样分析 ({samples} 个样本) ===")

    with rasterio.open(filepath) as src:
        height, width = src.shape

        # 随机选择一些窗口进行分析
        for i in range(samples):
            # 随机选择一个小窗口
            win_height = min(1000, height//10)  # 窗口高度不超过原图的/10
            win_width = min(1000, width//10)    # 窗口宽度不超过原图的/10

            start_row = np.random.randint(0, max(1, height - win_height))
            start_col = np.random.randint(0, max(1, width - win_width))

            window = Window(start_col, start_row, win_width, win_height)
            data = src.read(window=window)

            print(f"样本 {i+1}: 位置({start_row}:{start_row+win_height}, {start_col}:{start_col+win_width})")
            print(f"  形状: {data.shape}")

            if src.count == 1:
                # 单波段
                print(f"  数值范围: {data.min()} - {data.max()}")
                print(f"  平均值: {data.mean():.2f}")
            else:
                # 多波段
                for band_idx in range(min(src.count, 3)):  # 只显示前3个波段
                    band_data = data[band_idx]
                    print(f"  波段 {band_idx+1} 范围: {band_data.min()} - {band_data.max()}, 平均值: {band_data.mean():.2f}")


def split_and_save_tiles(image_path, label_path, output_dir, tile_size=256, min_positive_ratio=0.01, stride=None):
    """
    将大图像分割成小块并保存为PNG格式
    只保留标签中包含大于0值的窗口
    """
    if stride is None:
        stride = tile_size  # 默认无重叠

    print(f"\n=== 开始分割图像为 {tile_size}x{tile_size} 的块 (步长: {stride}) ===")

    # 创建输出目录
    train_img_dir = Path(output_dir) / "train" / "image"
    train_mask_dir = Path(output_dir) / "train" / "mask"
    val_img_dir = Path(output_dir) / "val" / "image"
    val_mask_dir = Path(output_dir) / "val" / "mask"

    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_path) as img_src:
        img_height, img_width = img_src.shape

    with rasterio.open(label_path) as lbl_src:
        lbl_height, lbl_width = lbl_src.shape

    print(f"图像尺寸: {img_width} x {img_height}")
    print(f"标签尺寸: {lbl_width} x {lbl_height}")

    # 计算共同覆盖区域（以较小范围为准）
    min_width = min(img_width, lbl_width)
    min_height = min(img_height, lbl_height)
    print(f"共同覆盖区域: {min_width} x {min_height}")

    # 根据步长计算需要多少个瓦片
    if stride is None or stride >= tile_size:
        # 无重叠模式，使用原始算法
        n_tiles_x = (min_width + tile_size - 1) // tile_size
        n_tiles_y = (min_height + tile_size - 1) // tile_size
        print(f"无重叠模式: 需要分割成 {n_tiles_x} x {n_tiles_y} = {n_tiles_x * n_tiles_y} 个小块")
    else:
        # 有重叠模式
        n_tiles_x = max(1, (min_width - tile_size) // stride + 1) if min_width >= tile_size else 1
        n_tiles_y = max(1, (min_height - tile_size) // stride + 1) if min_height >= tile_size else 1

        # 处理剩余空间，确保覆盖整个图像
        remaining_width = min_width - (n_tiles_x - 1) * stride
        remaining_height = min_height - (n_tiles_y - 1) * stride

        if remaining_width < tile_size and remaining_width > tile_size // 2:
            n_tiles_x += 1
        if remaining_height < tile_size and remaining_height > tile_size // 2:
            n_tiles_y += 1

        print(f"有重叠模式: 需要分割成 {n_tiles_x} x {n_tiles_y} = {n_tiles_x * n_tiles_y} 个小块")

    val_count = 0
    train_count = 0
    skipped_count = 0
    total_counter = 0  # 用于连续命名 0.png 到 n.png

    for y in range(n_tiles_y):
        for x in range(n_tiles_x):
            # 计算当前瓦片的边界（使用stride进行移动）
            left = x * stride if (x * stride + tile_size) <= min_width else min_width - tile_size
            top = y * stride if (y * stride + tile_size) <= min_height else min_height - tile_size
            right = min(left + tile_size, min_width)
            bottom = min(top + tile_size, min_height)

            # 确保瓦片是完整的
            if (right - left < tile_size) or (bottom - top < tile_size):
                print(f"跳过非完整瓦片 ({x}, {y}): 尺寸为 {right-left} x {bottom-top}")
                continue

            # 读取图像和标签块
            img_window = Window(left, top, tile_size, tile_size)
            lbl_window = Window(left, top, tile_size, tile_size)

            with rasterio.open(image_path) as img_src, rasterio.open(label_path) as lbl_src:
                image_tile = img_src.read(window=img_window)  # 形状为 (bands, height, width)
                label_tile = lbl_src.read(window=lbl_window, out_shape=(1, tile_size, tile_size))[0]  # 强制输出为单波段

            # 检查标签中是否包含大于0的像素
            positive_pixels = np.sum(label_tile > 0)
            total_pixels = label_tile.size
            positive_ratio = positive_pixels / total_pixels

            # 如果正样本比例低于阈值，跳过这个瓦片
            if positive_ratio < min_positive_ratio:
                skipped_count += 1
                continue

            print(f"处理瓦片 ({x}, {y}): 正样本比例 {positive_ratio:.4f} ({positive_pixels}/{total_pixels})")

            # 如果图像有多于3个通道，我们只取前3个通道
            if image_tile.shape[0] > 3:
                print(f"警告: 图像有 {image_tile.shape[0]} 个通道，只保留前3个通道")
                image_tile = image_tile[:3, :, :]

            # 如果图像只有1个通道，复制成3个通道
            if image_tile.shape[0] == 1:
                image_tile = np.repeat(image_tile, 3, axis=0)

            # 转换维度顺序: (C, H, W) -> (H, W, C)
            image_tile = np.transpose(image_tile, (1, 2, 0))

            # 转换为PIL图像
            image_pil = Image.fromarray(image_tile.astype('uint8'), mode='RGB' if image_tile.shape[2] == 3 else 'I')

            # 处理标签 - 如果标签已经是整数值，直接保存
            # 将标签中的1值放大到255，这样在PNG中更明显可见
            label_tile_vis = label_tile * 255  # 0->0, 1->255
            label_pil = Image.fromarray(label_tile_vis.astype('uint8'), mode='L')  # 使用灰度模式

            # 决定这是训练集还是验证集（这里简单地按比例分配）
            is_val = (train_count + val_count) % 10 == 0  # 大约10%的数据作为验证集

            # 使用连续数字作为文件名
            filename_base = f"{total_counter}"

            if is_val:
                img_path = val_img_dir / f"{filename_base}.png"
                mask_path = val_mask_dir / f"{filename_base}.png"
                val_count += 1
            else:
                img_path = train_img_dir / f"{filename_base}.png"
                mask_path = train_mask_dir / f"{filename_base}.png"
                train_count += 1

            total_counter += 1

            # 保存图像和标签
            image_pil.save(img_path)
            label_pil.save(mask_path)

            # 显示进度
            if (train_count + val_count) % 50 == 0:
                print(f"已处理 {train_count + val_count} 个瓦片... (训练: {train_count}, 验证: {val_count}, 跳过: {skipped_count})")

    print(f"分割完成! 总共 {train_count + val_count} 个有效瓦片 (训练: {train_count}, 验证: {val_count})")
    print(f"跳过了 {skipped_count} 个不包含目标的瓦片")
    print(f"输出保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="分析和处理遥感TIFF图像（含标签过滤）")
    parser.add_argument("--image-path", type=str, default="/home/ikun_server/clib/PycharmProjects/GLANet/data/image.tif",
                        help="输入图像TIFF文件路径")
    parser.add_argument("--label-path", type=str, default="/home/ikun_server/clib/PycharmProjects/GLANet/data/label.tif",
                        help="输入标签TIFF文件路径")
    parser.add_argument("--output-dir", type=str, default="/home/ikun_server/clib/PycharmProjects/GLANet/data/data_split",
                        help="输出目录")
    parser.add_argument("--analyze-only", action="store_true",
                        help="只分析文件，不进行分割")
    parser.add_argument("--tile-size", type=int, default=128,
                        help="分割的瓦片大小 (默认: 256)")
    parser.add_argument("--min-positive-ratio", type=float, default=0.01,
                        help="最小正样本比例阈值 (默认: 0.01 = 1%)")
    parser.add_argument("--stride", type=int, default=256,
                        help="滑动窗口步长，用于控制重叠 (默认: 128, 设置为tile-size则无重叠)")

    args = parser.parse_args()

    print("="*60)
    print("遥感图像TIFF分析和处理工具（含标签过滤）")
    print("="*60)

    # 分析图像文件
    analyze_tif_file(args.image_path, "图像")
    sample_analysis(args.image_path)

    # 分析标签文件
    analyze_tif_file(args.label_path, "标签")
    sample_analysis(args.label_path)

    if not args.analyze_only:
        # 执行分割
        split_and_save_tiles(
            image_path=args.image_path,
            label_path=args.label_path,
            output_dir=args.output_dir,
            tile_size=args.tile_size,
            min_positive_ratio=args.min_positive_ratio,
            stride=args.stride
        )

        print("\n=== 处理完成! ===")
        print(f"现在您可以修改 datasets/giddataset.py 中的 root 路径为: {args.output_dir}")
        print("确保修改后的路径与实际数据位置一致")


if __name__ == "__main__":
    main()