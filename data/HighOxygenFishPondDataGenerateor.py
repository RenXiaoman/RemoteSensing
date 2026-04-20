#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遥感图像TIFF文件读取和预处理脚本（鱼塘制氧机场景）
用于分析 image.tif 和 label3.tif 文件的特征，并将其分割成适合训练的小块。

筛选策略：
1. 图像切块策略基本沿用 split_with_filter.py。
2. 在每个标签块内部，按前景连通域筛选，只保留包含值为 2 的连通域。
3. 导出的标签为二值标签：
   - 0: 背景
   - 255: 前景（鱼塘和制氧机都映射为 255）
4. 也就是说，值 2 只用于判定某个子连通区域是否有效，不在最终标签中单独保留。
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window


def analyze_tif_file(filepath, name=""):
    """
    分析TIFF文件的基本属性。
    """
    print(f"\n=== 分析 {name} 文件: {filepath} ===")

    with rasterio.open(filepath) as src:
        print(f"文件形状: {src.shape}")
        print(f"波段数量 (通道数): {src.count}")
        print(f"数据类型: {src.dtypes[0]}")
        print(f"坐标参考系统 (CRS): {src.crs}")
        print(f"仿射变换矩阵: \n{src.transform}")
        print(f"nodata值: {src.nodata}")

        total_pixels = src.width * src.height
        if total_pixels < 10000 * 10000:
            data = src.read()
            for band_idx in range(src.count):
                band_data = data[band_idx]
                print(
                    f"波段 {band_idx + 1} - 最小值: {band_data.min()}, "
                    f"最大值: {band_data.max()}, 平均值: {band_data.mean():.2f}"
                )
        else:
            print("文件太大，无法整体分析，将进行分块采样分析")


def sample_analysis(filepath, samples=5):
    """
    对大文件进行采样分析。
    """
    print(f"\n=== 对 {Path(filepath).name} 进行采样分析 ({samples} 个样本) ===")

    with rasterio.open(filepath) as src:
        height, width = src.shape

        for i in range(samples):
            win_height = min(1000, height // 10)
            win_width = min(1000, width // 10)

            start_row = np.random.randint(0, max(1, height - win_height))
            start_col = np.random.randint(0, max(1, width - win_width))

            window = Window(start_col, start_row, win_width, win_height)
            data = src.read(window=window)

            print(f"样本 {i + 1}: 位置({start_row}:{start_row + win_height}, {start_col}:{start_col + win_width})")
            print(f"  形状: {data.shape}")

            if src.count == 1:
                print(f"  数值范围: {data.min()} - {data.max()}")
                print(f"  平均值: {data.mean():.2f}")
            else:
                for band_idx in range(min(src.count, 3)):
                    band_data = data[band_idx]
                    print(
                        f"  波段 {band_idx + 1} 范围: {band_data.min()} - {band_data.max()}, "
                        f"平均值: {band_data.mean():.2f}"
                    )


def retain_components_with_oxygen(label_tile, oxygen_class_value=2):
    """
    在当前 tile 内按前景连通域筛选，只保留包含制氧机的连通域。
    返回过滤后的标签（保留原始 1/2 值）和保留的连通域数量。
    """
    foreground_mask = label_tile > 0
    visited = np.zeros_like(foreground_mask, dtype=bool)
    filtered_label_tile = np.zeros_like(label_tile, dtype=label_tile.dtype)
    kept_component_count = 0
    height, width = label_tile.shape

    for row in range(height):
        for col in range(width):
            if not foreground_mask[row, col] or visited[row, col]:
                continue

            stack = [(row, col)]
            visited[row, col] = True
            component_pixels = []
            contains_oxygen = False

            while stack:
                cur_row, cur_col = stack.pop()
                component_pixels.append((cur_row, cur_col))
                if label_tile[cur_row, cur_col] == oxygen_class_value:
                    contains_oxygen = True

                for next_row, next_col in (
                    (cur_row - 1, cur_col),
                    (cur_row + 1, cur_col),
                    (cur_row, cur_col - 1),
                    (cur_row, cur_col + 1),
                ):
                    if 0 <= next_row < height and 0 <= next_col < width:
                        if foreground_mask[next_row, next_col] and not visited[next_row, next_col]:
                            visited[next_row, next_col] = True
                            stack.append((next_row, next_col))

            if contains_oxygen:
                kept_component_count += 1
                for comp_row, comp_col in component_pixels:
                    filtered_label_tile[comp_row, comp_col] = label_tile[comp_row, comp_col]

    return filtered_label_tile, kept_component_count


def save_overview_image(image_tile, binary_label_tile, original_label_tile, output_path):
    """
    使用 matplotlib 创建总览图：左侧原图，右侧标签。
    """
    if image_tile.ndim != 3:
        raise ValueError(f"Expected image_tile to be 3D, got shape {image_tile.shape}")

    # 同时兼容 CHW 和 HWC 两种输入格式，避免 overview 阶段重复转置。
    if image_tile.shape[0] in (1, 3) and image_tile.shape[-1] not in (1, 3):
        if image_tile.shape[0] > 3:
            image_tile = image_tile[:3, :, :]
        if image_tile.shape[0] == 1:
            image_tile = np.repeat(image_tile, 3, axis=0)
        image_rgb = np.transpose(image_tile, (1, 2, 0)).astype("uint8")
    else:
        image_rgb = image_tile.astype("uint8")
        if image_rgb.shape[-1] > 3:
            image_rgb = image_rgb[..., :3]
        if image_rgb.shape[-1] == 1:
            image_rgb = np.repeat(image_rgb, 3, axis=-1)

    label_vis = np.zeros((binary_label_tile.shape[0], binary_label_tile.shape[1], 3), dtype="uint8")
    label_vis[binary_label_tile > 0] = [255, 255, 255]
    label_vis[original_label_tile == 2] = [255, 0, 0]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(label_vis)
    axes[1].set_title("Label")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def split_and_save_tiles(
    image_path,
    label_path,
    output_dir,
    tile_size=224,
    stride=None,
    oxygen_class_value=2,
    overview=False,
    val_ratio=0.1
):
    """
    将大图像分割成小块并保存为 PNG 格式。
    仅保留标签块中“包含制氧机的连通域”。
    最终导出的标签为二值掩码：0/255。
    """
    if stride is None:
        stride = tile_size
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    print(f"\n=== 开始分割图像为 {tile_size}x{tile_size} 的块 (步长: {stride}) ===")
    print(f"有效块判定规则: 仅保留内部包含值为 {oxygen_class_value} 的前景连通域")
    print(f"训练/验证划分比例: {1 - val_ratio:.2%} / {val_ratio:.2%}")

    train_img_dir = Path(output_dir) / "train" / "image"
    train_mask_dir = Path(output_dir) / "train" / "mask"
    val_img_dir = Path(output_dir) / "val" / "image"
    val_mask_dir = Path(output_dir) / "val" / "mask"
    overview_train_dir = Path(output_dir) / "overview" / "train"
    overview_val_dir = Path(output_dir) / "overview" / "val"

    base_dirs = [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]
    if overview:
        base_dirs.extend([overview_train_dir, overview_val_dir])

    for directory in base_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_path) as img_src:
        img_height, img_width = img_src.shape

    with rasterio.open(label_path) as lbl_src:
        lbl_height, lbl_width = lbl_src.shape

    print(f"图像尺寸: {img_width} x {img_height}")
    print(f"标签尺寸: {lbl_width} x {lbl_height}")

    min_width = min(img_width, lbl_width)
    min_height = min(img_height, lbl_height)
    print(f"共同覆盖区域: {min_width} x {min_height}")

    if stride >= tile_size:
        n_tiles_x = (min_width + tile_size - 1) // tile_size
        n_tiles_y = (min_height + tile_size - 1) // tile_size
        print(f"无重叠模式: 需要分割成 {n_tiles_x} x {n_tiles_y} = {n_tiles_x * n_tiles_y} 个小块")
    else:
        n_tiles_x = max(1, (min_width - tile_size) // stride + 1) if min_width >= tile_size else 1
        n_tiles_y = max(1, (min_height - tile_size) // stride + 1) if min_height >= tile_size else 1

        remaining_width = min_width - (n_tiles_x - 1) * stride
        remaining_height = min_height - (n_tiles_y - 1) * stride

        if tile_size // 2 < remaining_width < tile_size:
            n_tiles_x += 1
        if tile_size // 2 < remaining_height < tile_size:
            n_tiles_y += 1

        print(f"有重叠模式: 需要分割成 {n_tiles_x} x {n_tiles_y} = {n_tiles_x * n_tiles_y} 个小块")

    val_count = 0
    train_count = 0
    skipped_count = 0
    total_counter = 0
    rng = np.random.default_rng(42)

    with rasterio.open(image_path) as img_src, rasterio.open(label_path) as lbl_src:
        for y in range(n_tiles_y):
            for x in range(n_tiles_x):
                left = x * stride if (x * stride + tile_size) <= min_width else min_width - tile_size
                top = y * stride if (y * stride + tile_size) <= min_height else min_height - tile_size
                right = min(left + tile_size, min_width)
                bottom = min(top + tile_size, min_height)

                if (right - left < tile_size) or (bottom - top < tile_size):
                    print(f"跳过非完整瓦片 ({x}, {y}): 尺寸为 {right - left} x {bottom - top}")
                    continue

                img_window = Window(left, top, tile_size, tile_size)
                lbl_window = Window(left, top, tile_size, tile_size)

                image_tile = img_src.read(window=img_window)
                label_tile = lbl_src.read(window=lbl_window, out_shape=(1, tile_size, tile_size))[0]

                if np.all(label_tile == 0):
                    skipped_count += 1
                    continue

                filtered_label_tile, kept_component_count = retain_components_with_oxygen(
                    label_tile,
                    oxygen_class_value=oxygen_class_value
                )
                oxygen_pixels = int(np.sum(filtered_label_tile == oxygen_class_value))
                has_oxygen = oxygen_pixels > 0

                if not has_oxygen:
                    skipped_count += 1
                    continue

                pond_pixels = int(np.sum(filtered_label_tile == 1))
                print(
                    f"处理瓦片 ({x}, {y}): 保留连通域 {kept_component_count} 个, "
                    f"包含制氧机像素 {oxygen_pixels} 个, 鱼塘像素 {pond_pixels} 个"
                )

                if image_tile.shape[0] > 3:
                    print(f"警告: 图像有 {image_tile.shape[0]} 个通道，只保留前 3 个通道")
                    image_tile = image_tile[:3, :, :]

                if image_tile.shape[0] == 1:
                    image_tile = np.repeat(image_tile, 3, axis=0)

                image_tile = np.transpose(image_tile, (1, 2, 0))
                image_pil = Image.fromarray(
                    image_tile.astype("uint8"),
                    mode="RGB" if image_tile.shape[2] == 3 else "I"
                )

                # 制氧机只用于筛选有效块，最终标签导出为二值 0/255。
                binary_label_tile = ((filtered_label_tile > 0).astype("uint8")) * 255
                label_pil = Image.fromarray(binary_label_tile, mode="L")

                is_val = rng.random() < val_ratio
                filename_base = f"{total_counter}"

                if is_val:
                    img_path = val_img_dir / f"{filename_base}.png"
                    mask_path = val_mask_dir / f"{filename_base}.png"
                    overview_path = overview_val_dir / f"{filename_base}.png"
                    val_count += 1
                else:
                    img_path = train_img_dir / f"{filename_base}.png"
                    mask_path = train_mask_dir / f"{filename_base}.png"
                    overview_path = overview_train_dir / f"{filename_base}.png"
                    train_count += 1

                total_counter += 1

                image_pil.save(img_path)
                label_pil.save(mask_path)
                if overview:
                    save_overview_image(image_tile, binary_label_tile, filtered_label_tile, overview_path)

                if (train_count + val_count) % 50 == 0:
                    print(
                        f"已处理 {train_count + val_count} 个瓦片..."
                        f" (训练: {train_count}, 验证: {val_count}, 跳过: {skipped_count})"
                    )

    print(f"分割完成! 总共 {train_count + val_count} 个有效瓦片 (训练: {train_count}, 验证: {val_count})")
    print(f"跳过了 {skipped_count} 个不包含制氧机的瓦片")
    print(f"输出保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="分析和处理鱼塘制氧机数据")
    parser.add_argument(
        "--image-path",
        type=str,
        default="/home/ikun_server/clib/PycharmProjects/GLANet/data/image.tif",
        help="输入图像 TIFF 文件路径"
    )
    parser.add_argument(
        "--label-path",
        type=str,
        default="/home/ikun_server/clib/PycharmProjects/GLANet/data/label3.tif",
        help="输入标签 TIFF 文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/ikun_server/clib/PycharmProjects/GLANet/data/high_oxygen_fish_pond_20Percent",
        help="输出目录"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="只分析文件，不进行分割"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="分割的瓦片大小 (默认: 256)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="滑动窗口步长，设置为 tile-size 则无重叠"
    )
    parser.add_argument(
        "--oxygen-class-value",
        type=int,
        default=2,
        help="表示制氧机的标签值 (默认: 2)"
    )
    parser.add_argument(
        "--overview",
        action="store_true",
        default=True,
        help="是否额外生成 overview 图，左侧为 image，右侧为对应 label"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="验证集比例，例如 0.1 表示 10%%，0.2 表示 20%%，0.15 表示 15%%"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("鱼塘制氧机数据生成工具")
    print("=" * 60)

    analyze_tif_file(args.image_path, "图像")
    sample_analysis(args.image_path)

    analyze_tif_file(args.label_path, "标签")
    sample_analysis(args.label_path)

    if not args.analyze_only:
        split_and_save_tiles(
            image_path=args.image_path,
            label_path=args.label_path,
            output_dir=args.output_dir,
            tile_size=args.tile_size,
            stride=args.stride,
            oxygen_class_value=args.oxygen_class_value,
            overview=args.overview,
            val_ratio=args.val_ratio
        )

        print("\n=== 处理完成! ===")
        print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
