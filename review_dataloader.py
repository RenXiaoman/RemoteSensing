"""
数据加载器审查脚本
此脚本将保存经过Transforms处理后的训练集和验证集图像，
以便检查数据加载器是否正确处理图像和标签。
"""
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as standard_transforms
import datasets.joint_transforms as joint_transforms
import datasets.transforms as extended_transforms
from torch.utils.data import DataLoader
import datasets.giddataset
import argparse
from tqdm import tqdm


def reverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反标准化函数，将标准化后的图像还原为原始范围"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def save_tensor_as_image(tensor, filepath):
    """将tensor保存为图像"""
    if isinstance(tensor, torch.Tensor):
        # 反标准化
        denorm_tensor = reverse_normalize(tensor)
        # 转换为numpy并调整维度顺序
        img_np = denorm_tensor.cpu().numpy().transpose(1, 2, 0)
        # 转换为0-255范围的uint8
        img_np = (img_np * 255).astype(np.uint8)
        # 保存为图像
        img = Image.fromarray(img_np)
        img.save(filepath)
    else:
        # 如果输入已经是numpy数组，则直接保存
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).astype(np.uint8)
        img = Image.fromarray(tensor)
        img.save(filepath)


def save_mask_as_image(mask, filepath):
    """将mask保存为图像"""
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask

    # 确保mask是整数类型
    mask_np = mask_np.astype(np.uint8)

    # 创建RGB图像以便更清晰地显示
    mask_rgb = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    # 类别0为黑色(0,0,0)，类别1为红色(255,0,0)
    mask_rgb[mask_np == 0] = [0, 0, 0]  # 黑色背景
    mask_rgb[mask_np == 1] = [255, 0, 0]  # 红色前景

    img = Image.fromarray(mask_rgb)
    img.save(filepath)


def main():
    parser = argparse.ArgumentParser(description="Review DataLoader processed images")
    parser.add_argument('--output-dir', type=str, default='./review_dataloader',
                        help='Directory to save reviewed images')
    parser.add_argument('--data-root', type=str,
                        default='/home/ikun_server/clib/PycharmProjects/GLANet/data/data_split_filtered',
                        help='Data root directory')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for dataloader')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to review from each set')

    args = parser.parse_args()

    # 设置数据集根路径
    datasets.giddataset.root = args.data_root

    # 设置变换 (与train.py保持一致)
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # 训练集的联合变换
    train_joint_transform_list = [
        joint_transforms.RandomHorizontallyFlip(),
    ]
    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)

    # 训练集的输入变换
    train_input_transform = standard_transforms.Compose([
        extended_transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
        extended_transforms.RandomGaussianBlur(),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    # 验证集的输入变换
    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    # 标签变换
    target_transform = extended_transforms.MaskToTensor()

    # 创建数据集
    print(f"Loading datasets from {args.data_root}")

    train_set = datasets.giddataset.Gids(
        'train',
        joint_transform=train_joint_transform,
        transform=train_input_transform,
        target_transform=target_transform
    )

    val_set = datasets.giddataset.Gids(
        'val',
        joint_transform=None,
        transform=val_input_transform,
        target_transform=target_transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=5,
        shuffle=False,  # 不打乱，以便我们可以选择前N个样本
        drop_last=False
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=5,
        shuffle=False,
        drop_last=False
    )

    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_output_dir = os.path.join(output_dir, 'train')
    val_output_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)

    print(f"Train dataset samples: {len(train_set)}")
    print(f"Val dataset samples: {len(val_set)}")
    print(f"Saving train images to: {train_output_dir}")
    print(f"Saving val images to: {val_output_dir}")

    # 处理训练集
    print("\nProcessing train dataset...")
    train_sample_count = 0
    for batch_idx, (imgs, masks, img_names) in enumerate(tqdm(train_loader, desc="Processing train")):
        for i in range(imgs.size(0)):
            if train_sample_count >= args.num_samples:
                break

            img = imgs[i]
            mask = masks[i]

            # 获取原始文件名
            img_name = f"train_batch{batch_idx}_idx{i}"

            # 保存图像
            img_path = os.path.join(train_output_dir, f"{img_name}_image.png")
            save_tensor_as_image(img, img_path)

            # 保存mask
            mask_path = os.path.join(train_output_dir, f"{img_name}_mask.png")
            save_mask_as_image(mask, mask_path)

            train_sample_count += 1

        if train_sample_count >= args.num_samples:
            break

    # 处理验证集
    print("\nProcessing val dataset...")
    val_sample_count = 0
    for batch_idx, (imgs, masks, img_names) in enumerate(tqdm(val_loader, desc="Processing val")):
        for i in range(imgs.size(0)):
            if val_sample_count >= args.num_samples:
                break

            img = imgs[i]
            mask = masks[i]

            # 获取原始文件名
            img_name = f"val_batch{batch_idx}_idx{i}"

            # 保存图像
            img_path = os.path.join(val_output_dir, f"{img_name}_image.png")
            save_tensor_as_image(img, img_path)

            # 保存mask
            mask_path = os.path.join(val_output_dir, f"{img_name}_mask.png")
            save_mask_as_image(mask, mask_path)

            val_sample_count += 1

        if val_sample_count >= args.num_samples:
            break

    print(f"\nCompleted! Saved {train_sample_count} train samples and {val_sample_count} val samples.")
    print(f"All images are saved in {output_dir}")
    print("Each sample includes both the normalized image and the corresponding mask.")


if __name__ == "__main__":
    main()