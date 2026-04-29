import argparse
import json
import os
import warnings
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import datasets
import torch.nn as nn


# 模型list
from GLANet import GLANet as GLANet
from baseline.UNet import UNet 
from baseline.DeepLab import deeplabv3plus_resnet50
from baseline.MAResUNet import MAResUNet
from baseline.GeleNet.GeleNet_models import GeleNet
from baseline.SwinUNet.vision_transformer import SwinUnet
from baseline.UNetFormer import UNetFormer
from baseline.CTCFNet import CTCFNet
# 模型list

from libs import metric
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



# 过滤第三方库的已知兼容性提示，避免验证输出被 warning 淹没。
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.registry is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Overwriting pvt_v2_b[0-5] in registry.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Mapping deprecated model name swsl_resnet18 to current .*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release.*",
    category=UserWarning,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Validate CTCFNet on validation set")
    parser.add_argument('--model-path', type=str, default='experiments/CTCFNet/epoch_97_f1_0.87290.pth', help='Path to trained model checkpoint')
    parser.add_argument('--batchsize', type=int, default=5, help='batchsize')
    parser.add_argument('--numclasses', type=int, default=2, help='number of classes')
    parser.add_argument('--gpu', type=int, default=0, help='the chosen gpu')
    parser.add_argument('--output-dir', type=str, default='./validation/CTCFNet', help='Directory to save results')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args


def overlay_labels_on_image(image, label, color_map, opacity=0.5):
    """将标签覆盖到图像上"""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()

    # 确保图像为uint8格式
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # 创建彩色标签图像
    colored_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(color_map):
        colored_label[label == idx] = color

    # 合成图像
    overlay = image.copy()
    mask = label > 0  # 只对非背景类进行叠加
    overlay[mask] = cv2.addWeighted(image, 1-opacity, colored_label, opacity, 0)[mask]

    return overlay


def create_comparison_image(original_img, pred_label, gt_label, color_map):
    """创建三联对比图像：原图、预测标签、真实标签"""
    # 转换图像格式并反归一化
    if isinstance(original_img, torch.Tensor):
        orig_img_np = original_img.cpu().numpy().transpose(1, 2, 0)
    else:
        orig_img_np = original_img

    # 反归一化处理（根据 __init__.py 中的标准化参数）
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 转换为uint8格式
    orig_img_np = (orig_img_np * 255).astype(np.uint8)

    # 处理标签
    if isinstance(pred_label, torch.Tensor):
        pred_label = pred_label.cpu().numpy()
    if isinstance(gt_label, torch.Tensor):
        gt_label = gt_label.cpu().numpy()

    # 创建彩色标签图
    pred_colored = np.zeros((pred_label.shape[0], pred_label.shape[1], 3), dtype=np.uint8)
    gt_colored = np.zeros((gt_label.shape[0], gt_label.shape[1], 3), dtype=np.uint8)

    for idx, color in enumerate(color_map):
        pred_colored[pred_label == idx] = color
        gt_colored[gt_label == idx] = color

    # 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    axes[0].imshow(orig_img_np)
    axes[0].set_title('Original Image (After Denormalization)')
    axes[0].axis('off')

    # 预测标签（红色高亮）
    axes[1].imshow(pred_colored)
    axes[1].set_title('Predicted Labels (Red for class 1)')
    axes[1].axis('off')

    # 真实标签（红色高亮）
    axes[2].imshow(gt_colored)
    axes[2].set_title('Ground Truth Labels (Red for class 1)')
    axes[2].axis('off')

    # 添加颜色图例
    red_patch = mpatches.Patch(color='red', label='Class 1 (Foreground)')
    black_patch = mpatches.Patch(color='black', label='Class 0 (Background)')
    axes[1].legend(handles=[red_patch, black_patch], loc='upper right', bbox_to_anchor=(1, 1))
    axes[2].legend(handles=[red_patch, black_patch], loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    return fig


def foreground_metrics(conf_mat):
    """只计算前景类（类别1）的 Precision / Recall / F1 / IoU。"""
    fg_idx = 1
    true_positive = conf_mat[fg_idx, fg_idx]
    false_positive = conf_mat[:, fg_idx].sum() - true_positive
    false_negative = conf_mat[fg_idx, :].sum() - true_positive

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = true_positive / (true_positive + false_positive + false_negative) if (true_positive + false_positive + false_negative) > 0 else 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'iou': float(iou),
    }


def build_summary(case_results):
    """汇总每个指标的均值和标准差，并生成简洁百分比显示"""
    metric_names = ('precision', 'recall', 'f1_score', 'iou')
    summary = {
        'num_cases': len(case_results),
        'metrics': {}
    }

    for metric_name in metric_names:
        values = np.array([case[metric_name] for case in case_results.values()], dtype=np.float64)
        mean_value = float(np.mean(values)) if values.size > 0 else 0.0
        std_value = float(np.std(values)) if values.size > 0 else 0.0

        # case_results 已经是百分比形式，这里直接汇总显示即可。
        summary['metrics'][metric_name] = f"{mean_value:.2f} ± {std_value:.2f}"

    return summary


def format_case_metrics(single_metrics):
    """将单个 case 的指标转成百分比并保留两位小数。"""
    return {
        'precision': round(single_metrics['precision'] * 100, 2),
        'recall': round(single_metrics['recall'] * 100, 2),
        'f1_score': round(single_metrics['f1_score'] * 100, 2),
        'iou': round(single_metrics['iou'] * 100, 2),
    }


def main():
    args = parse_args()

    # 加载数据集
    train_loader, val_loader = datasets.setup_loaders(args)


    # 加载模型
    # model = UNet(out_channel=args.numclasses)
    # model = GLANet(numclasses=args.numclasses) 
    # model = deeplabv3plus_resnet50(               1
    #         num_classes=2,
    #         output_stride=8,
    #         pretrained_backbone=False)
    
    # model = MAResUNet(num_channels=3,             2
    #                   num_classes=args.numclasses,
    #                   pretrained=False)
    
    # model = GeleNet(channel=32)                   3
    
    # model = SwinUnet(num_classes=args.numclasses, 4
    #                      img_size=256)
    
    # model = UNetFormer(
    #         decode_channels=64,
    #         dropout=0.1,
    #         backbone_name='swsl_resnet18',
    #         pretrained=False,
    #         window_size=8,
    #         num_classes=2)
    model = CTCFNet(img_size=256, in_chans=3, class_dim=2,
                  patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                    norm_layer=nn.LayerNorm, depths=[3, 3, 6, 3], sr_ratios=[8, 4, 2, 1])
    

    model = model.cuda(args.gpu)

    # 加载训练好的权重
    checkpoint = torch.load(args.model_path, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    # 定义颜色映射 (这里使用红黑，红色表示类别1，黑色表示类别0)
    color_map = [(0, 0, 0), (255, 0, 0)]  # Black for class 0, Red for class 1

    results = {}

    print("Starting validation...")
    with torch.no_grad():
        for i, (imgs, masks, _) in enumerate(tqdm(val_loader, desc="Validating")):
            imgs = imgs.cuda(args.gpu)
            masks = masks.cuda(args.gpu)

            # 获取模型预测
            outputs = model(imgs)
            outputs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # 对每个样本进行处理
            for j in range(imgs.size(0)):
                if i * args.batchsize + j >= len(val_loader.dataset):
                    break

                img = imgs[j]
                mask = masks[j]
                pred = preds[j]

                # 计算单个图像的指标
                conf_mat_single = metric.confusion_matrix(
                    pred=pred.cpu().numpy().flatten(),
                    label=mask.cpu().numpy().flatten(),
                    num_classes=args.numclasses
                )

                single_metrics = foreground_metrics(conf_mat_single)

                # 保存指标
                img_idx = i * args.batchsize + j
                results[f'image_{img_idx:04d}'] = format_case_metrics(single_metrics)

                # 创建对比图像
                comparison_fig = create_comparison_image(
                    original_img=img,
                    pred_label=pred,
                    gt_label=mask,
                    color_map=color_map
                )

                # 保存对比图像
                img_path = os.path.join(args.output_dir, f'comparison_{img_idx:04d}.png')
                comparison_fig.savefig(img_path, dpi=150, bbox_inches='tight')
                plt.close(comparison_fig)

    summary = build_summary(results)
    output_data = {
        'summary': summary,
        'cases': results
    }

    # 保存JSON结果
    json_path = os.path.join(args.output_dir, 'validation_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Validation completed. Results saved to {args.output_dir}")
    print(f"JSON metrics saved to {json_path}")

    # 输出总体统计信息
    if results:
        print(f"\nOverall Statistics:")
        print(f"Number of Cases: {summary['num_cases']}")
        print(f"Precision: {summary['metrics']['precision']}")
        print(f"Recall: {summary['metrics']['recall']}")
        print(f"F1 Score: {summary['metrics']['f1_score']}")
        print(f"IoU: {summary['metrics']['iou']}")


if __name__ == "__main__":
    main()
