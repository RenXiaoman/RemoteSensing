import argparse
import logging
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from baseline.UNet import UNet

from torch import nn
from libs import average_meter, metric
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import datasets
import warnings
from libs.metric import save_log

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="RemoteSensingSegmentation by PyTorch")
    parser.add_argument('--batchsize', type=int, default=10, help='batchsize')
    # model and classes
    parser.add_argument('--model', type=str, default='UNet', help='model name')
    parser.add_argument('--numclasses', type=int, default=2, help='number of classes')
    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='the chosen gpu')
    # learning_rate
    parser.add_argument('--base-lr', type=float, default=1e-4, metavar='M', help='')
    parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='M', help='weight-decay (default:1e-4)')
    # best result
    parser.add_argument('--best-f1-score', type=float, default=0)

    parser.add_argument('--total_epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 120)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start epoch (default:0)')

    args = parser.parse_args()

    directory = os.path.join("experiments", args.model)
    args.directory = directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    # 存储日志
    save_log('result', args.directory)
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.numclasses = args.numclasses
        self.dataset = datasets.giddataset

        train_loader, val_loader = datasets.setup_loaders(args)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss().cuda(args.gpu)

        model = UNet(out_channel=args.numclasses)
        self.model = model.cuda(args.gpu)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
        self.max_iter = args.total_epochs * len(self.train_loader)
        
        # 参数总数
        num_params = sum(p.numel() for p in model.parameters())
        logging.info('Model params = {:2.1f}M'.format(num_params / 1000000))

        # 用于存储训练过程中的指标，以便后续绘图
        self.train_metrics_history = {
            'loss': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'iou': [],
        }

        self.val_metrics_history = {
            'precision': [],
            'recall': [],
            'f1': [],
            'iou': [],
        }

    def _foreground_metrics(self, conf_mat):
        """只计算前景类（类别1）的分割指标。"""
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
            'f1': float(f1),
            'iou': float(iou),
        }

    def training(self, epoch):
        self.model.train()  # 将module设为训练模式，影响Dropout和BatchNorm等
        # 初始化
        train_loss = average_meter.AverageMeter()

        curr_iter = epoch * len(self.train_loader)
        lr = self.args.base_lr * (1 - float(curr_iter) / self.max_iter) ** 0.95

        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        conf_mat = np.zeros((args.numclasses, args.numclasses)).astype(np.int64)
        tbar = tqdm(self.train_loader)
    
        for index, data in enumerate(tbar):
            imgs = Variable(data[0])
            masks = Variable(data[1])
            
            imgs = imgs.cuda(args.gpu)  # [B, 3, 256, 256]
            masks = masks.cuda(args.gpu)  # [B, 256, 256]
            
            self.optimizer.zero_grad()
            
            outputs = self.model(imgs)
            _, preds = torch.max(outputs, 1)
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
            loss = self.criterion(outputs, masks)
            train_loss.update(loss, args.batchsize)
            loss.backward()
            self.optimizer.step()

            tbar.set_description('epoch {}, training loss {}, with learning rate {}.'.format(epoch, train_loss.avg, lr))
            masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
            conf_mat += metric.confusion_matrix(pred=preds.flatten(), label=masks.flatten(),
                                                num_classes=args.numclasses)
        metric.print_evaluate_results(conf_mat, np.diag(conf_mat) / (conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - np.diag(conf_mat)))
        train_metrics = self._foreground_metrics(conf_mat)

        print(
            f"Epoch [{epoch}] Train - Loss: {train_loss.avg:.4f}, "
            f"Precision: {train_metrics['precision']:.4f}, "
            f"Recall: {train_metrics['recall']:.4f}, "
            f"F1: {train_metrics['f1']:.4f}, "
            f"IoU: {train_metrics['iou']:.4f}"
        )

        # 保存训练指标历史（确保都是numpy标量）
        self.train_metrics_history['loss'].append(float(train_loss.avg))
        for key in ('precision', 'recall', 'f1', 'iou'):
            self.train_metrics_history[key].append(train_metrics[key])

        # 每个epoch后更新图表
        self.plot_metrics()

    def validating(self, epoch):
        self.model.eval()  # 将module设为预测模式，影响Dropout和BatchNorm等
        conf_mat = np.zeros((args.numclasses, args.numclasses)).astype(np.int64)
        tbar = tqdm(self.val_loader)
        for index, data in enumerate(tbar):
            imgs = Variable(data[0])
            masks = Variable(data[1])
            imgs = imgs.cuda(args.gpu)
            masks = masks.cuda(args.gpu)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(imgs)
            _, preds = torch.max(outputs, 1)
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
            masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
            conf_mat += metric.confusion_matrix(pred=preds.flatten(), label=masks.flatten(),
                                                num_classes=args.numclasses)

        metric.print_evaluate_results(conf_mat, np.diag(conf_mat) / (conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - np.diag(conf_mat)))
        val_metrics = self._foreground_metrics(conf_mat)
        # Validation metrics
        model_name = 'epoch_%d_f1_%.5f' % (epoch, val_metrics['f1'])
        latest_model_path = os.path.join(self.args.directory, 'latest.pth')
        torch.save(self.model.state_dict(), latest_model_path)
        # 模型保存
        if val_metrics['f1'] > self.args.best_f1_score:
            torch.save(self.model.state_dict(), os.path.join(self.args.directory, model_name + '.pth'))
            self.args.best_f1_score = val_metrics['f1']

        print(
            f"Epoch [{epoch}] Val - "
            f"Precision: {val_metrics['precision']:.4f}, "
            f"Recall: {val_metrics['recall']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}, "
            f"IoU: {val_metrics['iou']:.4f}"
        )

        # 保存验证指标历史（确保都是numpy标量）
        for key in ('precision', 'recall', 'f1', 'iou'):
            self.val_metrics_history[key].append(val_metrics[key])

        # 每个epoch后更新图表
        self.plot_metrics()

    def plot_metrics(self):
        """绘制所有指标的变化曲线"""
        epochs = range(len(self.train_metrics_history['loss']))
        val_epochs = range(len(self.val_metrics_history['precision']))

        metrics_to_plot = [
            ('precision', 'Precision'),
            ('recall', 'Recall'),
            ('f1', 'F1 Score'),
            ('iou', 'IoU'),
        ]

        for metric_key, metric_title in metrics_to_plot:
            plt.figure(figsize=(10, 6))

            # 绘制训练指标
            if metric_key in self.train_metrics_history and len(self.train_metrics_history[metric_key]) > 0:
                plt.plot(epochs, self.train_metrics_history[metric_key], 'orange', label=f'Train {metric_title}', linewidth=2)

            # 绘制验证指标
            if metric_key in self.val_metrics_history and len(self.val_metrics_history[metric_key]) > 0:
                plt.plot(val_epochs, self.val_metrics_history[metric_key], 'blue', label=f'Val {metric_title}', linewidth=2)

            # 计算最佳指标值和对应的epoch
            best_val_value = 0
            best_epoch = 0
            if metric_key in self.val_metrics_history and len(self.val_metrics_history[metric_key]) > 0:
                val_values = self.val_metrics_history[metric_key]
                best_val_value = max(val_values)
                best_epoch = val_values.index(best_val_value)

            # 设置标题显示最佳值和对应epoch
            plt.title(f'{metric_title} vs Epoch - Best {metric_title.split()[-1]} is {best_val_value:.4f} at Epoch {best_epoch}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_title)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 保存图片
            plt.savefig(os.path.join(self.args.directory, f'{metric_key}_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图形以释放内存


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args = parse_args()

    trainer = Trainer(args)
    print("Starting Epoch:", args.start_epoch)
    for epoch in range(args.start_epoch, args.total_epochs):
        trainer.training(epoch)
        trainer.validating(epoch)

    # 所有训练完成后，再次更新最终的图表
    trainer.plot_metrics()
