# coding=gbk
import argparse
import logging
import os
import torch

from GLANet import GLANet as GLANet

from torch import nn
from libs import average_meter, metric
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import datasets
from tensorboardX import SummaryWriter
import warnings
from libs.metric import save_log

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="RemoteSensingSegmentation by PyTorch")
    parser.add_argument('--batchsize', type=int, default=16, help='batchsize')
    # model and classes
    parser.add_argument('--model', type=str, default='GLANet', help='model name')
    parser.add_argument('--numclasses', type=int, default=6, help='number of classes')
    # GPU
    parser.add_argument('--gpu', type=int, default=3, help='the chosen gpu')
    # learning_rate
    parser.add_argument('--base-lr', type=float, default=0.1, metavar='M', help='')
    parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='M', help='weight-decay (default:1e-4)')
    # best result
    parser.add_argument('--best-kappa', type=float, default=0)

    parser.add_argument('--total-epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 120)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start epoch (default:0)')

    args = parser.parse_args()

    directory = "%s/" % (args.model)
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

        self.criterion = nn.CrossEntropyLoss(weight=None, reduction='mean', ignore_index=-1).cuda(args.gpu)

        model = GLANet(numclasses=args.numclasses)
        self.model = model.cuda(args.gpu)

        self.optimizer = torch.optim.Adadelta(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
        self.max_iter = args.total_epochs * len(self.train_loader)
        # 网络参数的数量
        num_params = sum(p.numel() for p in model.parameters())
        logging.info('Model params = {:2.1f}M'.format(num_params / 1000000))

    def training(self, epoch):
        self.model.train()  # 把module设成训练模式，对Dropout和BatchNorm有影响
        # 初始化
        train_loss = average_meter.AverageMeter()

        curr_iter = epoch * len(self.train_loader)
        lr = self.args.base_lr * (1 - float(curr_iter) / self.max_iter) ** 0.9

        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        conf_mat = np.zeros((args.numclasses, args.numclasses)).astype(np.int64)
        tbar = tqdm(self.train_loader)
        for index, data in enumerate(tbar):
            imgs = Variable(data[0])
            masks = Variable(data[1])
            imgs = imgs.cuda(args.gpu)
            masks = masks.cuda(args.gpu)
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
        train_acc, train_acc_per_class, train_acc_cls, train_IoU, train_mean_IoU, train_kappa = metric.evaluate(
            conf_mat)

        writer.add_scalar(tag='train_acc', scalar_value=train_acc, global_step=epoch, walltime=None)
        writer.add_scalar(tag='train_loss_per_epoch', scalar_value=train_loss.avg, global_step=epoch, walltime=None)

    def validating(self, epoch):
        self.model.eval()  # 把module设成预测模式，对Dropout和BatchNorm有影响
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

        val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_mean_IoU, val_kappa = metric.evaluate(conf_mat)
        # 存入验证OA
        writer.add_scalar(tag='val_acc', scalar_value=val_acc, global_step=epoch, walltime=None)
        model_name = 'epoch_%d_oa_%.5f_kappa_%.5f' % (epoch, val_acc, val_kappa)
        # 存最佳参数
        if val_kappa > self.args.best_kappa:
            torch.save(self.model.state_dict(), os.path.join(self.args.directory, model_name + '.pth'))
            self.args.best_kappa = val_kappa


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    writer = SummaryWriter(args.directory)

    trainer = Trainer(args)
    print("Starting Epoch:", args.start_epoch)
    for epoch in range(args.start_epoch, args.total_epochs):
        trainer.training(epoch)
        trainer.validating(epoch)