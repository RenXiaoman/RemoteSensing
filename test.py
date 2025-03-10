import argparse
import glob
import os
import torch

from GLANet import GLANet as GLANet
from PIL import Image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from datasets import joint_transforms, transforms as extended_transforms
import torchvision.transforms as standard_transforms
import warnings
warnings.filterwarnings("ignore")

palette = [255, 0, 0,
               255, 255, 255,
               0, 0, 255,
               0, 255, 255,
               0, 255, 0,
               255, 255, 0]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask
class transformimg():
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# 几何图像变换
    train_joint_transform_list = [
        #随机resize和裁剪
        #joint_transforms.RandomSizeAndCrop(size=256, crop_nopad=False, pre_size=None, scale_min=0.5,scale_max=2.0),
        #裁剪大小
        #joint_transforms.Resize(size=256),
        #随机操作
        joint_transforms.RandomHorizontallyFlip()]
    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)
    train_input_transform = []
    train_input_transform += [extended_transforms.ColorJitter(brightness=0.25,contrast=0.25,saturation=0.25,hue=0.25)]
    train_input_transform += [extended_transforms.RandomGaussianBlur()]
    train_input_transform += [standard_transforms.ToTensor(),standard_transforms.Normalize(*mean_std)]
    train_input_transform = standard_transforms.Compose(train_input_transform)
    val_input_transform = standard_transforms.Compose([standard_transforms.ToTensor(),standard_transforms.Normalize(*mean_std)])
    target_transform = extended_transforms.MaskToTensor()

class Gids(Dataset):
    def __init__(self,  joint_transform, transform, target_transform):

        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = sorted(glob.glob(os.path.join('/data/fywdata/Vaihingen/val/', 'image',"*.png")))
        self.label = sorted(glob.glob(os.path.join('/data/fywdata/Vaihingen/val/', 'mask', "*.png")))

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index], self.label[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        mask = np.array(mask)
        mask = np.max(mask, axis=2)

        mask = Image.fromarray(mask.astype(np.uint8))
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask, img_name

    def __len__(self):
        return len(self.imgs)


def parse_args():
    parser = argparse.ArgumentParser(description="RemoteSensingSegmentation by PyTorch")

    # output_save_path
    parser.add_argument('--save-pseudo-data-path', type=str, default='glanet')
    #chosen gpu
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--numclasses', type=int, default=6, help='number of classes')
    parser.add_argument("--model-path", type=str, default="/home/user/fyw/test/epoch_101_oa_0.85433_kappa_0.80770.pth")

    args = parser.parse_args()
    return args

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.resore_transform = transforms.Compose([
            DeNormalize([.485, .456, .406], [.229, .224, .225]),
            transforms.ToPILImage()
        ])
        #可视化
        self.visualize = transforms.Compose([transforms.ToTensor()])
        # 验证集
        tranformimage=transformimg()
        val_set = Gids(joint_transform=None, transform=tranformimage.val_input_transform,target_transform=tranformimage.target_transform)
        val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, num_workers=4)
        self.val_loader=val_loader
        model=GLANet(numclasses=args.numclasses)
        state_dict = torch.load(args.model_path,map_location='cuda:0')
        model.cuda(args.gpu).load_state_dict(state_dict,False)

        self.model = model
        #Adadelta优化器
        self.optimizer = torch.optim.Adadelta(model.parameters(),lr=0.1,weight_decay=0.0001)
        #存储结果
        self.save_pseudo_data_path = args.save_pseudo_data_path

    def validating(self):
        self.model.eval()  # 把module设成预测模式，对Dropout和BatchNorm有影响
        tbar = tqdm(self.val_loader)
        for index, data in enumerate(tbar):
            imgs = Variable(data[0])
            #masks = Variable(data[1])
            imgname = data[2]
            imgs = imgs.cuda(args.gpu)
            #masks = masks.cuda()
            self.optimizer.zero_grad()
            outputs= self.model(imgs)
            _, preds = torch.max(outputs, 1)
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
            #masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
            score = _.data.cpu().numpy()
            val_visual = []
            for i in range(score.shape[0]):
                img_pil = self.resore_transform(data[0][i])
                pred_vis_pil = colorize_mask(preds)
                gt_vis_pil = colorize_mask(data[1][i].numpy())

                val_visual.extend([self.visualize(img_pil.convert('RGB')),
                                           self.visualize(gt_vis_pil.convert('RGB')),
                                           self.visualize(pred_vis_pil.convert('RGB'))])

                rgb_save_path = os.path.join(self.save_pseudo_data_path, 'rgb')
                vis_save_path = os.path.join(self.save_pseudo_data_path, 'vis_label')
                gt_save_path = os.path.join(self.save_pseudo_data_path, 'gt')

                path_list = [rgb_save_path, vis_save_path, gt_save_path]

                for path in range(3):
                    if not os.path.exists(path_list[path]):
                        os.makedirs(path_list[path])
                img_pil.save(os.path.join(path_list[0], 'img_%s.jpg' % (imgname[i])))
                pred_vis_pil.save(os.path.join(path_list[1], 'vis_%s.png' % (imgname[i])))
                gt_vis_pil.save(os.path.join(path_list[2], 'gt_%s.png' % (imgname[i])))

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    trainer = Trainer(args)
    trainer.validating()

