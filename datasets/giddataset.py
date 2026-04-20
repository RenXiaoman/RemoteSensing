import os
import numpy as np
from PIL import Image
from torch.utils import data
import glob

# Update root path to point to your new data directory
root = '/home/ikun_server/clib/PycharmProjects/GLANet/data/high_oxygen_fish_pond_20Percent'
palette = [255, 0, 0,
               255, 255, 255,
               0, 0, 255,
               0, 255, 255,
               0, 255, 0,
               255, 0, 0]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    #加入调色板
    new_mask.putpalette(palette)
    return new_mask


class Gids(data.Dataset):
    def __init__(self,  mode, joint_transform, transform, target_transform):
        self.mode = mode
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        if mode=='train':
            self.imgs=sorted(glob.glob(os.path.join(root,'train','image',"*.png")))
            self.label = sorted(glob.glob(os.path.join(root, 'train', 'mask',"*.png")))
        elif mode=='val':
            self.imgs = sorted(glob.glob(os.path.join(root, 'val', 'image',"*.png")))
            self.label = sorted(glob.glob(os.path.join(root, 'val', 'mask',"*.png")))


    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index],self.label[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask=np.array(mask)  # [256, 256]
        
        # 如果mask是三维的（多通道），则取最大值；如果是二维的（灰度），则直接使用
        if len(mask.shape) == 3:
            mask=np.max(mask, axis=2)
        # 如果mask已经是二维的，就无需操作

        mask = Image.fromarray(mask.astype(np.uint8))
        # 图像变换
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
                
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)
        

        return img, mask, img_name

    def __len__(self):
        return len(self.imgs)


