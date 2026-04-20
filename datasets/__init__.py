import os
from datasets import giddataset, joint_transforms, transforms as extended_transforms
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

def setup_loaders(args):

    args.dataset_cls = giddataset
    batch_size = args.batchsize
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_joint_transform_list = [
        #随机操作
        joint_transforms.RandomHorizontallyFlip()#,
        #joint_transforms.Resize(256),
        #joint_transforms.RandomHorizontallyFlip()
        ]
    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)
    train_input_transform = []
    # train_input_transform += [extended_transforms.ColorJitter(brightness=0.25,contrast=0.25,saturation=0.25,hue=0.25)]
    # train_input_transform += [extended_transforms.RandomGaussianBlur()]
    # train_input_transform += [standard_transforms.ToTensor(),standard_transforms.Normalize(*mean_std)]
    train_input_transform += [standard_transforms.ToTensor()]
    train_input_transform = standard_transforms.Compose(train_input_transform)

    # val_input_transform = standard_transforms.Compose([standard_transforms.ToTensor(),standard_transforms.Normalize(*mean_std)])
    val_input_transform = standard_transforms.Compose([standard_transforms.ToTensor()])

    target_transform = extended_transforms.MaskToTensor()

    train_set = args.dataset_cls.Gids('train',joint_transform=train_joint_transform,transform=train_input_transform,target_transform=target_transform)
    val_set = args.dataset_cls.Gids('val', joint_transform =None,transform=val_input_transform,target_transform=target_transform)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers，单处理器一般设置为0
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = DataLoader(train_set, batch_size=batch_size,num_workers=nw, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size,num_workers=nw, shuffle=False)

    print("train_loader:"+str(len(train_loader)))
    print("val_loader:"+str(len(val_loader)))

    return train_loader, val_loader
