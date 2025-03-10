import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter
import math

# Squareplus
def Squareplus(x,b=1):
    x = x+torch.sqrt(torch.pow(x,2)+1)
    x=x/2

    return x


class SquareplusAtt(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(SquareplusAtt, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places

        # lecun_Norm
        self.Squareplus = Squareplus
        # SELU����
        #self.mySELU = torch.nn.SELU()
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

        # 1*1Conv
        # self.conv1by1 = nn.Conv2d(in_places+1, in_places, 1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape

        Q = (self.query_conv(x)).view(batch_size, -1, width * height)
        K = (self.key_conv(x)).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.Squareplus(Q).permute(-3, -1, -2)
        K = self.Squareplus(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)
        return (x + self.gamma * weight_value).contiguous()


#NA module
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    """3x3 convolution with padding bn relu and no bias"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                                   bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU(inplace=True))


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution and no bias; downsample 1/stride"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def atrous_conv3x3(in_planes, out_planes, rate=1, padding=1, stride=1):
    """3x3  atrous convolution and no bias"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                     dilation=rate, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, first_inplanes, inplanes, planes, rate=1, padding=1, stride=1, downsample=None):

        super().__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)  ####
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = atrous_conv3x3(planes, planes, rate, padding)  ####
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # only first layer1 block in_channel different
        if (first_inplanes != inplanes) and (downsample is not None):
            self.conv1 = conv1x1(first_inplanes, planes, stride)
            self.downsample = nn.Sequential(conv1x1(first_inplanes, planes * self.expansion, stride),
                                            nn.BatchNorm2d(planes * self.expansion))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MyNet(nn.Module):
    def __init__(self, block, layers, class_number, dropout_rate=0.2, in_channel=3):
        super().__init__()
        self.inplanes = 64
        self.conv1_1 = conv3x3_bn_relu(in_channel, 64, stride=2)
        self.conv1_2 = conv3x3_bn_relu(64, 64)
        self.conv1_3 = conv3x3_bn_relu(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(block, 128, 64, layers[0])  # 64 / 256
        self.layer2 = self._make_layer(block, 256, 128, layers[1], stride=2)  # 128 / 512
        self.layer3 = self._make_layer(block, 512, 256, layers[2], rate=2, padding=2)  # 256 / 1024
        self.layer4 = self._make_layer(block, 1024, 512, layers[3], rate=4, padding=4)  # 512 / 2048

        self.attention = SquareplusAtt(2048)
        self.conv5_4 = conv3x3_bn_relu(2048, 512)  ##if you want modify in_channel, need your own modify##

        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv6 = nn.Conv2d(512, class_number, 1, 1)

        ''' init weight '''
        # print('## init weight ##')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # no convtranspose linear

    def forward(self, x):
        size = x.shape[2:]
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention(x)
        x = self.conv5_4(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = F.upsample(x, size, mode='bilinear', align_corners=True)

        return x

    def _make_layer(self, block, first_inplanes, planes, blocks, rate=1, padding=1, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),  # with down stride same
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(first_inplanes, self.inplanes, planes, rate, padding, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes, planes, rate, padding))

        return nn.Sequential(*layers)


def GLANet(numclasses=6):
    model = MyNet(Bottleneck, layers=[3, 4, 6, 3], class_number=numclasses, dropout_rate=0.2)
    return model

