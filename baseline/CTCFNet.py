import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def to_2tuple(x):
    return tuple([x] * 2)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input

class PathchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).permute(0, 2, 1)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q.matmul(k.permute(0, 1, 3, 2))) * self.scale
        attn = F.softmax(attn, dim=1)
        attn = self.attn_drop(attn)
        x = (attn.matmul(v)).permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class  Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class BasicBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer='nn.LayerNorm', epsilon=1e-5):
        super().__init__()
        self.norm1 = eval(norm_layer)(dim, eps=epsilon)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = eval(norm_layer)(dim, eps=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def truncated_normal_(tensor,mean=0,std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, class_dim=1000, 
                embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False,
                qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_layer='nn.LayerNorm', epsilon=1e-5, **args):
        super().__init__()
        self.class_dim = class_dim
        self.patch_embed = PathchEmbed(img_size=img_size, patch_size=patch_size, 
                                        in_chans=in_chans,embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            BasicBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                epsilon=epsilon) for i in range(depth)
            ])
        self.norm = eval(norm_layer)(embed_dim, eps=epsilon)
        self.head = nn.Linear(embed_dim, class_dim) if class_dim > 0 else Identity()

        truncated_normal_(tensor=self.pos_embed)
        truncated_normal_(tensor=self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            truncated_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:,0]  
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Bi_DirectionalDecoder(nn.Module):
    def __init__(self, interpolate_size):
        super().__init__()
        self.interpolate_size = interpolate_size
        self.top_down_path_conv1x1_stage3 = nn.Conv2d(in_channels=512, out_channels=320, kernel_size=1)
        self.top_down_path_conv1x1_stage2 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=1)
        self.top_down_path_conv1x1_stage1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.bottom_up_path_conv1x1_stage2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.bottom_up_path_conv1x1_stage3 = nn.Conv2d(in_channels=128, out_channels=320, kernel_size=1)
        self.bottom_up_path_conv1x1_stage4 = nn.Conv2d(in_channels=320, out_channels=512, kernel_size=1)
        self.down2x_by_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.cbr = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=512,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
    
    def top_down_path(self, encoder1, encoder2, encoder3, encoder4,):
        stage4 = F.interpolate(encoder4, size=(self.interpolate_size, self.interpolate_size), mode='bilinear', align_corners=True)
        stage3 = self.top_down_path_conv1x1_stage3(stage4) + F.interpolate(encoder3, size=(self.interpolate_size, self.interpolate_size), mode='bilinear', align_corners=True)
        stage2 = self.top_down_path_conv1x1_stage2(stage3) + F.interpolate(encoder2, size=(self.interpolate_size, self.interpolate_size), mode='bilinear', align_corners=True)
        stage1 = self.top_down_path_conv1x1_stage1(stage2) + F.interpolate(encoder1, size=(self.interpolate_size, self.interpolate_size), mode='bilinear', align_corners=True)
        return torch.cat([stage1 ,stage2 , stage3, stage4],dim=1)
    
    def bottom_up_path(self, encoder1, encoder2, encoder3, encoder4):
        stage2 = encoder2+self.bottom_up_path_conv1x1_stage2(self.down2x_by_maxpool(encoder1))
        stage3 = encoder3+self.bottom_up_path_conv1x1_stage3(self.down2x_by_maxpool(stage2))
        stage4 = encoder4+self.bottom_up_path_conv1x1_stage4(self.down2x_by_maxpool(stage3))
        return torch.cat([F.interpolate(encoder1, size=(self.interpolate_size, self.interpolate_size), mode='bilinear',align_corners=True),\
                F.interpolate(stage2,size=(self.interpolate_size, self.interpolate_size), mode='bilinear',align_corners=True),\
                F.interpolate(stage3,size=(self.interpolate_size, self.interpolate_size), mode='bilinear',align_corners=True),\
                F.interpolate(stage4,size=(self.interpolate_size, self.interpolate_size), mode='bilinear',align_corners=True)],dim=1)
    
    def forward(self, input1, input2, input3, input4):
        top_down_feats = self.top_down_path(input1, input2, input3, input4)
        bottom_up_feats = self.bottom_up_path(input1, input2, input3, input4)
        return self.cbr(torch.cat([top_down_feats, bottom_up_feats], dim=1))
    
class StripPyramidPool(nn.Module):
    def __init__(self, in_chans, trans_chans) -> None:
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.maxpool_3x3 = nn.MaxPool2d(kernel_size=3)
        self.maxpool_5x5 = nn.MaxPool2d(kernel_size=5)
        self.maxpool_7x7 = nn.MaxPool2d(kernel_size=7)
        self.conv1x1 = nn.Conv2d(in_channels=in_chans,out_channels=in_chans//5, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels=(in_chans//5)*4 + in_chans, out_channels=trans_chans, kernel_size=3, padding=1)
    
    def forward(self,x):
        global_pool = F.interpolate(self.conv1x1(self.global_pool(x)),size=x.size()[2:],mode='bilinear',align_corners=True)
        maxpool_3x3 = F.interpolate(self.conv1x1(self.maxpool_3x3(x)),size=x.size()[2:],mode='bilinear',align_corners=True)
        maxpool_5x5 = F.interpolate(self.conv1x1(self.maxpool_5x5(x)),size=x.size()[2:],mode='bilinear',align_corners=True)
        maxpool_7x7 = F.interpolate(self.conv1x1(self.maxpool_7x7(x)),size=x.size()[2:],mode='bilinear',align_corners=True)
        return self.conv3x3(torch.cat([global_pool,maxpool_3x3,maxpool_5x5,maxpool_7x7,x],dim=1))
    
class FeatureAggregationModule(nn.Module):
    def __init__(self, cnn_chans, trans_chans) -> None:
        super().__init__()
        self.branch1 = StripPyramidPool(cnn_chans, trans_chans)
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels=cnn_chans+trans_chans, out_channels=trans_chans,kernel_size=3,padding=1),
                                     nn.BatchNorm2d(trans_chans),
                                     nn.GELU())
        self.branch3 = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                     nn.Conv2d(in_channels=trans_chans, out_channels=trans_chans // 4, kernel_size=1),
                                     nn.GELU(),
                                     nn.Conv2d(in_channels=trans_chans//4, out_channels=trans_chans, kernel_size=1),
                                     nn.Sigmoid())
        self.head = nn.Sequential(nn.Conv2d(in_channels=trans_chans*3, out_channels=trans_chans, kernel_size=3,padding=1),
                                  nn.BatchNorm2d(trans_chans),
                                  nn.GELU())
    
    def forward(self, cnn_block, trans_block):
        branch1 = self.branch1(cnn_block)
        branch2 = self.branch2(torch.cat([cnn_block, trans_block], dim=1))
        branch3 = self.branch3(branch2) * trans_block
        return self.head(torch.cat([branch1, branch2, branch3],dim=1))
    
class DepthwiseConv2d(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=1, stride=1,padding=0,dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_chans,
            out_channels=in_chans,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_chans
        )
        self.bn = nn.BatchNorm2d(num_features=in_chans)
        self.pointwise = nn.Conv2d(
            in_channels=in_chans,
            out_channels=out_chans,
            kernel_size=1
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x
    
    
class CNN_Block123(nn.Module):
    def __init__(self, inchans, outchans):
        super().__init__()
        self.stage = nn.Sequential(
            DepthwiseConv2d(in_chans=inchans, out_chans=outchans, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(outchans),
            nn.ReLU(),
            DepthwiseConv2d(in_chans=outchans, out_chans=outchans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchans),
            nn.ReLU(),
        )
        self.conv1x1 = DepthwiseConv2d(in_chans=inchans, out_chans=outchans, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    def forward(self, x):
        stage = self.stage(x)
        max = self.maxpool(x)
        max = self.conv1x1(max)
        stage = stage + max
        return stage

class CNN_Block45(nn.Module):
    def __init__(self, inchans, outchans):
        super().__init__()

        self.stage = nn.Sequential(
            DepthwiseConv2d(in_chans=inchans, out_chans=outchans, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(outchans),
            nn.ReLU(),
            DepthwiseConv2d(in_chans=outchans, out_chans=outchans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchans),
            nn.ReLU(),
            DepthwiseConv2d(in_chans=outchans, out_chans=outchans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchans),
            nn.ReLU(),
        )
        self.conv1x1 = DepthwiseConv2d(in_chans=inchans, out_chans=outchans, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    def forward(self, x):
        stage = self.stage(x)
        max = self.maxpool(x)
        max = self.conv1x1(max)
        stage = stage + max
        return stage

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q.matmul(k.permute(0,1,3,2))) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).permute(0,2,1,3).reshape(-1, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(BasicBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,attn_drop=0.,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, epsilon=1e-6, sr_ratio=1):
        super().__init__(dim, num_heads)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
    
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchEmbed(PathchEmbed):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super(PatchEmbed,self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).permute(0,2,1)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)

class CTCFNet(nn.Module):
    def __init__(self,img_size=224, patch_size=16, in_chans=3, embed_dims=[64,128,256,512],
                num_heads=[1,2,4,8], mlp_ratios=[4,4,4,4], qkv_bias=False, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                epsilon=1e-6, depths=[3,4,6,3], sr_ratios=[8, 4, 2, 1], class_dim=2):
        super().__init__()
        self.class_dim = class_dim
        self.depths = depths
        self.img_size = img_size
        self.interpolate_size = img_size // 4

        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                        in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2, 
                                        in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2, 
                                        in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2, 
                                        in_chans=embed_dims[2], embed_dim=embed_dims[3])
        
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(drop_rate)

        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))
        self.pos_drop2 = nn.Dropout(drop_rate)

        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, embed_dims[2]))
        self.pos_drop3 = nn.Dropout(drop_rate)

        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches, embed_dims[3]))
        self.pos_drop4 = nn.Dropout(drop_rate)

        dpr = np.linspace(0, drop_path_rate, sum(depths))
        cur = 0

        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, epsilon=epsilon,
                sr_ratio = sr_ratios[0]
            ) for i in range(depths[0])
        ])

        cur = cur + depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, epsilon=epsilon,
                sr_ratio = sr_ratios[1]
            ) for i in range(depths[1])
        ])

        cur = cur + depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, epsilon=epsilon,
                sr_ratio = sr_ratios[2]
            ) for i in range(depths[2])
        ])

        cur = cur+ depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, epsilon=epsilon,
                sr_ratio = sr_ratios[3]
            ) for i in range(depths[3])
        ])
        self.norm = norm_layer(embed_dims[3])

        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dims[3]))

        if class_dim > 0:
            self.head = nn.Linear(embed_dims[3], class_dim)

        self.cnn_branch1 = CNN_Block123(inchans=in_chans, outchans=64)
        self.cnn_branch2 = CNN_Block123(inchans=64, outchans=128)
        self.cnn_branch3 = CNN_Block123(inchans=64, outchans=128)
        self.cnn_branch4 = CNN_Block45(inchans=128, outchans=256)
        self.cnn_branch5 = CNN_Block45(inchans=320, outchans=640)
        self.conv1x1_4 = nn.Conv2d(in_channels=512, out_channels=320, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)

        self.CTmerge2 = FeatureAggregationModule(cnn_chans=128,trans_chans=64)
        self.CTmerge3 = FeatureAggregationModule(cnn_chans=128, trans_chans=128)
        self.CTmerge4 = FeatureAggregationModule(cnn_chans=256, trans_chans=320)
        self.CTmerge5 = FeatureAggregationModule(cnn_chans=640, trans_chans=512)

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=640, out_channels=320,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.detail_head = nn.Conv2d(in_channels=128, out_channels=self.class_dim, kernel_size=1)

        self.bi_directional_decoder = Bi_DirectionalDecoder(self.interpolate_size)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=self.class_dim, kernel_size=1)
        )
        
        truncated_normal_(self.pos_embed1)
        truncated_normal_(self.pos_embed2)
        truncated_normal_(self.pos_embed3)
        truncated_normal_(self.pos_embed4)
        truncated_normal_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            truncated_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)

    def reset_drop_path(self, drop_path_rate):
        dpr = np.linspace(0, drop_path_rate, sum(self.depths))
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur = cur+ self.depths[0]
        for i in range(self.depth[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]
        
        cur = cur+ self.depths[1]
        for i in range(self.depth[2]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur = cur+ self.depths[2]
        for i in range(self.depth[3]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

    
    def forward(self, x):
        B = x.shape[0]
        c_s1 = self.cnn_branch1(x)
        c_s2 = self.cnn_branch2(c_s1)
        # Stage 1
        x, (H, W) = self.patch_embed1(x)
        x = x + self.pos_embed1
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0,3,1,2)
        x = self.CTmerge2(c_s2, x)
        m_s1 = x  
        c_s3 = self.cnn_branch3(m_s1)
        # Stage 2
        x, (H, W) = self.patch_embed2(x)
        x = x + self.pos_embed2
        x = self.pos_drop2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0,3,1,2)
        x = self.CTmerge3(c_s3, x)
        m_s2 = x
        c_s4 = self.cnn_branch4(m_s2)
        # Stage 3
        x, (H, W) = self.patch_embed3(x)
        x = x + self.pos_embed3
        x = self.pos_drop3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0,3,1,2)
        x = self.CTmerge4(c_s4,x)
        m_s3 = x
        c_s5 = self.cnn_branch5(m_s3)
        # Stage 4
        x, (H, W) = self.patch_embed4(x)
        x = x + self.pos_embed4
        x = self.pos_drop4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0,3,1,2)
        x = self.CTmerge5(c_s5,x)
        m_s4 = x

        output = self.bi_directional_decoder(m_s1, m_s2, m_s3, m_s4)
        output = F.interpolate(output, size = self.img_size//2, mode='bilinear', align_corners=True)
        output = self.head(output)

        if self.training:
            return F.interpolate(output,size=self.img_size,mode='bilinear',align_corners=True),\
                self.detail_head(m_s2)
        else:
            return F.interpolate(output,size=self.img_size,mode='bilinear',align_corners=True)


# if __name__ == '__main__':
#     num_classes = 2
#     in_batch, inchannel, in_h, in_w = 10, 3, 256, 256
#     x = torch.randn(in_batch, inchannel, in_h, in_w)

#     net = CTCFNet(img_size=256, in_chans=inchannel, class_dim=num_classes,
#                   patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#                     norm_layer=nn.LayerNorm, depths=[3, 3, 6, 3], sr_ratios=[8, 4, 2, 1])

#     # eval() 模式下 forward 只返回主分割输出。
#     net.eval()
#     with torch.no_grad():
#         out = net(x)

#     print("CTCFNet input shape:", x.shape)
#     print("CTCFNet output shape:", out.shape)
