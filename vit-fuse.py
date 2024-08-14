from functools import partial
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 
#--------------------------------------#
#   Gelu激活函数的实现
#   利用近似的数学公式
#--------------------------------------#
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
 
    def forward(self, x):
        return 0.5 * x * (1 + F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x,3))))
 
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob       = 1 - drop_prob
    shape           = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor   = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_() 
    output          = x.div(keep_prob) * random_tensor
    return output
 
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
 
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
 
class PatchEmbed(nn.Module):
    def __init__(self, input_shape=[160, 160], patch_size=16, in_chans=256, num_features=256, norm_layer=None, flatten=True):
        super().__init__()
        self.num_patches    = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.flatten        = flatten
 
        self.proj = nn.Conv2d(in_chans, num_features, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(num_features) if norm_layer else nn.Identity()
 
    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
 
#--------------------------------------------------------------------------------------------------------------------#
#   Attention机制
#   将输入的特征qkv特征进行划分，首先生成query, key, value。query是查询向量、key是键向量、v是值向量。
#   然后利用 查询向量query 点乘 转置后的键向量key，这一步可以通俗的理解为，利用查询向量去查询序列的特征，获得序列每个部分的重要程度score。
#   然后利用 score 点乘 value，这一步可以通俗的理解为，将序列每个部分的重要程度重新施加到序列的值上去。
#--------------------------------------------------------------------------------------------------------------------#
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads  = num_heads
        self.scale      = (dim // num_heads) ** -0.5
 
        self.qkv        = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim)
        self.proj_drop  = nn.Dropout(proj_drop)
 
    def forward(self, x):
        B, N, C     = x.shape
        qkv = self.qkv(x)
        qkv         = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v     = qkv[0], qkv[1], qkv[2]
 
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
 
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
 
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs      = (drop, drop)
 
        self.fc1    = nn.Linear(in_features, hidden_features)
        self.act    = act_layer()
        self.drop1  = nn.Dropout(drop_probs[0])
        self.fc2    = nn.Linear(hidden_features, out_features)
        self.drop2  = nn.Dropout(drop_probs[1])
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
 
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1      = norm_layer(dim)
        self.attn       = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2      = norm_layer(dim)
        self.mlp        = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path  = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
        

class VisionTransformer(nn.Module):
    def __init__(
            self, input_shape=[160, 160], patch_size=16, in_chans=3, num_classes=1000, num_features=768,
            depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=GELU
        ):
        super().__init__()
        #-----------------------------------------------#
        #   224, 224, 3 -> 196, 768
        #-----------------------------------------------#
        self.patch_embed    = PatchEmbed(input_shape=input_shape, patch_size=patch_size, in_chans=in_chans, num_features=num_features)
        num_patches         = (160 // patch_size) * (160 // patch_size)
        self.num_features   = num_features
        self.new_feature_shape = [int(input_shape[0] // patch_size), int(input_shape[1] // patch_size)]
        self.old_feature_shape = [int(160 // patch_size), int(160 // patch_size)]
 
        #--------------------------------------------------------------------------------------------------------------------#
        #   classtoken部分是transformer的分类特征。用于堆叠到序列化后的图片特征中，作为一个单位的序列特征进行特征提取。
        #
        #   在利用步长为16x16的卷积将输入图片划分成14x14的部分后，将14x14部分的特征平铺，一幅图片会存在序列长度为196的特征。
        #   此时生成一个classtoken，将classtoken堆叠到序列长度为196的特征上，获得一个序列长度为197的特征。
        #   在特征提取的过程中，classtoken会与图片特征进行特征的交互。最终分类时，我们取出classtoken的特征，利用全连接分类。
        #--------------------------------------------------------------------------------------------------------------------#
        #   196, 768 -> 197, 768
        self.cls_token      = nn.Parameter(torch.zeros(1, 1, num_features))
        #--------------------------------------------------------------------------------------------------------------------#
        #   为网络提取到的特征添加上位置信息。
        #   以输入图片为224, 224, 3为例，我们获得的序列化后的图片特征为196, 768。加上classtoken后就是197, 768
        #   此时生成的pos_Embedding的shape也为197, 768，代表每一个特征的位置信息。
        #--------------------------------------------------------------------------------------------------------------------#
        #   197, 768 -> 197, 768
        self.pos_embed      = nn.Parameter(torch.zeros(1, num_patches + 1, num_features))
        self.pos_drop       = nn.Dropout(p=drop_rate)
 
        #-----------------------------------------------#
        #   197, 768 -> 197, 768  12次
        #-----------------------------------------------#
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim         = num_features, 
                    num_heads   = num_heads, 
                    mlp_ratio   = mlp_ratio, 
                    qkv_bias    = qkv_bias, 
                    drop        = drop_rate,
                    attn_drop   = attn_drop_rate, 
                    drop_path   = dpr[i], 
                    norm_layer  = norm_layer, 
                    act_layer   = act_layer
                )for i in range(depth)
            ]
        )
        self.norm = norm_layer(num_features)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
 
    def forward_features(self, x):
        x           = self.patch_embed(x)
        cls_token   = self.cls_token.expand(x.shape[0], -1, -1) 
        x           = torch.cat((cls_token, x), dim=1)
        
        cls_token_pe = self.pos_embed[:, 0:1, :]
        img_token_pe = self.pos_embed[:, 1: , :]
 
        img_token_pe = img_token_pe.view(1, *self.old_feature_shape, -1).permute(0, 3, 1, 2)
        img_token_pe = F.interpolate(img_token_pe, size=self.new_feature_shape, mode='bicubic', align_corners=False)
        img_token_pe = img_token_pe.permute(0, 2, 3, 1).flatten(1, 2)
        pos_embed = torch.cat([cls_token_pe, img_token_pe], dim=1)
 
        x = self.pos_drop(x + pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]
 
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
 
    def freeze_backbone(self):
        backbone = [self.patch_embed, self.cls_token, self.pos_embed, self.pos_drop, self.blocks[:8]]
        for module in backbone:
            try:
                for param in module.parameters():
                    param.requires_grad = False
            except:
                module.requires_grad = False
 
    def Unfreeze_backbone(self):
        backbone = [self.patch_embed, self.cls_token, self.pos_embed, self.pos_drop, self.blocks[:8]]
        for module in backbone:
            try:
                for param in module.parameters():
                    param.requires_grad = True
            except:
                module.requires_grad = True
 





class ViT(nn.Module):
    def __init__(self, input_shape=[160, 160], patch_size=16, in_chans=512, num_features=256, 
                 norm_layer=nn.LayerNorm, flatten=True, drop_rate=0.1):
        super().__init__()
        
        '''PatchEmbed'''
        self.num_patches    = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.flatten        = flatten
 
        self.proj = nn.Conv2d(in_chans, num_features, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(num_features) if norm_layer else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_features))
        num_patches = 4 * (input_shape[0] // patch_size) * (input_shape[0] // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, num_features))
        self.new_feature_shape = [int(input_shape[1] // patch_size), int(input_shape[1] // patch_size)]
        self.old_feature_shape = [int(input_shape[1] // patch_size), int(input_shape[1] // patch_size)]


        self.pos_drop = nn.Dropout(p=drop_rate)

        num_heads = 2
        mlp_ratio = 2
        qkv_bias = True
        attn_drop_rate = 0.1
        drop_path_rate = 0.1
        depth = 1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        act_layer = GELU

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim         = num_features, 
                    num_heads   = num_heads, 
                    mlp_ratio   = mlp_ratio, 
                    qkv_bias    = qkv_bias, 
                    drop        = drop_rate,
                    attn_drop   = attn_drop_rate, 
                    drop_path   = dpr[i], 
                    norm_layer  = norm_layer, 
                    act_layer   = act_layer
                )for i in range(depth)
            ]
        )


    def PatchEmbed(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    

    def forward(self, x):
        bs, ch, h, w = x[0].shape
        fea_1 = x[0]   # rgb_fea (tensor): dim:(B, C, H, W) [B, 256, 160, 160]
        fea_2 = x[1]   # polar_fea (tensor): dim:(B, C, H, W) [B, 256, 160, 160]
        fea_3 = x[2]   # ir_fea (tensor): dim:(B, C, H, W) [B, 256, 160, 160]
        fea_4 = x[3]   # decay_fea (tensor): dim:(B, C, H, W) [B, 256, 160, 160]

        assert fea_1.shape[0] == fea_2.shape[0] == fea_3.shape[0] == fea_4.shape[0]
        
        x_1 = self.PatchEmbed(fea_1)  # [b, 100, 256]
        x_2 = self.PatchEmbed(fea_2)  # [b, 100, 256]
        x_3 = self.PatchEmbed(fea_3)  # [b, 100, 256]
        x_4 = self.PatchEmbed(fea_4)  # [b, 100, 256]
        x = torch.cat([x_1, x_2, x_3, x_4], dim=1)  # concat [B, 400, 256]
        
        '''
        # p_x = torch.cat([fea_1, fea_2, fea_3, fea_4], dim=2)  # concat [B, 256, 640, 160]
        # x = self.PatchEmbed(p_x)  # [b, 400, 256]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [b, 1, 256]
        x = torch.cat((cls_token, x), dim=1)  # [b, 401, 256]
        cls_token_pe = self.pos_embed[:, 0:1, :]  # [1, 1, 256]
        img_token_pe = self.pos_embed[:, 1: , :]  # [1, 400, 256]
        
        img_token_pe = img_token_pe.view(1, *self.old_feature_shape, -1).permute(0, 3, 1, 2)  # [1, 1024, 10, 10], 256*4, 160/16, 160/16
        img_token_pe = F.interpolate(img_token_pe, size=self.new_feature_shape, mode='bicubic', align_corners=False)  # [1, 1024, 10, 10]
        img_token_pe = img_token_pe.permute(0, 2, 3, 1).flatten(1, 2)  # [1, 100, 1024]
        img_token_pe = img_token_pe.reshape(1, l*img_token_pe.shape[1], img_token_pe.shape[2]//l)  # [1, 400, 256]
        pos_embed = torch.cat([cls_token_pe, img_token_pe], dim=1)  # [1, 401, 256]
        x = self.pos_drop(x + pos_embed)  # [b, 401, 256]
        '''

        x = self.blocks(x)  # [b, 400, 256]
        x = x.view(bs, 4, x.shape[1]//4, ch)

        fea_1_new = x[:, 0, :].permute(0, 2, 1).view(bs, ch, int(sqrt(x.shape[2])), -1)
        fea_2_new = x[:, 1, :].permute(0, 2, 1).view(bs, ch, int(sqrt(x.shape[2])), -1)
        fea_3_new = x[:, 2, :].permute(0, 2, 1).view(bs, ch, int(sqrt(x.shape[2])), -1)
        fea_4_new = x[:, 3, :].permute(0, 2, 1).view(bs, ch, int(sqrt(x.shape[2])), -1)

        fea_1_out = F.interpolate(fea_1_new, size=([h, w]), mode='bilinear')
        fea_2_out = F.interpolate(fea_2_new, size=([h, w]), mode='bilinear')
        fea_3_out = F.interpolate(fea_3_new, size=([h, w]), mode='bilinear')
        fea_4_out = F.interpolate(fea_4_new, size=([h, w]), mode='bilinear')

        return fea_1_out, fea_2_out, fea_3_out, fea_4_out





    
def vit_b_16(input_shape=[160, 160], pretrained=False, num_classes=10):
    model = VisionTransformer(input_shape)
    
    if num_classes!=1000:
        model.head = nn.Linear(model.num_features, num_classes)
    return model




if __name__ == '__main__':
    input_x = []
    input_rgb = torch.Tensor(3, 256, 160, 160)
    input_polar = torch.Tensor(3, 256, 160, 160)
    input_ir = torch.Tensor(3, 256, 160, 160)
    input_decay = torch.Tensor(3, 256, 160, 160)
    input_x = [input_rgb, input_polar, input_ir, input_decay]

    input_shape = [input_rgb.shape[2], input_rgb.shape[3]]
    num_features = input_rgb.shape[1]
    patch_size=16
    in_chans=input_rgb.shape[1]
    num_features=256
    norm_layer=nn.LayerNorm
    flatten=True
    drop_rate=0.1

    model = ViT(input_shape, patch_size=16, in_chans=input_rgb.shape[1], 
                num_features=num_features, norm_layer=norm_layer, 
                flatten=flatten, drop_rate=drop_rate)

    model(input_x)


    # 假设输出128, 两个拼一起是256;
    # 4通道拼凑的是: 768*3, 改成256*4
    x = torch.zeros((4, 3, 160, 160))
    model(x)