import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.base_model import BaseModel
from typing import Optional, Callable
import math
from models._baseNet import BaseNet
from models._baseTrainer import Trainer
# from functools import partial
def trunc_normal_(tensor, mean=0., std=1., a=-2.,b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x/math.sqrt(2.)))/2
    l = norm_cdf((a-mean)/std)
    u = norm_cdf((b-mean)/std)
    with torch.no_grad():
        tensor.uniform_(2*l-1,2*u-1)
        tensor.erfinv_()
        tensor.mul_(std*math.sqrt(2.))
        tensor.clamp_(min=a,max=b)
    return tensor

class PatchEmbed(nn.Module):
    """ 2D iq to Patch Embedding
    """

    def __init__(
        self,
        patch_size: tuple = (2,8),
        in_chans: int = 1,
        embed_dim: int = 256,
        norm_layer: Optional[Callable] = None,
        bias: bool = True,
    ):
        super().__init__()
        # self.patch_size = patch_size
        # self.num_patches = 2//patch_size[0]* 128//patch_size[1]
        self.num_patches = 128 // patch_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        # self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        # B, C, H, W = x.shape
        # x = self.norm(x)
        x = self.proj(x)
        x = torch.tanh(self.norm(x))
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        
        return x

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
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

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, register_hook=False):
        B, N, C = x.shape
        # qkv = (
        #     self.qkv(x)
        #     .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        #     .permute(2, 0, 3, 1, 4)
        # )
        # q, k, v = (
        #     qkv[0],
        #     qkv[1],
        #     qkv[2],
        # )  # make torchscript happy (cannot use tensor as tuple)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)


        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.ReLU, # nn.GELU
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.dropout = nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, register_hook=False):
        # x = x + self.dropout(self.attn(self.norm1(x), register_hook=register_hook))
        # x = x + self.attn(self.norm1(x))
        # x = x + self.dropout(self.mlp(self.norm2(x)))
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x

class VitEncoder(nn.Module):
    def __init__(self,embed_dim=128,drop_rate=0.1,num_heads=2,mlp_ratio=2,qkv_bias=True,qk_scale=None,depth=3,patch_size=(2,2)):
        super(VitEncoder, self).__init__()
        # input(batch, 1, 2, 128)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(patch_size=patch_size,embed_dim=embed_dim,norm_layer=None)
        # norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        num_patches = self.patch_embed.num_patches
        
        # self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches+1, embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=drop_rate,
                    drop_path=drop_rate
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=0.02)
            # trunc_normal_(self.cls_token, std=0.02)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, register_blk=-1):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # cls_tokens = self.cls_token.expand(
        #     B, -1, -1
        # )  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        for i, blk in enumerate(self.blocks):
            x = blk(x, register_blk == i)
        x = self.norm(x)

        return x

class CTDNN(BaseNet):
    def __init__(self, hyper = None, logger = None):
        super().__init__(hyper, logger)  
        self.num_classes = hyper.num_classes
        self.encoder_emb = hyper.encoder_emb
        self.num_heads = hyper.num_heads
        self.mlp_ratio = hyper.mlp_ratio
        self.depth = hyper.depth
        self.patch_size = hyper.patch_size
        # input(batch, 1, 2, 128)

        self.steam_iq = VitEncoder(embed_dim=self.encoder_emb,
                                   drop_rate=0.1,
                                   num_heads=self.num_heads,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=True,
                                   qk_scale=None,
                                   depth=self.depth,
                                   patch_size=self.patch_size)
        seq_len = self.steam_iq.patch_embed.num_patches
        self.mlp= nn.Sequential(
            nn.Linear(self.encoder_emb*seq_len,self.encoder_emb*2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.encoder_emb*2,self.num_classes)
        )
        
        self.initialize_weight()
        self.to(self.hyper.device)
       

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        iq_feat = self.steam_iq(x)
        # iq_feat = iq_feat[:,0,:]
        iq_feat = iq_feat.flatten(1)
 
        out = self.mlp(iq_feat)
        return out

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def _xfit(self, train_loader, val_loader):
        net_trainer = CTDNN_Trainer(self, train_loader, val_loader, self.hyper, self.logger)
        net_trainer.loop()
        fit_info = net_trainer.epochs_stats
        return fit_info
    
    def predict(self, sample):     
        sample = sample.to(self.hyper.device)
        logit = self.forward(sample)
        pre_lab = torch.argmax(logit, 1).cpu()
        return pre_lab

class CTDNN_Trainer(Trainer):
    def __init__(self, model,train_loader,val_loader, cfg, logger):
        super().__init__(model,train_loader,val_loader,cfg,logger)

    def cal_loss_acc(self, sig_batch, lab_batch):
        logit = self.model(sig_batch)
        loss = self.criterion(logit, lab_batch)
        
        pre_lab = torch.argmax(logit, 1)
        acc = torch.sum(pre_lab == lab_batch.data).double(
        ).item() / lab_batch.size(0)
        
        return loss, acc

if __name__ == '__main__':
    model = CTDNN(11)
    x = torch.rand((4, 2, 128))
    y = model(x)
    print(y.shape)