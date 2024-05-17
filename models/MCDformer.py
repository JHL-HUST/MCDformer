import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.base_model import BaseModel
import math
import copy
import os


from models._baseNet import BaseNet
from models._baseTrainer import Trainer
from tqdm import tqdm





class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Conv_Block, self).__init__()
        self.in_c = in_channel
        self.out_c = out_channel
        kernel_size = kernel_size if kernel_size is not None else (1,3)

        self.conv_block = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(self.in_c, self.out_c, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c)
        )

    def forward(self, x):
        """
        x: [batchsize, C, H, W]
        """
        x = self.conv_block(x)
        # print(x.shape)

        return x
    
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x = self.downConv(x.permute(0, 2, 1))
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        # x = x.transpose(1,2)
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
        self.scale = qk_scale if qk_scale is not None else head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

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
        # print(q.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # if register_hook:
        #     self.save_attention_map(attn)
        #     attn.register_hook(self.save_attn_gradients)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        # x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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
        act_layer=nn.GELU,
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
        # d_ff = dim * 4
        # self.conv1 = nn.Conv1d(in_channels=dim, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=dim, kernel_size=1)
        self.dropout = nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        self.activation = act_layer()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, register_hook=False):
        x = x + self.attn(self.norm1(x), register_hook=register_hook)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        # x = self.norm2(x)
        return x

class TinyMLP(nn.Module):
    def __init__(self, N):
        super(TinyMLP, self).__init__()
        self.N = N

        self.mlp = nn.Sequential(
            nn.Linear(self.N, self.N // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.N // 4, self.N),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class FrequencyDomainDenoisingModule(nn.Module):
    def __init__(self, N):
        super(FrequencyDomainDenoisingModule, self).__init__()
        self.mlp = TinyMLP(N)

    def forward(self, x):
        # x:[N, C_out, 1, W]
        x_init = copy.deepcopy(x)
        
        # FFT(x) + FFT(j*y) = FFT(r)
        # x + j*y = r 
        r = x[:,:,0,:] + 1j*x[:,:,1,:]
        r = torch.fft.fft(r,dim=-1)
        x_fft = torch.zeros_like(x)
        x_fft[:,:,0,:] = torch.real(r)
        x_fft[:,:,1,:] = torch.imag(r)

        h = self.mlp(x_fft)
        x_h = torch.mul(h[:,:,0,:], x_fft[:,:,0,:]) + 1j * torch.mul(h[:,:,1,:], x_fft[:,:,1,:])
        x_h = torch.fft.ifft(x_h, dim=-1)
        r_ = torch.zeros_like(x)
        r_[:,:,0,:] = torch.real(x_h)
        r_[:,:,1,:] = torch.imag(x_h)

        x = r_ + x_init
        
        return x

class MCDformer(BaseNet):
    def __init__(self, hyper = None, logger = None):
        super().__init__(hyper, logger)  
        self.sig_len = hyper.sig_len
        self.extend_channel = hyper.extend_channel
        self.latent_dim = hyper.latent_dim
        self.num_classes = hyper.num_classes
        self.num_heads = hyper.num_heads
        self.conv_chan_list = hyper.conv_chan_list


        if self.conv_chan_list is None:
            self.conv_chan_list = [1,36, 64, 128, 256]
        self.stem_layers_num = len(self.conv_chan_list) - 1
        self.kernel_size_list = [(1,3),(2,3),(1,3),(1,3)]

        self.FDDM = FrequencyDomainDenoisingModule(self.sig_len)
        self.block = Block(
                    dim=self.sig_len, # 128
                    num_heads=self.num_heads,
                    mlp_ratio=1,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.5,
                    attn_drop=0.5,
                    drop_path=0.5
                )
        self.block2 = Block(
                    dim=int(self.sig_len/2), # 64
                    num_heads=self.num_heads,
                    mlp_ratio=1,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.5,
                    attn_drop=0.5,
                    drop_path=0.5
                )
        self.block_conv = ConvLayer(256)
        self.Conv_stem = nn.Sequential()

        for t in range(0, self.stem_layers_num):
            self.Conv_stem.add_module(
                f'conv_stem_{t}',Conv_Block(
                        self.conv_chan_list[t],
                        self.conv_chan_list[t + 1],
                        self.kernel_size_list[t]
                        )
                )

        # self.GAP = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(self.latent_dim, self.num_classes)
        )

        self.initialize_weight()
        self.to(self.hyper.device)

    def forward(self, x):
        # x = x / x.norm(p=2, dim=-1, keepdim=True)
        x = x.unsqueeze(1)
        x = self.FDDM(x)
        x = self.Conv_stem(x)
        x = x.squeeze(2)
        x = self.block(x.squeeze(2))
        x = self.block_conv(x)
        x = self.block2(x)
        # x = self.GAP(x)
        x = x[:,:,-1:]
        
        out = self.classifier(x.squeeze(2))
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
                nn.init.constant_(m.bias, 0)
    def _xfit(self, train_loader, val_loader):
        net_trainer = MCDformer_Trainer(self, train_loader, val_loader, self.hyper, self.logger)
        net_trainer.loop()
        fit_info = net_trainer.epochs_stats
        return fit_info
    
    def predict(self, sample):     
        sample = sample.to(self.hyper.device)
        logit = self.forward(sample)
        pre_lab = torch.argmax(logit, 1).cpu()
        return pre_lab

class MCDformer_Trainer(Trainer):
    def __init__(self, model,train_loader,val_loader, cfg, logger):
        super().__init__(model,train_loader,val_loader,                 cfg,logger)

    def cal_loss_acc(self, sig_batch, lab_batch):
        logit = self.model(sig_batch)
        loss = self.criterion(logit, lab_batch)
        
        pre_lab = torch.argmax(logit, 1)
        acc = torch.sum(pre_lab == lab_batch.data).double(
        ).item() / lab_batch.size(0)
        
        return loss, acc


if __name__ == '__main__':
    model = MCDformer(11, 128, 36,256)
    x = torch.rand((4,2, 128))
    y = model(x)
    print(y.shape)