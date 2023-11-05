import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as trans_fct
from torchvision import models
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
import copy


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((1, 2, 2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, decode_channels, encode_channels, attn_scale_factor, n_head, bilinear):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor = (1, 2, 2), mode = 'bilinear', align_corners = True)
            self.up_attn = nn.Upsample(scale_factor = attn_scale_factor, mode = 'bilinear', align_corners = True)
            self.conv = DoubleConv(decode_channels, encode_channels, decode_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(decode_channels, decode_channels, kernel_size=(1,2,2), stride= (1,2,2))
            self.up_attn = nn.Upsample(scale_factor = attn_scale_factor, mode = 'bilinear', align_corners = True)
            self.conv = DoubleConv(decode_channels+encode_channels, encode_channels)
        
        self.n_head = n_head
        
        n_neurons=[n_head * encode_channels, encode_channels]
        self.n_neurons = copy.deepcopy(n_neurons)
        self.mlp1 = nn.Linear(self.n_neurons[0], self.n_neurons[1])
        self.BatchNorm3d1 = nn.BatchNorm3d(self.n_neurons[1])
        self.ReLU = nn.ReLU()
        
    def forward(self, x1, x2, attn_weight):
        # x1 decode embed: [bs, d_in, T, h, w]
        # x2 encode embed: [bs, d_in, T, H, W]
        
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
    
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        bs, d_in, T, H, W = x2.shape
        x2 = x2.repeat(self.n_head, 1, 1, 1, 1)
        x2 = x2.permute(0, 3, 4, 2, 1).contiguous()
        x2_reshaped = x2.view(-1, T, d_in)                                              # x2_reshaped [n_head*bs*H*W, T, d_in]
        
        _, bs, h, w, T, _ = attn_weight.shape
        attn_weight = attn_weight.permute(0, 1, 4, 5, 2, 3).contiguous()        
        attn_weight = attn_weight.view(self.n_head*bs, T*T, h, w)               
        attn_weight = self.up_attn(attn_weight)
        attn_weight = attn_weight.permute(0, 2, 3, 1).contiguous().view(-1, T, T)  
        
        x2_attn = torch.matmul(attn_weight, x2_reshaped).view(self.n_head, bs, H, W, T, d_in).contiguous()

        # concatenate heads
        x2_attn = x2_attn.permute(1, 4, 2, 3, 5, 0).contiguous().view(bs, T, H, W, -1)  # attn_x2 [bs, T, h, w, d_in*n_head]
        x2_attn = self.mlp1(x2_attn).permute(0, 4, 1, 2, 3)                             # put C to 1st dim (mlp: bs, T, H, W, C)
        x2_attn = self.BatchNorm3d1(x2_attn).permute(0, 2, 3, 4, 1)                     # put C to 3rd dim (bnorm: bs, C, T, H, W)
        x2_attn = self.ReLU(x2_attn)

        x2_attn = x2_attn.permute(0, 4, 1, 2, 3)
        
        x = torch.cat([x2_attn, x1], dim=1)
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Attn_Unet(nn.Module):
    """
    Args:
        n_channels (int): Number of channels of the input
        n_head (int): Number of attention heads
        d_k (int): Dimension of the key and query vectors
    """
    def __init__(self, n_channels=3, bilinear=False, n_head=4, d_k=32):
        super(Attn_Unet, self).__init__()
        
        # n_channels => C
        self.n_channels = n_channels
        # "bottleneck" dimension
        self.attn_dim = 256
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.up1 = Up(256, 256, 2, n_head, bilinear)
        self.up2 = Up(256, 128, 4, n_head, bilinear)
        self.up3 = Up(128, 64, 8, n_head, bilinear)
        self.outc = OutConv(64, n_channels)
                
        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.attn_dim)
        
        self.outlayernorm = nn.LayerNorm(self.attn_dim)
        
        n_neurons=[n_head * self.attn_dim, 1024, self.attn_dim]
        self.n_neurons = copy.deepcopy(n_neurons)
        self.mlp1 = nn.Linear(self.n_neurons[0], self.n_neurons[1])
        self.BatchNorm3d1 = nn.BatchNorm3d(self.n_neurons[1])
        self.mlp2 = nn.Linear(self.n_neurons[1], self.n_neurons[2])
        self.BatchNorm3d2 = nn.BatchNorm3d(self.n_neurons[2])
        self.ReLU = nn.ReLU(inplace=True)
        # self.Sigmoid = nn.Sigmoid()
        
        dropout = 0.2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        x1 = self.inc(x)                                                    
        x2 = self.down1(x1)                                                 
        x3 = self.down2(x2)                                                 
        x4 = self.down3(x3)                                             
        bs, c, T, x4_h, x4_w = x4.shape
        
        attn_x4, attn_weight = self.attention_heads(x4, x4, x4)        
        
        # concatenate heads
        attn_x4 = attn_x4.permute(1, 4, 2, 3, 5, 0).contiguous().view(bs, T, x4_h, x4_w, -1) 
        attn_x4 = self.mlp1(attn_x4).permute(0, 4, 1, 2, 3)                     # put C to 1st dim (mlp: bs, T, H, W, C)
        attn_x4 = self.BatchNorm3d1(attn_x4).permute(0, 2, 3, 4, 1)             # put C to 3rd dim (bnorm: bs, C, T, H, W)
        attn_x4 = self.ReLU(attn_x4)
        attn_x4 = self.mlp2(attn_x4).permute(0, 4, 1, 2, 3)             
        attn_x4 = self.BatchNorm3d2(attn_x4).permute(0, 2, 3, 4, 1)    
        attn_x4 = self.ReLU(attn_x4)
        
        attn_x4 = self.dropout(attn_x4)
        attn_x4 = self.outlayernorm(attn_x4).permute(0, 4, 1, 2, 3)    
        
        # skip connection: concatenate n_heads * encoder embeds to the output
        # too large channel dimension.
        x = self.up1(attn_x4, x3, attn_weight)                         
        x = self.up2(x, x2, attn_weight)                               
        x = self.up3(x, x1, attn_weight)                              
        logits = self.outc(x)                                                   # logits [bs, 3, 10, 128, 128]

        
        return x1, logits
    
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v):
        
        q = q.permute(0, 3, 4, 2, 1)                                                # init qkv(x5) [bs, d_in, T, 16, 16]
        k = k.permute(0, 3, 4, 2, 1)                                                # permute qkv(x5) [bs, 16, 16, T, d_in]
        v = v.permute(0, 3, 4, 2, 1)
        
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
                
        bs, h, w, T, _ = q.size()
 
        q = self.fc1_q(q)
        q = q.view(bs, h, w, T, n_head, d_k).contiguous()                           # q [bs, 16, 16, T, n_head, d_k]
        q = q.permute(4, 0, 1, 2, 3, 5).contiguous().view(-1, h, w, T, d_k)         # q [n_head*bs, 16, 16, T, d_k]

        k = self.fc1_k(k).view(bs, h, w, T, n_head, d_k).contiguous()
        k = k.permute(4, 0, 1, 2, 3, 5).contiguous().view(-1, h, w, T, d_k)         # k [n_head*bs, 16, 16, T, d_k]
        
        # repeats the tensor v along its first dimension (batch size dimension) n_head times, 
        # while keeping the other dimensions unchanged.
        v = v.repeat(n_head, 1, 1, 1, 1)                                            # v [n_head*bs, 16, 16, T, d_in]

        output, attn = self.attention(q, k, v)                                      # output [n_head*bs*16*16, T, d_in]
        output = output.view(n_head, bs, h, w, T, d_in)                             # output [n_head, bs, 16, 16, T, d_in]
        attn = attn.view(n_head, bs, h, w, T, T)                                    # attn [n_head, bs, 16, 16, T, T]

        return output, attn



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        q_reshaped = q.view(-1, q.size(3), q.size(-1))                              # q_reshaped [n_head*bs*16*16, T, d_k]
        k_reshaped = k.view(-1, k.size(3), k.size(-1)).transpose(1, 2)              # k_reshaped [n_head*bs*16*16, d_k, T]
        v_reshaped = v.view(-1, v.size(3), v.size(-1))                              # v_reshaped [n_head*bs*16*16, T, d_in]

        attn = torch.matmul(q_reshaped, k_reshaped)                                 # attn [n_head*bs*16*16, T, T]
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v_reshaped)                                     # output [n_head*bs*16*16, T, d_in]
        
        return output, attn