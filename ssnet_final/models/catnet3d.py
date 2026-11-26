# -*- coding: utf-8 -*-
"""3-D CAT-Net integrated from the author-provided ``CATNet_3D.py``.

Only the main 3-D CAT-Net is kept; comparison and ablation variants are not
included.  Small portability edits remove optional helper-package dependencies
and expose a project-level output dictionary.
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange


def to_2tuple(value):
    return value if isinstance(value, tuple) else (value, value)


def trunc_normal_(tensor, mean=0.0, std=1.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std)


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res

class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.LeakyReLU()
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        # self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.LeakyReLU()

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        x_shape = list(x.shape)
        if (len(x_shape) == 5):
            [N, C, H, W, D] = x_shape
            new_shape = [N * H, C, W, D]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        batch = x.shape[0]


        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)
        if (len(x_shape) == 5):
            new_shape = [N, H] + list(output.shape)[1:]  # [N, H, C, W, D]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)  # [N, C, H, W, D]

        return output


class SeparableFourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, kernel_size=3):
        # bn_layer not used
        super(SeparableFourierUnit, self).__init__()
        self.groups = groups
        row_out_channels = out_channels // 2
        col_out_channels = out_channels - row_out_channels
        self.row_conv = torch.nn.Conv2d(in_channels=in_channels * 2,
                                        out_channels=row_out_channels * 2,
                                        kernel_size=(kernel_size, 1),  # kernel size is always like this, but the data will be transposed
                                        stride=1, padding=(kernel_size // 2, 0),
                                        padding_mode='reflect',
                                        groups=self.groups, bias=False)
        self.col_conv = torch.nn.Conv2d(in_channels=in_channels * 2,
                                        out_channels=col_out_channels * 2,
                                        kernel_size=(kernel_size, 1),  # kernel size is always like this, but the data will be transposed
                                        stride=1, padding=(kernel_size // 2, 0),
                                        padding_mode='reflect',
                                        groups=self.groups, bias=False)
        # self.row_bn = torch.nn.BatchNorm2d(row_out_channels * 2)
        # self.col_bn = torch.nn.BatchNorm2d(col_out_channels * 2)
        self.relu = torch.nn.LeakyReLU()

    def process_branch(self, x, conv):
        batch = x.shape[0]

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft(x, norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.relu(conv(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        output = torch.fft.irfft(ffted, s=x.shape[-1:], norm="ortho")
        return output


    def forward(self, x):
        rowwise = self.process_branch(x, self.row_conv)
        colwise = self.process_branch(x.permute(0, 1, 3, 2), self.col_conv).permute(0, 1, 3, 2)
        out = torch.cat((rowwise, colwise), dim=1)
        return out


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=False, separable_fu=False):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            # nn.BatchNorm2d(out_channels // 2),
            nn.LeakyReLU()
        )
        fu_class = SeparableFourierUnit if separable_fu else FourierUnit
        self.fu = fu_class(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = fu_class(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv3d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        """
        dropout_p: probability to be zeroed
        """
        super(ConvBlock, self).__init__()

        self.conv_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p))

        self.conv_1 = nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p))
        self.conv_2 = nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                                    nn.LeakyReLU())


        self.ffc_conv = SpectralTransform(out_channels, out_channels)

    def forward(self, x):
        x1 = self.conv_conv(x)
        x2 = self.conv_2(x1) + self.ffc_conv(x1)

        return x1+x2


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x):
        x_shape = list(x.shape)
        if (len(x_shape) == 5):
            [N, C, H, W, D] = x_shape
            new_shape = [N * H, C, W, D]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        output = self.maxpool(x)

        if (len(x_shape) == 5):
            new_shape = [N, H] + list(output.shape)[1:]  # [N, H, C, W, D]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)  # [N, C, H, W, D]

        return self.conv(output)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim, elementwise_affine=False)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, elementwise_affine=False)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        # x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.drop_path(self.mlp(x))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, self.dim_scale*dim, bias=False) if dim_scale else nn.Identity()
        # self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,C//(self.dim_scale**2))
        # x= self.norm(x)

        return x


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None
        self.concat_back_dim = nn.Linear(dim, int(dim/2))

    def forward(self, x, x1):

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        x = torch.cat([x, x1], -1)
        x = self.concat_back_dim(x)
        return x


class CATNet3D(nn.Module):
    bottleneck_channels = 256
    supports_semi_supervision = False

    def __init__(self, params):
        super(CATNet3D, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        self.input_resolution = self.params['input_resolution']
        self.depths = self.params['depth']
        self.num_heads = self.params['num_heads']
        self.window_size = self.params['window_size']
        self.qkv_bias = self.params['qkv_bias']
        self.qk_scale = self.params['qk_scale']
        self.drop_rate = self.params['drop_rate']
        self.attn_drop_rate = self.params['attn_drop_rate']
        self.drop_path_rate = self.params['drop_path_rate']
        self.mlp_ratio = self.params['mlp_ratio']
        self.dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        self.use_checkpoint = self.params['use_checkpoint']
        self.is_training = self.params['is_training']
        self.is_anchor = self.params['is_anchor']
        self.T = params['T']




        assert (len(self.ft_chns) == 5)

        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        self.up1 = BasicLayer_up(dim=self.ft_chns[4], input_resolution=tuple([int(x/16) for x in self.input_resolution]), depth=self.depths[3],
                                          num_heads=self.num_heads[3], window_size=self.window_size, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                                          drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                                          drop_path=self.dpr[6:],
                                          norm_layer=nn.LayerNorm, upsample=PatchExpand, use_checkpoint=self.use_checkpoint)
        self.up2 = BasicLayer_up(self.ft_chns[3], input_resolution=tuple([int(x/8) for x in self.input_resolution]), depth=self.depths[2],
                                          num_heads=self.num_heads[2], window_size=self.window_size, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                                          drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                                          drop_path=self.dpr[4:6],
                                          norm_layer=nn.LayerNorm, upsample=PatchExpand, use_checkpoint=self.use_checkpoint)
        self.up3 = BasicLayer_up(self.ft_chns[2], input_resolution=tuple([int(x/4) for x in self.input_resolution]), depth=self.depths[1],
                                          num_heads=self.num_heads[1], window_size=self.window_size, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                                          drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                                          drop_path=self.dpr[2:4],
                                          norm_layer=nn.LayerNorm, upsample=PatchExpand, use_checkpoint=self.use_checkpoint)
        self.up4 = BasicLayer_up(self.ft_chns[1], input_resolution=tuple([int(x/2) for x in self.input_resolution]), depth=self.depths[0],
                                          num_heads=self.num_heads[0], window_size=self.window_size, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                                          drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                                          drop_path=self.dpr[0:2],
                                          norm_layer=nn.LayerNorm, upsample=PatchExpand, use_checkpoint=self.use_checkpoint)


        self.up_branch_1 = PatchExpand(input_resolution=tuple([int(x / 8) for x in self.input_resolution]),
                                       dim=self.ft_chns[3], dim_scale=8, norm_layer=None)
        self.up_branch_2 = PatchExpand(input_resolution=tuple([int(x / 4) for x in self.input_resolution]),
                                       dim=self.ft_chns[2], dim_scale=4, norm_layer=None)
        self.up_branch_3 = PatchExpand(input_resolution=tuple([int(x / 2) for x in self.input_resolution]),
                                       dim=self.ft_chns[1], dim_scale=2, norm_layer=None)

        self.out_conv = nn.Sequential(nn.Conv3d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1),
                                      nn.Sigmoid())
        self.out_conv_1 = nn.Sequential(nn.Conv3d(self.ft_chns[0], self.n_class,
                                                kernel_size=3, padding=1),
                                      nn.Sigmoid())
        self.out_conv_2 = nn.Sequential(nn.Conv3d(self.ft_chns[0], self.n_class,
                                                kernel_size=3, padding=1),
                                      nn.Sigmoid())
        self.out_conv_3 = nn.Sequential(nn.Conv3d(self.ft_chns[0], self.n_class,
                                                kernel_size=3, padding=1),
                                      nn.Sigmoid())

        self.out_conv_a = nn.Sequential(nn.Conv3d(self.ft_chns[4], self.n_class + 1,
                                                  kernel_size=1, padding=0),
                                        )

        # The author-provided CAT-Net keeps several historical/reference
        # submodules in the class definition that are not used by its actual
        # forward path (e.g. ConvBlock.conv_1, DownBlock.maxpool_conv,
        # up_branch_1 and out_conv_1). DataParallel silently tolerates these
        # trainable-but-unused parameters, while DDP expects every trainable
        # parameter to participate in gradient reduction. Marking only those
        # inactive reference branches non-trainable preserves the effective
        # CAT-Net computation and makes the model DDP-safe without enabling
        # per-iteration unused-parameter graph traversal.
        self._freeze_inactive_reference_modules()

    @staticmethod
    def _freeze_module(module):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def _freeze_inactive_reference_modules(self):
        for module in self.modules():
            if isinstance(module, ConvBlock):
                self._freeze_module(module.conv_1)
            elif isinstance(module, DownBlock):
                self._freeze_module(module.maxpool_conv)

        self._freeze_module(self.up_branch_1)
        self._freeze_module(self.out_conv_1)

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":
            x = self.up(x)
            x = x.view(B,4*H,4*W,-1)
            x = x.permute(0,3,1,2) #B,C,H,W
            x = self.out_conv(x)
        return x

    def forward(self, x, feature_module=None):
        """Run the author's 3-D CAT-Net on a 3-D LA volume.

        The original implementation pads the depth from 80 to 112 so that the
        2-D transformer expansion operates on 112x112 planes while the encoder
        remains volumetric (Conv3d).  This version keeps that computation but
        removes the hard-coded batch size so it can live inside the shared
        project runner.
        """
        if x.dim() != 5:
            raise ValueError(f"CATNet3D expects [B, C, H, W, D], got {tuple(x.shape)}")

        batch_size, _, full_h, full_w, original_depth = x.shape
        target_depth = int(self.input_resolution[1])
        if original_depth > target_depth:
            raise ValueError(
                f"CATNet3D input depth {original_depth} exceeds configured transformer depth {target_depth}"
            )
        pad_total = target_depth - original_depth
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        if pad_total:
            x = F.pad(x, (pad_left, pad_right, 0, 0, 0, 0))

        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        anchor_out = self.out_conv_a(x4)
        anchor_out_soft = F.softmax(anchor_out / self.T, dim=1)
        uncertainty = -torch.sum(
            anchor_out_soft * torch.log(anchor_out_soft + 1e-6), dim=1, keepdim=True
        )
        x4 = uncertainty * x4

        fsl_original = None
        fsl_transformed = None
        if feature_module is not None:
            x4, fsl_original, fsl_transformed = feature_module(x4)

        def flatten_depth_plane(value):
            n, c, h, w, d = value.shape
            value = value.transpose(1, 2).reshape(n * h, c, w, d)
            value = value.permute(0, 2, 3, 1).contiguous()
            return value.view(value.size(0), -1, value.size(3))

        x4 = flatten_depth_plane(x4)
        x3 = flatten_depth_plane(x3)
        x2 = flatten_depth_plane(x2)
        x1 = flatten_depth_plane(x1)
        x0 = flatten_depth_plane(x0)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        up_branch_2 = self.up_branch_2(x)
        x = self.up3(x, x1)
        up_branch_3 = self.up_branch_3(x)
        x = self.up4(x, x0)

        padded_depth = original_depth + pad_total
        x = x.view(batch_size * full_h, full_w, padded_depth, -1).permute(0, 3, 1, 2)
        up_branch_3 = up_branch_3.view(batch_size * full_h, full_w, padded_depth, -1).permute(0, 3, 1, 2)
        up_branch_2 = up_branch_2.view(batch_size * full_h, full_w, padded_depth, -1).permute(0, 3, 1, 2)

        def restore_volume(value):
            value = value.reshape(batch_size, full_h, *value.shape[1:])
            return value.transpose(1, 2).contiguous()

        x = restore_volume(x)
        up_branch_3 = restore_volume(up_branch_3)
        up_branch_2 = restore_volume(up_branch_2)

        output_4 = self.out_conv(x)
        output_3 = self.out_conv_3(up_branch_3)
        output_2 = self.out_conv_2(up_branch_2)
        output = (output_2 + output_3 + output_4) / 3.0

        # Match the author's test-time behavior: remove the depth padding.
        if not self.training and pad_total:
            output = output[..., pad_left : pad_left + original_depth]

        return {
            "branches": [output],
            "embeddings": [],
            "anchor_logits": anchor_out,
            "fsl_original": fsl_original,
            "fsl_transformed": fsl_transformed,
            "binary_probability": True,
            "depth_padding": (pad_left, pad_right),
        }

