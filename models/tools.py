"""MAIN-VC tools
    Modified from: https://github.com/jjery2243542/adaptive_voice_conversion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn


def pad_layer(inData, layer, pad_mode="reflect"):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size // 2, kernel_size // 2 - 1)
    else:
        pad = (kernel_size // 2, kernel_size // 2)
    inData = F.pad(inData, pad=pad, mode=pad_mode)
    outData = layer(inData)
    return outData


def pad_layer_2d(inData, layer, pad_mode="reflect"):
    kernel_size = layer.kernel_size
    if kernel_size[0] % 2 == 0:
        pad_x = [kernel_size[0] // 2, kernel_size[0] // 2 - 1]
    else:
        pad_x = [kernel_size[0] // 2, kernel_size[0] // 2]
    if kernel_size[1] % 2 == 0:
        pad_y = [kernel_size[1] // 2, kernel_size[1] // 2 - 1]
    else:
        pad_y = [kernel_size[1] // 2, kernel_size[1] // 2]
    pad = tuple(pad_x + pad_y)
    inData = F.pad(inData, pad=pad, mode=pad_mode)
    outData = layer(inData)
    return outData


def pixel_shuffle_1d(inData, scale_factor=2):
    batch_size, channels, in_width = inData.size()
    channels //= scale_factor
    out_width = in_width * scale_factor
    in_view = inData.contiguous().view(batch_size, channels, scale_factor, in_width)
    shuffle_out = in_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out


def upsample(x, scale_factor=2):
    return F.interpolate(x, scale_factor=scale_factor, mode="nearest")


def flatten(x):
    return x.contiguous().view(x.size(0), -1)


def adaIn(z_c, z_s):
    """AdaIN
    z_c: content embedding
    z_s: speaker embedding
    """
    p = z_s.size(1) // 2
    mu, sigma = z_s[:, :p], z_s[:, p:]
    outData = z_c * sigma.unsqueeze(dim=2) + mu.unsqueeze(dim=2)
    return outData


def get_act_func(func_name):
    if func_name == "lrelu":
        return nn.LeakyReLU()
    return nn.ReLU()


def cc(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return net.to(device)


def infinite_iter(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)
