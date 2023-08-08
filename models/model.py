"""MAIN-VC model
    Modified from: https://github.com/jjery2243542/adaptive_voice_conversion
"""

import sys
import torch
import torch.nn as nn
sys.path.append("..")
from tools import *

import yaml # for test

class SpeakerEncoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, kernel_size,
                 bank_size, bank_scale, c_bank, 
                 n_conv_blocks, n_dense_blocks, 
                 subsample, act, dropout_rate):
        super(SpeakerEncoder, self).__init__()

        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = get_act_func(act)

        self.APC_module = nn.ModuleList([
            nn.Conv1d(c_in, c_bank, kernel_size=3, padding=1, dilation=1, padding_mode='reflect'),
            nn.Conv1d(c_in, c_bank, kernel_size=3, padding=2, dilation=2, padding_mode='reflect'),
            nn.Conv1d(c_in, c_bank, kernel_size=3, padding=4 , dilation=4, padding_mode='reflect'),
            nn.Conv1d(c_in, c_bank, kernel_size=3, padding=6 , dilation=6, padding_mode='reflect'),
            nn.Conv1d(c_in, c_bank, kernel_size=3, padding=8 , dilation=8, padding_mode='reflect')
        ])

        in_channels = 400
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)

        self.first_conv_layers = nn.ModuleList([
            nn.Conv1d(c_h, c_h, kernel_size=kernel_size)
                for _ in range(n_conv_blocks)
        ])

        self.second_conv_layers = nn.ModuleList([
            nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
                for sub, _ in zip(subsample, range(n_conv_blocks))
        ])

        self.pooling_layer = nn.AdaptiveAvgPool1d(1)

        self.first_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])
        self.second_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])

        self.output_layer = nn.Linear(c_h, c_out)
        self.dropout_layer = nn.Dropout(p=dropout_rate)


    def conv_blocks(self, inData):
        outData = inData
        for l in range(self.n_conv_blocks):
            y = pad_layer(outData, self.first_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                outData = F.avg_pool1d(outData, kernel_size=self.subsample[l], ceil_mode=True)
            outData = y + outData
        return outData


    def dense_blocks(self, inp):
        out = inp
        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out


    def APC(self, inp, act):
        out_list = []
        for layer in self.APC_module:
            out_list.append(act(layer(inp)))
        outData = torch.cat(out_list+[inp], dim=1)
        return outData


    def forward(self, x):
        # APC
        out = self.APC(x, act=self.act)
        # dimension reduction
        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
        # conv blocks
        out = self.conv_blocks(out)
        # avg pooling
        out = self.pooling_layer(out).squeeze(2)
        # dense blocks
        out = self.dense_blocks(out)
        out = self.output_layer(out)
        return out


class ContentEncoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, kernel_size,
            bank_size, bank_scale, c_bank, 
            n_conv_blocks, subsample, 
            act, dropout_rate):
        super(ContentEncoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.subsample = subsample
        self.act = get_act_func(act)
        self.conv_bank = nn.ModuleList([nn.Conv1d(c_in, c_bank, kernel_size=k)
                                        for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size)
                                                for _ in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
                                                 for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False) # IN
        self.mean_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.std_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)


    def forward(self, inData):
        outData = conv_bank(inData, self.conv_bank, act=self.act)
        outData = pad_layer(outData, self.in_conv_layer)
        outData = self.norm_layer(outData)
        outData = self.act(outData)
        outData = self.dropout_layer(outData)
        for l in range(self.n_conv_blocks):
            y = pad_layer(outData, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                outData = F.avg_pool1d(outData, kernel_size=self.subsample[l], ceil_mode=True)
            outData = y + outData

        mu = pad_layer(outData, self.mean_layer)
        sigma = pad_layer(outData, self.std_layer)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, 
            c_in, c_cond, c_h, c_out, 
            kernel_size,
            n_conv_blocks, upsample, act, sn, dropout_rate):
        super(Decoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.upsample = upsample
        self.act = get_act_func(act)
        f = nn.utils.spectral_norm if sn else lambda x: x
        self.in_conv_layer = f(nn.Conv1d(c_in, c_h, kernel_size=1))
        self.first_conv_layers = nn.ModuleList([f(nn.Conv1d(c_h, c_h, kernel_size=kernel_size))
                                                for _ in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([f(nn.Conv1d(c_h, c_h * up, kernel_size=kernel_size))
                                                 for _, up in zip(range(n_conv_blocks), self.upsample)])
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.conv_affine_layers = nn.ModuleList([f(nn.Linear(c_cond, c_h * 2))
                                                 for _ in range(n_conv_blocks*2)])
        self.out_conv_layer = f(nn.Conv1d(c_h, c_out, kernel_size=1))
        self.dropout_layer = nn.Dropout(p=dropout_rate)


    def forward(self, z, cond):
        out = pad_layer(z, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = adaIn(y, self.conv_affine_layers[l*2](cond))
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            if self.upsample[l] > 1:
                y = pixel_shuffle_1d(y, scale_factor=self.upsample[l])
            y = self.norm_layer(y)
            y = adaIn(y, self.conv_affine_layers[l*2+1](cond))
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.upsample[l] > 1:
                out = y + upsample(out, scale_factor=self.upsample[l]) 
            else:
                out = y + out
        out = pad_layer(out, self.out_conv_layer)
        return out


class MAINVC(nn.Module):
    def __init__(self, config):
        super(MAINVC, self).__init__()
        self.speaker_encoder = SpeakerEncoder(**config['SpeakerEncoder'])
        self.content_encoder = ContentEncoder(**config['ContentEncoder'])
        self.decoder = Decoder(**config['Decoder'])


    def forward(self, x, x_sf):
        emb = self.speaker_encoder(x_sf)
        mu, log_sigma = self.content_encoder(x)
        eps = log_sigma.new(*log_sigma.size()).normal_(0, 1)
        dec = self.decoder(mu + torch.exp(log_sigma/2) * eps, emb)
        return mu, log_sigma, emb, dec


    def inference(self, x, x_cond):
        emb = self.speaker_encoder(x_cond)
        mu, _ = self.content_encoder(x)
        dec = self.decoder(mu, emb)
        return dec


    def get_speaker_embedding(self, x):
        emb = self.speaker_encoder(x)
        return emb


#__________test__________
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

Es = SpeakerEncoder(**config['SpeakerEncoder'])
Ec = ContentEncoder(**config['ContentEncoder'])
D  = Decoder(**config['Decoder'])

x = torch.randn(2, 80, 500)
y = torch.randn(2, 80, 480)
cond = Es(x)
mu = Ec(y)[0]
dec = D(mu, cond)
print(cond.shape)
print(mu.shape)
print(dec.shape)


