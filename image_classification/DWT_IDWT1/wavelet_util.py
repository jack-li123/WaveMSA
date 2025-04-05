# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

import torch.nn as nn

from wavelet_util.DWT_IDWT1.DWT_IDWT_layer import *


class Downsample(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsample, self).__init__()
        self.dwt = DWT_2D_tiny(wavename = wavename)


    def forward(self, input):
        LL = self.dwt(input)
        return LL

class Downsample_v1(nn.Module):
    """
        for ResNet_C
        X --> torch.cat(X_ll, X_lh, X_hl, X_hh)
    """
    def __init__(self, wavename = 'haar'):
        super(Downsample_v1, self).__init__()
        self.dwt = DWT_2D(wavename = wavename)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        # return torch.cat((LL, LH, HL, HH), dim = 1)
        return LL, LH, HL, HH

class Downsample_v2(nn.Module):
    """
        for ResNet_A
        X --> 1/4*(X_ll + X_lh + X_hl + X_hh)
    """
    def __init__(self, wavename = 'haar'):
        super(Downsample_v2, self).__init__()
        self.dwt = DWT_2D(wavename = wavename)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        return (LL + LH + HL + HH) / 4

class Upsample_v1(nn.Module):
    """
      (X_ll, X_lh, X_hl, X_hh) ----> x
    """
    def __init__(self, wavename = 'haar'):
        super( Upsample_v1, self).__init__()
        self.idwt = IDWT_2D(wavename=wavename)

    def forward(self,  LL, LH, HL, HH):
        x = self.idwt(LL, LH, HL, HH)
        return x

class Downsample__v1(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsample__v1, self).__init__()
        self.dwt = DWT_1D(wavename = wavename)


    def forward(self, input):
        L,H = self.dwt(input)

        return L