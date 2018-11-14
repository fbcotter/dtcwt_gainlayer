import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from pytorch_wavelets import DTCWTForward, DTCWTInverse


def init_dtcwt_coeffs(C, F, lp_size, bp_sizes):
    """ Initializes the lowpass and bandpass filters for a dtcwt
    wavelet pyramid"""
    if not (isinstance(F, tuple) or isinstance(F, list)):
        F = [F,] * (len(bp_sizes) + 1)

    bp = [None,]*len(bp_sizes)
    for j, (s, f) in enumerate(zip(bp_sizes, F[:-1])):
        if s != 0:
            # std = 1/s
            std = 2/np.sqrt(s*s*C)
            bp[j] = torch.tensor(std) * torch.randn(f, C, 6, s, s, 2,
                                                    requires_grad=True)

    s = lp_size
    if s == 0:
        lp = None
    else:
        # std = 1/s
        std = 2/np.sqrt(s*s*C)
        lp = torch.tensor(std) * torch.randn(F[-1], C, s, s, requires_grad=True)

    return lp, bp


class WaveGainLayer(nn.Module):
    """ Create gains and apply them to each orientation independently

    Inputs:
        C: Number of input channels
        F: number of output channels
        lp_size: Spatial size of lowpass filter
        bp_sizes: Spatial size of bandpass filters
    """
    def __init__(self, C, F, lp_size=1, bp_sizes=(3,1), lp_stride=1,
                 bp_strides=(1,1)):
        super().__init__()
        self.C = C
        self.F = F
        self.lp_size = lp_size
        self.bp_sizes = bp_sizes
        self.J = len(bp_sizes)
        self.lp, self.bp = init_dtcwt_coeffs(
            C, F, lp_size, bp_sizes)
        self.lp = nn.Parameter(self.lp)
        self.bp = nn.ParameterList([nn.Parameter(bp) for bp in self.bp])
        self.lp_scales = [1, 2, 4, 8]
        self.bp_scales = [2, 4, 8, 16]
        #  self.bp_scales = [1, 4, 8, 16]
        self.lp_stride = lp_stride
        self.bp_strides = bp_strides

        # Calculate the pad widths
        if lp_size is not None:
            self.lp_pad = lp_size // 2
        else:
            self.lp_pad = None
        self.bp_pad = []
        for s in bp_sizes:
            if s is not None:
                self.bp_pad.append(s // 2)
            else:
                self.bp_pad.append(None)

    def forward(self, x):
        yl, yh = x
        assert len(yh) == len(self.bp), "Number of bandpasses must " + \
            "match number of filters"

        if self.lp is not None:
            s = self.lp_stride
            yl2 = self.lp_scales[self.J-1] * func.conv2d(
                yl, self.lp, padding=self.lp_pad, stride=s)
        else:
            yl2 = torch.zeros_like(yl)

        yh2 = []
        for bp, level, pad, scale, s in zip(
                self.bp, yh, self.bp_pad, self.bp_scales[:self.J],
                self.bp_strides):
            if bp is not None and not bp.shape == torch.Size([0]):
                angles = []
                for l in range(6):
                    in_r, in_i = level[:,:,l,:,:,0], level[:,:,l,:,:,1]
                    w_r, w_i = bp[:,:,l,:,:,0], bp[:,:,l,:,:,1]

                    # real output = r*r - i*i
                    out_r = func.conv2d(in_r, w_r, padding=pad, stride=s) - \
                        func.conv2d(in_i, w_i, padding=pad, stride=s)
                    # imag output = r*i + i*r
                    out_i = func.conv2d(in_r, w_i, padding=pad, stride=s) + \
                        func.conv2d(in_i, w_r, padding=pad, stride=s)

                    angles.append(torch.stack((out_r, out_i), dim=-1))
                yh2.append(scale * torch.stack(angles, dim=2))
            else:
                yh2.append(torch.zeros_like(level))

        return yl2, yh2


class WaveConvLayer(nn.Module):
    """ Decomposes a signal into a DTCWT pyramid and learns weights for each
    scale.

    This type takes a forward DTCWT of a signal and then learns convolutional
    kernels in each layer, before taking an Inverse DTCWT. The number of scales
    taken in the pyramid decomposition will be the length of the input bp_sizes.
    You can not learn parameters for a layer by setting the kernel size to 0.

    Inputs:
        C: Number of input channels
        F: number of output channels
        lp_size: Spatial size of lowpass filter
        bp_sizes: Spatial size of bandpass filters
    """
    def __init__(self, C, F, lp_size, bp_sizes, stride=1, biort='near_sym_a',
                 qshift='qshift_a'):
        super().__init__()
        self.C = C
        self.F = F
        self.lp_size = lp_size
        self.bp_sizes = bp_sizes
        skip_hps = False
        if self.bp_sizes[0] == 0:
            skip_hps = True
        self.J = len(bp_sizes)
        self.XFM = DTCWTForward(
            biort=biort, qshift=qshift, J=self.J, skip_hps=skip_hps)
        self.IFM = DTCWTInverse(
            biort=biort, qshift=qshift, J=self.J)
        self.GainLayer = WaveGainLayer(C, F, lp_size, bp_sizes)
        self.stride = stride

    def forward(self, X):
        yl, yh = self.XFM(X)
        yl2, yh2 = self.GainLayer((yl, yh))
        return self.IFM((yl2, yh2))[...,::self.stride, ::self.stride]
