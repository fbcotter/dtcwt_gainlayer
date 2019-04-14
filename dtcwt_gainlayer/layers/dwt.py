import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as func
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse


class WaveGainLayer(nn.Module):
    """ Create gains and apply them to each orientation independently

    Inputs:
        C: Number of input channels
        F: number of output channels
        lp_size: Spatial size of lowpass filter
        bp_sizes: Spatial size of bandpass filters

    Can specify None for any of the sizes, in which case, the convolution is not
    done, and zeros are passed through. This may be particularly useful say if
    you wanted to only apply gains to the second scale and not to the first.
    """
    def __init__(self, C, F, lp_size=1, bp_sizes=(1,)):
        super().__init__()
        self.C = C
        self.F = F
        self.lp_size = lp_size
        self.bp_sizes = bp_sizes
        self.J = len(bp_sizes)

        # Create the lowpass gain
        if lp_size is None or lp_size == 0:
            self.lp_pad = None
            self.g_lp = nn.Parameter(torch.tensor([]), requires_grad=False)
        else:
            self.lp_pad = (lp_size - 1) // 2
            self.g_lp = nn.Parameter(torch.randn(F, C, lp_size, lp_size))

        # Create the banpass gains
        self.bp_pad = []
        bps = []
        for s in bp_sizes:
            if s is None or s == 0:
                self.bp_pad.append(None)
                bps.append(nn.Parameter(torch.tensor([]), requires_grad=False))
            else:
                self.bp_pad.append((s - 1) // 2)
                bps.append(nn.Parameter(torch.randn(3, F, C, s, s)))
        self.g = nn.ParameterList(bps)
        self.init()

    def forward(self, coeffs):
        # Pull out the lowpass and the bandpass coefficients
        u_lp, u = coeffs
        assert len(u) == len(self.g), "Number of bandpasses must " + \
            "match number of filters"

        if self.g_lp is None or self.g_lp.shape == torch.Size([0]):
            v_lp = torch.zeros_like(u_lp)
        else:
            v_lp = func.conv2d(u_lp, self.g_lp, padding=self.lp_pad)

        v = []
        for i in range(self.J):
            g_i = self.g[i]
            u_i = u[i]
            pad = self.bp_pad[i]
            if g_i is None or g_i.shape == torch.Size([0]):
                v.append(torch.zeros_like(u_i))
            else:
                v_i1 = func.conv2d(u_i[:,:,0], g_i[0], padding=pad)
                v_i2 = func.conv2d(u_i[:,:,1], g_i[1], padding=pad)
                v_i3 = func.conv2d(u_i[:,:,2], g_i[2], padding=pad)
                # Stack up the bands along the third dimension to match the
                # input style
                v.append(torch.stack((v_i1, v_i2, v_i3), dim=2))

        return v_lp, v

    def init(self, gain=1, method='xavier_uniform'):
        lp_scales = np.array([1, 2, 4, 8]) * gain
        bp_scales = np.array([2, 4, 8, 16]) * gain
        # Choose the initialization scheme
        if method == 'xavier_uniform':
            fn = init.xavier_uniform_
        else:
            fn = init.xavier_normal_

        if not (self.g_lp is None or self.g_lp.shape == torch.Size([0])):
            fn(self.g_lp, gain=lp_scales[self.J-1])

        for j in range(self.J):
            g_i = self.g[j]
            if not (g_i is None or g_i.shape == torch.Size([0])):
                fn(g_i, gain=bp_scales[j])

    def extra_repr(self):
        return '(g_lp): Parameter of type {} with size: {}'.format(
            self.g_lp.type(), 'x'.join([str(x) for x in self.g_lp.shape]))


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
        q: the proportion of actiavtions to keep
        mode: the padding mode for convolution
    """
    def __init__(self, C, F, lp_size=3, bp_sizes=(1,), q=1.0, wave='db2',
                 mode='zero'):
        super().__init__()
        self.C = C
        self.F = F
        self.J = len(bp_sizes)

        self.XFM = DWTForward(J=self.J, mode=mode, wave=wave)
        if q < 0:
            self.shrink = ReLUWaveCoeffs()
        else:
            self.shrink = lambda x: x
        self.GainLayer = WaveGainLayer(C, F, lp_size, bp_sizes)
        self.IFM = DWTInverse(mode=mode, wave=wave)

    def forward(self, X):
        yl, yh = self.XFM(X)
        yl, yh = self.GainLayer((yl, yh))
        yl, yh = self.shrink((yl, yh))
        return self.IFM((yl, yh))

    def init(self, gain=1, method='xavier_uniform'):
        self.GainLayer.init(gain, method)


class ReLUWaveCoeffs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """ Bandpass input comes in as a tensor of shape (N, C, 3, H, W).
        Need to do the relu independently on real and imaginary parts """
        yl, yh = x
        yl = func.relu(yl)
        yh = [func.relu(b) for b in yh]
        return yl, yh
