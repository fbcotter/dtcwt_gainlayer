import torch
import torch.nn as nn
import torch.nn.functional as func
import math
import numpy as np
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from dtcwt_gainlayer.layers.nonlinear import WaveNonLinearity


class WaveGainLayer(nn.Module):
    """ Create gains and apply them to each orientation independently

    Inputs:
        C: Number of input channels
        F: number of output channels
        lp_size: Spatial size of lowpass filter
        bp_sizes: Spatial size of bandpass filters
        wd: l2 weight decay for lowpass gain
        wd1: l1 weight decay for bandpass gain. If None, uses wd and l2.


    The forward pass should be provided with a tuple of wavelet coefficients.
    The tuple should have length 2 and be the lowpass and bandpass coefficients.
    The lowpass coefficients should have shape (N, C, H, W).
    The bandpass coefficients should also be a tuple/list and have length J.
    Each entry in the bandpass list should have 6 dimensions, and should be of
    shape (N, C, 6, H', W', 2).

    Returns a tuple of tensors of the same form as the input, but with F output
    channels.
    """
    def __init__(self, C, F, lp_size=3, bp_sizes=(1,), lp_stride=1,
                 bp_strides=(1,), wd=0, wd1=None):
        super().__init__()
        self.wd = wd
        self.wd1 = wd1
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

        # Create the bandpass gains
        self.bp_pad = []
        bps = []
        for s in bp_sizes:
            if s is None or s == 0:
                self.bp_pad.append(None)
                bps.append(nn.Parameter(torch.tensor([]), requires_grad=False))
            else:
                self.bp_pad.append((s - 1) // 2)
                bps.append(nn.Parameter(torch.randn(6, 2, F, C, s, s)))
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
        for j in range(self.J):
            g_j = self.g[j]
            u_j = u[j]
            pad = self.bp_pad[j]
            if g_j is None or g_j.shape == torch.Size([0]):
                v.append(torch.zeros_like(u_j))
            else:
                # Do the mixing for each orientation independently
                bands = []
                for l in range(6):
                    u_jl_real, u_jl_imag = u_j[:,:,l,:,:,0], u_j[:,:,l,:,:,1]
                    g_jl_real, g_jl_imag = g_j[l, 0], g_j[l, 1]
                    # real output = r*r - i*i
                    v_jl_real = (func.conv2d(u_jl_real, g_jl_real, padding=pad)
                               - func.conv2d(u_jl_imag, g_jl_imag, padding=pad)) # noqa
                    # imag output = r*i + i*r
                    v_jl_imag = (func.conv2d(u_jl_real, g_jl_imag, padding=pad)
                               + func.conv2d(u_jl_imag, g_jl_real, padding=pad)) # noqa
                    bands.append(torch.stack((v_jl_real, v_jl_imag), dim=-1))
                # Stack up the 6 bands along the third dimension again
                v.append(torch.stack(bands, dim=2))

        return v_lp, v

    def init(self, gain=1, method='xavier_uniform'):
        lp_scales = np.array([1, 2, 4, 8]) * gain
        #  bp_scales = np.array([1, 2, 4, 8]) * gain
        bp_scales = np.array([2, 4, 8, 16]) * gain

        # Calculate the fan in and fan out manually - the gain are in odd shapes
        # so won't work with the default functions
        if not (self.g_lp is None or self.g_lp.shape == torch.Size([0])):
            s = self.g_lp.shape
            fan_in, fan_out = s[1]*s[2]*s[3], s[0]*s[2]*s[3]
            std = lp_scales[self.J-1] * math.sqrt(2.0 / (fan_in + fan_out))
            a = math.sqrt(3.0) * std
            with torch.no_grad():
                self.g_lp.uniform_(-a, a)

        for j in range(self.J):
            g_j = self.g[j]
            if not (g_j is None or g_j.shape == torch.Size([0])):
                s = g_j.shape
                fan_in, fan_out = s[3]*s[4]*s[5], s[2]*s[4]*s[5]
                std = bp_scales[j] * math.sqrt(2.0 / (fan_in + fan_out))
                a = math.sqrt(3.0) * std
                with torch.no_grad():
                    g_j.uniform_(-a, a)

    def get_reg(self):
        a = self.wd*0.5*torch.sum(self.g_lp**2)
        if self.wd1 is not None:
            for g in self.g:
                a += self.wd1 * torch.sum(torch.sqrt(g[:,0]**2 + g[:,1]**2))
        else:
            for g in self.g:
                a += 0.5 * self.wd * torch.sum(g**2)
        return a

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
        xfm: true indicating should take dtcwt of input, false to skip
        ifm: true indicating should take inverse dtcwt of output, false to skip
        wd: l2 weight decay for lowpass gain
        wd1: l1 weight decay for bandpass gain. If None, uses wd and l2.
        lp_nl: nonlinearity to use for the lowpass. See
          :py:class:`dtcwt_gainlayer.layers.nonlinear.WaveNonLinearity`.
        bp_nl: nonlinearities to use for the bandpasses. See
          :py:class:`dtcwt_gainlayer.layers.nonlinear.WaveNonLinearity`.
        lp_nl_kwargs: keyword args for the lowpass nonlinearity
        bp_nl_kwargs: keyword args for the lowpass nonlinearity
    """
    def __init__(self, C, F, lp_size=3, bp_sizes=(1,), biort='near_sym_a',
                 qshift='qshift_a', xfm=True, ifm=True, wd=0, wd1=None,
                 lp_nl=None, bp_nl=None, lp_nl_kwargs={}, bp_nl_kwargs={}):
        super().__init__()
        self.C = C
        self.F = F
        # If any of the mixing for a scale is 0, don't calculate the dtcwt at
        # that scale
        skip_hps = [True if s == 0 else False for s in bp_sizes]
        self.J = len(bp_sizes)
        self.wd = wd
        self.wd1 = wd1

        # The forward transform
        if xfm:
            self.XFM = DTCWTForward(
                biort=biort, qshift=qshift, J=self.J, skip_hps=skip_hps,
                o_dim=2, ri_dim=-1)
        else:
            self.XFM = lambda x: x

        # The mixing
        self.GainLayer = WaveGainLayer(C, F, lp_size, bp_sizes, wd=wd, wd1=wd1)

        # The nonlinearity
        if not isinstance(bp_nl, (list, tuple)):
            bp_nl = [bp_nl,] * self.J
        self.NL = WaveNonLinearity(F, lp_nl, bp_nl, lp_nl_kwargs, bp_nl_kwargs)

        # The inverse
        if ifm:
            self.IFM = DTCWTInverse(biort=biort, qshift=qshift, o_dim=2,
                                    ri_dim=-1)
        else:
            self.IFM = lambda x: x

    def forward(self, x):
        lp, bp = self.XFM(x)
        lp, bp = self.GainLayer((lp, bp))
        lp, bp = self.NL((lp, bp))
        y = self.IFM((lp, bp))
        return y

    def init(self, gain=1, method='xavier_uniform'):
        self.GainLayer.init(gain, method)



class WaveParamLayer(nn.Module):
    """ Parameterizes gains in the DTCWT domain

    Inputs:
        C: Number of input channels
        F: number of output channels
        k: a power of 2
        J: an integer
    """
    def __init__(self, C, F, k=4, stride=1, J=1, wd=0, wd1=None, right=True):
        super().__init__()
        self.wd = wd
        if wd1 is None:
            self.wd1 = wd
        else:
            self.wd1 = wd1
        self.C = C
        self.F = F
        x = torch.zeros(F, C, k, k)
        torch.nn.init.xavier_uniform_(x)
        xfm = DTCWTForward(J=J)
        self.ifm = DTCWTInverse()
        yl, yh = xfm(x)
        self.J = J
        if k == 4 and J == 1:
            self.downsample = True
            yl = func.avg_pool2d(yl, 2)
            self.gl = nn.Parameter(torch.zeros_like(yl))
            self.gh = nn.Parameter(torch.zeros_like(yh[0]))
            self.gl.data = yl.data
            self.gh.data = yh[0].data
            if right:
                self.pad = (1, 2, 1, 2)
            else:
                self.pad = (2, 1, 2, 1)
        elif k == 4 and J == 2:
            self.downsample = False
            self.gl = nn.Parameter(torch.zeros_like(yl))
            self.gh = nn.Parameter(torch.zeros_like(yh[0]))
            self.gl.data = yl.data
            self.gh.data = yh[1].data
            if right:
                self.pad = (1, 2, 1, 2)
            else:
                self.pad = (2, 1, 2, 1)
        elif k == 8 and J == 1:
            self.downsample = True
            yl = func.avg_pool2d(yl, 2)
            self.gl = nn.Parameter(torch.zeros_like(yl))
            self.gh = nn.Parameter(torch.zeros_like(yh[0]))
            self.gl.data = yl.data
            self.gh.data = yh[0].data
            if right:
                self.pad = (3, 4, 3, 4)
            else:
                self.pad = (4, 3, 4, 3)
        elif k == 8 and J == 2:
            self.downsample = False
            self.gl = nn.Parameter(torch.zeros_like(yl))
            #  self.gh = nn.Parameter(torch.zeros_like(yh[0]))
            self.gh = nn.Parameter(torch.zeros_like(yh[1]))
            self.gl.data = yl.data
            self.gh.data = yh[1].data
            #  self.gh1.data = yh[1].data
            if right:
                self.pad = (3, 4, 3, 4)
            else:
                self.pad = (4, 3, 4, 3)
        elif k == 8 and J == 3:
            self.downsample = False
            self.gl = nn.Parameter(torch.zeros_like(yl))
            #  self.gh = nn.Parameter(torch.zeros_like(yh[0]))
            self.gh = nn.Parameter(torch.zeros_like(yh[2]))
            self.gl.data = yl.data
            self.gh.data = yh[2].data
            #  self.gh1.data = yh[1].data
            if right:
                self.pad = (3, 4, 3, 4)
            else:
                self.pad = (4, 3, 4, 3)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.downsample:
            gl = func.interpolate(self.gl, scale_factor=2, mode='bilinear',
                                  align_corners=False)
        else:
            gl = self.gl

        if self.J == 1:
            h = self.ifm((gl, (self.gh,)))
        elif self.J == 2:
            h = self.ifm((gl, (None, self.gh)))
        elif self.J == 3:
            h = self.ifm((gl, (None, None, self.gh)))
        x = torch.nn.functional.pad(x, self.pad)
        y = func.conv2d(x, h)
        return y

    def get_reg(self):
        a = self.wd*0.5*torch.sum(self.gl**2)
        a += self.wd1*torch.sum(torch.abs(self.gh))
        #  if hasattr(self, 'gh1'):
            #  a += self.wd1*torch.sum(torch.abs(self.gh1))
        return a

    def extra_repr(self):
        return '(gl): Parameter of type {} with size: {}\n' \
               '(gh): Parameter of type {} with size: {}'.format(
                   self.gl.type(), 'x'.join([str(x) for x in self.gl.shape]),
                   self.gh.type(), 'x'.join([str(x) for x in self.gh.shape]))
