import torch
import torch.nn as nn
import torch.nn.functional as func
import math
import numpy as np
from pytorch_wavelets import DTCWTForward, DTCWTInverse


class WaveGainLayer(nn.Module):
    """ Create gains and apply them to each orientation independently

    Inputs:
        C: Number of input channels
        F: number of output channels
        lp_size: Spatial size of lowpass filter
        bp_sizes: Spatial size of bandpass filters


    The forward pass should be provided with a tuple of wavelet coefficients.
    The tuple should have length 2 and be the lowpass and bandpass coefficients.
    The lowpass coefficients should have shape (N, C, H, W).
    The bandpass coefficients should also be a tuple/list and have length J.
    Each entry in the bandpass list should have 6 dimensions, and should be of
    shape (N, C, 6, H', W', 2).

    Returns a tuple of tensors of the same form as the input, but with F output
    channels.
    """
    def __init__(self, C, F, lp_size=3, bp_sizes=(1,1), lp_stride=1,
                 bp_strides=(1,1)):
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
            s = u_lp.shape
            v_lp = torch.zeros((s[0], self.F, s[2], s[3]), device=u_lp.device)
        else:
            v_lp = func.conv2d(u_lp, self.g_lp, padding=self.lp_pad)

        v = []
        for j in range(self.J):
            g_j = self.g[j]
            u_j = u[j]
            pad = self.bp_pad[j]
            if g_j is None or g_j.shape == torch.Size([0]):
                s = u_j.shape
                v.append(torch.zeros((s[0], self.F, s[2], s[3], s[4], s[5]),
                                     device=u_j.device))
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
    """
    def __init__(self, C, F, lp_size=3, bp_sizes=(1,), q=1.0,
                 biort='near_sym_a', qshift='qshift_a',
                 xfm=True, ifm=True):
        super().__init__()
        self.C = C
        self.F = F
        # If any of the mixing for a scale is 0, don't calculate the dtcwt at
        # that scale
        skip_hps = [True if s == 0 else False for s in bp_sizes]
        self.J = len(bp_sizes)

        # The forward transform
        if xfm:
            self.XFM = DTCWTForward(
                biort=biort, qshift=qshift, J=self.J, skip_hps=skip_hps)
        else:
            self.XFM = lambda x: x

        # The nonlinearity
        if 0.0 < q < 1.0:
            self.shrink = SparsifyWaveCoeffs_std(self.J, F, q, 0.9)
        elif q <= 0.0:
            self.shrink = ReLUWaveCoeffs()
        else:
            self.shrink = lambda x: x

        # The mixing
        self.GainLayer = WaveGainLayer(C, F, lp_size, bp_sizes)

        # The inverse
        if ifm:
            self.IFM = DTCWTInverse(biort=biort, qshift=qshift, J=self.J)
        else:
            self.IFM = lambda x: x

    def forward(self, x):
        u_lp, u = self.XFM(x)
        v_lp, v = self.GainLayer((u_lp, u))
        u_lp2, u2 = self.shrink((v_lp, v))
        y = self.IFM((u_lp2, u2))
        return y

    def init(self, gain=1, method='xavier_uniform'):
        self.GainLayer.init(gain, method)


class ReLUWaveCoeffs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """ Bandpass input comes in as a tensor of shape (N, C, 6, H, W, 2).
        Need to do the relu independently on real and imaginary parts """
        yl, yh = x
        yl = func.relu(yl)
        yh = [func.relu(b) for b in yh]
        return yl, yh


class SparsifyWaveCoeffs_std(nn.Module):
    r""" Sparsifies complex wavelet coefficients by shrinking their magnitude

    Given a quantile value, shrinks the wavelet coefficients so that (1-q) are
    non zero. E.g. if q = 0.9, only 10 percent of coefficients will remain.

    Uses the rayleigh distribution to estimate what the threshold should be.

    Args:
        J (int): number of scales
        C (int): number of channels
        q (float): quantile value.
        alpha (float): exponential moving average value. must be between 0 and 1
    """
    def __init__(self, J, C, q=0.5, alpha=.9, soft=True):
        super().__init__()
        assert alpha >= 0 and alpha <= 1
        # Examine initial thresholds
        self.J = J
        self.C = C
        self.alpha = alpha
        # Keep track of the old value for the ema
        self.ema = nn.Parameter(torch.zeros((J, C, 6)), requires_grad=False)
        self.k = np.sqrt(-2 * np.log(1-q+1e-6))
        #  self.k = k
        self.soft = soft

    @property
    def threshs(self):
        return self.k * self.ema

    def constrain(self):
        pass

    def forward(self, x):
        r""" Sparsify wavelet coefficients coming from DTCWTForward

        Args:
            x (tuple): tuple of yl, yh, where yh is a list/tuple of length J
            representing the coefficients for scale1, scale2, ... scaleJ

        Returns:
            y (tuple): tuple of yl, yh, where yh is the shrunk wavelet
            coefficients
        """
        yl, yh = x
        assert len(yh) == self.J
        yh2 = []

        # For each scale, calculate max(r-t, 0)/r, as the 'gain' to be applied
        # to each input.
        for j in range(self.J):
            if yh[j].shape == torch.Size([0]):
                yh2.append(yh[j])
            else:
                r2 = torch.sum(yh[j]**2, dim=-1, keepdim=True)
                r = torch.sqrt(r2)

                # Calculate the std of each channel
                sz = r.shape[0] * r.shape[3] * r.shape[4]
                # If the real and imaginary parts both have variance 1,
                # then the mean should be situated around sqrt(pi/2) = 1.25
                mu = 1/sz * torch.sum(r.data, dim=(0,3,4,5))
                m2 = 1/sz * torch.sum(r2.data, dim=(0,3,4,5))
                # Similarly, the std should be sqrt((4-pi)/2) = .65
                std = torch.sqrt(m2-mu**2)

                # Update the estimate
                # self.ema[j] = self.alpha * std + (1-self.alpha)*self.ema[j]
                self.ema[j] = self.alpha * self.ema[j] + (1-self.alpha) * std

                # Calculate the new threshold
                thresh = self.ema[j] * self.k

                # Review the threshold to match the size of yh:
                # (N, C, 6, H, W, 2)
                t = thresh.view(self.C, 6, 1, 1, 1)

                if self.soft:
                    yh2.append(_mag_shrink(yh[j], r, t))
                else:
                    yh2.append(_mag_shrink_hard(yh[j], r, t))

        return yl, yh2


def _mag_shrink_hard(x, r, t):
    gain = (r >= t).float()

    return x * gain


def _mag_shrink(x, r, t):
    # Calculate the numerator
    r_new = func.relu(r - t)

    # Add 1 to the denominator if the numerator will be 0
    r1 = r + (r.data <= t).float()

    return x * r_new/r1


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
