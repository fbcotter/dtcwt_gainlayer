import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init
import numpy as np
from pytorch_wavelets import DTCWTForward, DTCWTInverse


class WaveGainLayer(nn.Module):
    """ Create gains and apply them to each orientation independently

    Inputs:
        C: Number of input channels
        F: number of output channels
        lp_size: Spatial size of lowpass filter
        bp_sizes: Spatial size of bandpass filters
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
                               - func.conv2d(u_jl_imag, g_jl_imag, padding=pad))
                    # imag output = r*i + i*r
                    v_jl_imag = (func.conv2d(u_jl_real, g_jl_imag, padding=pad)
                               + func.conv2d(u_jl_imag, g_jl_real, padding=pad))
                    bands.append(torch.stack((v_jl_real, v_jl_imag), dim=-1))
                # Stack up the 6 bands along the third dimension again
                v.append(torch.stack(bands, dim=2))

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
            g_j = self.g[j]
            if not (g_j is None or g_j.shape == torch.Size([0])):
                fn(g_j, gain=bp_scales[j])

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
    """
    def __init__(self, C, F, lp_size=3, bp_sizes=(1,), q=1.0,
                 biort='near_sym_a', qshift='qshift_a'):
        super().__init__()
        self.C = C
        self.F = F
        # If any of the mixing for a scale is 0, don't calculate the dtcwt at
        # that scale
        skip_hps = [True if s == 0 else False for s in bp_sizes]
        self.J = len(bp_sizes)

        self.XFM = DTCWTForward(
            biort=biort, qshift=qshift, J=self.J, skip_hps=skip_hps)
        # The nonlinearity
        if 0.0 < q < 1.0:
            self.shrink = SparsifyWaveCoeffs_std(self.J, C, q, 0.9)
        elif q <= 0.0:
            self.shrink = ReLUWaveCoeffs()
        else:
            self.shrink = lambda x: x
        self.GainLayer = WaveGainLayer(C, F, lp_size, bp_sizes)
        self.IFM = DTCWTInverse(
            biort=biort, qshift=qshift, J=self.J)

    def forward(self, x):
        u_lp, u = self.XFM(x)
        v_lp, v = self.GainLayer((u_lp, u))
        u_lp2, u2 = self.shrink((v_lp, v))
        y = self.IFM((u_lp2, u2))
        return y


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
