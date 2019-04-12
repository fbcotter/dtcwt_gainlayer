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
    def __init__(self, C, F, lp_size=3, bp_sizes=(1,1), lp_stride=1,
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

    def extra_repr(self):
        return '(lp): Parameter of type {} with size: {}'.format(
            self.lp.type(), 'x'.join([str(x) for x in self.lp.shape]))


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
        self.lp_size = lp_size
        self.bp_sizes = bp_sizes
        skip_hps = False
        if self.bp_sizes[0] == 0:
            skip_hps = True
        self.J = len(bp_sizes)

        self.XFM = DTCWTForward(
            biort=biort, qshift=qshift, J=self.J, skip_hps=skip_hps)
        if 0.0 < q < 1.0:
            self.shrink = SparsifyWaveCoeffs_std(self.J, C, q, 0.9)
        elif q <= 0.0:
            self.shrink = ReLUWaveCoeffs()
        else:
            self.shrink = lambda x: x
        self.GainLayer = WaveGainLayer(C, F, lp_size, bp_sizes)
        self.IFM = DTCWTInverse(
            biort=biort, qshift=qshift, J=self.J)

    def forward(self, X):
        yl, yh = self.XFM(X)
        yl, yh = self.GainLayer((yl, yh))
        yl, yh = self.shrink((yl, yh))
        return self.IFM((yl, yh))


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
