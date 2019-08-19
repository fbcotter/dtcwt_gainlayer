import torch
import torch.nn as nn
import torch.nn.functional as func
from dtcwt_gainlayer.layers.shrink import SparsifyWaveCoeffs_std, mag, SoftShrink


class PassThrough(nn.Module):
    def forward(self, x):
        return x


class WaveNonLinearity(nn.Module):
    """ Performs a wavelet-based nonlinearity.

    Args:
        C (int): Number of input channels. Some of the nonlinearities have batch
            norm, so need to know this.
        lp (str): Nonlinearity to use for the lowpass coefficients
        bp (list(str)): Nonlinearity to use for the bandpass coefficients.
        lp_q (float): Quantile value for sparsity threshold for lowpass.
            1 keeps all coefficients and 0 keeps none. Only valid if lp is
            'softshrink_std' or 'hardshrink_std'. See
            :class:`SparsifyWaveCoeffs_std`.
        bp_q (float): Quantile value for sparsity threshold for bandpass
            coefficients. Only valid if bp is 'softshrink_std' or
            'hardshrink_std'.

    The options for the lowpass are:

    - none
    - relu (as you'd expect)
    - relu2 - applies batch norm + relu
    - softshrink - applies soft shrinkage with a learnable threshold
    - hardshrink_std - applies hard shrinkage. The 'std' implies that it
      tracks the standard deviation of the activations, and sets a threshold
      attempting to reach a desired sparsity level. This assumes that the
      lowpass coefficients follow a laplacian distribution. See
      :class:`dtcwt_gainlayer.layers.shrink.SparsifyWaveCoeffs_std`.
    - softshrink_std - same as hardshrink std except uses soft shrinkage.

    The options for the bandpass are:

    - none
    - relu (applied indepently to the real and imaginary components)
    - relu2 - applies batch norm + relu to the magnitude of the bandpass
      coefficients
    - softshrink - applies shoft shrinkage to the magnitude of the bp
      coefficietns with a learnable threshold
    - hardshrink_std - applies hard shrinkage by tracking the standard
      deviation. Assumes the bp distributions follow an exponential
      distribution. See
      :class:`dtcwt_gainlayer.layers.shrink.SparsifyWaveCoeffs_std`.
    - softshrink_std - same as hardshrink_std but with soft shrinkage.

    """
    def __init__(self, C, lp=None, bp=(None,), lp_q=0.8, bp_q=0.8):
        super().__init__()
        if lp is None or lp == 'none':
            self.lp = PassThrough()
        elif lp == 'relu':
            self.lp = nn.ReLU()
        elif lp == 'relu2':
            self.lp = BNReLUWaveCoeffs(C, bp=False)
        elif lp == 'softshrink':
            self.lp = SoftShrink(C, complex=False)
        elif lp == 'hardshrink_std':
            self.lp = SparsifyWaveCoeffs_std(C, lp_q, bp=False, soft=False)
        elif lp == 'softshrink_std':
            self.lp = SparsifyWaveCoeffs_std(C, lp_q, bp=False, soft=True)
        else:
            raise ValueError("Unkown nonlinearity {}".format(lp))

        fs = []
        for b in bp:
            if b is None or b == 'none':
                f = PassThrough()
            elif b == 'relu':
                f = nn.ReLU()
            elif b == 'relu2':
                f = BNReLUWaveCoeffs(C, bp=True)
            elif b == 'softshrink':
                f = SoftShrink(C, complex=True)
            elif b == 'hardshrink_std':
                f = SparsifyWaveCoeffs_std(C, bp_q, bp=True, soft=False)
            elif b == 'softshrink_std':
                f = SparsifyWaveCoeffs_std(C, bp_q, bp=True, soft=True)
            else:
                raise ValueError("Unkown nonlinearity {}".format(lp))
            fs.append(f)
        self.bp = nn.ModuleList(fs)

    def forward(self, x):
        """ Applies the selected lowpass and bandpass nonlinearities to the
        input x.

        Args:
            x (tuple): tuple of (lowpass, bandpasses)

        Returns:
            y (tuple): tuple of (lowpass, bandpasses)
        """
        yl, yh = x
        yl = self.lp(yl)
        yh = [bp(y) if y.shape != torch.Size([0]) else y
              for bp, y in zip(self.bp, yh)]
        return (yl, yh)


class BNReLUWaveCoeffs(nn.Module):
    """ Applies batch normalization followed by a relu

    Args:
        C (int): number of channels
        bp (bool): If true, applies bn+relu to the magnitude of the bandpass
            coefficients. If false, is applying bn+relu to the lowpass coeffs.
    """
    def __init__(self, C, bp=True):
        super().__init__()
        self.bp = bp
        if bp:
            self.BN = nn.BatchNorm2d(6*C)
        else:
            self.BN = nn.BatchNorm2d(C)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        """ Applies nonlinearity to the input x """
        if self.bp:
            s = x.shape
            # Move the orientation dimension to the channel
            x = x.view(s[0], s[1]*s[2], s[3], s[4], s[5])
            θ = torch.atan2(x.data[..., 1], x.data[..., 0])
            r = mag(x, complex=True)
            r_new = self.ReLU(self.BN(r))
            y = torch.stack((r_new * torch.cos(θ), r_new * torch.sin(θ)), dim=-1)
            # Reshape to a 6D tensor again
            y = y.view(s[0], s[1], s[2], s[3], s[4], s[5])
        else:
            y = self.ReLU(self.BN(x))
        return y
