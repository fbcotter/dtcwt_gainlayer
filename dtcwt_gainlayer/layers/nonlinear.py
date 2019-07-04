import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np


class PassThrough(nn.Module):
    def forward(self, x):
        return x


class WaveNonLinearity(nn.Module):
    def __init__(self, C, lp=None, bp=(None,), lpkwargs={}, bpkwargs={}):
        super().__init__()
        if lp is None or lp == 'none':
            self.lp = PassThrough()
        elif lp == 'relu':
            self.lp = nn.ReLU()
        elif lp == 'hardshrink':
            thresh = lpkwargs.get('thresh', 1)
            self.lp = nn.Hardshrink(thresh)
        elif lp == 'softshrink':
            thresh = lpkwargs.get('thresh', 1)
            self.lp = nn.Softshrink(thresh)
        elif lp == 'hardshrink_std':
            q = lpkwargs.get('q', 0.8)
            self.lp = SparsifyWaveCoeffs_std(C, q, bp=False, soft=False)
        elif lp == 'softshrink_std':
            q = lpkwargs.get('q', 0.8)
            self.lp = SparsifyWaveCoeffs_std(C, q, bp=False, soft=True)
        else:
            raise NotImplementedError

        fs = []
        for b in bp:
            if b is None or b == 'none':
                f = PassThrough()
            elif b == 'relu':
                f = nn.ReLU()
            elif b == 'hardshrink':
                thresh = lpkwargs.get('thresh', 1)
                f = nn.Hardshrink(thresh)
            elif b == 'softshrink':
                thresh = lpkwargs.get('thresh', 1)
                f = nn.Softshrink(thresh)
            elif b == 'hardshrink_std':
                q = lpkwargs.get('q', 0.8)
                f = SparsifyWaveCoeffs_std(C, q, bp=True, soft=False)
            elif b == 'softshrink_std':
                q = lpkwargs.get('q', 0.8)
                f = SparsifyWaveCoeffs_std(C, q, bp=True, soft=True)
            elif b == 'mag':
                f = MagWaveCoeffs()
            else:
                raise ValueError
            fs.append(f)
        self.bp = nn.ModuleList(fs)

    def forward(self, x):
        yl, yh = x
        yl = self.lp(yl)
        yh = [bp(y) if y.shape != torch.Size([0]) else y
              for bp, y in zip(self.bp, yh)]
        return (yl, yh)


class MagWaveCoeffs(nn.Module):
    """ Returns the magnitude of complex wavelets. Still returns an imaginary
    componenet but it is zero """
    def forward(self, g):
        r = torch.sqrt(g[...,0]**2 + g[...,1]**2)
        return torch.stack((r, torch.zeros_like(r)), dim=-1)


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


def _mag_shrink_hard(x, r, t):
    gain = (r >= t).float()

    return x * gain


def _mag_shrink(x, r, t):
    # Calculate the numerator
    r_new = func.relu(r - t)

    # Add 1 to the denominator if the numerator will be 0
    r1 = r + (r.data <= t).float()

    return x * r_new/r1


class SparsifyWaveCoeffs_std(nn.Module):
    r""" Sparsifies complex wavelet coefficients by shrinking their magnitude

    Given a quantile value, shrinks the wavelet coefficients so that (1-q) are
    non zero. E.g. if q = 0.9, only 10 percent of coefficients will remain.

    Uses an exponential distribution to estimate the bandpass thresholds, and a
    laplacian to estimate the lowpass thresholds.

    Args:
        J (int): number of scales
        C (int): number of channels
        q (float): quantile value, if 1, keep all activations. if 0, keep none.
        alpha (float): exponential moving average value. must be between 0 and 1
    """
    def __init__(self, C, q=0.5, bp=True, alpha=.9, soft=True):
        super().__init__()
        assert alpha >= 0 and alpha <= 1
        # Examine initial thresholds
        self.C = C
        self.alpha = alpha
        if bp:
            # Keep track of the std
            self.ema = nn.Parameter(torch.zeros((C, 6)), requires_grad=False)
            #  self.k = np.sqrt(-2 * np.log(1-q+1e-6))
            self.k = -np.log(1-q+1e-6)
        else:
            F = (1-0.5*q)
            self.ema = nn.Parameter(torch.zeros(C), requires_grad=False)
            self.k = -np.log(2-2*F)/np.sqrt(2)

        self.soft = soft
        self.bp = bp

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
        if x.shape == torch.Size([0]):
            y = x
        else:
            if self.bp:
                r2 = torch.sum(x**2, dim=-1, keepdim=True)
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
                self.ema.data = self.alpha * self.ema.data + (1-self.alpha) * std

                # Calculate the new threshold
                thresh = self.ema * self.k

                # Review the threshold to match the size of yh:
                # (N, C, 6, H, W, 2)
                t = thresh.view(self.C, 6, 1, 1, 1)

                if self.soft:
                    y = _mag_shrink(x, r, t)
                else:
                    y = _mag_shrink_hard(x, r, t)
            else:
                # Calculate the std of each channel
                sz = x.shape[0] * x.shape[2] * x.shape[3]
                mu = 1/sz * torch.sum(x.data, dim=(0,2,3))
                m2 = 1/sz * torch.sum(x.data**2, dim=(0,2,3))
                std = torch.sqrt(m2-mu**2)

                # Update the std estimate
                self.ema.data = self.alpha * self.ema.data + (1-self.alpha) * std

                # Calculate the new threshold
                thresh = self.ema * self.k
                t = thresh.view(-1, 1, 1)
                mu = mu.view(-1, 1, 1)
                if self.soft:
                    y = _mag_shrink(x, torch.abs(x-mu), t)
                else:
                    y = _mag_shrink_hard(x, torch.abs(x-mu), t)

        return y


#  class SparsifyWaveCoeffs_std(nn.Module):
    #  r""" Sparsifies complex wavelet coefficients by shrinking their magnitude

    #  Given a quantile value, shrinks the wavelet coefficients so that (1-q) are
    #  non zero. E.g. if q = 0.9, only 10 percent of coefficients will remain.

    #  Uses the rayleigh distribution to estimate what the threshold should be.

    #  Args:
        #  J (int): number of scales
        #  C (int): number of channels
        #  q (float): quantile value.
        #  alpha (float): exponential moving average value. must be between 0 and 1
    #  """
    #  def __init__(self, J, C, q=0.5, alpha=.9, soft=True):
        #  super().__init__()
        #  assert alpha >= 0 and alpha <= 1
        #  # Examine initial thresholds
        #  self.J = J
        #  self.C = C
        #  self.alpha = alpha
        #  # Keep track of the old value for the ema
        #  self.ema = nn.Parameter(torch.zeros((J, C, 6)), requires_grad=False)
        #  self.k = np.sqrt(-2 * np.log(1-q+1e-6))
        #  #  self.k = k
        #  self.soft = soft

    #  @property
    #  def threshs(self):
        #  return self.k * self.ema

    #  def constrain(self):
        #  pass

    #  def forward(self, x):
        #  r""" Sparsify wavelet coefficients coming from DTCWTForward

        #  Args:
            #  x (tuple): tuple of yl, yh, where yh is a list/tuple of length J
            #  representing the coefficients for scale1, scale2, ... scaleJ

        #  Returns:
            #  y (tuple): tuple of yl, yh, where yh is the shrunk wavelet
            #  coefficients
        #  """
        #  yl, yh = x
        #  assert len(yh) == self.J
        #  yh2 = []

        #  # For each scale, calculate max(r-t, 0)/r, as the 'gain' to be applied
        #  # to each input.
        #  for j in range(self.J):
            #  if yh[j].shape == torch.Size([0]):
                #  yh2.append(yh[j])
            #  else:
                #  r2 = torch.sum(yh[j]**2, dim=-1, keepdim=True)
                #  r = torch.sqrt(r2)

                #  # Calculate the std of each channel
                #  sz = r.shape[0] * r.shape[3] * r.shape[4]
                #  # If the real and imaginary parts both have variance 1,
                #  # then the mean should be situated around sqrt(pi/2) = 1.25
                #  mu = 1/sz * torch.sum(r.data, dim=(0,3,4,5))
                #  m2 = 1/sz * torch.sum(r2.data, dim=(0,3,4,5))
                #  # Similarly, the std should be sqrt((4-pi)/2) = .65
                #  std = torch.sqrt(m2-mu**2)

                #  # Update the estimate
                #  # self.ema[j] = self.alpha * std + (1-self.alpha)*self.ema[j]
                #  self.ema[j] = self.alpha * self.ema[j] + (1-self.alpha) * std

                #  # Calculate the new threshold
                #  thresh = self.ema[j] * self.k

                #  # Review the threshold to match the size of yh:
                #  # (N, C, 6, H, W, 2)
                #  t = thresh.view(self.C, 6, 1, 1, 1)

                #  if self.soft:
                    #  yh2.append(_mag_shrink(yh[j], r, t))
                #  else:
                    #  yh2.append(_mag_shrink_hard(yh[j], r, t))

        #  return yl, yh2


