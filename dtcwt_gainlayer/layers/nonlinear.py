import torch
import torch.nn as nn
import torch.nn.functional as func
from .shrink import SparsifyWaveCoeffs_std, mag, SoftShrink


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
        elif lp == 'relu2':
            self.lp = BNReLUWaveCoeffs(C, bp=False)
        elif lp == 'hardshrink':
            thresh = lpkwargs.get('thresh', 1)
            self.lp = nn.Hardshrink(thresh)
        elif lp == 'softshrink':
            self.lp = SoftShrink(C, complex=False)
            #  thresh = lpkwargs.get('thresh', 1)
            #  self.lp = nn.Softshrink(thresh)
        elif lp == 'hardshrink_std':
            q = lpkwargs.get('q', 0.8)
            self.lp = SparsifyWaveCoeffs_std(C, q, bp=False, soft=False)
        elif lp == 'softshrink_std':
            q = lpkwargs.get('q', 0.8)
            self.lp = SparsifyWaveCoeffs_std(C, q, bp=False, soft=True)
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
            elif b == 'hardshrink':
                f = nn.Hardshrink(thresh)
            elif b == 'softshrink':
                f = SoftShrink(C, complex=True)
                #  thresh = lpkwargs.get('thresh', 1)
                #  f = nn.Softshrink(thresh)
            elif b == 'hardshrink_std':
                q = lpkwargs.get('q', 0.8)
                f = SparsifyWaveCoeffs_std(C, q, bp=True, soft=False)
            elif b == 'softshrink_std':
                q = lpkwargs.get('q', 0.8)
                f = SparsifyWaveCoeffs_std(C, q, bp=True, soft=True)
            elif b == 'mag':
                f = MagWaveCoeffs()
            else:
                raise ValueError("Unkown nonlinearity {}".format(lp))
            fs.append(f)
        self.bp = nn.ModuleList(fs)

    def forward(self, x):
        yl, yh = x
        yl = self.lp(yl)
        yh = [bp(y) if y.shape != torch.Size([0]) else y
              for bp, y in zip(self.bp, yh)]
        return (yl, yh)


class BNReLUWaveCoeffs(nn.Module):
    def __init__(self, C, bp=True):
        super().__init__()
        self.bp = bp
        if bp:
            self.BN = nn.BatchNorm2d(6*C)
        else:
            self.BN = nn.BatchNorm2d(C)
        self.ReLU = nn.ReLU()

    def forward(self, x):
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


class MagWaveCoeffs(nn.Module):
    """ Returns the magnitude of complex wavelets. Still returns an imaginary
    componenet but it is zero """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, g):
        r = torch.sqrt(g[...,0]**2 + g[...,1]**2 + self.epsilon)
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
