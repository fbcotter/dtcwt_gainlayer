""" This module implements soft and hard shrinkage functions for wavelet
coefficients """
import torch
import torch.nn.functional as func
from torch.autograd import Function
import torch.nn as nn
import numpy as np


class HardShrink1(Function):
    """ Autograd defined class to do hard shrinkage on complex coeffs """
    @staticmethod
    def forward(ctx, x1, t1, complex=False):
        if complex:
            t1_r = t1.view(-1, 1, 1, 1, 1)
        else:
            t1_r = t1.view(-1, 1, 1)
        m1 = mag(x1, complex, keepdim=True)
        gain1 = (m1 > t1_r).float()

        ctx.save_for_backward(gain1)
        return x1 * gain1

    @staticmethod
    def backward(ctx, grad_y1):
        gain1, = ctx.saved_tensors
        grad_x1 = None
        grad_t1 = None
        if ctx.needs_input_grad[0]:
            grad_x1 = grad_y1 * gain1

        return grad_x1, grad_t1, None


def _soft_thresh_grad(grad_y, x, m, gain, complex):
    if not complex:
        return grad_y * (gain > 0).float()
    else:
        """ Assume y = c+id = S(x) = S(a+ib) """
        common = 1/m**2 * ((gain > 0).float() - gain)
        dc = torch.cat((gain, torch.zeros_like(gain)), dim=-1) + \
            torch.unsqueeze(x[...,0], dim=-1) * x * common
        dd = torch.cat((torch.zeros_like(gain), gain), dim=-1) + \
            torch.unsqueeze(x[...,1], dim=-1) * x * common
        return torch.unsqueeze(grad_y[...,0], dim=-1) * dc + \
            torch.unsqueeze(grad_y[...,1], dim=-1) * dd


class SoftShrink1(Function):
    """ Autograd defined class to do soft shrinkage on complex coeffs """
    @staticmethod
    def forward(ctx, x, t, complex=False):
        ctx.complex = complex
        # Do some shape manipulation for the channels
        if complex:
            t2 = t.view(-1, 6, 1, 1)
        else:
            t2 = t.view(-1, 1, 1)

        r = mag(x, complex=complex)
        # When the magnitude is below the threshold, add an offset
        denom = r + (r < t2).float()
        gain = torch.relu(r - t2)/denom

        if complex:
            y = x * gain[..., None]
        else:
            y = x * gain

        if ctx.needs_input_grad[0]:
            if complex:
                ctx.save_for_backward(x, gain, r, t)
            else:
                ctx.save_for_backward(torch.sign(y))
        return y

    @staticmethod
    def backward(ctx, dy):
        complex = ctx.complex
        grad_x = None
        grad_t = None

        if ctx.needs_input_grad[0]:
            if complex:
                x, gain, r, t = ctx.saved_tensors
                # y = c + jd = S(x) = S(a+jb)
                t = t.view(-1, 6, 1, 1)
                dc, dd = dy[..., 0], dy[..., 1]
                a, b = x[..., 0], x[..., 1]
                f = t * (r > t).float()/r**3
                h = -(r > t).float()/r
                dcda = gain + a**2*f
                dcdb = a*b*f
                ddda = dcdb
                dddb = gain + b**2*f
                dcdt = a * h
                dddt = b * h
                da = dc*dcda + dd*ddda
                db = dc*dcdb + dd*dddb
                dt = torch.sum(dc*dcdt + dd*dddt, dim=(0, 3, 4))
                grad_x = torch.stack((da, db), dim=-1)
                grad_t = dt
            else:
                sign, = ctx.saved_tensors
                grad_x = dy * (sign != 0).float()
                grad_t = -(dy * sign).sum(dim=(0,2,3))

        return grad_x, grad_t, None


def mag(y, complex=False, keepdim=False, epsilon=1e-6):
    """ Calculates the magnitude of the vector y. Can accept complex (last
    dimension is 2) or real inputs.
    """
    # Calculate the complex magnitude if complex, else just the absolute value
    if complex:
        r = torch.sqrt(y[...,0]**2 + y[...,1]**2 + epsilon)
        if keepdim:
            r = torch.unsqueeze(r, dim=-1)
    else:
        r = torch.abs(y)
    return r


def _mag_shrink_hard(x, r, t):
    """ x is the input, r is the magnitude and t is the threshold
    """
    gain = (r >= t).float()

    return x * gain


def _mag_shrink(x, r, t):
    """ x is the input, r is the magnitude and t is the threshold
    """
    # Calculate the numerator
    r_new = func.relu(r - t)

    # Add 1 to the denominator if the numerator will be 0
    r1 = r + (r.data <= t).float()

    return x * r_new/r1


class HardShrink(nn.Module):
    r""" Performs element-wise hard shrinkage

    Can work on complex or real inputs. If complex, it is assumed the last
    dimension is 2. Creates C thresholds, one for each input channel. C must be
    the second dimension.

    .. math::

        y = x \cdot \mathbb{I}(|x| - t > 0)

    Where :math:`\mathbb{I}` is the indicator function.

    The gradients are:

    .. math::

        \delta x_i &=& \delta y_i \mathbb{I}(|x| - t > 0) \\
        \delta t &=& 0

    Note:
        That while the gradients are calculated for this block,
        :math:`\delta t = 0 \forall x`.

    Note:
        The threshold should be strictly positive. To constrain this ,
        we allow the thresh parameter to be real and take the absolute value of
        it.
    """
    def __init__(self, C, t_init=1.0, complex=False):
        super().__init__()
        if hasattr(t_init, '__iter__'):
            assert len(t_init) == C
        else:
            t_init = [t_init,] * C
        self.thresh = nn.Parameter(torch.tensor(t_init).float())
        self.complex = complex
        self.C = C

    @property
    def thresh(self):
        return torch.abs(self._thresh)

    def forward(self, x):
        """ Applies Hard Thresholding to x """
        if x.shape == torch.Size([0]):
            return x
        else:
            assert x.shape[1] == self.C
            return HardShrink1.apply(x, self.thresh, self.complex)


class SoftShrink(nn.Module):
    r""" Performs element-wise soft shrinkage

    Can work on complex or real inputs. If complex, it is assumed the last
    dimension is 2.

    .. math::

        y = x \frac{\text{Relu}(|x| - t)}{|x|} = x \cdot g


    The gradients are:

    .. math::

        \delta x_i &=& \delta y_i \cdot \mathbb{I}(|y_i| > 0) \\
        \delta t &=& -\sum_i \delta y_i \text{sign}(y_i)

    Where :math:`\mathbb{I}` is the indicator function.

    Note:
        Unlike the hard shrinkage nonlinearity, :math:`\delta t` is not 0 for
        all t, note that it is 0 if t becomes too big and :math:`\mathbb{I}
        \rightarrow 0`.

    Note:
        The threshold should be strictly positive. To constrain this ,
        we allow the thresh parameter to be real and take the absolute value of
        it.
    """
    def __init__(self, C, complex=False):
        super().__init__()
        if complex:
            self._thresh = nn.Parameter(0.1*torch.rand(C, 6))
        else:
            self._thresh = nn.Parameter(0.1*torch.rand(C))
        self.complex = complex
        self.C = C

    @property
    def thresh(self):
        return torch.abs(self._thresh)

    def forward(self, x):
        """ Applies Soft Thresholding to x """
        if x.shape == torch.Size([0]):
            return x
        else:
            assert x.shape[1] == self.C
            return SoftShrink1.apply(x, self.thresh, self.complex)


class SparsifyWaveCoeffs_std(nn.Module):
    r""" Sparsifies complex wavelet coefficients by shrinking their magnitude

    Given a quantile value, shrinks the wavelet coefficients so that (1-q) are
    non zero. E.g. if q = 0.9, only 10 percent of coefficients will remain.

    Uses an exponential distribution to estimate the bandpass thresholds, and a
    laplacian to estimate the lowpass thresholds. I.e. for complex bandpass coefficients
    with standard deviation :math:`\sigma`, the threshold on the magnitudes is set to:

    .. math::

        t_{r} = -\sigma \log(1-q)

    and for lowpass coefficients, the threshold on is set to:

    .. math::

        t_{lp} = -\sigma \frac{\log q}{\sqrt{2}}

    Args:
        C (int): number of channels
        q (float): quantile value, if 1, keep all activations. if 0, keep none.
        bp (bool): True if this is tracking complex bandpass coefficients.
        alpha (float): exponential moving average value. must be between 0 and 1
        soft (bool): True if you want to use soft thresholding
        warmup (int): number of steps to before applying the shrinkage

    Note:
        The standard deviation will be updated unless the class is in evaluation
        mode. This should be handled when you are testing automatically, so long
        as you call `net.eval()`.
    """
    def __init__(self, C, q=0.5, bp=True, alpha=.9, soft=True, warmup=500):
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

        self.warmup = nn.Parameter(torch.tensor(warmup), requires_grad=False)
        self.step = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.soft = soft
        self.bp = bp

    @property
    def threshs(self):
        return self.k * self.ema

    def forward(self, x):
        r""" Sparsify wavelet coefficients as a function of their :math:`\sigma`

        Args:
            x: wavelet lowpass or bandpass coefficients
        """
        if x.shape == torch.Size([0]):
            y = x
        else:
            if self.bp:
                if self.training:
                    r = mag(x, complex=True, keepdim=True)
                    r2 = r**2

                    # Calculate the std of each channel
                    sz = r.shape[0] * r.shape[3] * r.shape[4]

                    # If the real and imaginary parts both have variance 1,
                    # then the mean should be situated around sqrt(pi/2) = 1.25
                    mu = 1/sz * torch.sum(r.data, dim=(0,3,4,5))
                    m2 = 1/sz * torch.sum(r2.data, dim=(0,3,4,5))
                    # Similarly, the std should be sqrt((4-pi)/2) = .65
                    std = torch.sqrt(torch.relu(m2-mu**2))

                    # Update the estimate
                    # self.ema[j] = self.alpha * std + (1-self.alpha)*self.ema[j]
                    self.ema.data = self.alpha * self.ema.data + (1-self.alpha) * std

                thresh = self.ema * self.k

                # Review the threshold to match the size of yh:
                # (N, C, 6, H, W, 2)
                t = thresh.view(self.C, 6, 1, 1, 1)

                if self.step >= self.warmup:
                    if self.soft:
                        #  y = _mag_shrink(x, r, t)
                        y = SoftShrink1.apply(x, t, True)
                    else:
                        #  y = _mag_shrink_hard(x, r, t)
                        y = HardShrink1.apply(x, t, True)
                else:
                    y = x
            else:
                if self.training:
                    # Calculate the std of each channel
                    sz = x.shape[0] * x.shape[2] * x.shape[3]
                    mu = 1/sz * torch.sum(x.data, dim=(0,2,3))
                    m2 = 1/sz * torch.sum(x.data**2, dim=(0,2,3))
                    std = torch.sqrt(m2-mu**2)

                    # Update the std estimate
                    self.ema.data = self.alpha * self.ema.data + (1-self.alpha) * std

                thresh = self.ema * self.k
                t = thresh.view(-1, 1, 1)
                mu = 0
                if self.step >= self.warmup:
                    if self.soft:
                        #  y = SoftShrink1.apply(x, t, False)
                        y = _mag_shrink(x, torch.abs(x-mu), t)
                    else:
                        #  y = HardShrink1.apply(x, t, False)
                        y = _mag_shrink_hard(x, torch.abs(x-mu), t)
                else:
                    y = x

        self.step.data += 1
        return y
