import torch
import torch.nn as nn
import torch.nn.functional as func
import math
import numpy as np
import pytorch_wavelets
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from dtcwt_gainlayer.layers.nonlinear import WaveNonLinearity
from torch.autograd import Function

ORIENTATION_DIM = 2
REAL_IMAG_DIM = -1


class ComplexL1(Function):
    r""" Applies complex L1 regularization. Whenever the input is zero, sets the
    gradient to be 0 """
    @staticmethod
    def forward(ctx, z):
        mag = torch.sqrt(z[..., 0]**2 + z[..., 1]**2)
        phase = torch.atan2(z[..., 1], z[..., 0])
        # Mark the locations where the input was zero with nans
        phase[(z[..., 0] == 0) & (z[..., 1] == 0)] = torch.tensor(np.nan)
        ctx.save_for_backward(phase)
        return torch.sum(mag)

    @staticmethod
    def backward(ctx, dy):
        phase, = ctx.saved_tensors
        dz = torch.stack((dy*torch.cos(phase), dy*torch.sin(phase)), dim=-1)
        # Wherever we have nans in the output (nans have the unique property
        # that they never equal each other), set the gradient to 0.
        dz[dz != dz] = 0
        return dz


class SmoothMagFn(torch.autograd.Function):
    """ Class to do complex magnitude """
    @staticmethod
    def forward(ctx, x, y, b):
        r = torch.sqrt(x**2 + y**2 + b**2)
        if x.requires_grad:
            dx = x/r
            dy = y/r
            ctx.save_for_backward(dx, dy)

        return r - b

    @staticmethod
    def backward(ctx, dr):
        dx = None
        if ctx.needs_input_grad[0]:
            drdx, drdy = ctx.saved_tensors
            dx = drdx * dr
            dy = drdy * dr
        return dx, dy, None


class WaveGainLayer(nn.Module):
    """ Create wavelet gains and apply them to each orientation independently

    Args:
        C (int): Number of input channels
        F (int): number of output channels
        lp_size (int): Spatial size of lowpass filter
        bp_sizes (tuple(int)): Spatial size of bandpass filters. The length of
            the tuple indicates the number of scales we want to use. Can have
            zeros in the tuple to not do processing at that scale.
        wd (float): l2 weight decay for lowpass gain
        wd1 (float, optional): l1 weight decay for bandpass gains. If None, uses
            wd and l2.

    Attributes:
        J (int): lenght of bp_sizes tuple
        g_lp (torch.Tensor): Learnable lowpass gain
        g (:class:`~torch.nn.ParameterList`): Tuple of complex bandpass gains

    Note:
        We recommend you use the class' init method to initialize the gains, as
        the complex weights are 6-D tensors, and the in-built glorot
        initialization will incorrectly estimate the fan in and fan out.
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
                bps.append(nn.Parameter(torch.randn(6, F, C, s, s, 2)))
        self.g = nn.ParameterList(bps)
        self.init()

    def forward(self, coeffs):
        r""" Applies the wavelet gains to the provided coefficients.

        Args:
            coeffs (tuple): tuple of (lowpass, bandpass) coefficients, where
                the bandpass coefficients is also a tuple of length J. This is
                the default return style from
                :class:`pytorch_wavelets.dtcwt.transform2d.DTCWTForward`, so
                you can easily connect the output of the DTCWT to the gain
                layer.

        Returns:
            coeffs (tuple): tuple of (lowpass, bandpass) coefficients in same
                form as input but with the channels mixed.
        """
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
                    g_jl_real, g_jl_imag = g_j[l, ..., 0], g_j[l, ..., 1]
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

    def init(self, gain=1):
        r""" Initializes the gain layer to follow the xavier uniform method

        Args:
            gain (float): Xavier/Glorot gain a.

        Have to calculate the fan in and fan out directly from the tensors. Then
        sets the gains to be randomly drawn from a uniform:

        .. math::

            g \sim U\left[-g \sqrt{ \frac{6}{fan\_in + fan\_out} },\
                 g \sqrt{\frac{6}{fan\_in+fan\_out}} \right]

        where g is the provided gain
        """
        lp_scales = np.array([1, 2, 4, 8]) * gain
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
                fan_in, fan_out = s[2]*s[3]*s[4], s[1]*s[3]*s[4]
                std = bp_scales[j] * math.sqrt(2.0 / (fan_in + fan_out))
                a = math.sqrt(3.0) * std
                with torch.no_grad():
                    g_j.uniform_(-a, a)

    def get_reg(self):
        r""" Returns regularization function applied to the lowpass and bandpass
        gains

        Returned value is differentiable, so can call .backward() on it.
        """
        a = self.wd*0.5*torch.sum(self.g_lp**2)
        if self.wd1 is not None:
            for g in self.g:
                a += self.wd1 * ComplexL1.apply(g)
        else:
            for g in self.g:
                a += 0.5 * self.wd * torch.sum(g**2)
        return a

    def extra_repr(self):
        return '(g_lp): Parameter of type {} with size: {}'.format(
            self.g_lp.type(), 'x'.join([str(x) for x in self.g_lp.shape]))


class WaveConvLayer(nn.Module):
    r""" Takes the input into the DTCWT domain before learning complex mixing
    gains and optionally applying a nonliearity.

    This is a convenience class that combines the
    :class:`dtcwt_gainlayer.WaveGainLayer`,
    :class:`pytorch_wavelets.DTCWTForward`,
    :class:`pytorch_wavelets.DTCWTInverse`,
    and :class:`dtcwt_gainlayer.WaveNonLinearity` classes.
    Can build your own class to combine these individually for more flexibility
    (e.g. if you did not want to use the Inverse DTCWT after the nonlinearity).

    Args:
        C (int): Number of input channels. See
            :class:`~dtcwt_gainlayer.WaveGainLayer`.
        F (int): number of output channels. See
            :class:`~dtcwt_gainlayer.WaveGainLayer`.
        lp_size: Spatial size of lowpass filter. See
            :class:`~dtcwt_gainlayer.WaveGainLayer`.
        bp_sizes: Spatial size of bandpass filters. See
            :class:`~dtcwt_gainlayer.WaveGainLayer`.
        biort: Biorthogonal filters to use for the DTCWT and inverse DTCWT.
        qshift: Quarter shift filters to use for the DTCWT and inverse DTCWT.
        xfm: true indicating should take dtcwt of input, false to skip
        ifm: true indicating should take inverse dtcwt of output, false to skip
        wd: l2 weight decay for lowpass gain. See
            :class:`~dtcwt_gainlayer.WaveGainLayer`.
        wd1: l1 weight decay for bandpass gain. If None, uses wd and l2. See
            :class:`~dtcwt_gainlayer.WaveGainLayer`.
        lp_nl: nonlinearity to use for the lowpass. See
            :class:`~dtcwt_gainlayer.WaveNonLinearity`.
        bp_nl: nonlinearities to use for the bandpasses. See
            :class:`~dtcwt_gainlayer.WaveNonLinearity`.
        lp_nl_kwargs: keyword args for the lowpass nonlinearity
        bp_nl_kwargs: keyword args for the lowpass nonlinearity
    """
    def __init__(self, C, F, lp_size=3, bp_sizes=(1,), biort='near_sym_a',
                 qshift='qshift_a', xfm=True, ifm=True, wd=0, wd1=None,
                 lp_nl=None, bp_nl=None, lp_nl_kwargs=None, bp_nl_kwargs=None):
        super().__init__()
        self.C = C
        self.F = F
        # If any of the mixing for a scale is 0, don't calculate the dtcwt at
        # that scale
        skip_hps = [True if s == 0 else False for s in bp_sizes]
        self.J = len(bp_sizes)
        self.wd = wd
        self.wd1 = wd1
        self.do_xfm = xfm
        self.do_ifm = ifm

        self.XFM = DTCWTForward(
            biort=biort, qshift=qshift, J=self.J, skip_hps=skip_hps,
            o_dim=ORIENTATION_DIM, ri_dim=REAL_IMAG_DIM)
        self.GainLayer = WaveGainLayer(C, F, lp_size, bp_sizes, wd=wd, wd1=wd1)
        if not isinstance(bp_nl, (list, tuple)):
            bp_nl = [bp_nl, ] * self.J
        self.NL = WaveNonLinearity(F, lp_nl, bp_nl, lp_nl_kwargs, bp_nl_kwargs)
        self.IFM = DTCWTInverse(
            biort=biort, qshift=qshift, o_dim=ORIENTATION_DIM,
            ri_dim=REAL_IMAG_DIM)

    def forward(self, x):
        """ Applies the wavelet gain layer to the provided coefficients.

        Args:
            x: If xfm is true, x should be a pixel representation of an input.
               If it is false, it should be tuple of (lowpass, bandpass)
               coefficients, where the bandpass coefficients is also a tuple of
               length J.

        Returns:
            y: If ifm is true, y will be in the pixel domain. If false, will be
                a tuple of (lowpass, bandpass) coefficients.
        """
        if self.do_xfm:
            y = self.XFM(x)
        else:
            y = x
        y = self.GainLayer(y)
        y = self.NL(y)
        if self.do_ifm:
            y = self.IFM(y)

        return y

    def init(self, gain=1, method='xavier_uniform'):
        """ Initializes the Gain layer weights
        """
        self.GainLayer.init(gain, method)

    def get_reg(self):
        """ Returns regularization function applied to the gain layer """
        return self.Gainlayer.get_reg()


class WaveParamLayer(nn.Module):
    r""" Parameterizes gains in the DTCWT domain before doing convolution in the
    spatial domain.

    Currently only supports convolutional kernels with size 4, or 8. For an
    8x8 kernel, can specify to use only the lower frequency regions. Also, only
    uses the coarsest scale bandpass coefficients. I.e. if J=3, will not learn
    scale 1 and 2 coefficients, but scale 3 and the lowpass.

    Args:
        C: Number of input channels
        F: number of output channels
        k: kernel size
        J: number of scales
        wd: l2 weight decay to use for lowpass weights
        wd1: l1 weight decay to use for bandpass gains. If None, will use l2
            weight decay and the wd term.
        right: As the convolutional kernels are even, this flag shifts the
            output by 1 pixel to the right (and down), and if it is off, the
            output is shifted one pixel to the left (and up).
    """
    def __init__(self, C, F, k=4, stride=1, J=1, wd=0, wd1=None, right=True):
        super().__init__()
        self.wd = wd
        if wd1 is None:
            self.wd1 = wd
        else:
            self.wd1 = wd1
        self.k = k
        self.C = C
        self.F = F
        self.J = J
        self.right = right
        self.ifm = DTCWTInverse()
        self._init()

    def _init(self):
        # To get the right coeff sizes, perform a forward dtcwt on a kernel size
        # you ultimately want after reconstruction.
        x = torch.zeros(self.F, self.C, self.k, self.k)
        torch.nn.init.xavier_uniform_(x)
        xfm = DTCWTForward(J=self.J)
        yl, yh = xfm(x)
        self.downsample = False

        if self.k == 4:
            if self.J == 1:
                self.downsample = True
                yl = func.avg_pool2d(yl, 2)
            self.u_lp = nn.Parameter(torch.zeros_like(yl))
            self.uj = nn.Parameter(torch.zeros_like(yh[-1]))
            self.u_lp.data = yl.data
            self.uj.data = yh[-1].data
            if self.right:
                self.pad = (1, 2, 1, 2)
            else:
                self.pad = (2, 1, 2, 1)
        elif self.k == 8:
            if self.J == 1:
                self.downsample = True
                yl = func.avg_pool2d(yl, 2)
            self.u_lp = nn.Parameter(torch.zeros_like(yl))
            self.uj = nn.Parameter(torch.zeros_like(yh[-1]))
            self.u_lp.data = yl.data
            self.uj.data = yh[-1].data
            if self.right:
                self.pad = (3, 4, 3, 4)
            else:
                self.pad = (4, 3, 4, 3)
        else:
            raise NotImplementedError
        return yl, yh

    @property
    def filt(self):
        """ Gets the spatial domain representation of the filter.

        Inverts the learned gains from DTCWT coefficients to wavelet coeffs
        """
        if self.downsample:
            u_lp = func.interpolate(self.u_lp, scale_factor=2, mode='bilinear',
                                    align_corners=False)
        else:
            u_lp = self.u_lp

        if self.J == 1:
            h = self.ifm((u_lp, (self.uj,)))
        elif self.J == 2:
            h = self.ifm((u_lp, (None, self.uj)))
        elif self.J == 3:
            h = self.ifm((u_lp, (None, None, self.uj)))
        return h

    def forward(self, x):
        """ Convolves an input by the reconstructed kernel. Does the inverse
        DTCWT on the weights and then applies the convolutional kernel in the
        spatial domain.
        """
        x = torch.nn.functional.pad(x, self.pad)
        y = func.conv2d(x, self.h)
        return y

    def get_reg(self):
        """ Returns regularization function applied to the lowpass and bandpass
        gains

        Returned value is differentiable, so can call .backward() on it.
        """
        a = self.wd*0.5*torch.sum(self.u_lp**2)
        if self.wd1 is not None:
            for g in self.g:
                a += self.wd1 * ComplexL1.apply(g)
        else:
            for g in self.g:
                a += 0.5 * self.wd * torch.sum(g**2)
        return a

    def extra_repr(self):
        return '(u_lp): Parameter of type {} with size: {}\n' \
               '(uj): Parameter of type {} with size: {}'.format(
                   self.u_lp.type(), 'x'.join([str(x) for x in self.u_lp.shape]),
                   self.uj.type(), 'x'.join([str(x) for x in self.uj.shape]))


class WaveMaxPoolJ1(nn.Module):
    """ Performs max pooling across DTCWT orientations by taking the largest
    magnitude and adding the lowpass coefficients.
    """
    def forward(self, x):
        yl, (yh, ) = x
        r = SmoothMagFn.apply(yh[..., 0], yh[..., 1], 1e-2)
        r = torch.max(r, dim=ORIENTATION_DIM)[0]
        return r + torch.nn.functional.avg_pool2d(yl, 2)
