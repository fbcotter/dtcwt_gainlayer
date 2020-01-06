""" This module builds different networks with nonlinearities
"""
import torch
import torch.nn as nn
import torch.nn.functional as func
from dtcwt_gainlayer import WaveConvLayer
from dtcwt_gainlayer.layers.dtcwt import WaveMaxPoolJ1
from collections import OrderedDict

nets = {
    'ref': ['conv', 'pool', 'conv', 'pool', 'conv'],
    'waveA': ['gain1', 'wpool', 'conv', 'pool', 'conv'],
    'waveB': ['conv', 'pool', 'gain1', 'wpool', 'conv'],
    'waveC': ['conv', 'pool', 'conv', 'pool', 'gain'],
    'waveD': ['gain1', 'wpool', 'gain', 'wpool', 'conv'],
    'waveE': ['conv', 'pool', 'gain1', 'wpool', 'gain'],
    'waveF': ['gain1', 'wpool', 'gain1', 'wpool', 'gain'],
}


class PassThrough(nn.Module):
    def forward(self, x):
        """ No nonlinearity """
        return x


class NonlinearNet2(nn.Module):
    """ Builds a VGG-like network with gain layers with nonlinearities

    Args:
        dataset (str): cifar10, cifar100, tiny_imagenet. Needed to know
            how to shape the network backend.
        type (str): key into the nets dictionary defining the layer order
        num_channels (int): number of output channels for the first scale. This
            value doubles after pooling.
        wd (float): l2 weight decay for pixel and lowpass gains.
        wd1 (float): l1 weight decay for complex bandpass gains
        pixel_k (int): pixel convolution kernel size
        lp_k (int): lowpass convolution kernel size
        bp_ks (tuple(int)): bandpass convolution kernel sizes. Length of this
            tuple defines how many wavelet scales to take. If you want to skip
            the first scale, you can set bp_ks=(0,1)
        pixel_nl (str): pixel nonlinearity. See
            :class:`dtcwt_gainlayer.WaveNonlinearity`.
        lp_nl (str): lowpass nonlinearity. See
            :class:`dtcwt_gainlayer.WaveNonlinearity`.
        pixel_nl (str): banpass nonlinearity. See
            :class:`dtcwt_gainlayer.WaveNonlinearity`.

    Note:
        The wd and wd1 parameters prime the network for when you
        call :meth:`NonlinearNet.get_reg` to get the regularization
        term.

    """
    def __init__(self, dataset, type, num_channels=64, wd=1e-4, wd1=None,
                 pixel_k=5, lp_k=3, bp_ks=(1,), pixel_nl='none', lp_nl='relu',
                 bp_nl='relu2'):
        super().__init__()

        if dataset == 'cifar10':
            self.num_classes = 10
        elif dataset == 'cifar100':
            self.num_classes = 100
        elif dataset == 'tiny_imagenet':
            self.num_classes = 200
        self.wd = wd
        self.wd1 = wd1
        self._wave_params = []
        self._default_params = []
        layers = nets[type]

        WaveLayer = lambda Cin, Cout, ifm: WaveConvLayer(
            Cin, Cout, lp_k, bp_ks, wd=wd, wd1=wd1, lp_nl=lp_nl, bp_nl=(bp_nl,),
            ifm=ifm)

        if pixel_nl == 'relu':
            σ_pixel = lambda C: nn.Sequential(
                nn.BatchNorm2d(C),
                nn.ReLU())
        else:
            σ_pixel = lambda C: nn.BatchNorm2d(C)

        # Build the main part of the network from the dictionary definition
        blks = []
        layer, pool = 0, 1
        Cin, Cout = 3, num_channels
        for blk in layers:
            if blk == 'conv':
                name = 'conv' + chr(ord('A') + layer)
                blk = nn.Sequential(
                    nn.Conv2d(Cin, Cout, pixel_k, padding=2, stride=1),
                    nn.BatchNorm2d(Cout), nn.ReLU())
                self._default_params.extend(list(blk.parameters()))
                Cin = Cout
                layer += 1
            elif blk.startswith('gain'):
                if blk.endswith('1'):
                    blk = WaveLayer(Cin, Cout, ifm=False)
                else:
                    blk = nn.Sequential(WaveLayer(Cin, Cout, ifm=True), σ_pixel(Cout))
                name = 'wave' + chr(ord('A') + layer)
                self._wave_params.extend(list(blk.parameters()))
                Cin = Cout
                layer += 1
            elif blk == 'pool':
                name = 'pool' + str(pool)
                blk = nn.MaxPool2d(2)
                pool += 1
                Cout = 2*Cin
            elif blk == 'wpool':
                name = 'wpool' + str(pool)
                blk = WaveMaxPoolJ1()
                pool += 1
                Cout = 2*Cin
            blks.append((name, blk))

        # Build the backend of the network
        if dataset == 'cifar10' or dataset == 'cifar100':
            self.net = nn.Sequential(OrderedDict(blks))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(Cout, self.num_classes)
        elif dataset == 'tiny_imagenet':
            blk1 = nn.MaxPool2d(2)
            blk2 = nn.Sequential(
                nn.Conv2d(Cout, 2*Cout, pixel_k, padding=2, stride=1),
                nn.BatchNorm2d(2*Cout),
                nn.ReLU())
            blks = blks + [
                ('pool3', blk1),
                ('conv_final', blk2),]
            self.net = nn.Sequential(OrderedDict(blks))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(2*Cout, self.num_classes)
            self._default_params.extend(list(blk2.parameters()))
        self._default_params.extend(list(self.fc1.parameters()))

    def parameters(self):
        """ Return all parameters that do not belong to any wavelet based
        learning """
        return self._default_params

    def wave_parameters(self):
        """ Return all parameters that belong to wavelet based learning """
        return self._wave_params

    def get_reg(self):
        """ Applies custom regularization.

        The default in pytorch is to apply l2 to everything with the same weight
        decay. We allow for more customizability.
        """
        loss = 0
        for name, m in self.net.named_children():
            if name.startswith('wave'):
                loss += m[0].GainLayer.get_reg()
            elif name.startswith('conv'):
                loss += 0.5 * self.wd * torch.sum(m[0].weight**2)
        loss += 0.5 * self.wd * torch.sum(self.fc1.weight**2)
        return loss

    def clip_grads(self, value=1):
        """ Clips gradients to be in the range [-value, value].

        Can be useful to do if you are getting nans in training. Also sets nans
        to be 0.
        """
        grads = []
        for name, m in self.net.named_children():
            if name.startswith('wave'):
                grads.extend([g for g in m[0].GainLayer.g])
        # Set nans in grads to 0
        for g in filter(lambda g: g.grad is not None, grads):
            g.grad.data[g.grad.data != g.grad.data] = 0
        torch.nn.utils.clip_grad_value_(grads, value)

    def forward(self, x):
        """ Apply the spatial learning, the average pooling and a final fully
        connected layer """
        out = self.net(x)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return func.log_softmax(out, dim=-1)
