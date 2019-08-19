""" This module builds different networks with nonlinearities
"""
import torch
import torch.nn as nn
import torch.nn.functional as func
from dtcwt_gainlayer import WaveConvLayer
from collections import OrderedDict

nets = {
    'ref': ['conv', 'pool', 'conv', 'pool', 'conv'],
    'waveA': ['gain', 'pool', 'conv', 'pool', 'conv'],
    'waveB': ['conv', 'pool', 'gain', 'pool', 'conv'],
    'waveC': ['conv', 'pool', 'conv', 'pool', 'gain'],
    'waveD': ['gain', 'pool', 'gain', 'pool', 'conv'],
    'waveE': ['conv', 'pool', 'gain', 'pool', 'gain'],
    'waveF': ['gain', 'pool', 'gain', 'pool', 'gain'],
}


class PassThrough(nn.Module):
    def forward(self, x):
        """ No nonlinearity """
        return x


class NonlinearNet(nn.Module):
    """ MixedNet allows custom definition of conv/inv layers as you would
    a normal network. You can change the ordering below to suit your
    task
    """
    def __init__(self, dataset, type, num_channels=64, wd=1e-4, wd1=None,
                 pixel_nl='none', lp_nl='relu', bp_nl='relu2'):
        super().__init__()

        # Define the number of scales and classes dependent on the dataset
        if dataset == 'cifar10':
            self.num_classes = 10
        elif dataset == 'cifar100':
            self.num_classes = 100
        elif dataset == 'tiny_imagenet':
            self.num_classes = 200
        self.wd = wd
        self.wd1 = wd1
        C = num_channels
        self._wave_params = []
        self._default_params = []

        WaveLayer = lambda C1, C2: WaveConvLayer(
            C1, C2, 3, (1,), wd=wd, wd1=wd1, lp_nl=lp_nl, bp_nl=(bp_nl,))

        if pixel_nl == 'relu':
            σ_pixel = lambda C: nn.Sequential(
                nn.BatchNorm2d(C),
                nn.ReLU())
        else:
            σ_pixel = lambda C: nn.BatchNorm2d(C)

        layers = nets[type]
        blks = []
        # A letter counter for the layer number
        layer = 0
        # The number of input (C1) and output (C2) channels. The channels double
        # after a pooling layer
        C1 = 3
        C2 = num_channels
        # A number for the pooling layer
        pool = 1
        for blk in layers:
            if blk == 'conv':
                name = 'conv' + chr(ord('A') + layer)
                blk = nn.Sequential(
                    nn.Conv2d(C1, C2, 5, padding=2, stride=1),
                    nn.BatchNorm2d(C2),
                    nn.ReLU())
                self._default_params.extend(list(blk.parameters()))
                C1 = C2
                layer += 1
            elif blk == 'gain':
                name = 'wave' + chr(ord('A') + layer)
                blk = nn.Sequential(WaveLayer(C1, C2), σ_pixel(C2))
                self._wave_params.extend(list(blk.parameters()))
                C1 = C2
                layer += 1
            elif blk == 'pool':
                name = 'pool' + str(pool)
                blk = nn.MaxPool2d(2)
                pool += 1
                C2 = 2*C1
            blks.append((name, blk))

        # F is the last output size from first 6 layers
        if dataset == 'cifar10' or dataset == 'cifar100':
            # Network is 3 stages of convolution
            self.net = nn.Sequential(OrderedDict(blks))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(4*C, self.num_classes)
            self._default_params.extend(list(self.fc1.parameters()))
        elif dataset == 'tiny_imagenet':
            blk1 = nn.MaxPool2d(2)
            blk2 = nn.Sequential(
                nn.Conv2d(4*C, 8*C, 5, padding=2, stride=1),
                nn.BatchNorm2d(8*C),
                nn.ReLU())
            self._default_params.extend(list(blk2[0].parameters()))
            blks = blks + [
                ('pool3', blk1),
                ('conv_final', blk2),]
            self.net = nn.Sequential(OrderedDict(blks))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(8*C, self.num_classes)
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
