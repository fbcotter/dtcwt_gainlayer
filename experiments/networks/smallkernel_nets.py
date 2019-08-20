""" This module builds different networks with nonlinearities
"""
import torch
import torch.nn as nn
import torch.nn.functional as func
from dtcwt_gainlayer import WaveConvLayer
from collections import OrderedDict

nets = {
    'ref': ['conv', 'conv', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv'],
    'gainA': ['gain', 'conv', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv'],
    'gainB': ['conv', 'gain', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv'],
    'gainC': ['conv', 'conv', 'pool', 'gain', 'conv', 'pool', 'conv', 'conv'],
    'gainD': ['conv', 'conv', 'pool', 'conv', 'gain', 'pool', 'conv', 'conv'],
    'gainE': ['conv', 'conv', 'pool', 'conv', 'conv', 'pool', 'gain', 'conv'],
    'gainF': ['conv', 'conv', 'pool', 'conv', 'conv', 'pool', 'conv', 'gain'],
    'gainAB': ['gain', 'gain', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv'],
    'gainBC': ['conv', 'gain', 'pool', 'gain', 'conv', 'pool', 'conv', 'conv'],
    'gainCD': ['conv', 'conv', 'pool', 'gain', 'gain', 'pool', 'conv', 'conv'],
    'gainDE': ['conv', 'conv', 'pool', 'conv', 'gain', 'pool', 'gain', 'conv'],
    'gainAC': ['gain', 'conv', 'pool', 'gain', 'conv', 'pool', 'conv', 'conv'],
    'gainBD': ['conv', 'gain', 'pool', 'conv', 'gain', 'pool', 'conv', 'conv'],
    'gainCE': ['conv', 'conv', 'pool', 'gain', 'conv', 'pool', 'gain', 'conv'],
}


class GainLayerNet(nn.Module):
    """ MixedNet allows custom definition of conv/inv layers as you would
    a normal network. You can change the ordering below to suit your
    task
    """
    def __init__(self, dataset, type, q=1., use_dwt=False, num_channels=96):
        super().__init__()

        # Define the number of scales and classes dependent on the dataset
        if dataset == 'cifar10':
            self.num_classes = 10
            self.S = 3
        elif dataset == 'cifar100':
            self.num_classes = 100
            self.S = 3
        elif dataset == 'tiny_imagenet':
            self.num_classes = 200
            self.S = 4

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

        # Call the DWT or the DTCWT conv layer
        if use_dwt:
            WaveLayer = lambda x, y: WaveConvLayer_dwt(x, y, 3, (1,))
        else:
            WaveLayer = lambda x, y: WaveConvLayer(x, y, 1, (1,))

        for blk in layers:
            if blk == 'conv':
                name = 'conv' + chr(ord('A') + layer)
                # Add a triple of layers for each convolutional layer
                blk = nn.Sequential(
                    nn.Conv2d(C1, C2, 3, padding=1, stride=1),
                    nn.BatchNorm2d(C2),
                    nn.ReLU())
                # The next layer's input channels becomes this layer's output
                # channels
                C1 = C2
                # Increase the layer counter
                layer += 1
            elif blk == 'gain':
                name = 'gain' + chr(ord('A') + layer)
                blk = nn.Sequential(
                    WaveLayer(C1, C2),
                    nn.BatchNorm2d(C2),
                    nn.ReLU())
                C1 = C2
                layer += 1
            elif blk == 'pool':
                name = 'pool' + str(pool)
                blk = nn.MaxPool2d(2)
                pool += 1
                C2 = 2*C1
            # Add the name and block to the list
            blks.append((name, blk))

        # F is the last output size from first 6 layers
        if dataset == 'cifar10' or dataset == 'cifar100':
            # Network is 3 stages of convolution
            self.net = nn.Sequential(OrderedDict(blks))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(C2, self.num_classes)
        elif dataset == 'tiny_imagenet':
            blk1 = nn.MaxPool2d(2)
            blk2 = nn.Sequential(
                nn.Conv2d(C2, 2*C2, 3, padding=1, stride=1),
                nn.BatchNorm2d(2*C2),
                nn.ReLU())
            blk3 = nn.Sequential(
                nn.Conv2d(2*C2, 2*C2, 3, padding=1, stride=1),
                nn.BatchNorm2d(2*C2),
                nn.ReLU())
            blks = blks + [
                ('pool3', blk1),
                ('convG', blk2),
                ('convH', blk3)]
            self.net = nn.Sequential(OrderedDict(blks))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(2*C2, self.num_classes)

    def forward(self, x):
        """ Define the default forward pass"""
        out = self.net(x)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return func.log_softmax(out, dim=-1)


