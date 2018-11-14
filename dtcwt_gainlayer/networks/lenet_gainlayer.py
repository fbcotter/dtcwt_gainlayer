import torch.nn as nn
import torch.nn.functional as F
from dtcwt_gainlayer.layers.dtcwt import WaveConvLayer
from dtcwt_gainlayer.networks.module import MyModule


def net_init(m):
    pass


class LeNet_GainLayer(MyModule):
    def __init__(self, num_classes):
        self.C1 = 6
        self.C2 = 16
        super(LeNet_GainLayer, self).__init__()
        self.conv1 = WaveConvLayer(
            C=3, F=self.C1, lp_size=3, bp_sizes=(1, ))
        self.conv2 = WaveConvLayer(
            C=6, F=self.C2, lp_size=3, bp_sizes=(1, ))
        self.bn1 = nn.BatchNorm2d(self.C1)
        self.bn2 = nn.BatchNorm2d(self.C2)
        self.fc1 = nn.Linear(self.C2*7*7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out
