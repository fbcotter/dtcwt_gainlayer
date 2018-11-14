import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from dtcwt_gainlayer.networks.module import MyModule


def net_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        try:
            init.constant_(m.bias, 0)
        except AttributeError:
            pass


class LeNet(MyModule):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        # don't need bias as we're using batch norm
        self.C1 = 48
        self.C2 = 16
        self.conv1 = nn.Conv2d(3, self.C1, 5, bias=False)
        self.conv2 = nn.Conv2d(self.C1, self.C2, 5, bias=False)
        self.bn1 = nn.BatchNorm2d(self.C1)
        self.bn2 = nn.BatchNorm2d(self.C2)
        self.fc1 = nn.Linear(self.C2*5*5, 120)
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

        return(out)
