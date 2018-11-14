# Fergal Cotter
#

# Future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def log_info(self, *args, **kwargs):
        pass

    def burn_in(self, *args, **kwargs):
        pass

    def after_update(self, *args, **kwargs):
        pass

    @staticmethod
    def mag(a):
        if a.shape[-1] != 2:
            return torch.abs(a)
        else:
            return torch.sqrt(a[..., 0]**2 + a[..., 1]**2)
