"""
This script allows you to run a host of tests on the invariant layer and
slightly different variants of it on MNIST.
"""
import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
from torchvision import datasets, transforms
from dtcwt_gainlayer.layers.dtcwt import WaveGainLayer, WaveConvLayer
import time
import torch.nn.functional as func
import numpy as np
import random
from tune_trainer import BaseClass, get_hms

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('outdir', type=str, help='experiment directory')
parser.add_argument('--seed', type=int, default=None, metavar='S',
                    help='random seed (default: None)')
parser.add_argument('--no-scheduler', action='store_true')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--smoke-test', action="store_true",
                    help="Finish quickly for testing")
parser.add_argument('--type', default=None, type=str, nargs='+',
                    help='''Model type(s) to build. If left blank, will run the
standard gainlayer''')


class WaveGainNet(nn.Module):
    def __init__(self, C1=7, C2=49, q=1.):
        super().__init__()
        self.conv1 = WaveConvLayer(1, C1, 3, (1,), q)
        self.conv2 = WaveConvLayer(C1, C2, 3, (1,), q)

        # Create the projection layer that doesn't need learning
        self.fc1 = nn.Linear(7*7*C2, 10)
        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc1.bias.data.zero_()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = func.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = func.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        y = func.relu(self.fc1(x))
        y = self.fc2(y)
        return func.log_softmax(y, dim=1)

    def init(self, std=1):
        for child in self.children():
            try:
                child.init(std)
            except AttributeError:
                pass


class ConvNet(nn.Module):
    def __init__(self, C1=7, C2=49, k=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, C1, k, 1, padding=(k-1)//2)
        self.conv2 = nn.Conv2d(C1, C2, k, 1, padding=(k-1)//2)
        self.fc1 = nn.Linear(7*7*C2, 10)
        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        y = func.relu(self.fc1(x))
        y = self.fc2(y)
        return func.log_softmax(y, dim=1)

    def init(self, std=1):
        for child in self.children():
            classname = child.__class__.__name__
            if classname.find('Conv') != -1:
                init.xavier_uniform_(child.weight, gain=std)


class TrainNET(BaseClass):
    """ This class handles model training and scheduling for our mnist networks.

    The config dictionary setup in the main function defines how to build the
    network. Then the experiment handler calles _train and _test to evaluate
    networks one epoch at a time.

    If you want to call this without using the experiment, simply ensure
    config is a dictionary with keys::

        - args: The parser arguments
        - type: The network type, one of
        - lr (optional): the learning rate
        - momentum (optional): the momentum
        - wd (optional): the weight decay
        - std (optional): the initialization variance
    """
    def _setup(self, config):
        args = config.pop("args")
        vars(args).update(config)
        type_ = config.get('type')
        args.cuda = torch.cuda.is_available()
        if hasattr(args, 'verbose'):
            self._verbose = args.verbose

        if args.seed is not None:
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.cuda:
                torch.cuda.manual_seed(args.seed)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                '~/data',
                train=True,
                download=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307, ), (0.3081, ))
                ])),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                '~/data',
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307, ), (0.3081, ))
                ])),
            batch_size=100,
            shuffle=True,
            **kwargs)

        # Build the network based on the type parameter. θ are the optimal
        # hyperparameters found by cross validation.
        C1 = 7
        C2 = 49
        if type_ == 'conv':
            self.model = ConvNet(C1, C2)
            θ = (0.1, 0.5, 1e-5, 1)
        elif type_ == 'gain':
            q = 1.0
            q = config.get('q', q)
            self.model = WaveGainNet(C1, C2, q=q)
            θ = (0.032, 0.9, 1e-4, 1)
        else:
            raise ValueError('Unknown type')

        lr, mom, wd, std = θ
        # If the parameters were provided as an option, use them
        lr = config.get('lr', lr)
        mom = config.get('mom', mom)
        wd = config.get('wd', wd)
        std = config.get('std', std)
        self.model.init(std)
        self.model.cuda()

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
        self.args = args
        if self.verbose:
            print(self.model)


if __name__ == "__main__":
    datasets.MNIST('~/data', train=True, download=True)
    args = parser.parse_args()

    if args.no_scheduler:
        args.verbose = True
        outdir = os.path.join(os.environ['HOME'], 'nonray_results', args.outdir)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        if args.type is None:
            type_ = 'gain'
        else:
            type_ = args.type[0]
        cfg = {'args': args, 'type': type_}
        trn = TrainNET(cfg)
        elapsed_time = 0

        best_acc = 0
        for epoch in range(20):
            print('| Learning rate: {}'.format(trn.optimizer.param_groups[0]['lr']))
            print('| Momentum : {}'.format(trn.optimizer.param_groups[0]['momentum']))
            start_time = time.time()
            trn._train()
            results = trn._test()
            acc1 = results['mean_accuracy']
            if acc1 > best_acc:
                print('| Saving Best model...\t\t\tTop1 = {:.2f}%'.format(acc1))
                trn._save(outdir)
            best_acc = acc1

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d\t Epoch time: %.1fs' % (
              get_hms(elapsed_time) + (epoch_time,)))

    else:

        args.verbose = False
        import ray
        from ray import tune
        from ray.tune.schedulers import AsyncHyperBandScheduler
        from shutil import copyfile

        ray.init()
        exp_name = args.outdir
        outdir = os.path.join(os.environ['HOME'], 'ray_results', exp_name)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        # Copy this source file to the output directory for record keeping
        copyfile(__file__, os.path.join(outdir, 'search.py'))

        sched = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            reward_attr="neg_mean_loss",
            max_t=80,
            grace_period=5)

        tune.run_experiments(
            {
                exp_name: {
                    "stop": {
                        #  "mean_accuracy": 0.95,
                        "training_iteration": 1 if args.smoke_test else 20,
                    },
                    "resources_per_trial": {
                        "cpu": 1,
                        "gpu": 0.3,
                    },
                    "run": TrainNET,
                    #  "num_samples": 1 if args.smoke_test else 40,
                    "num_samples": 3,
                    "checkpoint_at_end": True,
                    "config": {
                        "args": args,
                        "type": tune.grid_search([
                            'gain']),
                        #  "type": tune.grid_search(['conv_wide']),
                        "lr": tune.grid_search([0.0316, 0.1]),
                        "mom": tune.grid_search([0.7, 0.9]),
                        "wd": tune.grid_search([1e-5, 1e-4]),
                        "q": tune.grid_search([0.3, 0.5, 0.8, 1.]),
                        #  "std": tune.grid_search([0.5, 1., 1.5, 2.0])
                    }
                }
            },
            verbose=1,
            scheduler=sched)
