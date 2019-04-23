"""
This script allows you to run a host of tests on the invariant layer and
slightly different variants of it on CIFAR.
"""
from shutil import copyfile
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import time
from dtcwt_gainlayer.layers.dtcwt import WaveConvLayer
from dtcwt_gainlayer.layers.dwt import WaveConvLayer as WaveConvLayer_dwt
import torch.nn.functional as func
import numpy as np
import random
from collections import OrderedDict
from dtcwt_gainlayer.data import cifar, tiny_imagenet
from dtcwt_gainlayer import optim
from math import sqrt
from tune_trainer import BaseClass, get_hms
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('outdir', type=str, help='experiment directory')
parser.add_argument('-C', type=int, default=96, help='number channels')
parser.add_argument('--dwt', action='store_true', help='use the dwt')
parser.add_argument('--seed', type=int, default=None, metavar='S',
                    help='random seed (default: None)')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--smoke-test', action="store_true",
                    help="Finish quickly for testing")
parser.add_argument('--datadir', type=str, default='/scratch/share/cifar',
                    help='Default location for the dataset')
parser.add_argument('--dataset', default='cifar100', type=str,
                    help='which dataset to use',
                    choices=['cifar10', 'cifar100', 'tiny_imagenet'])
parser.add_argument('--trainsize', default=-1, type=int,
                    help='size of training set')
parser.add_argument('--no-comment', action='store_true',
                    help='Turns off prompt to enter comments about run.')
parser.add_argument('--nsamples', type=int, default=0,
                    help='The number of runs to test.')
parser.add_argument('--exist-ok', action='store_true',
                    help='If true, is ok if output directory already exists')
parser.add_argument('--epochs', default=120, type=int, help='num epochs')
parser.add_argument('--cpu', action='store_true', help='Do not run on gpus')
parser.add_argument('--num-gpus', type=float, default=0.5)
parser.add_argument('--no-scheduler', action='store_true')
parser.add_argument('--type', default=None, type=str, nargs='+',
                    help='''Model type(s) to build. If left blank, will run 14
networks consisting of those defined by the dictionary "nets" (0, 1, or 2
invariant layers at different depths). Can also specify to run "nets1" or
"nets2", which swaps out the invariant layers for other iterations.
Alternatively can directly specify the layer name, e.g. "invA", or "invB2".''')

# Core hyperparameters
parser.add_argument('--reg', default='l2', type=str, help='regularization term')
parser.add_argument('--steps', default=[60,80,100], type=int, nargs='+')
parser.add_argument('--gamma', default=0.2, type=float, help='Lr decay')


def net_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0) or (classname.find('Linear') == 0):
        init.xavier_uniform_(m.weight, gain=sqrt(2))
        try:
            init.constant_(m.bias, 0)
        # Can get an attribute error if no bias to learn
        except AttributeError:
            pass


class MyModule(nn.Module):
    """ This is a wrapper for our networks that has some useful functions"""
    def __init__(self, dataset):
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

    def init(self, std=1):
        """ Define the default initialization scheme """
        self.apply(net_init)
        # Try do any custom layer initializations
        for child in self.net.modules():
            try:
                child.init(std)
            except AttributeError:
                pass

    def forward(self, x):
        """ Define the default forward pass"""
        out = self.net(x)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return func.log_softmax(out, dim=-1)


# Define the options of networks.
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


class NetBuilder(MyModule):
    """ NetBuilder allows custom definition of conv/inv layers as you would
    a normal network. You can change the ordering below to suit your
    task
    """
    def __init__(self, dataset, type, use_dwt=False, num_channels=96):
        super().__init__(dataset)
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


class TrainNET(BaseClass):
    """ This class handles model training and scheduling for our mnist networks.

    The config dictionary setup in the main function defines how to build the
    network. Then the experiment handler calles _train and _test to evaluate
    networks one epoch at a time.

    If you want to call this without using the experiment, simply ensure
    config is a dictionary with keys::

        - args: The parser arguments
        - type: The network type, a letter value between 'A' and 'N'. See above
            for what this represents.
        - lr (optional): the learning rate
        - momentum (optional): the momentum
        - wd (optional): the weight decay
        - std (optional): the initialization variance
    """
    def _setup(self, config):
        args = config.pop("args")
        vars(args).update(config)
        type_ = config.get('type', 'gainA')
        use_dwt = config.get('dwt', args.dwt)
        C = config.get('C', args.C)
        dataset = config.get('dataset', args.dataset)
        if hasattr(args, 'verbose'):
            self._verbose = args.verbose

        if args.seed is not None:
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            if self.use_cuda:
                torch.cuda.manual_seed(args.seed)

        # ######################################################################
        #  Data
        kwargs = {'num_workers': 0, 'pin_memory': True} if self.use_cuda else {}
        if dataset.startswith('cifar'):
            self.train_loader, self.test_loader = cifar.get_data(
                32, args.datadir, dataset=dataset,
                batch_size=args.batch_size, trainsize=args.trainsize,
                seed=args.seed, **kwargs)
        elif dataset == 'tiny_imagenet':
            self.train_loader, self.test_loader = tiny_imagenet.get_data(
                64, args.datadir, val_only=False,
                batch_size=args.batch_size, trainsize=args.trainsize,
                seed=args.seed, distributed=False, **kwargs)

        # ######################################################################
        # Build the network based on the type parameter. θ are the optimal
        # hyperparameters found by cross validation.
        if type_.startswith('ref'):
            θ = (0.1, 0.9, 1e-4, 1)
        elif type_ in nets.keys():
            θ = (0.1, 0.85, 1e-4, 0.5)
        else:
            θ = (0.45, 0.8, 1e-4, 1)
            #  raise ValueError('Unknown type')
        lr, mom, wd, q = θ
        # If the parameters were provided as an option, use them
        lr = config.get('lr', lr)
        mom = config.get('mom', mom)
        wd = config.get('wd', wd)
        q = config.get('q', q)
        std = config.get('std', 1.0)

        # Build the network
        self.model = NetBuilder(dataset, type_, use_dwt, C)
        self.model.init(std)

        # Split across GPUs
        if torch.cuda.device_count() > 1 and args.num_gpus > 1:
            self.model = nn.DataParallel(self.model)
            model = self.model.module
        else:
            model = self.model
        if self.use_cuda:
            self.model.cuda()

        # ######################################################################
        # Build the optimizer - use separate parameter groups for the gain
        # and convolutional layers
        default_params = {'params': list(model.fc1.parameters()),
                          'lr': lr, 'mom': mom, 'wd': wd}
        gain_params = {'params': [], 'lr': lr, 'mom': mom, 'wd': wd}
        for name, module in model.net.named_children():
            if name.startswith('gain'):
                gain_params['params'] += list(module.parameters())
            else:
                default_params['params'] += list(module.parameters())

        self.optimizer, self.scheduler = optim.get_optim(
            'sgd', [default_params, gain_params], init_lr=0.1,
            steps=args.steps, wd=1e-4, gamma=args.gamma, momentum=0.9,
            max_epochs=args.epochs)

        if self.verbose:
            print(self.model)


def linear_func(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    b = y1 - m*x1
    return m, b


if __name__ == "__main__":
    args = parser.parse_args()

    # If we don't use a scheduler, just train 1 network in a simple loop
    if args.no_scheduler:
        # Create reporting objects
        args.verbose = True
        outdir = os.path.join(os.environ['HOME'], 'nonray_results', args.outdir)
        tr_writer = SummaryWriter(os.path.join(outdir, 'train'))
        val_writer = SummaryWriter(os.path.join(outdir, 'val'))
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        # Copy this source file to the output directory for record keeping
        copyfile(__file__, os.path.join(outdir, 'search.py'))

        # Choose the model to run and build it
        if args.type is None:
            type_ = 'ref2'
        else:
            type_ = args.type[0]
        cfg = {'args': args, 'type': type_}
        trn = TrainNET(cfg)
        trn._final_epoch = args.epochs

        # Train for set number of epochs
        elapsed_time = 0
        best_acc = 0
        for epoch in range(trn.final_epoch):
            print("\n| Training Epoch #{}".format(epoch))
            print('| Learning rate: {}'.format(
                trn.optimizer.param_groups[0]['lr']))
            print('| Momentum : {}'.format(
                trn.optimizer.param_groups[0]['momentum']))
            start_time = time.time()
            # Update the scheduler
            trn.step_lr()

            # Train for one iteration and update
            trn_results = trn._train_iteration()
            tr_writer.add_scalar('loss', trn_results['mean_loss'], epoch)
            tr_writer.add_scalar('acc', trn_results['mean_accuracy'], epoch)
            tr_writer.add_scalar('acc5', trn_results['acc5'], epoch)

            # Validate
            val_results = trn._test()
            val_writer.add_scalar('loss', val_results['mean_loss'], epoch)
            val_writer.add_scalar('acc', val_results['mean_accuracy'], epoch)
            val_writer.add_scalar('acc5', val_results['acc5'], epoch)
            acc = val_results['mean_accuracy']
            if acc > best_acc:
                print('| Saving Best model...\t\t\tTop1 = {:.2f}%'.format(acc))
                trn._save(outdir, 'model_best.pth')
                best_acc = acc

            trn._save(outdir, name='model_last.pth')
            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
            print('| Elapsed time : %d:%02d:%02d\t Epoch time: %.1fs' % (
                  get_hms(elapsed_time) + (epoch_time,)))

    # We are using a scheduler
    else:
        # Create the training object
        args.verbose = False
        import ray
        from ray import tune
        from ray.tune.schedulers import AsyncHyperBandScheduler
        ray.init()
        exp_name = args.outdir
        outdir = os.path.join(os.environ['HOME'], 'ray_results', exp_name)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        # Copy this source file to the output directory for record keeping
        copyfile(__file__, os.path.join(outdir, 'search.py'))

        # Build the scheduler
        sched = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            reward_attr="neg_mean_loss",
            max_t=200,
            grace_period=120)

        # Select which networks to run
        if args.type is not None:
            if len(args.type) == 1 and args.type[0] == 'nets':
                type_ = list(nets.keys()) + ['ref']
            else:
                type_ = args.type
        else:
            type_ = list(nets.keys()) + ['ref',]

        tune.run_experiments(
            {
                exp_name: {
                    "stop": {
                        #  "mean_accuracy": 0.95,
                        "training_iteration": (1 if args.smoke_test
                                               else args.epochs),
                    },
                    "resources_per_trial": {
                        "cpu": 1,
                        "gpu": 0 if args.cpu else args.num_gpus
                    },
                    "run": TrainNET,
                    "num_samples": 10 if args.nsamples == 0 else args.nsamples,
                    "checkpoint_at_end": True,
                    "config": {
                        "args": args,
                        "dataset": args.dataset,
                        "type": tune.grid_search(type_),
                        "dwt": args.dwt,
                        "C": args.C,
                    }
                }
            },
            verbose=1,
            scheduler=sched)
