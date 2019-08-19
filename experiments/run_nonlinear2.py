"""
This script allows you to run a host of tests on the invariant layer and
slightly different variants of it on CIFAR.
"""
from save_exp import save_experiment_info, save_acc
import argparse
import os
import torch
import torch.nn as nn
import time
from dtcwt_gainlayer.layers.dtcwt import WaveConvLayer
from dtcwt_gainlayer.layers.nonlinear import PassThrough

import torch.nn.functional as func
import numpy as np
import random
from collections import OrderedDict
from dtcwt_gainlayer.data import cifar, tiny_imagenet
from dtcwt_gainlayer import optim
from tune_trainer import BaseClass, get_hms, net_init
from tensorboardX import SummaryWriter
import py3nvml
from math import ceil

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('outdir', type=str, help='experiment directory')
parser.add_argument('-C', type=int, default=64, help='number channels')
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
parser.add_argument('--resume', action='store_true',
                    help='Rerun from a checkpoint')
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
                    help="Model type(s) to build")

# Core hyperparameters
parser.add_argument('--lr', default=0.5, type=float, help='learning rate')
parser.add_argument('--lr1', default=None, type=float, help='learning rate for gainlayer')
parser.add_argument('--mom', default=0.85, type=float, help='momentum')
parser.add_argument('--mom1', default=None, type=float, help='momentum for gainlayer')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--wd1', default=1e-5, type=float, help='l1 weight decay')
parser.add_argument('--reg', default='l2', type=str, help='regularization term')
parser.add_argument('--steps', default=[60,80,100], type=int, nargs='+')
parser.add_argument('--gamma', default=0.2, type=float, help='Lr decay')
parser.add_argument('--opt1', default='sgd', type=str, help='gainlayer opt')
parser.add_argument('--pixel-nl', default='relu', type=str)
parser.add_argument('--lp-nl', default='none', type=str)
parser.add_argument('--bp-nl', default='none', type=str)
parser.add_argument('--lp-q', default=0.8, type=float)
parser.add_argument('--bp-q', default=0.8, type=float)


nets = {
    'ref': ['conv', 'pool', 'conv', 'pool', 'conv'],
    'waveA': ['gain', 'pool', 'conv', 'pool', 'conv'],
    'waveB': ['conv', 'pool', 'gain', 'pool', 'conv'],
    'waveC': ['conv', 'pool', 'conv', 'pool', 'gain'],
    'waveD': ['gain', 'pool', 'gain', 'pool', 'conv'],
    'waveE': ['conv', 'pool', 'gain', 'pool', 'gain'],
    'waveF': ['gain', 'pool', 'gain', 'pool', 'gain'],
}


class MixedNet(nn.Module):
    """ MixedNet allows custom definition of conv/inv layers as you would
    a normal network. You can change the ordering below to suit your
    task
    """
    def __init__(self, dataset, type, num_channels=64, wd=1e-4, wd1=None,
                 pixel_nl='none', lp_nl='relu', bp_nl='relu2',
                 lp_nl_kwargs={}, bp_nl_kwargs={}):
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

        # Call the DTCWT conv layer
        WaveLayer = lambda C1, C2: WaveConvLayer(
            C1, C2, 3, (1,), wd=wd, wd1=wd1, lp_nl=lp_nl, bp_nl=(bp_nl,),
            lp_nl_kwargs=lp_nl_kwargs, bp_nl_kwargs=bp_nl_kwargs)

        if pixel_nl == 'relu':
            σ_pixel = lambda C: nn.Sequential(
                nn.BatchNorm2d(C),
                nn.ReLU())
        elif pixel_nl == 'none' and lp_nl == 'relu2':
            σ_pixel = lambda C: PassThrough()
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
                C1 = C2
                layer += 1
            elif blk == 'gain':
                name = 'wave' + chr(ord('A') + layer)
                blk = nn.Sequential(WaveLayer(C1, C2), σ_pixel(C2))
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
        elif dataset == 'tiny_imagenet':
            blk1 = nn.MaxPool2d(2)
            blk2 = nn.Sequential(
                nn.Conv2d(4*C, 8*C, 5, padding=2, stride=1),
                nn.BatchNorm2d(8*C),
                nn.ReLU())
            blks = blks + [
                ('pool3', blk1),
                ('conv_final', blk2),]
                #  ('convH', blk3)]
            self.net = nn.Sequential(OrderedDict(blks))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(8*C, self.num_classes)

    def get_reg(self):
        loss = 0
        for name, m in self.net.named_children():
            if name.startswith('wave'):
                loss += m[0].GainLayer.get_reg()
            elif name.startswith('conv'):
                loss += 0.5 * self.wd * torch.sum(m[0].weight**2)
        loss += 0.5 * self.wd * torch.sum(self.fc1.weight**2)
        return loss

    def clip_grads(self, value=1):
        grads = []
        for name, m in self.net.named_children():
            if name.startswith('wave'):
                grads.extend([g for g in m[0].GainLayer.g])
        # Set nans in grads to 0
        for g in filter(lambda g: g.grad is not None, grads):
            g.grad.data[g.grad.data != g.grad.data] = 0
        torch.nn.utils.clip_grad_value_(grads, value)

    def forward(self, x):
        """ Define the default forward pass"""
        out = self.net(x)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return func.log_softmax(out, dim=-1)


class TrainNET(BaseClass):
    """ This class handles model training and scheduling for our mnist networks.

    The config dictionary setup in the main function defines how to build the
    network. Then the experiment handler calles _train and _test to evaluate
    networks one epoch at a time.

    If you want to call this without using the experiment, simply ensure
    config is a dictionary with keys::

        - args: The parser arguments
    """
    def _setup(self, config):
        args = config.pop("args")
        vars(args).update(config)
        # Parameters like learning rate and momentum can be speicified by the
        # config search space. If not specified, fall back to the args
        type = config.get('type', 'ref')
        lr = config.get('lr', args.lr)
        mom = config.get('mom', args.mom)
        wd = config.get('wd', args.wd)
        C = config.get('num_channels', args.C)
        dataset = config.get('dataset', args.dataset)
        num_gpus = config.get('num_gpus', args.num_gpus)

        # Get optimizer parameters for gainlayer
        mom1 = config.get('mom1', args.mom1)
        lr1 = config.get('lr1', args.lr1)
        wd1 = config.get('wd1', args.wd1)
        opt1 = config.get('opt1', args.opt1)
        gamma1 = config.get('gamma1', 0.2)
        if mom1 is None:
            mom1 = mom
        if lr1 is None:
            lr1 = lr

        # Get nonlinearity options
        pixel_nl = config.get('pixel_nl', args.pixel_nl)
        lp_nl = config.get('lp_nl', args.lp_nl)
        bp_nl = config.get('bp_nl', args.bp_nl)
        lp_q = config.get('lp_q', args.lp_q)
        bp_q = config.get('bp_q', args.bp_q)
        lp_thresh = config.get('lp_thresh', 1)
        bp_thresh = config.get('bp_thresh', 1)

        if hasattr(args, 'verbose'):
            self._verbose = args.verbose

        num_workers = 4
        if args.seed is not None:
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            num_workers = 0
            if self.use_cuda:
                torch.cuda.manual_seed(args.seed)
        else:
            args.seed = random.randint(0, 10000)

        # ######################################################################
        #  Data
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if self.use_cuda else {}
        if dataset.startswith('cifar'):
            self.train_loader, self.test_loader = cifar.get_data(
                32, args.datadir, dataset=dataset,
                batch_size=args.batch_size, trainsize=args.trainsize,
                **kwargs)
        elif dataset == 'tiny_imagenet':
            self.train_loader, self.test_loader = tiny_imagenet.get_data(
                64, args.datadir, val_only=False,
                batch_size=args.batch_size, trainsize=args.trainsize,
                distributed=False, **kwargs)

        # ######################################################################
        # Build the network based on the type parameter. θ are the optimal
        # hyperparameters found by cross validation.

        # Build the network
        self.model = MixedNet(dataset, type, C, wd, wd1,
                              pixel_nl=pixel_nl, lp_nl=lp_nl, bp_nl=bp_nl,
                              lp_nl_kwargs=dict(q=lp_q, thresh=lp_thresh),
                              bp_nl_kwargs=dict(q=bp_q, thresh=bp_thresh))
        init = lambda x: net_init(x, 1.0)
        self.model.apply(init)

        # Split across GPUs
        if torch.cuda.device_count() > 1 and num_gpus > 1:
            self.model = nn.DataParallel(self.model)
            model = self.model.module
        else:
            model = self.model
        if self.use_cuda:
            self.model.cuda()

        # ######################################################################
        # Build the optimizer - use separate parameter groups for the gain
        # and convolutional layers
        default_params = list(model.fc1.parameters())
        gain_params = []
        for name, module in model.net.named_children():
            params = [p for p in module.parameters() if p.requires_grad]
            if name.startswith('wave'):
                gain_params += params
            else:
                default_params += params

        self.optimizer, self.scheduler = optim.get_optim(
            'sgd', default_params, init_lr=lr,
            steps=args.steps, wd=0, gamma=0.2, momentum=mom,
            max_epochs=args.epochs)

        if len(gain_params) > 0:
            # Do not use the optimizer's weight decay, call a special method to
            # do it.
            self.optimizer1, self.scheduler1 = optim.get_optim(
                opt1, gain_params, init_lr=lr1,
                steps=args.steps, wd=0, gamma=gamma1, momentum=mom1,
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
        outdir = os.path.join(os.environ['HOME'], 'gainlayer_results', args.outdir)
        tr_writer = SummaryWriter(os.path.join(outdir, 'train'))
        val_writer = SummaryWriter(os.path.join(outdir, 'val'))
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.type is None:
            type = 'ref'
        else:
            type = args.type[0]

        py3nvml.grab_gpus(ceil(args.num_gpus))
        cfg = {'args': args, 'type': type, 'num_gpus': args.num_gpus,
               'C': args.C, 'lr': args.lr, 'lr1': args.lr1, 'mom': args.mom,
               'mom1': args.mom1, 'wd': args.wd, 'wd1': args.wd1,
               'opt1': args.opt1, 'pixel_nl': args.pixel_nl,
               'lp_nl': args.lp_nl, 'bp_nl': args.bp_nl}
        trn = TrainNET(cfg)
        trn._final_epoch = args.epochs

        # Copy this source file to the output directory for record keeping
        if args.resume:
            trn._restore(os.path.join(outdir, 'model_last.pth'))
        else:
            save_experiment_info(outdir, args.seed, args.no_comment, trn.model)

        if args.seed is not None and trn.use_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Train for set number of epochs
        elapsed_time = 0
        best_acc = 0
        trn.step_lr()
        for epoch in range(trn.last_epoch, trn.final_epoch):
            print("\n| Training Epoch #{}".format(epoch))
            print('| Learning rate: {}'.format(
                trn.optimizer.param_groups[0]['lr']))
            print('| Momentum : {}'.format(
                trn.optimizer.param_groups[0]['momentum']))
            start_time = time.time()

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
            # Update the scheduler
            trn.step_lr()

        save_acc(outdir, best_acc, acc)
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
        save_experiment_info(outdir, args.seed, args.no_comment)

        # Build the scheduler
        sched = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            reward_attr="neg_mean_loss",
            max_t=200,
            grace_period=120)

        # Select which networks to run
        if args.type is not None:
            if len(args.type) == 1 and args.type[0] == 'nets':
                type = list(nets.keys())
            else:
                type = args.type
        else:
            type = list(nets.keys())

        tune.run(
            TrainNET,
            name=exp_name,
            scheduler=sched,
            stop={#  "mean_accuracy": 0.95,
                  "training_iteration": (1 if args.smoke_test else args.epochs),
            },
            resources_per_trial={
                "cpu": 1,
                "gpu": 0 if args.cpu else args.num_gpus
            },
            num_samples=(10 if args.nsamples == 0 else args.nsamples),
            checkpoint_at_end=True,
            config={
                "args": args, "dataset": args.dataset,
                "type": tune.grid_search(type),
                "lr": 0.5, "mom": 0.85, "wd": 3e-4, "wd1": 1e-5,
                "num_channels": 64,
                #  "pixel_nl": 'none',
                #  "lp_nl": 'relu',
                #  "bp_nl": 'relu2',
                "pixel_nl": tune.grid_search(['none', 'relu']),
                "lp_nl": tune.grid_search(['none', 'relu', 'relu2', 'softshrink']),
                "bp_nl": tune.grid_search(['none', 'relu', 'relu2', 'softshrink'])
            },
            verbose=1)
