"""
This script allows you to run a host of tests on the invariant layer and
slightly different variants of it on CIFAR.
"""
from shutil import copyfile
import argparse
import os
import torch
import torch.nn as nn
import time
from dtcwt_gainlayer.layers.dtcwt import WaveConvLayer
from dtcwt_gainlayer.layers.dwt import WaveConvLayer as WaveConvLayer_dwt
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
parser.add_argument('-C', type=int, default=32, help='number channels')
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
                    help='''Model type(s) to build. If left blank, will run 14
networks consisting of those defined by the dictionary "nets" (0, 1, or 2
invariant layers at different depths). Can also specify to run "nets1" or
"nets2", which swaps out the invariant layers for other iterations.
Alternatively can directly specify the layer name, e.g. "invA", or "invB2".''')

# Core hyperparameters
parser.add_argument('--lr', default=0.45, type=float, help='learning rate')
parser.add_argument('--mom', default=0.8, type=float, help='momentum')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--reg', default='l2', type=str, help='regularization term')
parser.add_argument('--steps', default=[60,80,100], type=int, nargs='+')
parser.add_argument('--gamma', default=0.2, type=float, help='Lr decay')
parser.add_argument('-q', default=1, type=float,
                    help='proportion of activations to keep')


# Define the options of networks.
# 'gain' = xfm + gain + inv + nonlinear
# 'gain1' = xfm + gain + nonlinear
# 'gain2' = gain + nonlinear
# 'gain3' = gain + nonlinear + inv
# 'gain4' = gain + inv + nonlinear
nets = {
    'ref': ['conv', 'conv', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv'],
    'type1': ['gain', 'conv', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv'],
    'type2': ['conv', 'gain', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv'],
    'type3': ['gain1', 'gain3', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv'],
    'type4': ['gain1', 'gain4', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv'],
    'type5': ['gain1', 'gain2', 'gain3', 'pool', 'conv', 'pool', 'conv', 'conv'],
    'type6': ['gain1', 'gain2', 'gain4', 'pool', 'conv', 'pool', 'conv', 'conv'],
    'type7': ['gain1', 'gain3', 'pool', 'gain1', 'gain3', 'pool', 'conv', 'conv'],
}


class MixedNet(nn.Module):
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
        layer_num = 0
        # The number of input (C1) and output (C2) channels. The channels double
        # after a pooling layer
        C1 = 3
        C2 = num_channels
        # A number for the pooling layer
        pool = 1

        # Call the DWT or the DTCWT conv layer
        if use_dwt:
            WaveLayer = lambda x, y, q, xfm, ifm: WaveConvLayer_dwt(
                x, y, 3, (1,), q=q, xfm=xfm, ifm=ifm)
        else:
            WaveLayer = lambda x, y, q, xfm, ifm: WaveConvLayer(
                x, y, 1, (1,), q=q, xfm=xfm, ifm=ifm)

        for layer in layers:
            if layer == 'conv' or layer.startswith('gain'):
                if layer == 'conv':
                    # Add a triple of layers for each convolutional layer
                    blk = nn.Sequential(
                        nn.Conv2d(C1, C2, 3, padding=1, stride=1),
                        nn.BatchNorm2d(C2),
                        nn.ReLU())
                elif layer == 'gain':
                    blk = nn.Sequential(
                        WaveLayer(C1, C2, 1.0, True, True),
                        nn.ReLU())
                elif layer == 'gain1':
                    blk = nn.Sequential(
                        WaveLayer(C1, C2, q, True, False))
                elif layer == 'gain2':
                    blk = nn.Sequential(
                        WaveLayer(C1, C2, q, False, False))
                elif layer == 'gain3':
                    blk = nn.Sequential(
                        WaveLayer(C1, C2, q, False, True))
                elif layer == 'gain4':
                    blk = nn.Sequential(
                        WaveLayer(C1, C2, 1.0, False, True),
                        nn.ReLU())
                else:
                    raise ValueError
                name = layer + chr(ord('A') + layer_num)
                # The next layer's input channels becomes this layer's output
                # channels
                C1 = C2
                # Increase the layer counter
                layer_num += 1
            elif layer == 'pool':
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
        use_dwt = config.get('dwt', False)
        C = config.get('C', 96)
        dataset = config.get('dataset', args.dataset)
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
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # ######################################################################
        #  Data
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if self.use_cuda else {}
        if args.dataset.startswith('cifar'):
            self.train_loader, self.test_loader = cifar.get_data(
                32, args.datadir, dataset=dataset,
                batch_size=args.batch_size, trainsize=args.trainsize,
                **kwargs)
        elif args.dataset == 'tiny_imagenet':
            self.train_loader, self.test_loader = tiny_imagenet.get_data(
                64, args.datadir, val_only=False,
                batch_size=args.batch_size, trainsize=args.trainsize,
                distributed=False, **kwargs)

        # ######################################################################
        # Build the network based on the type parameter. θ are the optimal
        # hyperparameters found by cross validation.
        if type_.startswith('ref'):
            θ = (0.1, 0.9, 1e-4, 1)
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
        self.model = MixedNet(args.dataset, type_, q, use_dwt, C)
        init = lambda x: net_init(x, std)
        self.model.apply(init)

        # Split across GPUs
        if torch.cuda.device_count() > 1 and config.get('num_gpus', 0) > 1:
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
            if name.startswith('gain'):
                gain_params += params
            else:
                default_params += params

        self.optimizer, self.scheduler = optim.get_optim(
            'sgd', default_params, init_lr=lr,
            steps=args.steps, wd=wd, gamma=0.2, momentum=mom,
            max_epochs=args.epochs)

        if len(gain_params) > 0:
            # Get special optimizer parameters
            lr1 = config.get('lr1', lr)
            gamma1 = config.get('gamma1', 0.2)
            mom1 = config.get('mom1', mom)
            wd1 = config.get('wd1', wd)

            self.optimizer1, self.scheduler1 = optim.get_optim(
                'sgd', gain_params, init_lr=lr1,
                steps=args.steps, wd=wd1, gamma=gamma1, momentum=mom1,
                max_epochs=args.epochs)



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

        # Choose the model to run and build it
        if args.type is None:
            type_ = 'ref2'
        else:
            type_ = args.type[0]
        py3nvml.grab_gpus(ceil(args.num_gpus))
        cfg = {'args': args, 'type': type_, 'num_gpus': args.num_gpus,
               'dwt': args.dwt, 'C': args.C,
               'lr': args.lr, 'mom': args.mom, 'wd': args.wd, 'q': args.q}
        trn = TrainNET(cfg)
        trn._final_epoch = args.epochs

        # Copy this source file to the output directory for record keeping
        if args.resume:
            trn._restore(os.path.join(outdir, 'model_last.pth'))
        else:
            save_experiment_info(outdir, args.seed, args.no_comment, trn.model)

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
                type_ = list(nets.keys())
            else:
                type_ = args.type
        else:
            type_ = list(nets.keys())

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
                        #  "lr": tune.sample_from(lambda spec: np.random.uniform(
                            #  0.1, 0.7
                        #  )),
                        #  "mom": tune.sample_from(
                            #  lambda spec: m*spec.config.lr + b +
                                #  0.05*np.random.randn()),
                        #  "wd": tune.sample_from(lambda spec: np.random.uniform(
                           #  1e-5, 5e-4
                        #  ))
                        "lr": tune.grid_search([0.45]),
                        "mom": tune.grid_search([0.8]),
                        "q": tune.grid_search([1]),
                        #  "wd": tune.grid_search([1e-5, 1e-1e-4]),
                        #  "std": tune.grid_search([0.5, 1.5])
                        "dwt": args.dwt,
                        "C": args.C,
                    }
                }
            },
            verbose=1,
            scheduler=sched)
