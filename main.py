"""
Generic main file for creating/loading a neural network and training it.

    pre: parse arguments
    step 1: builds a model
    step 2: loads data
    step 3: creates an optimizer and gets the parameters to optimize
    step 4: loop through train/val functions
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import py3nvml
from tensorboardX import SummaryWriter
import time

import random

import os
import sys
import argparse
import numpy as np

from dtcwt_gainlayer.networks import getNetwork
from dtcwt_gainlayer import cifar
from dtcwt_gainlayer.utils import get_hms
from dtcwt_gainlayer.save_exp import save_experiment_info, save_acc
import dtcwt_gainlayer.learn as learn

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('exp_dir', type=str,
                    help='Output directory for the experiment')
parser.add_argument('--data_dir', default='DATA', type=str,
                    help='Directory in which to find the cifar10/100 data')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='lenet_gainlayer', type=str,
                    help='model')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true',
                    help='Test mode with the saved model')
parser.add_argument('--num_gpus', default=1, type=int,
                    help='number of gpus to use')
parser.add_argument('--summary_freq', default=4, type=int,
                    help='number of updates per epoch')
parser.add_argument('--wd', default=1e-5, type=float,
                    help='weight decay')
parser.add_argument('--optim', default='sgd', type=str,
                    help='The optimizer to use')
parser.add_argument('--seed', default=-1, type=int,
                    help='random seed')
parser.add_argument('--verbose', action='store_true',
                    help='Make plots during training')
parser.add_argument('--epochs', default=200, type=int,
                    help='num epochs')
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch size')
parser.add_argument('--trainsize', default=50000, type=int,
                    help='size of training set')
parser.add_argument('--eval_period', default=1, type=int,
                    help='after how many train epochs to run validation')
parser.add_argument('--no_comment', action='store_true',
                    help='Turns off prompt to enter comments about run.')
parser.add_argument('--exist_ok', action='store_true',
                    help='If true, is ok if output directory already exists')
args = parser.parse_args()

# If seed was not provided, create one and seed numpy and pytorch
if args.seed < 0:
    args.seed = np.random.randint(1 << 16)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)

# Hyperparameter settings
py3nvml.grab_gpus(args.num_gpus, gpu_fraction=0.95)
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, batch_size = 1, args.batch_size


# ##############################################################################
#  Model
print('\n[Phase 1] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    chkpt_dir = os.path.join(args.exp_dir, 'chkpt')
    assert os.path.isdir(chkpt_dir), 'Error: No checkpoint directory found!'
    _, file_name, _, in_size = getNetwork(args)
    checkpoint = torch.load(os.path.join(chkpt_dir, file_name + '.t7'))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
elif args.testOnly:
    print('\n[Test Phase] : Model setup')
    chkpt_dir = os.path.join(args.exp_dir, 'chkpt')
    assert os.path.isdir(chkpt_dir), 'Error: No checkpoint directory found!'
    _, file_name, _, in_size = getNetwork(args)
    checkpoint = torch.load(os.path.join(chkpt_dir, file_name + '.t7'))
    net = checkpoint['net']
else:
    print('| Building net type [' + args.net_type + ']...')
    chkpt_dir = os.path.join(args.exp_dir, 'chkpt')
    net, file_name, net_init, in_size = getNetwork(args)
    net.apply(net_init)
    save_experiment_info(args.exp_dir, args.net_type, args.seed,
                         args.no_comment, args.exist_ok, net)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(args.num_gpus))
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()

# ##############################################################################
#  Data
print('\n[Phase 2] : Data Preparation')
print("| Preparing dataset...")
trainloader, testloader = cifar.get_data(
    args.data_dir, in_size, args.dataset, args.batch_size, args.trainsize,
    args.seed)
num_iter = len(trainloader)

# Test only option
if args.testOnly:
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total
    print("| Test Result\tAcc@1: %.2f%%" % (acc))

    sys.exit(0)

# ##############################################################################
#  Optimizer
print('\n[Phase 3] : Building Optimizer')
print('| Training Epochs = ' + str(args.epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(args.optim))

tr_writer = SummaryWriter(os.path.join(args.exp_dir, 'train'))
te_writer = SummaryWriter(os.path.join(args.exp_dir, 'test'))
elapsed_time = 0
# Get the parameters to optimize
try:
    params = net.param_groups()
except:
    params = net.parameters()

if args.optim == 'sgd':
    optimizer = optim.SGD(
        params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60,120, 160], gamma=0.2)
elif args.optim == 'adam':
    optimizer = optim.Adam(
        params, lr=args.lr, weight_decay=args.wd,
        betas=(0.9, .999))
    # lambda1 = lambda epoch: 1/np.sqrt(epoch+1)
    lambda1 = lambda epoch: 1
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1)
    # scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[100, 150, 180], gamma=0.2)
else:
    raise ValueError('Unknown optimizer')

# ##############################################################################
#  Train
print('\n[Phase 4] : Training')
# Get one batch of validation data for logging
x, y = next(iter(testloader))
if use_cuda:
    x = x.cuda()

for epoch in range(start_epoch, start_epoch+args.epochs):
    start_time = time.time()
    scheduler.step()

    learn.train(trainloader, net, criterion, optimizer, epoch, args.epochs,
                use_cuda, tr_writer, summary_freq=1)

    # After training, log some stats about the net
    try:
        net.log_info(args.exp_dir, x, epoch)
    except AttributeError:
        net.module.log_info(args.exp_dir, x, epoch)

    if epoch % args.eval_period == 0:
        sys.stdout.write('\n| Validating...')
        sys.stdout.flush()
        acc = learn.validate(testloader, net, criterion, use_cuda, epoch,
                             te_writer)
        if acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
            state = {
                'net': net.module if use_cuda else net,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(chkpt_dir):
                os.mkdir(chkpt_dir)
            save_point = os.path.join(chkpt_dir, file_name + '.t7')
            torch.save(state, save_point)
            best_acc = acc

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d\t Epoch time: %.1fs' % (
        get_hms(elapsed_time) + (epoch_time,)))

print('\n[Phase 5] : Results')
print('* Test results : Acc@1 = %.2f%%' % best_acc)
save_acc(args.exp_dir, best_acc)
