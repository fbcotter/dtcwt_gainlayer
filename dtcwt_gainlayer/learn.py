"""
Module to do generic training/validating of a pytorch network. On top of the
loss function, will report back accuracy.
"""
# Future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import torch
import torch.utils.data
import torch.autograd as autograd
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_plot_steps(num_iter, pts):
    startpoint = 1/pts * (num_iter-1)
    update_steps = np.linspace(startpoint, num_iter-1, pts, endpoint=True)
    return update_steps.astype('int')


def get_lr(optim):
    lrs = []
    for p in optim.param_groups:
        lrs.append(p['lr'])
    if len(lrs) == 1:
        return lrs[0]
    else:
        return lrs


def train(loader, net, loss_fn, optimizer, epoch=0, epochs=0,
          use_cuda=True, writer=None, summary_freq=4):
    """ Train a model with the given loss functions

    Args:
        loader: pytorch data loader. needs to spit out a triple of
            (x, target) where target is an int representing the class number.
        net: nn.Module that spits out a prediction
        loss_fn: Loss function to apply to model output. Loss function should
            accept 3 inputs - (output, months, target).
        optimizer: any pytorch optimizer
        epoch (int): current epoch
        epochs (int): max epochs
        use_cuda (bool): true if want to use gpu
        writer: tensorboard writer
        summary_freq: number of times to update the
    """
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    losses = AverageMeter()
    top1 = AverageMeter()
    num_iter = len(loader)
    update_steps = np.linspace(0, num_iter-1, summary_freq).astype('int')

    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, get_lr(optimizer)))
    with autograd.detect_anomaly():
        for batch_idx, (inputs, targets) in enumerate(loader):
            # GPU settings
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # Forward and Backward
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            # Apply after update steps - nonstandard. I use it to rescale some
            # weights
            try:
                net.module.after_update()
            except AttributeError:
                net.after_update()

            # Plotting/Reporting
            train_loss += loss.item()
            losses.update(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            score = predicted.eq(targets.data).cpu().sum()
            correct += score
            top1.update(score.item()/targets.size(0))

            # Output summaries
            if batch_idx in update_steps and writer is not None:
                global_step = 100*epoch + int(100*batch_idx/num_iter)
                writer.add_scalar('acc', 100.*top1.avg, global_step)
                writer.add_scalar('loss', losses.avg, global_step)
                top1.reset()
                losses.reset()

            sys.stdout.write('\r')
            sys.stdout.write(
                '| Epoch [{:3d}/{:3d}] Iter[{:3d}/{:3d}]\t\tLoss: {:.4f} '
                'Acc@1: {:.3f}%'.format(
                    epoch, epochs, batch_idx+1, num_iter, loss.item(),
                    100. * correct.item()/total))
            sys.stdout.flush()


def validate(loader, net, loss_fn=None, use_cuda=True, epoch=-1, writer=None):
    """ Validate a model with the given loss functions

    Args:
        loader: pytorch data loader. needs to spit out a tuple of
            (x, target) where target is an integer representing the class of
            the input.
        net: nn.Module that spits out a prediction
        loss_fn: Loss function to apply to model output. Can be none and loss
            reporting won't be done for validation steps.
        use_cuda (bool): if true, will put things on the gpu
        epoch: current epoch (used only for print and logging purposes)
        writer: tensorboard writer

    Returns:
        acc: current epoch accuracy
    """
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            if loss_fn is not None:
                loss = loss_fn(outputs, targets)
                test_loss += loss.item()
            else:
                test_loss = torch.tensor(0)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct.item()/total
    sys.stdout.write('\r')
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %
          (epoch, loss.item(), acc))

    if writer is not None and epoch >= 0:
        writer.add_scalar('loss', loss.item(), 100*epoch)
        writer.add_scalar('acc', acc, 100*epoch)

    return acc
