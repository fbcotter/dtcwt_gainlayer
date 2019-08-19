import torch
import time
import random
import torch.nn.functional as func
import torch.nn as nn
from dtcwt_gainlayer.data import cifar, tiny_imagenet
import numpy as np
import sys
import os


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def num_correct(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res, batch_size


class TrainingObject(object):
    """ This class handles model training and scheduling for our networks.

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
    def __init__(self, model, dataset, datadir, optim, scheduler, optim2=None,
                 scheduler2=None, batch_size=128, seed=None, num_gpus=1,
                 verbose=False):
        # Parameters like learning rate and momentum can be speicified by the
        # config search space. If not specified, fall back to the args
        self.dataset = dataset
        self.datadir = datadir
        self.num_gpus = num_gpus
        self.verbose = verbose
        self.model = model
        self.optimizer = optim
        self.scheduler = scheduler
        self.optimizer2 = optim2
        self.scheduler2 = scheduler2

        num_workers = 4
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            num_workers = 0
            if self.use_cuda:
                torch.cuda.manual_seed(seed)
        else:
            seed = random.randint(0, 10000)

        # ######################################################################
        #  Data
        kwargs = {}
        if self.use_cuda:
            kwargs = {'num_workers': num_workers, 'pin_memory': True}
        if dataset.startswith('cifar'):
            self.train_loader, self.test_loader = cifar.get_data(
                32, datadir, dataset=dataset,
                batch_size=batch_size, **kwargs)
        elif dataset == 'tiny_imagenet':
            self.train_loader, self.test_loader = tiny_imagenet.get_data(
                64, datadir, val_only=False,
                batch_size=batch_size, **kwargs)

        # ######################################################################
        # Split across GPUs
        if torch.cuda.device_count() > 1 and num_gpus > 1:
            self.model = nn.DataParallel(self.model)
            model = self.model.module
        else:
            model = self.model
        if self.use_cuda:
            self.model.cuda()

    @property
    def use_cuda(self):
        if not hasattr(self, '_use_cuda'):
            self._use_cuda = torch.cuda.is_available()
        return self._use_cuda

    @property
    def last_epoch(self):
        return self.scheduler.last_epoch

    @property
    def final_epoch(self):
        if hasattr(self, '_final_epoch'):
            return self._final_epoch
        else:
            return 120

    def step_lr(self):
        self.scheduler.step()
        if self.scheduler2 is not None:
            self.scheduler2.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
        if self.optimizer2 is not None:
            self.optimizer2.zero_grad()

    def opt_step(self):
        self.optimizer.step()
        if self.optimizer2 is not None:
            self.optimizer2.step()

    def _train_iteration(self):
        self.model.train()
        top1_update = 0
        top1_epoch = 0
        top5_update = 0
        top5_epoch = 0
        loss_update = 0
        loss_epoch = 0
        update = 0
        epoch = 0
        num_iter = len(self.train_loader)
        start = time.time()
        update_steps = np.linspace(
            int(1/4 * num_iter), num_iter-1, 4).astype('int')

        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            self.zero_grad()

            output = self.model(data)
            loss = func.nll_loss(output, target)
            if torch.isnan(loss):
                raise ValueError(
                    "Nan found in training at epoch {}".format(self.last_epoch))
            loss += self.model.get_reg()
            loss.backward()
            self.model.clip_grads()
            self.opt_step()

            corrects, bs = num_correct(output.data, target, topk=(1,5))
            top1_epoch += corrects[0].item()
            top5_epoch += corrects[1].item()
            loss_epoch += loss.item()*bs
            epoch += bs

            # Plotting/Reporting
            if self.verbose:
                update += bs
                top1_update += corrects[0].item()
                top5_update += corrects[1].item()
                loss_update += loss.item()*bs

                sys.stdout.write('\r')
                sys.stdout.write(
                    '| Epoch [{:3d}/{:3d}] Iter[{:3d}/{:3d}]\t\t'
                    'Loss: {:.4f}\tAcc@1: {:.3f}%\tAcc@5: {:.3f}%\t'
                    'Elapsed Time: {:.1f}min'.format(
                        self.last_epoch, self.final_epoch, batch_idx+1,
                        num_iter, loss_update/update,
                        100. * top1_update/update,
                        100. * top5_update/update,
                        (time.time()-start)/60))
                sys.stdout.flush()
                # Every update_steps, print a new line
                if batch_idx in update_steps:
                    top1_update = 0
                    top5_update = 0
                    loss_update = 0
                    update = 0
                    print()
        loss_epoch /= epoch
        top1_epoch = 100. * top1_epoch/epoch
        top5_epoch = 100. * top5_epoch/epoch
        return {"mean_loss": loss_epoch, "mean_accuracy": top1_epoch, "acc5":
                top5_epoch}

    def _test(self):
        self.model.eval()
        test_loss = 0
        top1_correct = 0
        top5_correct = 0
        epoch = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = self.model(data)
                # sum up batch loss
                loss = func.nll_loss(output, target, reduction='sum')

                # get the index of the max log-probability
                corrects, bs = num_correct(output.data, target, topk=(1, 5))
                test_loss += loss.item()
                top1_correct += corrects[0].item()
                top5_correct += corrects[1].item()
                epoch += bs

        test_loss /= epoch
        acc1 = 100. * top1_correct/epoch
        acc5 = 100. * top5_correct/epoch
        if self.verbose:
            # Save checkpoint when best model
            print("|\n| Validation Epoch #{}\t\t\tLoss: {:.4f}\tAcc@1: {:.2f}%"
                  "\tAcc@5: {:.2f}%".format(
                      self.last_epoch, test_loss, acc1, acc5))
        return {"mean_loss": test_loss, "mean_accuracy": acc1, "acc5": acc5}

    def _train(self):
        if not hasattr(self, '_last_epoch'):
            self._last_epoch = 0
        else:
            self._last_epoch += 1
        self.step_lr()
        self._train_iteration()
        return self._test()

    def _save(self, checkpoint_dir, name='model.pth'):
        checkpoint_path = os.path.join(checkpoint_dir, name)
        model = self.model.state_dict()
        opt = self.optimizer.state_dict()
        sch = self.scheduler.state_dict()
        opt2 = None
        sch2 = None
        if self.optimizer2 is not None:
            opt2 = self.optimizer2.state_dict()
        if self.scheduler2 is not None:
            sch2 = self.scheduler2.state_dict()
        torch.save({
            'model_state_dict': model,
            'optimizer_state_dict': opt,
            'scheduler_state_dict': sch,
            'optimizer2_state_dict': opt2,
            'scheduler2_state_dict': sch2,
        }, checkpoint_path)

        return checkpoint_path

    def _restore(self, checkpoint_path):
        chk = torch.load(checkpoint_path)
        self.model.load_state_dict(chk['model_state_dict'])
        self.optimizer.load_state_dict(chk['optimizer_state_dict'])

        if 'scheduler_state_dict' in chk.keys():
            self.scheduler.load_state_dict(chk['scheduler_state_dict'])

        if 'optimizer2_state_dict' in chk.keys() and \
                chk['optimizer2_state_dict'] is not None:
            if self.optimizer2 is None:
                raise ValueError('Loading from a checkpoint with a second '
                                 'optimizer, but we dont have one')
            else:
                self.optimizer2.load_state_dict(chk['optimizer2_state_dict'])

        if 'scheduler2_state_dict' in chk.keys() and \
                chk['scheduler2_state_dict'] is not None:
            if self.scheduler2 is None:
                raise ValueError('Loading from a checkpoint with a second '
                                 'scheduler, but we dont have one')
            else:
                self.scheduler2.load_state_dict(chk['scheduler2_state_dict'])
