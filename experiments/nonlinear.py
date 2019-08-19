from save_exp import save_experiment_info, save_acc
import argparse
import os
import torch
import time
from networks.nonlinear_nets import NonlinearNet
from utils import get_hms, TrainingObject
from optim import get_optim
from tensorboardX import SummaryWriter
import py3nvml
from math import ceil

# Training settings
parser = argparse.ArgumentParser(description='Nonlinear example')
parser.add_argument('outdir', type=str, help='experiment directory')
parser.add_argument('--type', default=None, type=str, nargs='+')
parser.add_argument('-C', type=int, default=96, help='number channels')
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
# Core hyperparameters
parser.add_argument('--lr', default=0.5, type=float, help='learning rate')
parser.add_argument('--lr1', default=None, type=float, help='learning rate for wavelet domain')
parser.add_argument('--mom', default=0.85, type=float, help='momentum')
parser.add_argument('--mom1', default=None, type=float, help='momentum for wavelet domain')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--wd1', default=1e-5, type=float, help='l1 weight decay')
parser.add_argument('--reg', default='l2', type=str, help='regularization term')
parser.add_argument('--steps', default=[60,80,100], type=int, nargs='+')
parser.add_argument('--gamma', default=0.2, type=float, help='Lr decay')
parser.add_argument('--pixel-nl', default='relu', type=str)
parser.add_argument('--lp-nl', default='none', type=str)
parser.add_argument('--bp-nl', default='none', type=str)


if __name__ == "__main__":
    args = parser.parse_args()

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
    model = NonlinearNet(args.dataset, type, num_channels=args.C,
                         wd=args.wd, wd1=args.wd1, pixel_nl=args.pixel_nl,
                         lp_nl=args.lp_nl, bp_nl=args.bp_nl)

    # ######################################################################
    # Build the optimizer - use separate parameter groups for the gain
    # and convolutional layers
    default_params = model.parameters()
    wave_params = model.wave_parameters()
    optim, sched = get_optim(
        'sgd', default_params, init_lr=args.lr,
        steps=args.steps, wd=0, gamma=args.gamma, momentum=args.mom,
        max_epochs=args.epochs)

    if len(wave_params) > 0:
        if args.lr1 is None:
            args.lr1 = args.lr
        if args.mom1 is None:
            args.mom1 = args.mom
        optim2, sched2 = get_optim(
            'sgd', wave_params, init_lr=args.lr1,
            steps=args.steps, wd=0, gamma=args.gamma, momentum=args.mom1,
            max_epochs=args.epochs)
    else:
        optim2, sched2 = None, None

    trn = TrainingObject(model, args.dataset, args.datadir, optim, sched,
                         optim2, sched2, args.batch_size, args.seed,
                         args.num_gpus, args.verbose)
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
