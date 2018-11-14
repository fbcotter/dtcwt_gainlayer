# Fergal Cotter
#

# Future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
from py3nvml.utils import get_free_gpus
import numpy as np
import time
import argparse

NUM_RUNS = 5
TRAINSET_SIZES = [1000, 2000, 5000, 10000, 20000, 50000]
DATA_DIR = 'DATA'

parser = argparse.ArgumentParser(description='''
PyTorch CIFAR10/CIFAR100 Training with standard and wavelet based convolutional
layers. Designed to run on a multi-gpu system, and will run each experiment on a
free gpu, one after another until all gpus are taken. Can be run on a cpu, but
will be slow (perhaps restrict to low dataset sizes). Needs py3nvml to query the
GPUs. For a multiple GPU system, will spin up subprocesses and use each one to
run an experiment. For a CPU only test, will run one after the other.\n

The output will have directory structure:

    exp_dir/<dataset>/<layer_type>/<trainset_size>/<run_number>/

In each experiment directory, you will see the saved source (not useful for
this experiment), a file called stdout (can inspect to see the printouts),
the best checkpoint parameters, and tensorboard logs.

It is recommended to use tensorboard to view the run results.
''', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--exp_dir', type=str, default='OUTDIR',
                    help='Output directory for the experiment')
parser.add_argument('--train_sizes', type=int, nargs='+',
                    default=[1000, 2000, 5000, 10000, 20000, 50000],
                    help='Size of train sets to run experiments on. Provide as '
                         'a list after this flag.')
parser.add_argument('--datasets', type=str, nargs='+',
                    default=['cifar100', 'cifar10'],
                    choices=['cifar100', 'cifar10'],
                    help='List of datasets to run experiments on.')
parser.add_argument('-N', type=int, default=5, help='Number of runs')
parser.add_argument('--cpu', action='store_true', help='Run only on cpu')


def main(args):
    print('-' * 100)
    for dataset in args.datasets:
        print('Running {} tests'.format(dataset))
        print('-' * 100)
        for ts in args.train_sizes:
            ts_str = '{}k'.format(ts//1000)
            print('-' * 50)
            print('Trainset size: {}'.format(ts_str))
            print('-' * 50)
            for N in range(args.N):
                print('LeNet {}'.format(N))
                outdir = os.path.join(
                    args.exp_dir, dataset, 'lenet', ts_str, str(N))
                os.makedirs(outdir, exist_ok=True)
                stdout_file = open(os.path.join(outdir, 'stdout'), 'w')
                cmd = ['python', 'main.py', outdir,
                       '--data_dir', DATA_DIR,
                       '--net_type', 'lenet',
                       '--trainsize', str(ts),
                       '--eval_period', '4',
                       '--dataset', dataset,
                       '--no_comment', '--exist_ok',
                       '--optim', 'adam', '--lr', '0.001']
                if args.cpu:
                    subprocess.run(cmd, stdout=stdout_file)
                else:
                    num_gpus = np.array(get_free_gpus()).sum()
                    while num_gpus < 1:
                        time.sleep(10)
                        num_gpus = np.array(get_free_gpus()).sum()
                    subprocess.Popen(cmd, stdout=stdout_file)
                    # Give the processes time to start
                    time.sleep(20)

                print('WaveLeNet {}'.format(N))
                outdir = os.path.join(
                    args.exp_dir, dataset, 'wavelenet', ts_str, str(N))
                cmd = ['python', 'main.py', outdir,
                       '--data_dir', DATA_DIR,
                       '--net_type', 'lenet_gainlayer',
                       '--trainsize', str(ts),
                       '--eval_period', '4',
                       '--dataset', dataset,
                       '--no_comment', '--exist_ok',
                       '--optim', 'adam', '--lr', '0.001']
                os.makedirs(outdir, exist_ok=True)
                stdout_file = open(os.path.join(outdir, 'stdout'), 'w')
                if args.cpu:
                    subprocess.run(cmd, stdout=stdout_file)
                else:
                    num_gpus = np.array(get_free_gpus()).sum()
                    while num_gpus < 1:
                        time.sleep(10)
                        num_gpus = np.array(get_free_gpus()).sum()
                    subprocess.Popen(cmd, stdout=stdout_file)
                    # Give the processes time to start
                    time.sleep(20)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
