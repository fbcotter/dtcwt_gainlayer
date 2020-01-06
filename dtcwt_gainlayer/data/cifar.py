"""
Module to load in CIFAR10/100
"""
import torchvision
from torchvision import transforms
import numpy as np
import torch
import os

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}


def subsample(cifar10=True, size=50000):
    """ Subsamples cifar10/cifar100 so the entire dataset has size <size> but
    with equal classes."""
    basedir = os.path.dirname(__file__)
    if cifar10:
        class_sz = size // 10
        idx = np.load(os.path.join(basedir, 'cifar10_idxs.npy'))
        return np.sort(idx[:, :class_sz].ravel())
    else:
        class_sz = size // 100
        idx = np.load(os.path.join(basedir, 'cifar100_idxs.npy'))
        return np.sort(idx[:, :class_sz].ravel())


def get_data(in_size, data_dir, dataset='cifar10', batch_size=128, trainsize=-1,
             perturb=True, double_size=False, pin_memory=True, num_workers=0):
    """ Provides a pytorch loader to load in cifar10/100
    Args:
        in_size (int): the input size - can be used to scale the spatial size
        data_dir (str): the directory where the data is stored
        dataset (str): 'cifar10' or 'cifar100'
        batch_size (int): batch size for train loader. the val loader batch
            size is always 100
        trainsize (int): size of the training set. can be used to subsample it
        perturb (bool): whether to do data augmentation on the training set
        double_size (bool): whether to double the input size
    """
    if double_size:
        resize = transforms.Resize(in_size*2)
    else:
        resize = transforms.Resize(in_size*1)

    if perturb:
        transform_train = transforms.Compose([
            transforms.RandomCrop(in_size, padding=4),
            resize,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean[dataset], std[dataset])
        ])
    else:
        transform_train = transforms.Compose([
            transforms.CenterCrop(in_size),
            resize,
            transforms.ToTensor(),
            transforms.Normalize(mean[dataset], std[dataset])
        ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(in_size),
        resize,
        transforms.ToTensor(),
        transforms.Normalize(mean[dataset], std[dataset]),
    ])

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True,
            transform=transform_train)
        if trainsize > 0:
            idxs = subsample(False, trainsize)
            trainset = torch.utils.data.Subset(trainset, idxs)
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=False,
            transform=transform_test)

    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True,
            transform=transform_train)
        if trainsize > 0:
            idxs = subsample(False, trainsize)
            trainset = torch.utils.data.Subset(trainset, idxs)
        testset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=False,
            transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory)

    return trainloader, testloader
