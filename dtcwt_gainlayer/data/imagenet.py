from torchvision import transforms, datasets
import torch
import os
import sys
import random
import torch.utils.data


def subsample(data_dir, sz):
    dest = os.path.join(data_dir, 'train{}'.format(sz))
    os.makedirs(dest)
    classes = os.listdir(os.path.join(data_dir, 'train'))
    for c in classes:
        if os.path.isdir(os.path.join(data_dir, 'train', c)):
            os.makedirs(os.path.join(dest, c))
            files = os.listdir(os.path.join(data_dir, 'train', c, 'images'))
            for i in range(sz):
                os.symlink(os.path.join(data_dir, 'train', c, 'images', files[i]),
                           os.path.join(dest, c, files[i]))


def get_data(data_dir, val_only=False, batch_size=128, trainsize=-1,
             perturb=True, num_workers=0, iter_size=1, distributed=False,
             pin_memory=False):
    """ Provides a pytorch loader to load in imagenet
    Args:
        in_size (int): the input size - can be used to scale the spatial size
        data_dir (str): the directory where the data is stored
        val_only (bool): Whether to load only the validation set
        batch_size (int): batch size for train loader. the val loader batch
            size is always 100
        class_sz (int): size of the training set. can be used to subsample it
        perturb (bool): whether to do data augmentation on the training set
        num_workers (int): how many workers to load data
        iter_size (int):
    """
    # Data loading code
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if not os.path.exists(valdir):
        raise ValueError(
            'Could not find the val folder in the Imagenet directory.')

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    # Get the test loader
    testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transform_test),
        batch_size=batch_size//iter_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)

    if val_only:
        trainloader = None
    else:
        if 0 < trainsize < 1300000:
            class_sz = trainsize // 1000
            traindir = os.path.join(data_dir, 'train{}'.format(class_sz))
        else:
            traindir = os.path.join(data_dir, 'train')
        if not os.path.exists(traindir):
            subsample(data_dir, class_sz)
            assert os.path.exists(traindir)
        # Get the train loader
        if perturb:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

        trainset = datasets.ImageFolder(traindir, transform_train)

        if distributed:
            trainsampler = torch.utils.data.distributed.DistributedSampler(
                trainset)
        else:
            trainsampler = None

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size // iter_size,
            shuffle=(trainsampler is None), num_workers=num_workers,
            pin_memory=pin_memory, sampler=trainsampler,
            )

    sys.stdout.write("| loaded imagenet")
    return trainloader, testloader

