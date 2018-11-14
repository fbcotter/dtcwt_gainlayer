from .lenet import LeNet
from .lenet_gainlayer import LeNet_GainLayer
import dtcwt_gainlayer
import sys


# Return network & file name
def getNetwork(args):
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100

    if args.net_type == 'lenet':
        net = LeNet(num_classes)
        net_init = dtcwt_gainlayer.networks.lenet.net_init
        file_name = 'lenet'
        in_size = 32
    elif args.net_type == 'lenet_gainlayer':
        net = LeNet_GainLayer(num_classes)
        net_init = dtcwt_gainlayer.networks.lenet_gainlayer.net_init
        file_name = 'lenet_gainlayer'
        in_size = 28
    else:
        print('Error : Network should be either [lenet / lenet_gainlayer]')
        sys.exit(0)

    return net, file_name, net_init, in_size
