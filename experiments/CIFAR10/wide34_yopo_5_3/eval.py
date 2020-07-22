import argparse
import torch
# import numpy as np
import os

from experiments.cifar10.wide34_yopo_5_3.config import config
from experiments.cifar10.wide34_yopo_5_3.dataset import create_test_dataset
from experiments.cifar10.wide34_yopo_5_3.network import create_network

from lib.training.train import eval_one_epoch
from lib.utils.misc import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', '--resume', default='log/models/last.checkpoint',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default:log/last.checkpoint)')
    parser.add_argument('-d', type=int, default=0, help='Which gpu to use')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    net = create_network()
    net.to(device)

    ds_eval = create_test_dataset(100)

    attack_method = config.create_evaluation_attack_method(device)

    if os.path.isfile(args.resume):
        load_checkpoint(args.resume, net)

    print('Evaluating')
    clean_acc, adv_acc = eval_one_epoch(net, ds_eval, device, attack_method)
    print('clean acc: {}; adv acc: {}'.format(clean_acc, adv_acc))


if __name__ == '__main__':
    main()
