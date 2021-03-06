import torch
import sys
import torch.nn as nn
# import numpy as np
# import argparse
import os


# from collections import OrderedDict


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-3])
lib_dir = os.path.join(root_path, 'lib')
add_path(root_path)

from experiments.cifar10.wide34_pgd10.config import config, args
from experiments.dataset import create_train_dataset, create_test_dataset
from experiments.cifar10.wide34_pgd10.network import create_network, DeepInfoMaxLoss

from lib.utils.misc import save_args, save_checkpoint, load_checkpoint
from lib.training.train import train_one_epoch, train_hloss, train_mi, eval_one_epoch


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    net = create_network()
    net2 = DeepInfoMaxLoss()

    if torch.cuda.device_count() > 1:
        print('Let\'s use {} GPUs!'.format(torch.cuda.device_count()))
        net = nn.DataParallel(net)
        net2 = nn.DataParallel(net2)

    net.to(device)
    net2.to(device)
    criterion = config.create_loss_function().to(device)

    optimizer = config.create_optimizer(net.parameters())
    lr_scheduler = config.create_lr_scheduler(optimizer)

    ds_train = create_train_dataset(args.batch_size)
    ds_eval = create_test_dataset(args.batch_size)

    train_attack = config.create_attack_method(device)
    eval_attack = config.create_evaluation_attack_method(device)

    now_epoch = 0

    if args.auto_continue:
        args.resume = os.path.join(config.model_dir, 'last.checkpoint')
    if args.resume is not None and os.path.isfile(args.resume):
        now_epoch = load_checkpoint(args.resume, net, optimizer, lr_scheduler)

    for i in range(now_epoch, config.num_epochs):

        descr_str = 'Epoch:{}/{} -- lr:{}'.format(i, config.num_epochs,
                                                  lr_scheduler.get_last_lr()[0])
        cleanacc, advacc = train_mi(net, net2, ds_train, optimizer, criterion, device,
                                    descr_str, train_attack, adv_coef=args.adv_coef)
        tb_train_dic = {'Acc': cleanacc, 'YofoAcc': advacc}
        print('Train: {}'.format(tb_train_dic))

        if config.eval_interval > 0 and i % config.eval_interval == 0:
            acc, advacc = eval_one_epoch(net, ds_eval, device, eval_attack)
            tb_val_dic = {'Acc': acc, 'AdvAcc': advacc}
            print('Eval: {}'.format(tb_val_dic))

        lr_scheduler.step()

    save_checkpoint(i, net, optimizer, lr_scheduler,
                    file_name=os.path.join(config.model_dir, 'mi.{}'.format(i)))
    acc, advacc = eval_one_epoch(net, ds_eval, device, eval_attack)
    tb_val_dic = {'Acc': acc, 'AdvAcc': advacc}
    print('Eval: {}'.format(tb_val_dic))
    file1 = open(os.path.join(config.model_dir, 'mi_results.txt'), 'w')
    file1.write('MI Eval: {}'.format(tb_val_dic))
    file1.close()


if __name__ == '__main__':
    main()
