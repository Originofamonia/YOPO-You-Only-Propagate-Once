from __future__ import print_function
import os
from tqdm import tqdm
from collections import OrderedDict
from time import time
import json

import argparse
import torch
import sys
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-3])
lib_dir = os.path.join(root_path, 'lib')
add_path(root_path)

from experiments.cifar10_trades.baseline_respre18_trades_10step.config import config, args
from experiments.cifar10_trades.baseline_respre18_trades_10step.network import create_network
from experiments.cifar10_trades.baseline_respre18_trades_10step.trades import trades_loss
from experiments.dataset import create_train_dataset, create_test_dataset
from lib.training.train import eval_one_epoch
from lib.utils.misc import torch_accuracy, AvgMeter


def train(args, model, device, train_loader, optimizer, epoch, descrip_str='Training'):
    model.train()
    pbar = tqdm(train_loader)
    pbar.set_description(descrip_str)

    CleanAccMeter = AvgMeter()
    TradesAccMeter = AvgMeter()
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss, cleanloss, klloss, cleanacc, tradesacc = trades_loss(model=model,
                                                                   x_natural=data,
                                                                   y=target,
                                                                   optimizer=optimizer,
                                                                   device=device,
                                                                   step_size=args.step_size,
                                                                   epsilon=args.epsilon,
                                                                   perturb_steps=args.num_steps,
                                                                   beta=args.beta, )
        loss.backward()
        optimizer.step()

        CleanAccMeter.update(cleanacc)
        TradesAccMeter.update(tradesacc)

        pbar_dic = OrderedDict()
        pbar_dic['cleanloss'] = '{:.3f}'.format(cleanloss)
        pbar_dic['klloss'] = '{:.3f}'.format(klloss)
        pbar_dic['CleanAcc'] = '{:.2f}'.format(CleanAccMeter.mean)
        pbar_dic['TradesAcc'] = '{:.2f}'.format(TradesAccMeter.mean)
        pbar.set_postfix(pbar_dic)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = create_network().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    eval_attack = config.create_evaluation_attack_method(device)

    train_loader = create_train_dataset(args.batch_size)
    test_loader = create_test_dataset(args.batch_size)

    now_train_time = 0
    for i in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, i)

        s_time = time()
        descr_str = 'Epoch: {}/{}'.format(i, args.epochs)
        # adversarial training
        train(args, model, device, train_loader, optimizer, i, descr_str)
        now_train_time += time() - s_time
        if args.log_interval > 0 and i % args.log_interval == 0:
            acc, advacc = eval_one_epoch(model, test_loader, device, eval_attack)
            tb_val_dic = {'Acc': acc, 'AdvAcc': advacc}
            print('Eval: {}'.format(tb_val_dic))

    torch.save(model.state_dict(),
               os.path.join(config.model_dir, 'model-wideres-epoch{}.pt'.format(i)))

    acc, advacc = eval_one_epoch(model, test_loader, device, eval_attack)
    tb_val_dic = {'Acc': acc, 'AdvAcc': advacc}
    print('Eval: {}'.format(tb_val_dic))


if __name__ == '__main__':
    main()
