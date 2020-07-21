import torch
# import json
# import numpy as np
from tensorboardX import SummaryWriter
# import argparse

# import torch.nn as nn
import torch.optim as optim
import os
# from collections import OrderedDict

from experiments.CIFAR10.wide34_yopo_5_3.config import config, args
from experiments.CIFAR10.wide34_yopo_5_3.dataset import create_train_dataset, create_test_dataset
from experiments.CIFAR10.wide34_yopo_5_3.network import create_network

from lib.utils.misc import save_args, save_checkpoint, load_checkpoint
from lib.training.train import eval_one_epoch
from experiments.CIFAR10.wide34_yopo_5_3.loss import Hamiltonian, CrossEntropyWithWeightPenlty
from experiments.CIFAR10.wide34_yopo_5_3.training_function import train_one_epoch, FastGradientLayerOneTrainer


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(log_dir=config.log_dir)

    net = create_network()
    net.to(device)
    criterion = config.create_loss_function().to(device)
    # criterion = CrossEntropyWithWeightPenlty(net.other_layers, DEVICE, config.weight_decay)#.to(DEVICE)
    # ce_criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = config.create_optimizer(net.other_layers.parameters())
    lr_scheduler = config.create_lr_scheduler(optimizer)

    # Make Layer One trainer  This part of code should be writen in config.py

    hamiltonian_func = Hamiltonian(net.layer_one, config.weight_decay)
    layer_1_optimizer = optim.SGD(net.layer_one.parameters(), lr=lr_scheduler.get_last_lr()[0], momentum=0.9,
                                  weight_decay=5e-4)
    layer_1_optimizer_lr_scheduler = optim.lr_scheduler.MultiStepLR(layer_1_optimizer,
                                                                    milestones=[30, 34, 36], gamma=0.1)
    layer_1_trainer = FastGradientLayerOneTrainer(hamiltonian_func, layer_1_optimizer,
                                                  config.inner_iters, config.sigma, config.eps)

    ds_train = create_train_dataset(args.batch_size)
    ds_val = create_test_dataset(args.batch_size)

    # TrainAttack = config.create_attack_method(device)
    eval_attack = config.create_evaluation_attack_method(device)

    now_epoch = 0

    if args.auto_continue:
        args.resume = os.path.join(config.model_dir, 'last.checkpoint')
    if args.resume is not None and os.path.isfile(args.resume):
        now_epoch = load_checkpoint(args.resume, net, optimizer, lr_scheduler)

    for i in range(now_epoch, config.num_epochs):
        # if now_epoch > config.num_epochs:
        #     break
        # now_epoch = now_epoch + 1

        descrip_str = 'Training epoch:{}/{} -- lr:{}'.format(i, config.num_epochs,
                                                             lr_scheduler.get_last_lr()[0])
        acc, yofoacc = train_one_epoch(net, ds_train, optimizer, criterion, layer_1_trainer, config.K,
                                       device, descrip_str)
        tb_train_dic = {'Acc': acc, 'YofoAcc': yofoacc}
        print(tb_train_dic)
        writer.add_scalars('Train', tb_train_dic, i)
        if config.eval_interval > 0 and i % config.eval_interval == 0:
            acc, advacc = eval_one_epoch(net, ds_val, device, eval_attack)
            tb_val_dic = {'Acc': acc, 'AdvAcc': advacc}
            writer.add_scalars('Val', tb_val_dic, i)

        lr_scheduler.step()
        layer_1_optimizer_lr_scheduler.step()

    save_checkpoint(now_epoch, net, optimizer, lr_scheduler,
                    file_name=os.path.join(config.model_dir, 'epoch-{}.checkpoint'.format(now_epoch)))


if __name__ == '__main__':
    main()
