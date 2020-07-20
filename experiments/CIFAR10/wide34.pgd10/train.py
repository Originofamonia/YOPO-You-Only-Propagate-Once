from config import config, args
from dataset import create_train_dataset, create_test_dataset
from network import create_network

from utils.misc import save_args, save_checkpoint, load_checkpoint
from training.train import train_one_epoch, eval_one_epoch

import torch
import json
import time
import numpy as np
from tensorboardX import SummaryWriter
import argparse

import os
from collections import OrderedDict


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    net = create_network()
    net.to(device)
    criterion = config.create_loss_function().to(device)

    optimizer = config.create_optimizer(net.parameters())
    lr_scheduler = config.create_lr_scheduler(optimizer)

    ds_train = create_train_dataset(args.batch_size)
    ds_val = create_test_dataset(args.batch_size)

    train_attack = config.create_attack_method(device)
    eval_attack = config.create_evaluation_attack_method(device)

    now_epoch = 0

    if args.auto_continue:
        args.resume = os.path.join(config.model_dir, 'last.checkpoint')
    if args.resume is not None and os.path.isfile(args.resume):
        now_epoch = load_checkpoint(args.resume, net, optimizer, lr_scheduler)

    while True:
        if now_epoch > config.num_epochs:
            break
        now_epoch = now_epoch + 1

        descrip_str = 'Training epoch:{}/{} -- lr:{}'.format(now_epoch, config.num_epochs,
                                                             lr_scheduler.get_last_lr()[0])
        train_one_epoch(net, ds_train, optimizer, criterion, device,
                        descrip_str, train_attack, adv_coef=args.adv_coef)
        if config.val_interval > 0 and now_epoch % config.val_interval == 0:
            eval_one_epoch(net, ds_val, device, eval_attack)

        lr_scheduler.step()

        save_checkpoint(now_epoch, net, optimizer, lr_scheduler,
                        file_name=os.path.join(config.model_dir, 'epoch-{}.checkpoint'.format(now_epoch)))


if __name__ == '__main__':
    main()
