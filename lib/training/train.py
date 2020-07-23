import os
import sys

from lib.utils.misc import torch_accuracy, AvgMeter
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm

father_dir = os.path.join('/', *os.path.realpath(__file__).split(os.path.sep)[:-2])
# print(father_dir)
if father_dir not in sys.path:
    sys.path.append(father_dir)


def train_one_epoch(net, batch_generator, optimizer, criterion, device,
                    descr_str='Training', attack_method=None, adv_coef=1.0):
    """
    :param attack_freq:  Frequencies of training with adversarial examples. -1 indicates natural training
    :param attack_method: the attack method, None represents natural training
    :return: clean_acc, adv_acc
    """
    net.train()
    pbar = tqdm(batch_generator)
    advacc = -1
    advloss = -1
    cleanacc = -1
    cleanloss = -1
    pbar.set_description(descr_str)
    for i, (data, label) in enumerate(pbar):
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()

        pbar_dic = OrderedDict()
        TotalLoss = 0

        if attack_method is not None:
            adv_inp = attack_method.attack(net, data, label)
            optimizer.zero_grad()
            net.train()
            pred = net(adv_inp)
            loss = criterion(pred, label)

            acc = torch_accuracy(pred, label, (1,))
            advacc = acc[0].item()
            advloss = loss.item()
            # TotalLoss = TotalLoss + loss * adv_coef
            (loss * adv_coef).backward()

        pred = net(data)

        loss = criterion(pred, label)
        # TotalLoss = TotalLoss + loss
        loss.backward()
        # TotalLoss.backward()
        # param = next(net.parameters())
        # grad_mean = torch.mean(param.grad)

        optimizer.step()
        acc = torch_accuracy(pred, label, (1,))
        cleanacc = acc[0].item()
        cleanloss = loss.item()
        # pbar_dic['grad'] = '{}'.format(grad_mean)
        pbar_dic['Acc'] = '{:.2f}'.format(cleanacc)
        pbar_dic['loss'] = '{:.2f}'.format(cleanloss)
        pbar_dic['AdvAcc'] = '{:.2f}'.format(advacc)
        pbar_dic['Advloss'] = '{:.2f}'.format(advloss)
        pbar.set_postfix(pbar_dic)

    return cleanacc, advacc


def train_hloss(net, batch_generator, optimizer, criterion, device,
                descr_str='Training', attack_method=None, adv_coef=1.0):
    """
    :return: clean_acc, adv_acc
    """
    net.train()
    pbar = tqdm(batch_generator)
    advacc = -1
    advloss = -1
    cleanacc = -1
    hloss_criterion = nn.MSELoss()
    pbar.set_description(descr_str)
    for i, (data, label) in enumerate(pbar):
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()

        pbar_dic = OrderedDict()

        if attack_method is not None:
            adv_inp = attack_method.attack(net, data, label)
            optimizer.zero_grad()
            net.train()
            h1a, h2a, h3a, h4a, ya = net(adv_inp)
            xent_loss = criterion(ya, label)

            acc = torch_accuracy(ya, label, (1,))
            advacc = acc[0].item()
            # advloss = xent_loss.item()
            # (xent_loss * adv_coef).backward()

        h1, h2, h3, h4, y = net(data)

        h_loss = hloss_criterion(h4a, h4)
        loss = h_loss * adv_coef + xent_loss
        loss.backward()

        optimizer.step()
        acc = torch_accuracy(y, label, (1,))
        cleanacc = acc[0].item()

        pbar_dic['Acc'] = '{:.2f}'.format(cleanacc)
        pbar_dic['hloss'] = '{:.2f}'.format(h_loss)
        pbar_dic['AdvAcc'] = '{:.2f}'.format(advacc)
        pbar_dic['xentloss'] = '{:.2f}'.format(xent_loss)
        pbar.set_postfix(pbar_dic)

    return cleanacc, advacc


def eval_one_epoch(net, batch_generator, device, attack_method=None):
    net.eval()
    pbar = tqdm(batch_generator)
    clean_accuracy = AvgMeter()
    adv_accuracy = AvgMeter()

    pbar.set_description('Evaluating')
    for (data, label) in pbar:
        data, label = data.to(device), label.to(device)

        with torch.no_grad():
            h1, h2, h3, h4, y = net(data)
            acc = torch_accuracy(y, label, (1,))
            clean_accuracy.update(acc[0].item())

        if attack_method is not None:
            adv_inp = attack_method.attack(net, data, label)

            with torch.no_grad():
                y = net(adv_inp)
                acc = torch_accuracy(y, label, (1,))
                adv_accuracy.update(acc[0].item())

        pbar_dic = OrderedDict()
        pbar_dic['CleanAcc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['AdvAcc'] = '{:.2f}'.format(adv_accuracy.mean)

        pbar.set_postfix(pbar_dic)

        adv_acc = adv_accuracy.mean if attack_method is not None else 0
    return clean_accuracy.mean, adv_acc
