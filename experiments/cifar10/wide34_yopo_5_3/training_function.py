import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm

from experiments.cifar10.wide34_yopo_5_3.config import config
# from experiments.CIFAR10.wide34_yopo_5_3.loss import Hamiltonian, cal_l2_norm
from lib.utils.misc import torch_accuracy, AvgMeter


class FastGradientLayerOneTrainer(object):

    def __init__(self, Hamiltonian_func, param_optimizer,
                 inner_steps=2, sigma=0.008, eps=0.03):
        self.inner_steps = inner_steps
        self.sigma = sigma
        self.eps = eps
        self.Hamiltonian_func = Hamiltonian_func
        self.param_optimizer = param_optimizer

    def step(self, inp, p, eta):
        """
        Perform Iterative Sign Gradient on eta
        ret: inp, eta
        """

        p = p.detach()

        for i in range(self.inner_steps):
            tmp_inp = inp + eta
            tmp_inp = torch.clamp(tmp_inp, 0, 1)
            H = self.Hamiltonian_func(tmp_inp, p)

            eta_grad_sign = torch.autograd.grad(H, eta, only_inputs=True, retain_graph=False)[0].sign()

            eta = eta - eta_grad_sign * self.sigma

            eta = torch.clamp(eta, -1.0 * self.eps, self.eps)
            eta = torch.clamp(inp + eta, 0.0, 1.0) - inp
            eta = eta.detach()
            eta.requires_grad_()
            eta.retain_grad()

        # self.param_optimizer.zero_grad()

        yofo_inp = eta + inp
        yofo_inp = torch.clamp(yofo_inp, 0, 1)

        loss = -1.0 * self.Hamiltonian_func(yofo_inp, p)

        loss.backward()
        # self.param_optimizer.step()
        # self.param_optimizer.zero_grad()

        return yofo_inp, eta


def train_one_epoch(net, batch_generator, optimizer, criterion, LayerOneTrainer, K,
                    device, descr_str='Training'):
    """
    :return: clean_acc, adv_acc
    """
    net.train()
    pbar = tqdm(batch_generator)
    yofoacc = -1
    cleanacc = -1
    cleanloss = -1
    pbar.set_description(descr_str)
    for i, (data, label) in enumerate(pbar):
        data, label = data.to(device), label.to(device)

        eta = torch.cuda.FloatTensor(*data.shape).uniform_(-config.eps, config.eps)
        eta = eta.to(device)
        eta.requires_grad_()

        optimizer.zero_grad()
        LayerOneTrainer.param_optimizer.zero_grad()

        for j in range(K):
            # optimizer.zero_grad()

            pbar_dic = OrderedDict()
            TotalLoss = 0

            h1, h2, h3, h4, y = net(data + eta.detach())

            loss = criterion(y, label)
            TotalLoss = TotalLoss + loss
            wgrad = net.conv1.weight.grad
            # bgrad = net.conv1.bias.grad
            TotalLoss.backward()
            net.conv1.weight.grad = wgrad
            # net.conv1.bias.grad = bgrad
            # param = next(net.parameters())
            # grad_mean = torch.mean(param.grad)

            # optimizer.step()
            # optimizer.zero_grad()

            p = -1.0 * net.layer_one_out.grad
            yofo_inp, eta = LayerOneTrainer.step(data, p, eta)

            with torch.no_grad():
                if j == 0:
                    acc = torch_accuracy(y, label, (1,))
                    cleanacc = acc[0].item()
                    cleanloss = loss.item()

                if j == K - 1:
                    h1, h2, h3, h4, y = net(yofo_inp)
                    yofoacc = torch_accuracy(y, label, (1,))[0].item()
            # pbar_dic['grad'] = '{}'.format(grad_mean)

        optimizer.step()
        LayerOneTrainer.param_optimizer.step()
        optimizer.zero_grad()
        LayerOneTrainer.param_optimizer.zero_grad()
        pbar_dic['Acc'] = '{:.2f}'.format(cleanacc)
        pbar_dic['loss'] = '{:.2f}'.format(cleanloss)
        pbar_dic['YofoAcc'] = '{:.2f}'.format(yofoacc)
        pbar.set_postfix(pbar_dic)

    return cleanacc, yofoacc
