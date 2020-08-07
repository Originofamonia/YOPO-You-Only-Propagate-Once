import torch
import torch.nn as nn
import torch.nn.functional as F
# import experiments.cifar10.wide34_pgd10.config
from lib.base_model.wide_resnet import WideResNet


class GlobalD(nn.Module):
    def __init__(self, in_channel1, in_channel2):
        super(GlobalD, self).__init__()
        self.in_channel1 = in_channel1
        self.in_channel2 = in_channel2

        self.c0 = nn.Conv2d(self.in_channel1, 64, kernel_size=3)
        self.c2 = nn.Conv2d(self.in_channel2, 64, kernel_size=4, stride=2)
        self.l0 = nn.Linear(512, 512)

        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.c3 = nn.Conv2d(64, 32, kernel_size=4, stride=2)
        self.l1 = nn.Linear(1152, 512)

        self.l2 = nn.Linear(1024, 1)

    def forward(self, inputs1, inputs2):
        h1 = F.relu(self.c0(inputs1))
        h1 = self.c1(h1)
        h1 = h1.view(h1.shape[0], -1)
        h1 = F.relu(self.l0(h1))

        h2 = F.relu(self.c2(inputs2))
        h2 = self.c3(h2)
        h2 = h2.view(h2.shape[0], -1)
        h2 = F.relu(self.l1(h2))
        h = torch.cat((h1, h2), dim=1)

        h = self.l2(h)
        return h


class LocalD(nn.Module):
    def __init__(self, in_channel):
        super(LocalD, self).__init__()

        self.in_channel = in_channel
        self.c0 = nn.Conv2d(self.in_channel, 512, kernel_size=1)
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        hidden = F.relu(self.c0(x))
        hidden = F.relu(self.c1(hidden))
        return self.c2(hidden)


class PriorD(nn.Module):
    def __init__(self, in_channel):
        super(PriorD, self).__init__()
        self.l0 = nn.Linear(in_channel, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        x = x.view(-1, 40960)
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1):
        super(DeepInfoMaxLoss, self).__init__()
        self.global_d = GlobalD(640, 16)  # in_channel (y, M)
        self.local_d = LocalD(656)  # in_channel (h_cat)
        self.prior_d = PriorD(40960)  # in_channel (y)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, h4, h1, h1_prime):  # (y, M, M_prime)
        h4_exp = h4.repeat(1, 1, 4, 4)
        hcat = torch.cat((h4_exp, h1), dim=1)
        hcat_prime = torch.cat((h4_exp, h1_prime), dim=1)

        Ej = -F.softplus(-self.local_d(hcat)).mean()
        Em = F.softplus(self.local_d(hcat_prime)).mean()
        _local = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(h4, h1)).mean()
        Em = F.softplus(self.global_d(h4, h1_prime)).mean()
        _global = (Em - Ej) * self.alpha

        prior = torch.rand_like(h4)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(h4)).mean()
        _prior = - (term_a + term_b) * self.gamma

        return _local + _global + _prior


def create_network():
    return WideResNet()


def test():
    net = create_network()
    y = net((torch.randn(1, 3, 32, 32)))
    print(y.size())
