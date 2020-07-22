import torch
# import experiments.cifar10_trades.baseline_respre18_trades_10step.config
# from lib.base_model.wide_resnet import WideResNet
from lib.base_model.preact_resnet import PreActResNet18


def create_network():
    # return WideResNet(34)
    return PreActResNet18()


def test():
    net = create_network()
    y = net((torch.randn(1, 3, 32, 32)))
    print(y.size())
