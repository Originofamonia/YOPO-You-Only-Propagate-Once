import torch
import torch.optim as optim
import os
import sys


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)


abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-3])
lib_dir = os.path.join(root_path, 'lib')
add_path(root_path)


from experiments.cifar10_trades.preres18_trades_yopo_2_5.config import config, args
from experiments.dataset import create_train_dataset, create_test_dataset
from experiments.cifar10_trades.preres18_trades_yopo_2_5.network import create_network

from lib.utils.misc import save_args, save_checkpoint, load_checkpoint
from lib.training.train import eval_one_epoch
from experiments.cifar10_trades.preres18_trades_yopo_2_5.loss import Hamiltonian, CrossEntropyWithWeightPenalty
from experiments.cifar10_trades.preres18_trades_yopo_2_5.training_function import train_one_epoch, \
    FastGradientLayerOneTrainer


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = create_network()
    net.to(device)

    criterion = config.create_loss_function().to(device)
    optimizer = config.create_optimizer(net.other_layers.parameters())
    lr_scheduler = config.create_lr_scheduler(optimizer)

    Hamiltonian_func = Hamiltonian(net.layer_one, config.weight_decay)
    layer_one_optimizer = optim.SGD(net.layer_one.parameters(), lr=lr_scheduler.get_last_lr()[0], momentum=0.9,
                                    weight_decay=5e-4)
    lyaer_one_optimizer_lr_scheduler = optim.lr_scheduler.MultiStepLR(layer_one_optimizer,
                                                                      milestones=[70, 90, 100], gamma=0.1)
    LayerOneTrainer = FastGradientLayerOneTrainer(Hamiltonian_func, layer_one_optimizer,
                                                  config.inner_iters, config.sigma, config.eps)

    ds_train = create_train_dataset(args.batch_size)
    ds_eval = create_test_dataset(args.batch_size)

    eval_attack = config.create_evaluation_attack_method(device)

    now_epoch = 0
    if args.auto_continue:
        args.resume = os.path.join(config.model_dir, 'last.checkpoint')
    if args.resume is not None and os.path.isfile(args.resume):
        now_epoch = load_checkpoint(args.resume, net, optimizer, lr_scheduler)

    for i in range(now_epoch, config.num_epochs):

        descrip_str = 'Training epoch:{}/{} -- lr:{}'.format(i, config.num_epochs,
                                                             lr_scheduler.get_last_lr()[0])
        acc, yofoacc = train_one_epoch(net, ds_train, optimizer, criterion, LayerOneTrainer, config.K,
                                       device, descrip_str)
        tb_train_dic = {'Acc': acc, 'YofoAcc': yofoacc}
        print('Train: {}'.format(tb_train_dic))

        if config.eval_interval > 0 and i % config.eval_interval == 0:
            acc, advacc = eval_one_epoch(net, ds_eval, device, eval_attack)
            tb_val_dic = {'Acc': acc, 'AdvAcc': advacc}
            print('Eval: {}'.format(tb_val_dic))

        lr_scheduler.step()
        lyaer_one_optimizer_lr_scheduler.step()

    save_checkpoint(i, net, optimizer, lr_scheduler,
                    file_name=os.path.join(config.model_dir, 'epoch-{}.checkpoint'.format(i)))

    acc, advacc = eval_one_epoch(net, ds_eval, device, eval_attack)
    tb_val_dic = {'Acc': acc, 'AdvAcc': advacc}
    print('Eval: {}'.format(tb_val_dic))


if __name__ == '__main__':
    main()
