# YOPO (You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle)

## Prerequisites
* Python
* pytorch
* torchvision
* numpy
* tqdm

### Download this repo
```
git clone https://github.com/Originofamonia/YOPO-You-Only-Propagate-Once.git
cd YOPO-You-Only-Propagate-Once
```
Recommended: create a pytorch virtual environment
```
python3 -m venv --system-site-packages ~/torch  # or other names you like
```
Activate your virtual environment (venv):
```
source ~/torch/bin/activate  # replace torch with your own venv name
```
Install prerequisites in your venv:
```
pip3 install -r requirements.txt --user
```
Run MI loss:
```
cd YOPO-You-Only-Propagate-Once/experiments/cifar10/wide34_pgd10
python train_mi.py  # train
python eval.py  # eval
``` 

### YOPO training
Go to directory `experiments/CIFAR10/wide34.yopo-5-3`
run `python train.py -d <whcih_gpu>`

You can change all the hyper-parameters in `config.py`. And change network in `network.py`
Runing this code for the first time will dowload the dataset in `./experiments/CIFAR10/data/`, you can modify the path in `dataset.py`

<!--
## Experiment results

<center class="half">
    <img src="https://s2.ax1x.com/2019/05/16/EbamrT.jpg" width="300"/><img src="https://s2.ax1x.com/2019/05/16/EbatsK.jpg" width="300"/>
</center>
-->

