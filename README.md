# Federated Averaging 

![visitors](https://visitor-badge.glitch.me/badge?page_id=MrZhang1994.mobile-federated-learning)

## Requirements

1. Make sure GPU is avaible and `CUDA>=11.0` has been installed on your computer. You can check it with
    ```bash
        nvidia-smi
    ```
2. Download the repo with following command:
    ```bash
        git clone --recurse-submodules https://github.com/MrZhang1994/mobile-federated-learning.git
    ```
3. Follow the instruction or documentation of [FedML](https://github.com/FedML-AI/FedML) to install required package in python environment. Or, you can simply create an virtural environment with `python>=3.8` and run `pip install -r requirements.txt` to download the required packages. If you use `anaconda3` or `miniconda`, you can run following instructions to download the required packages in python. 
    ```bash
        conda create -y -n mfl python=3.8
        conda activate mfl
        pip install pip --upgrade
        pip install -r requirements.txt
    ```
    
## Experimental Tracking Platform 

report real-time result to wandb.com, please change ID to your own

```
wandb login `Your ID`
```

## Experiment Scripts

To test whether program are correctly configured, you can run following commands to see whether training process starts correctly.

```
sh begin.sh -[VERBOSE]
```

Or, you can try other heterogeneous distribution (Non-IID) experiment:
``` 
## MNIST (non-i.i.d)
sh run_fedavg_standalone_pytorch.sh 0 mnist ../FedML/data/MNIST lr hetero 59361 0.03 sgd sch_random -v

## shakespeare (non-i.i.d LEAF)
sh run_fedavg_standalone_pytorch.sh 0 shakespeare ../FedML/data/shakespeare rnn hetero 59361 0.03 sgd sch_random -v

# fed_shakespeare (non-i.i.d Google)
sh run_fedavg_standalone_pytorch.sh 0 fed_shakespeare ../FedML/data/fed_shakespeare rnn hetero 59361 0.03 sgd sch_random -v

## Federated EMNIST (non-i.i.d)
sh run_fedavg_standalone_pytorch.sh 0 femnist ../FedML/data/FederatedEMNIST/datasets cnn hetero 59361 0.03 sgd sch_random -v

## Fed_CIFAR100 (non-i.i.d)
sh run_fedavg_standalone_pytorch.sh 0 fed_cifar100 ../FedML/data/fed_cifar100 resnet18_gn hetero 59361 0.03 adam sch_random -v

# Stackoverflow_LR (non-i.i.d)
sh run_fedavg_standalone_pytorch.sh 0 stackoverflow_lr ../FedML/data/stackoverflow lr hetero 59361 0.03 sgd sch_random -v

# Stackoverflow_NWP (non-i.i.d)
sh run_fedavg_standalone_pytorch.sh 0 stackoverflow_nwp ../FedML/data/stackoverflow rnn hetero 59361 0.03 sgd sch_random -v
 
# CIFAR10 (non-i.i.d) 
sh run_fedavg_standalone_pytorch.sh 0 cifar10 ../FedML/data/cifar10 resnet56 hetero 59361 0.03 sgd sch_random -v
```

All above datasets are heterogeneous non-i.i.d distributed.

### Clean Partial Results
There might be some partial results generated during training which are stored in following directories, including `./src/wandb`, `./src/__pycache`, `./src/runs` and `./src/result`. They may not be used anymore. If you want to remove them from the current workspace and make sure to backup the dir `./src/result` which stores almost all the information of training process to dir `./src/outs`, you can simply run the following instruction. 

```bash
    sh clean.sh
```

Or, you want to specify the name of the backup file, you can run

```bash
    sh clean.sh [NAME]
```

One usage example can be shown by run `sh clean.sh help`.

### Benchmark Results
We publish benchmark experimental results at wanb.com: \
https://app.wandb.ai/automl/fedml
