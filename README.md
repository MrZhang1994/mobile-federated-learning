# Federated Averaging ![visitors](https://visitor-badge.glitch.me/badge?page_id=MrZhang1994.mobile-federated-learning)

## Requirements

1. Make sure GPU is avaible and `CUDA>=11.0` has been installed on your computer. You can check it with
    ```bash
        nvidia-smi
    ```
2. Download the repo on Github [FedML](https://github.com/FedML-AI/FedML)
    ```bash
        git clone https://github.com/FedML-AI/FedML.git
    ```
3. Follow the instruction or documentation of [FedML](https://github.com/FedML-AI/FedML) to install required package in python environment. Or, you can simply create an virtural environment with `python>=3.8` and run `pip install -r requirements.txt` to download the required packages. If you use `anaconda3` or `miniconda`, you can run following instructions to download the required packages in python. 
    ```bash
        conda create -y -n mfl python=3.8
        conda activate mfl
        pip install pip --upgrade
        pip install -r requirements.txt
    ```
4. Change the root dir in `main_fedavg.py` to the absolute path of `FedML`. For example,
    ```python
        sys.path.insert(0, os.path.abspath("/home/zzp1012/FedML")) # add the root dir of FedML
    ```

## Experimental Tracking Platform 

report real-time result to wandb.com, please change ID to your own

```
wandb login `Your ID`
```

Also, we integrate tensorboard into our training process. With the instruction in [Tensorboard Instructions](https://www.jianshu.com/p/46eb3004beca), you can easily check the learning curve in tensorboard.


## Experiment Scripts

To test whether program are correctly configured, you can run following commands to see whether training process starts correctly.

```
sh begin.sh -[VERBOSE]
```

Or, you can try other heterogeneous distribution (Non-IID) experiment:
``` 
## MNIST (non-i.i.d)
sh run_fedavg_standalone_pytorch.sh 0 100 mnist [Your PATH to FedML]/data/MNIST lr hetero 59361 0.03 sgd sch_random -f -v

## shakespeare (non-i.i.d LEAF)
sh run_fedavg_standalone_pytorch.sh 0 100 shakespeare [Your PATH to FedML]/data/shakespeare rnn hetero 59361 0.03 sgd sch_random -f -v

# fed_shakespeare (non-i.i.d Google)
sh run_fedavg_standalone_pytorch.sh 0 100 fed_shakespeare [Your PATH to FedML]/data/fed_shakespeare rnn hetero 59361 0.03 sgd sch_random -f -v

## Federated EMNIST (non-i.i.d)
sh run_fedavg_standalone_pytorch.sh 0 100 femnist [Your PATH to FedML]/data/FederatedEMNIST/datasets cnn hetero 59361 0.03 sgd sch_random -f -v

## Fed_CIFAR100 (non-i.i.d)
sh run_fedavg_standalone_pytorch.sh 0 100 fed_cifar100 [Your PATH to FedML]/data/fed_cifar100 resnet18_gn hetero 59361 0.03 adam sch_random -f -v

# Stackoverflow_LR (non-i.i.d)
sh run_fedavg_standalone_pytorch.sh 0 100 stackoverflow_lr [Your PATH to FedML]/data/stackoverflow lr hetero 59361 0.03 sgd sch_random -f -v

# Stackoverflow_NWP (non-i.i.d)
sh run_fedavg_standalone_pytorch.sh 0 100 stackoverflow_nwp [Your PATH to FedML]/data/stackoverflow rnn hetero 59361 0.03 sgd sch_random -f -v
 
# CIFAR10 (non-i.i.d) 
sh run_fedavg_standalone_pytorch.sh 0 100 cifar10 [Your PATH to FedML]/data/cifar10 resnet56 hetero 59361 0.03 sgd sch_random -f -v
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
