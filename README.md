# Federated Averaging

## Requirements

1. Download the repo on Github [FedML](https://github.com/FedML-AI/FedML)
2. Change the root dir in `main_fedavg.py` to the absolute path of `FedML`. For example,
    ```python
        sys.path.insert(0, os.path.abspath("/home/zzp1012/FedML")) # add the root dir of FedML
    ```
3. Follow the instruction or documentation of [FedML](https://github.com/FedML-AI/FedML) to install required package in python environment.

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
sh run_fedavg_standalone_pytorch.sh 0 100 mnist [Your PATH to FedML]/data/MNIST lr hetero 10000 0.03 sgd sch_random -f -v

## shakespeare (non-i.i.d LEAF)
sh run_fedavg_standalone_pytorch.sh 0 100 shakespeare [Your PATH to FedML]/data/shakespeare rnn hetero 10000 0.03 sgd sch_random -f -v

# fed_shakespeare (non-i.i.d Google)
sh run_fedavg_standalone_pytorch.sh 0 100 fed_shakespeare [Your PATH to FedML]/data/fed_shakespeare rnn hetero 10000 0.03 sgd sch_random -f -v

## Federated EMNIST (non-i.i.d)
sh run_fedavg_standalone_pytorch.sh 0 100 femnist [Your PATH to FedML]/data/FederatedEMNIST/datasets cnn hetero 10000 0.03 sgd sch_random -f -v

## Fed_CIFAR100 (non-i.i.d)
sh run_fedavg_standalone_pytorch.sh 0 100 fed_cifar100 [Your PATH to FedML]/data/fed_cifar100 resnet18_gn hetero 10000 0.03 adam sch_random -f -v

# Stackoverflow_LR (non-i.i.d)
sh run_fedavg_standalone_pytorch.sh 0 100 stackoverflow_lr [Your PATH to FedML]/data/stackoverflow lr hetero 10000 0.03 sgd sch_random -f -v

# Stackoverflow_NWP (non-i.i.d)
sh run_fedavg_standalone_pytorch.sh 0 100 stackoverflow_nwp [Your PATH to FedML]/data/stackoverflow rnn hetero 10000 0.03 sgd sch_random -f -v
 
# CIFAR10 (non-i.i.d) 
sh run_fedavg_standalone_pytorch.sh 0 100 cifar10 [Your PATH to FedML]/data/cifar10 resnet56 hetero 10000 0.03 sgd sch_random -f -v
```

All above datasets are heterogeneous non-i.i.d distributed.

### Benchmark Results
We publish benchmark experimental results at wanb.com: \
https://app.wandb.ai/automl/fedml
