#!/usr/bin/env bash
cd ./src
sh run_fedavg_standalone_pytorch.sh 0 10 femnist /zzp/FedML/data/FederatedEMNIST/datasets cnn hetero 10000 0.03 sgd sch_random -v