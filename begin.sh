#!/usr/bin/env bash
cd ./src
sh run_fedavg_standalone_pytorch.sh 1 10 femnist /home/zzp1012/FedML/data/FederatedEMNIST/datasets cnn hetero 500 1 0.03 sgd sch_mpn_empty