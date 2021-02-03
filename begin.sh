#!/usr/bin/env bash
cd ./src
VERBOSE=$1
sh run_fedavg_standalone_pytorch.sh 0 femnist ../FedML/data/FederatedEMNIST/datasets cnn hetero 59361 0.03 sgd sch_pn_method_3 -c $VERBOSE