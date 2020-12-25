#!/usr/bin/env sh

cd src && ./run_fedavg_standalone_pytorch.sh 0 10 femnist ../FedML/data/FederatedEMNIST/datasets cnn hetero 500 1 0.01 sgd 0 sch_mpn
