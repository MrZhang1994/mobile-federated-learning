#!/usr/bin/env bash

sh run_fedavg_standalone_pytorch.sh 0 10 femnist /zzp/FedML/data/FederatedEMNIST/datasets cnn hetero 500 1 0.03 sgd 0 sch_mpn