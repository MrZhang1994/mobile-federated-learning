#!/usr/bin/env bash

sh run_fedavg_standalone_pytorch.sh 0 10 femnist /csh/mobile-FL/FedML-master/data/FederatedEMNIST/ cnn hetero 500 1 0.03 sgd 0 sch_mpn