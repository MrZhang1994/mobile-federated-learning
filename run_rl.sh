#!/usr/bin/env bash

cd src
python3 ./main_fedavg.py \
    --gpu 0 \
    --dataset femnist \
    --data_dir ../FedML/data/FederatedEMNIST/datasets \
    --model cnn \
    --partition_method hetero  \
    --comm_round 59361 \
    --client_optimizer sgd \
    --lr 0.03 \
    --method sch_pn_method_1
