#!/usr/bin/env bash

cd src
for method in ddpg pg random ddpg_baseline pg_amender pg_noamender
do
    RL_PRESET=$method python3 ./main_fedavg.py \
    --gpu 0 \
    --dataset femnist \
    --data_dir ../FedML/data/FederatedEMNIST/datasets \
    --model cnn \
    --partition_method hetero  \
    --comm_round 59361 \
    --batch_size 100 \
    --client_optimizer sgd \
    --lr 0.03 \
    --method sch_pn_method_1 \
    -f
done
