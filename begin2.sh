#!/usr/bin/env bash
cd ./src

# python main_fedavg.py\
#  --gpu 0 \
#  --dataset femnist \
#  --data_dir /csh/mobile-FL/FedML-master/data/FederatedEMNIST/datasets \
#  --model cnn \
#  --partition_method hetero \
#  --comm_round 100000 \
#  --client_optimizer sgd \
#  --lr 0.03 \
#  --method sch_channel

python main_fedavg.py\
 --gpu 1 \
 --dataset femnist \
 --data_dir /csh/mobile-FL/FedML-master/data/FederatedEMNIST/datasets \
 --model cnn \
 --partition_method hetero \
 --comm_round 100000 \
 --client_optimizer sgd \
 --lr 0.03 \
 --method sch_rrobin

 python main_fedavg.py\
 --gpu 1 \
 --dataset femnist \
 --data_dir /csh/mobile-FL/FedML-master/data/FederatedEMNIST/datasets \
 --model cnn \
 --partition_method hetero \
 --comm_round 100000 \
 --client_optimizer sgd \
 --lr 0.03 \
 --method sch_loss

 # python main_fedavg.py\
 # --gpu 0 \
 # --dataset femnist \
 # --data_dir /csh/mobile-FL/FedML-master/data/FederatedEMNIST/datasets \
 # --model cnn \
 # --partition_method hetero \
 # --comm_round 100000 \
 # --client_optimizer sgd \
 # --lr 0.03 \
 # --method sch_random


