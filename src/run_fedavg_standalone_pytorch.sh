#!/usr/bin/env bash

GPU=$1

DATASET=$2

DATA_PATH=$3

MODEL=$4

DISTRIBUTION=$5

ROUND=$6

LR=$7

OPT=$8

METHOD=$9

VERBOSE=$10 #-v

if [ $METHOD = "all" ]; then
    python3 ./main_fedavg.py \
    --gpu $GPU \
    --dataset $DATASET \
    --data_dir $DATA_PATH \
    --model $MODEL \
    --partition_method $DISTRIBUTION  \
    --comm_round $ROUND \
    --client_optimizer $OPT \
    --lr $LR \
    --method "sch_mpn" \
    $VERBOSE

    python3 ./main_fedavg.py \
    --gpu $GPU \
    --dataset $DATASET \
    --data_dir $DATA_PATH \
    --model $MODEL \
    --partition_method $DISTRIBUTION  \
    --comm_round $ROUND \
    --client_optimizer $OPT \
    --lr $LR \
    --method "sch_random" \
    $VERBOSE

    python3 ./main_fedavg.py \
    --gpu $GPU \
    --dataset $DATASET \
    --data_dir $DATA_PATH \
    --model $MODEL \
    --partition_method $DISTRIBUTION  \
    --comm_round $ROUND \
    --client_optimizer $OPT \
    --lr $LR \
    --method "sch_channel" \
    $VERBOSE

    python3 ./main_fedavg.py \
    --gpu $GPU \
    --dataset $DATASET \
    --data_dir $DATA_PATH \
    --model $MODEL \
    --partition_method $DISTRIBUTION  \
    --comm_round $ROUND \
    --client_optimizer $OPT \
    --lr $LR \
    --method "sch_rrobin" \
    $VERBOSE

    python3 ./main_fedavg.py \
    --gpu $GPU \
    --dataset $DATASET \
    --data_dir $DATA_PATH \
    --model $MODEL \
    --partition_method $DISTRIBUTION  \
    --comm_round $ROUND \
    --client_optimizer $OPT \
    --lr $LR \
    --method "sch_loss" \
    $VERBOSE

else
    python3 ./main_fedavg.py \
    --gpu $GPU \
    --dataset $DATASET \
    --data_dir $DATA_PATH \
    --model $MODEL \
    --partition_method $DISTRIBUTION  \
    --comm_round $ROUND \
    --client_optimizer $OPT \
    --lr $LR \
    --method $METHOD \
    $VERBOSE
fi