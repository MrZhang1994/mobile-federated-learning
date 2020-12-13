#!/usr/bin/env bash

GPU=$1

BATCH_SIZE=$2

DATASET=$3

DATA_PATH=$4

MODEL=$5

DISTRIBUTION=$6

ROUND=$7

EPOCH=$8

LR=$9

OPT=$10

CI=$11

METHOD=$12

VERBOSE=$13 #-v

if [ $METHOD = "all" ]; then
    python3 ./main_fedavg.py \
    --gpu $GPU \
    --dataset $DATASET \
    --data_dir $DATA_PATH \
    --model $MODEL \
    --partition_method $DISTRIBUTION  \
    --comm_round $ROUND \
    --epochs $EPOCH \
    --batch_size $BATCH_SIZE \
    --client_optimizer $OPT \
    --lr $LR \
    --ci $CI \
    --method "sch_mpn" \
    $VERBOSE

    python3 ./main_fedavg.py \
    --gpu $GPU \
    --dataset $DATASET \
    --data_dir $DATA_PATH \
    --model $MODEL \
    --partition_method $DISTRIBUTION  \
    --comm_round $ROUND \
    --epochs $EPOCH \
    --batch_size $BATCH_SIZE \
    --client_optimizer $OPT \
    --lr $LR \
    --ci $CI \
    --method "sch_random" \
    $VERBOSE


    python3 ./main_fedavg.py \
    --gpu $GPU \
    --dataset $DATASET \
    --data_dir $DATA_PATH \
    --model $MODEL \
    --partition_method $DISTRIBUTION  \
    --comm_round $ROUND \
    --epochs $EPOCH \
    --batch_size $BATCH_SIZE \
    --client_optimizer $OPT \
    --lr $LR \
    --ci $CI \
    --method "sch_channel" \
    $VERBOSE

    python3 ./main_fedavg.py \
    --gpu $GPU \
    --dataset $DATASET \
    --data_dir $DATA_PATH \
    --model $MODEL \
    --partition_method $DISTRIBUTION  \
    --comm_round $ROUND \
    --epochs $EPOCH \
    --batch_size $BATCH_SIZE \
    --client_optimizer $OPT \
    --lr $LR \
    --ci $CI \
    --method "sch_rrobin" \
    $VERBOSE


    python3 ./main_fedavg.py \
    --gpu $GPU \
    --dataset $DATASET \
    --data_dir $DATA_PATH \
    --model $MODEL \
    --partition_method $DISTRIBUTION  \
    --comm_round $ROUND \
    --epochs $EPOCH \
    --batch_size $BATCH_SIZE \
    --client_optimizer $OPT \
    --lr $LR \
    --ci $CI \
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
    --epochs $EPOCH \
    --batch_size $BATCH_SIZE \
    --client_optimizer $OPT \
    --lr $LR \
    --ci $CI \
    --method $METHOD \
    $VERBOSE
fi