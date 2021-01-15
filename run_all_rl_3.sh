#!/usr/bin/env bash

for method in random random_5
do
    RL_PRESET=$method sh run_rl.sh
done
