#!/usr/bin/env bash

for method in pg_amender ac_amender
do
    RL_PRESET=$method sh run_rl.sh
done
