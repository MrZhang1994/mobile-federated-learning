#!/usr/bin/env bash

for method in pg_noamender ac_noamender
do
    RL_PRESET=$method sh run_rl.sh
done
