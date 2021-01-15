#!/usr/bin/env bash

for method in pg_5 ac_5 pg_amender_5 ac_amender_5 pg_noamender_5 ac_noamender_5
do
    RL_PRESET=$method sh run_rl.sh
done
