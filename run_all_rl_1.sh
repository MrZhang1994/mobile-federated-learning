#!/usr/bin/env bash

for method in pg ac pg_amender ac_amender pg_noamender ac_noamender
do
    RL_PRESET=$method sh run_rl.sh
done
