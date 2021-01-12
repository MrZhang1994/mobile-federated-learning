#!/usr/bin/env bash

for method in random pg ac
do
    RL_PRESET=$method sh run_rl.sh
done
