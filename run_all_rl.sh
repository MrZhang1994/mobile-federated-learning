#!/usr/bin/env bash

for method in ddpg pg random ddpg_baseline pg_amender pg_noamender
do
    RL_PRESET=$method ./begin.sh
done
