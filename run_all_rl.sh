#!/usr/bin/env bash

for method in ddpg pg random ddpg_baseline pg_amender
do
  RL_PRESET=$method sh begin.sh
done

