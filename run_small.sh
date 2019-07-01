#!/bin/bash
env=$1
envname=${1//_/-}
expname=${2//_/-}
time python scripts/mbexp.py \
    -env $env \
    -o ctrl_cfg.opt_cfg.cfg.popsize 200 \
    -o ctrl_cfg.opt_cfg.cfg.num_elites 20 \
    -o ctrl_cfg.opt_cfg.plan_hor 5 \
    -logdir log/dev--$envname--$expname
