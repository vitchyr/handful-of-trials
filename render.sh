#!/bin/bash
dir=$1
env=$2
iter=$3
time python scripts/render.py \
    -env $2 \
    -model-dir $1 \
    -logdir $1/renderlog \
    -init-iter 20 \
    -last-iter 21 \

    #-o ctrl_cfg.opt_cfg.cfg.popsize 5 \
    #-o ctrl_cfg.opt_cfg.cfg.num_elites 1 \
    #-o ctrl_cfg.opt_cfg.plan_hor 1 \
    #-o exp_cfg.sim_cfg.task_hor 2 \
