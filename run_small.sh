#!/bin/bash
time python scripts/mbexp.py \
    -env $1 \
    -o ctrl_cfg.opt_cfg.cfg.popsize 200 \
    -o ctrl_cfg.opt_cfg.cfg.num_elites 20 \
    -o ctrl_cfg.opt_cfg.plan_hor 5 \
    -logdir log/test-$1
