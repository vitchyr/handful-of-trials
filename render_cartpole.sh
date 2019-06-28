#!/bin/bash
time python scripts/render.py \
    -env cartpole \
    -o ctrl_cfg.opt_cfg.cfg.popsize 5 \
    -o ctrl_cfg.opt_cfg.cfg.num_elites 1 \
    -o ctrl_cfg.opt_cfg.plan_hor 1 \
    -o exp_cfg.sim_cfg.task_hor 2 \
    -model-dir /home/vitchyr/git/handful-of-trials/log/test/2019-06-28--13:15:24/ \
    -logdir /home/vitchyr/git/handful-of-trials/log/test/2019-06-28--13:15:24/renderlog

