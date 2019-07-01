#!/bin/bash
time python scripts/mbexp.py \
    -env pointmass_reach_fixed_point \
    -o ctrl_cfg.opt_cfg.cfg.popsize 50 \
    -o ctrl_cfg.opt_cfg.cfg.num_elites 5 \
    -o ctrl_cfg.opt_cfg.plan_hor 5 \
    -logdir log/test-$1
    #-env PointmassUWallTestEnvBig-v1 \
