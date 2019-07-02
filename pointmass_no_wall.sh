#!/bin/bash
time python scripts/mbexp.py \
    -env pointmass_no_walls \
    -o exp_cfg.log_cfg.nrecord 1 \
    -logdir log/pointmass-no-walls-cartpole-settings-squared-loss
    #-env PointmassUWallTestEnvBig-v1 \
