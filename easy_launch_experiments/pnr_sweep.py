from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import argparse
import pythonplusplus.machine_learning.hyperparameter as hyp
import tensorflow as tf
from dotmap import DotMap
from easy_launcher.launcher_util import run_experiment

from dmbrl.config import create_config
from dmbrl.controllers.MPC import MPC
from dmbrl.misc.MBExp import MBExperiment


def exp(variant):
    from easy_logger import logger
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="2"

    tf.reset_default_graph()
    config_module_kwargs = {}
    overrides = []
    for override_key, value in variant['override_params'].items():
        overrides.append([override_key.replace('-', '.'), value])
    cfg = create_config(
        env_name="sawyer_push_and_reach",
        ctrl_type="MPC",
        ctrl_args=DotMap(),
        overrides=overrides,
        logdir=logger.get_snapshot_dir(),
        config_module_kwargs=config_module_kwargs,
    )
    # HACK
    cfg.ctrl_cfg.opt_cfg.task_hor = cfg.exp_cfg.sim_cfg.task_hor
    cfg.exp_cfg.log_cfg.logdir = logger.get_snapshot_dir()

    cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)
    exp = MBExperiment(cfg.exp_cfg)

    config_dict = cfg.toDict()
    config_dict['config_module_kwargs'] = config_module_kwargs
    logger.log_variant(
        os.path.join(exp.logdir, 'complete_variant.json'),
        config_dict,
    )

    print("log dir:", exp.logdir)

    exp.run_experiment()


def main():
    n_seeds = 1
    mode = 'local_docker'
    # mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )
    mode = 'here_no_doodad'
    exp_prefix = 'dev-time'

    n_seeds = 1
    # mode = 'sss'
    exp_prefix = 'deepthought-neur-rebut-pets-pnr-take3'

    search_space = {
        "override_params.exp_cfg-exp_cfg-nrollouts_per_iter": [10],
        "override_params.exp_cfg-log_cfg-nrecord": [0],
        "override_params.exp_cfg-log_cfg-record_period": [1],
        "override_params.exp_cfg-log_cfg-neval": [0],
        "override_params.exp_cfg-log_cfg-nrecord_eval_mode": [0],
        "override_params.exp_cfg-log_cfg-neval_eval_mode": [0],
        "override_params.ctrl_cfg-per": [5],
        "override_params.exp_cfg-exp_cfg-ntrain_iters": [2000],
        "override_params.ctrl_cfg-opt_cfg-cfg-popsize": [200],
        "override_params.ctrl_cfg-opt_cfg-cfg-num_elites": [20],
        "override_params.ctrl_cfg-opt_cfg-cfg-max_iters": [5],
        "override_params.ctrl_cfg-max_num_data": [100000],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=dict(
            steps_needed_to_solve=5,
        ),
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                exp,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
                snapshot_mode='gap_and_last',
                snapshot_gap=50,
                use_gpu=True,
                time_in_mins=int(2.9*24*60),
            )

if __name__ == "__main__":
    main()
