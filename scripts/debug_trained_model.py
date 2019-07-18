from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint

from dotmap import DotMap

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config


def main(
        env,
        ctrl_type,
        ctrl_args,
        overrides,
        model_dir,
        logdir,
        config_module_kwargs,
):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})

    overrides += [
        ["ctrl_cfg.prop_cfg.model_init_cfg.model_dir", model_dir],
        ["ctrl_cfg.prop_cfg.model_init_cfg.load_model", "True"],
        ["ctrl_cfg.prop_cfg.model_pretrained", "True"],
        ["exp_cfg.sim_cfg.task_hor", "12"],
        ["exp_cfg.exp_cfg.ninit_rollouts", "0"],
        ["exp_cfg.exp_cfg.ntrain_iters", "1"],
        # ["exp_cfg.exp_cfg.nrollouts_per_iter", "1"],
        # ["exp_cfg.log_cfg.nrecord", "1"],
        # ["exp_cfg.log_cfg.neval", "1"],
        ["exp_cfg.exp_cfg.nrollouts_per_iter", "0"],
        ["exp_cfg.log_cfg.nrecord", "0"],
        ["exp_cfg.log_cfg.neval", "0"],
        # ["exp_cfg.log_cfg.nrecord_eval_mode", "0"],
        # ["exp_cfg.log_cfg.neval_eval_mode", "0"],
        ["exp_cfg.log_cfg.nrecord_eval_mode", "0"],
        ["exp_cfg.log_cfg.neval_eval_mode", "1"],
        ["ctrl_cfg.per", "5"],
        ["ctrl_cfg.opt_cfg.cfg.popsize", "30000"],
        ["ctrl_cfg.opt_cfg.cfg.num_elites", "1"],
        ["ctrl_cfg.opt_cfg.cfg.max_iters", "1"],
        ["ctrl_cfg.opt_cfg.cfg.alpha", "0"],
        # ["ctrl_cfg.opt_cfg.cfg.popsize", "4"e],
        # ["ctrl_cfg.opt_cfg.cfg.num_elites", "4"],
        # ["ctrl_cfg.opt_cfg.cfg.max_iters", "5"],
        ["ctrl_cfg.opt_cfg.plan_hor", "5"],
        ["ctrl_cfg.opt_cfg.init_var_scale", "1."],
        ["ctrl_cfg.log_cfg.log_particles", "True"],
        ["ctrl_cfg.log_cfg.log_traj_preds", "True"],
    ]
    # overrides.append(["exp_cfg.log_cfg.rawdir", str(rawdir)])

    cfg = create_config(
        env, ctrl_type, ctrl_args, overrides, logdir,
        config_module_kwargs=config_module_kwargs
    )
    # HACK
    cfg.ctrl_cfg.opt_cfg.task_hor = cfg.exp_cfg.sim_cfg.task_hor
    cfg.pprint()

    if ctrl_type == "MPC":
        cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)
    exp = MBExperiment(cfg.exp_cfg)

    if os.path.exists(exp.logdir):
        overwrite = user_prompt(
            "{} already exists. Overwrite?".format(exp.logdir)
        )
        if not overwrite:
            return
    else:
        os.makedirs(exp.logdir)
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))

    exp.run_experiment()
    print("Saved to")
    print(exp.logdir)

    # import ipdb; ipdb.set_trace()
    # policy = exp.policy
    # env = exp.env
    # agent = exp.agent
    # H = exp.task_hor
    # samples = []
    # for i in range(nrecord):
    #     iter_dir = os.path.join(logdir, "train_iter%d" % (i + 1))
    #     os.makedirs(iter_dir, exist_ok=True)
    #     samples.append(
    #         agent.sample(
    #             H, policy,
    #             os.path.join(iter_dir, "rollout%d.mp4" % j)
    #         )
    #     )


def user_prompt(question: str) -> bool:
    """

    Prompt the yes/no-*question* to the user.
    https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    from distutils.util import strtobool

    while True:
        user_input = input(question + " [y/n]: ").lower()
        try:
            result = strtobool(user_input)
            return result
        except ValueError:
            print("Please use y/n or yes/no.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-env', type=str, required=True)
    # parser.add_argument('-model-dir', type=str, required=True)
    # parser.add_argument('-logdir', type=str, required=True)
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2,
                        default=[])
    parser.add_argument('-o', '--override', action='append', nargs=2,
                        default=[])
    parser.add_argument('-init-iter', type=int, default=0)
    parser.add_argument('-last-iter', type=int, default=1)
    parser.add_argument('-nrecord', type=int, default=1)
    parser.add_argument('-no-raw-dir', action='store_true')
    args = parser.parse_args()
    base_dir = "/home/vitchyr/git/handful-of-trials/log/point-uwall-sweep-after-terminal-cost-fix-2/2019-07-10--15:49:40/"

    main(
        env="pointmass_u_wall",
        ctrl_type="MPC",
        ctrl_args=args.ctrl_arg,
        overrides=args.override,
        model_dir=base_dir,
        logdir=base_dir + "debug_log",
        config_module_kwargs={
            "planning_horizon": 4,
            "steps_needed_to_solve": 4,
            "task_horizon_factor": 4
        },
    )
