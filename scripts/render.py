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
        ctrl_type, ctrl_args, overrides, model_dir, logdir,
        init_iter,
        last_iter,
        nrecord,
        rawdir,
):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})

    overrides.append(["ctrl_cfg.prop_cfg.model_init_cfg.model_dir", model_dir])
    overrides.append(["ctrl_cfg.prop_cfg.model_init_cfg.load_model", "True"])
    overrides.append(["ctrl_cfg.prop_cfg.model_pretrained", "True"])
    overrides.append(["exp_cfg.exp_cfg.ninit_rollouts", "0"])
    overrides.append(["exp_cfg.exp_cfg.init_iter", str(init_iter)])
    overrides.append(["exp_cfg.exp_cfg.ntrain_iters", str(last_iter)])
    overrides.append(["exp_cfg.log_cfg.nrecord", str(nrecord)])
    overrides.append(["exp_cfg.log_cfg.rawdir", str(rawdir)])

    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
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
    parser.add_argument('-env', type=str, required=True)
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2,
                        default=[])
    parser.add_argument('-o', '--override', action='append', nargs=2,
                        default=[])
    parser.add_argument('-model-dir', type=str, required=True)
    parser.add_argument('-logdir', type=str, required=True)
    parser.add_argument('-init-iter', type=int, default=0)
    parser.add_argument('-last-iter', type=int, default=1)
    parser.add_argument('-nrecord', type=int, default=1)
    parser.add_argument('-raw-dir', action='store_false')
    args = parser.parse_args()

    main(
        args.env,
        "MPC",
        args.ctrl_arg,
        args.override,
        args.model_dir,
        args.logdir,
        args.init_iter,
        args.last_iter,
        args.nrecord,
        args.raw_dir,
    )
