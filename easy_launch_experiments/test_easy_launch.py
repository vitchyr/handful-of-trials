from easy_launcher.launcher_util import run_experiment


def test(variant):
    import tensorflow
    from easy_logger import logger
    import mujoco_py
    print(tensorflow.__version__)
    print(mujoco_py)
    print(variant)
    print(logger.get_snapshot_dir())
    return


run_experiment(
    test,
    # exp_prefix='generate_pets_ami',
    mode='local_docker',
    variant=dict(
        a=123,
    ),
    verbose=True,
    use_gpu=True,
)
