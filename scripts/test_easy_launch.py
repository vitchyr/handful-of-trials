from easy_launcher.launcher_util import run_experiment


def test(variant):
    import tensorflow
    import mujoco_py
    print(tensorflow.__version__)
    print(mujoco_py.__version__)
    return


run_experiment(
    test,
    mode='local_docker',
    variant=dict(
        a=123,
    ),
    verbose=True,
    use_gpu=True,
)
