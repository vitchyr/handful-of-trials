import json
import os.path as osp
from collections import namedtuple, OrderedDict, defaultdict
from enum import Enum
from numbers import Number

import numpy as np

GitInfo = namedtuple(
    'GitInfo',
    [
        'directory',
        'code_diff',
        'code_diff_staged',
        'commit_hash',
        'branch_name',
    ],
)


def save_git_info(logdir):
    git_infos = get_git_info()
    if git_infos is not None:
        for (
                directory, code_diff, code_diff_staged, commit_hash, branch_name
        ) in git_infos:
            if directory[-1] == '/':
                diff_file_name = directory[1:-1].replace("/", "-") + ".patch"
                diff_staged_file_name = (
                        directory[1:-1].replace("/", "-") + "_staged.patch"
                )
            else:
                diff_file_name = directory[1:].replace("/", "-") + ".patch"
                diff_staged_file_name = (
                        directory[1:].replace("/", "-") + "_staged.patch"
                )
            if code_diff is not None and len(code_diff) > 0:
                with open(osp.join(logdir, diff_file_name), "w") as f:
                    f.write(code_diff + '\n')
            if code_diff_staged is not None and len(code_diff_staged) > 0:
                with open(osp.join(logdir, diff_staged_file_name), "w") as f:
                    f.write(code_diff_staged + '\n')
            with open(osp.join(logdir, "git_infos.txt"), "a") as f:
                f.write("directory: {}".format(directory))
                f.write('\n')
                f.write("git hash: {}".format(commit_hash))
                f.write('\n')
                f.write("git branch name: {}".format(branch_name))
                f.write('\n\n')


def get_git_info():
    try:
        import git
        dirs = [
            '/home/vitchyr/git/handful-of-trials/',
            '/home/vitchyr/git/multiworld/',
        ]

        git_infos = []
        for directory in dirs:
            # Idk how to query these things, so I'm just doing try-catch
            try:
                repo = git.Repo(directory)
                try:
                    branch_name = repo.active_branch.name
                except TypeError:
                    branch_name = '[DETACHED]'
                git_infos.append(GitInfo(
                    directory=directory,
                    code_diff=repo.git.diff(None),
                    code_diff_staged=repo.git.diff('--staged'),
                    commit_hash=repo.head.commit.hexsha,
                    branch_name=branch_name,
                ))
            except git.exc.InvalidGitRepositoryError:
                pass
    except ImportError:
        git_infos = None
    return git_infos


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        try:
            return json.JSONEncoder.default(self, o)
        except TypeError as e:
            if isinstance(o, object):
                return {
                    '$object': str(o)
                }
            else:
                raise e


def get_generic_path_information(paths, stat_prefix=''):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    if len(paths) == 0:
        return statistics
    returns = [sum(path["rewards"]) for path in paths]

    rewards = np.vstack([path["rewards"] for path in paths])
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))
    actions = [path["ac"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["ac"] for path in paths])
    else:
        actions = np.vstack([path["ac"] for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))

    statistics[stat_prefix + 'Num Paths'] = len(paths)
    for info_key in ['env_infos']:
        if info_key in paths[0]:
            all_env_infos = [
                list_of_dicts__to__dict_of_lists(p[info_key])
                for p in paths
            ]
            for k in all_env_infos[0].keys():
                final_ks = np.array([info[k][-1] for info in all_env_infos])
                first_ks = np.array([info[k][0] for info in all_env_infos])
                all_ks = np.concatenate([info[k] for info in all_env_infos])
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    final_ks,
                    stat_prefix='{}/final/'.format(info_key),
                ))
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    first_ks,
                    stat_prefix='{}/initial/'.format(info_key),
                ))
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    all_ks,
                    stat_prefix='{}/'.format(info_key),
                ))

    return statistics


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data).astype(np.float32)
        stats[name + ' Min'] = np.min(data).astype(np.float32)
    return stats


def list_of_dicts__to__dict_of_lists(lst):
    """
    ```
    x = [
        {'foo': 3, 'bar': 1},
        {'foo': 4, 'bar': 2},
        {'foo': 5, 'bar': 3},
    ]
    ppp.list_of_dicts__to__dict_of_lists(x)
    # Output:
    # {'foo': [3, 4, 5], 'bar': [1, 2, 3]}
    ```
    """
    if len(lst) == 0:
        return {}
    keys = lst[0].keys()
    output_dict = defaultdict(list)
    for d in lst:
        assert set(d.keys()) == set(keys)
        for k in keys:
            output_dict[k].append(d[k])
    return output_dict

