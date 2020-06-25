from collections import namedtuple
from Configs.UDRL import *

import numpy as np


Episode = namedtuple('Episode',
                     field_names=['states',
                                  'actions',
                                  'rewards',
                                  'init_command',
                                  'total_return',
                                  'length',
                                  ])


def sample_command(buffer, last_few):
    if len(buffer) == 0:
        return [1, 1]

    # 1.
    episodes = buffer.top_episodes(last_few)

    # 2.
    lengths = [episode.length for episode in episodes]
    desired_horizon = min(np.round(np.mean(lengths)), MAX_STEPS)

    # 3.
    returns = [episode.total_return for episode in episodes]
    mean_return, std_return = np.mean(returns), np.std(returns)
    desired_return = np.random.uniform(mean_return, mean_return+std_return)

    return [desired_horizon, desired_return]


def generate_command(target_horizon, target_reward_mean, target_reward_std):
    target_horizon = min(target_horizon, MAX_STEPS)
    target_reward = round(np.random.random_sample() * target_reward_std + target_reward_mean, 0)
    return [target_horizon, target_reward]


def log_episode(experiment, context, episode: Episode, step):
    with experiment.context_manager(context):
        experiment.log_metric("episode reward", episode.total_return, step=step)
        experiment.log_metric("episode length", episode.length, step=step)
        experiment.log_metric("command horizon", episode.init_command[0], step=step)
        experiment.log_metric("command return", episode.init_command[1], step=step)


