from util.util import Episode
from Agents.Behavior import Behavior
from typing import List

import torch


def run_episode(env, policy, init_command: List[float], render=False) -> Episode:
    desired_horizon = init_command[0]
    desired_return = init_command[1]

    states = []
    actions = []
    rewards = []

    time_steps = 0
    done = False

    state = env.reset().tolist()
    total_reward = 0

    while not done:
        with torch.no_grad():
            action = policy(state, [desired_horizon, desired_return])
        next_state, reward, done, _ = env.step(action)

        # making the environment sparse here
        total_reward += reward
        desired_return -= total_reward * done

        states.append(state)
        actions.append(action)
        rewards.append(total_reward * done)
        state = next_state.tolist()

        # Make sure it's always a valid horizon
        desired_horizon = max(desired_horizon - 1, 1)

        time_steps += 1
        if render:
            env.render()

    return Episode(states, actions, rewards, init_command, total_reward, time_steps)


def generate_exploratory_episode(env, behavior: Behavior, init_command: List[float], render=False) -> Episode:
    return run_episode(env, behavior.exploratory_action, init_command, render)


def generate_sample_episode(env, behavior: Behavior, init_command: List[float], render=False) -> Episode:
    return run_episode(env, behavior.sample_action, init_command, render)


def generate_greedy_episode(env, behavior: Behavior, init_command: List[float], render=False) -> Episode:
    return run_episode(env, behavior.greedy_action, init_command, render)


def generate_random_episode(env, init_command, render=False) -> Episode:
    random_policy = lambda state, command: env.action_space.sample()
    return run_episode(env, random_policy, init_command, render)











