import gym
from torch.distributions import Categorical

from Agents.Behavior import Behavior
import torch
from gym import wrappers


print("CUDA availability: ", torch.cuda.is_available())
device = torch.device("cpu")

RETURN_SCALE = 0.01  # Scaling factor for desired horizon input (reward)
HORIZON_SCALE = 0.01  # Scaling factor for desired horizon input (steps)


# env = gym.make("LunarLander-v2")
env_to_wrap = gym.make("LunarLander-v2")
env = wrappers.Monitor(env_to_wrap, 'videos/', force=True)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy = Behavior(state_dim, 2, action_dim, 0.05, device)
policy.load_state_dict(torch.load("models/my_udrl_800"))


for idx in range(10):
    state = env.reset().tolist()
    done = False
    command_horizon, command_reward = 300, 317
    step = 0
    ep_rew = 0
    while not done:

        with torch.no_grad():
            action = policy.sample_action(state, [command_horizon * HORIZON_SCALE, command_reward * RETURN_SCALE])
        state, reward, done, info = env.step(action)
        state = state.tolist()
        ep_rew += reward
        command_horizon = max(command_horizon - 1, 1)
        env.render()
    print(ep_rew)

env.close()
env_to_wrap.close()
