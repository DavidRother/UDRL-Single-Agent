from comet_ml import Experiment
from Agents.Behavior import Behavior
from util.ReplayBuffer import ReplayBuffer
from util.util import sample_command, log_episode, generate_command
from Reinforce.training import Trainer
from Reinforce.episode import generate_exploratory_episode, generate_random_episode, generate_sample_episode
from Configs.UDRL import *
from Configs.Logging import *
from Configs.Environment import *

import numpy as np
import torch
import warnings
import gym

experiment = Experiment(api_key=COMET_API_KEY, project_name=COMET_PROJECT_NAME, workspace=COMET_WORKSPACE)
# log hyperparameters
experiment.log_parameters({
    "MAIN_LOOP_ITERATIONS": MAIN_LOOP_ITERATIONS,
    "BATCH_SIZE": BATCH_SIZE,
    "HORIZON_SCALE": HORIZON_SCALE,
    "RETURN_SCALE": RETURN_SCALE,
    "LAST_FEW": LAST_FEW,
    "LEARNING_RATE": LEARNING_RATE,
    "NUM_EXPLORATORY_EPISODES": NUM_EXPLORATORY_EPISODES,
    "NUM_UPDATES_PER_TRAINING_EPOCH": NUM_UPDATES_PER_TRAINING_EPOCH,
    "NUM_WARM_UP_EPISODES": NUM_WARM_UP_EPISODES,
    "REPLAY_BUFFER_SIZE": REPLAY_BUFFER_SIZE,
    "EVALUATION_FREQUENCY": EVALUATION_FREQUENCY,
    "MAX_STEPS": MAX_STEPS,
    "NUM_EVALUATIONS": NUM_EVALUATIONS,
    "EPSILON_GREEDY": EPSILON_GREEDY,
    "RANDOM_SEED": RANDOM_SEED})

# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)

warnings.filterwarnings("ignore")

# Traing on GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

env = gym.make('LunarLander-v2')  # RocketLander-v0 | LunarLander-v2 | MountainCar-v0 | CartPole-v0
# _ = env.seed(RANDOM_SEED)
# RANDOM_SEED = 12321345
# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)
# _ = env.seed(RANDOM_SEED)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
command_size = 2  # desired horizon and return


buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

# Sample random episodes
init_command = sample_command(buffer, LAST_FEW)
for idx in range(NUM_WARM_UP_EPISODES):
    buffer.add_episode(generate_random_episode(env, init_command, render=False))

exploratory_episode_counter = 0
evaluation_episode_counter = 0

behavior = Behavior(state_size, command_size, action_size, epsilon=EPSILON_GREEDY, device=device)
trainer = Trainer(behavior.parameters(), LEARNING_RATE)

for epoch in range(1, MAIN_LOOP_ITERATIONS + 1):

    episodes_to_train = buffer.sample_episodes(LAST_FEW)
    mean_loss = trainer.train(behavior, episodes_to_train, NUM_UPDATES_PER_TRAINING_EPOCH, BATCH_SIZE)

    experiment.log_metric("Mean Loss", mean_loss, step=epoch)

    top_episodes = buffer.top_episodes(LAST_FEW)  # [(S,A,R,S_), ... ]
    tgt_horizon = int(np.mean([x.length for x in top_episodes]))
    tgt_reward_mean = np.mean([x.total_return for x in top_episodes])
    tgt_reward_std = np.std([x.total_return for x in top_episodes])

    experiment.log_metric("tgt_reward_mean", tgt_reward_mean, step=exploratory_episode_counter)
    # Sample exploratory commands and generate episodes
    for idx in range(NUM_EXPLORATORY_EPISODES):
        init_command = generate_command(tgt_horizon, tgt_reward_mean, tgt_reward_std)
        episode = generate_sample_episode(env, behavior, init_command)
        log_episode(experiment, "Exploratory", episode, exploratory_episode_counter)
        buffer.add_episode(episode)
        exploratory_episode_counter += 1

    if (epoch % EVALUATION_FREQUENCY) == 0:
        behavior.save(f"models/my_udrl_{epoch}")
        with torch.no_grad():
            behavior.eval()
            for e in range(NUM_EVALUATIONS):
                command = generate_command(tgt_horizon, tgt_reward_mean, tgt_reward_std)
                episode = generate_sample_episode(env, behavior, command, False)
                log_episode(experiment, "Evaluation", episode, evaluation_episode_counter)
                evaluation_episode_counter += 1
            behavior.train()

behavior.save(f"models/my_udrl_final")
