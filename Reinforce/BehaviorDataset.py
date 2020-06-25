from torch.utils.data import Dataset as TorchDataset

import numpy as np
import random
import torch


class BehaviorDataset(TorchDataset):
    """ Samples behavior segments for supervised learning
    from given input episodes.
    """
    def __init__(self, episodes, size, horizon_scale, return_scale, device):
        super(BehaviorDataset, self).__init__()
        self.episodes = episodes
        self.horizon_scale = horizon_scale
        self.return_scale = return_scale
        self.size = size
        self.device = device

    def __len__(self):
        # just returning a placeholder number for now
        return self.size

    def __getitem__(self, idx):
        # get episode
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]

        # randomly sample an episode
        episode = random.choice(self.episodes)

        # extract behavior segment
        start_index = np.random.choice(episode.length - 1)  # ensures cmd_steps >= 1
        command_horizon = (episode.length - start_index - 1)
        command_return = np.sum(episode.rewards[start_index:])
        command = command_horizon * self.horizon_scale, command_return * self.return_scale

        state = episode.states[start_index].copy()
        state.extend(command)

        # construct sample
        features = state
        label = episode.actions[start_index]               # action taken
        sample = {
            'features': torch.tensor(features, dtype=torch.float).to(self.device),
            'label': torch.tensor(label, dtype=torch.long).to(self.device)  # categorical val
        }
        return sample

