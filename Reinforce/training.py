from Configs.UDRL import *
from Reinforce.BehaviorDataset import BehaviorDataset
from udrl.utils import BehaviorDataset as BD
from torch.utils.data import DataLoader as TorchDataLoader

import numpy as np
import torch


class Trainer:

    def __init__(self, parameters, learning_rate):
        self.loss_func = torch.nn.NLLLoss()
        self.optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    def train(self, behavior, buffer, n_updates, batch_size):
        all_loss = []
        train_dset = BehaviorDataset(buffer,
                                     size=batch_size * n_updates,
                                     horizon_scale=HORIZON_SCALE,
                                     return_scale=RETURN_SCALE, device=behavior.device)
        training_behaviors = TorchDataLoader(train_dset,
                                             batch_size=batch_size, shuffle=True, num_workers=0)

        if not behavior.training:
            behavior.train()
        for idx, behavior_batch in enumerate(training_behaviors):  # this runs for NUM_UPDATES_PER_ITER rounds
            behavior.zero_grad()
            logprobs = behavior(behavior_batch['features'])
            loss = self.loss_func(logprobs, behavior_batch['label'])
            loss.backward()
            self.optimizer.step()
            all_loss.append(loss.cpu().detach())

        return np.mean(all_loss)
