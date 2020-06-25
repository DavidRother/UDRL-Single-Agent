from torch import nn
from torch.distributions import Categorical
from typing import List
from Configs.UDRL import *

import torch
import numpy as np
import random


class Behavior(nn.Module):

    def __init__(self, state_size, command_size, action_size, epsilon=0.05, device='cpu'):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size + command_size, 64),  # 2 = command_dim
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
            nn.LogSoftmax(dim=-1)
        )

        self.to(device)
        self.device = device
        self.epsilon = epsilon
        self.state_size = state_size
        self.command_size = command_size
        self.action_size = action_size

    def forward(self, features: torch.Tensor):
        return self.model(features)

    def sample_action(self, state: List[float], command: List[float]):
        return self.action_distribution(state, command).sample().item()

    def greedy_action(self, state: List[float], command: List[float]):
        probabilities = self.action_log_probabilities(state, command)
        return np.argmax(probabilities.detach().cpu().numpy())

    def exploratory_action(self, state: List[float], command: List[float]):
        if random.uniform(0, 1) <= self.epsilon:
            return np.random.randint(self.action_size)
        return self.action_distribution(state, command).sample().item()

    def action_distribution(self, state: List[float], command: List[float]):
        action_log_probabilities = self.action_log_probabilities(state, command)
        dist = Categorical(logits=action_log_probabilities)
        return dist

    def action_log_probabilities(self, state: List[float], command: List[float]):
        command[0] = command[0] * HORIZON_SCALE
        command[1] = command[1] * RETURN_SCALE
        features = state + command
        state = torch.FloatTensor(features).to(self.device)
        return self.forward(state)

    def save(self, filename: str):
        torch.save(self.state_dict(), filename)

    def load(self, filename: str, device='cpu'):
        self.load_state_dict(torch.load(filename))
        self.device = device
        self.to(device)
