import random
import util.util as util
from sortedcontainers import SortedList
from typing import List


class ReplayBuffer(object):
    """Implemented as a priority queue, where the priority value is
    set to be episode's total reward. Note that unlike usual RL buffers,
    we store entire 'trajectories' together, instead of just transitions.
    """

    def __init__(self, size):
        self.size = size
        self.buffer = SortedList([], key=lambda x: -x.total_return)  # order in descending ordering (best at idx 0)

    def __getitem__(self, key):
        return self.buffer[key]

    def __len__(self):
        return len(self.buffer)

    def add_episode(self, episode: util.Episode):
        if episode.length < 2:  # ignore episodes that only last 1 step
            return
        self.buffer.add(episode)
        if len(self.buffer) > self.size:
            self.buffer.pop(-1)

    def top_episodes(self, num_episodes: int) -> List[util.Episode]:
        return self.buffer[:num_episodes]

    def sample_episodes(self, num_episodes: int) -> List[util.Episode]:
        return random.choices(self.buffer, k=num_episodes)
