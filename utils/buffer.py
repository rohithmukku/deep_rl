from collections import deque, namedtuple
import numpy as np
import torch
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):
    def __init__(self, capacity) -> None:
        super().__init__()

        self.memory = deque([], maxlen=capacity)
    
    def __len__(self):
        return len(self.memory)

    def add(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)