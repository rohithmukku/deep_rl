from utils.buffer import ReplayBuffer, Transition
from agent import Agent
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

from utils.mlp import Network

class DQNAgent(Agent):

    """DQN Agent Implementation"""

    def __init__(self, device, obs_dim, act_dim, buffer_size, min_size,
                    batch_size, epsilon, epsilon_decay, min_epsilon, gamma,
                    alpha, target_update) -> None:
        super().__init__()

        self.device = device
        self.buffer_size = buffer_size
        self.min_size = min_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.target_update = target_update
        self.update_count = 0

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = 2 * max(self.obs_dim, self.act_dim)
        self.hidden_dim = max(16, self.hidden_dim)

        # Behaviour Network
        self.qf_behaviour = Network(self.obs_dim, self.hidden_dim,
                                    self.act_dim).to(self.device)
        # Target Network
        self.qf_target = Network(self.obs_dim, self.hidden_dim,
                                 self.act_dim).to(self.device)
        
        self.replay_buffer = ReplayBuffer(max_capacity=self.buffer_size, min_capacity=self.min_size)

        self.optimizer = optim.Adam(self.qf_behaviour.parameters(), lr=self.alpha)
        self.qf_target.load_state_dict(self.qf_behaviour.state_dict())

    def train(self, _):
        if self.replay_buffer.__len__() < self.replay_buffer.min_capacity:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack([torch.from_numpy(item).float() for item in batch.state]).to(self.device)
        action_batch = torch.tensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        # next_state_batch = torch.stack([torch.from_numpy(item).float() for item in batch.state]).to(self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([torch.from_numpy(item).float() for item in batch.next_state
                                                if item is not None]).to(self.device)
        
        state_action_values = self.qf_behaviour(state_batch).gather(1, action_batch.view(-1,1))

        next_state_values = torch.empty(self.batch_size, device=self.device).fill_(-10.)
        next_state_values[non_final_mask] = self.qf_target(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.mse_loss(state_action_values.squeeze(1), expected_state_action_values)

        print(loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def observe(self, state, action, next_state, reward):
        if not isinstance(next_state, np.ndarray):
            print(next_state)
            reward = -1
        self.replay_buffer.add(state, action, next_state, reward)

    def act(self, state):
        self.epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)
        print(self.epsilon)
        action = None
        if random.random() < self.epsilon:
            # Random action (Exploration)
            action = np.random.randint(self.act_dim)
        else:
            # Pick action with best Q value (Exploitation)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.qf_behaviour(state).max(1)[1]
            action = action.detach().cpu().item()
        return action

    def update(self, _):
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.qf_target.load_state_dict(self.qf_behaviour.state_dict())
    
    def update_batch(self, done):
        return