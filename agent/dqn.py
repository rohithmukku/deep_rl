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

    def __init__(self, device, obs_dim, act_dim, buffer_size, batch_size, epsilon, gamma, alpha, target_update) -> None:
        super().__init__()

        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.target_update = target_update
        self.update_count = 0

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = 2 * max(self.obs_dim, self.act_dim)
        self.hidden_dim = max(20, self.hidden_dim)

        # Behaviour Network
        self.qf_behaviour = Network(self.obs_dim, self.hidden_dim,
                                    self.act_dim).to(self.device)
        # Target Network
        self.qf_target = Network(self.obs_dim, self.hidden_dim,
                                 self.act_dim).to(self.device)
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.optimizer = optim.Adam(self.qf_behaviour.parameters(), lr=self.alpha)
        self.qf_target.load_state_dict(self.qf_behaviour.state_dict())

    def train(self, _):
        if self.batch_size > self.replay_buffer.__len__():
            # print("memory len: " + str(self.replay_buffer.__len__()))
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # print(batch.reward)

        state_batch = torch.stack([torch.from_numpy(item).float() for item in batch.state]).to(self.device)
        action_batch = torch.tensor(batch.action).to(self.device)
        # next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([torch.from_numpy(item).float() for item in batch.next_state
                                                if item is not None]).to(self.device)
        
        state_action_values = self.qf_behaviour(state_batch).gather(1, action_batch.view(-1,1))

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.qf_target(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        print(loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.qf_behaviour.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def observe(self, state, action, next_state, reward):
        self.replay_buffer.add(state, action, next_state, reward)

    def act(self, state):
        # TODO: Decaying epsilon implementation
        action = None
        if random.random() < self.epsilon:
            # Random action (Exploration)
            action = np.random.randint(self.act_dim)
        else:
            # Pick action with best Q value (Exploitation)
            action = torch.argmax(self.qf_behaviour(torch.Tensor(state).to(self.device)))
            action = action.detach().cpu().item()
        return action

    def update(self, _):
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.qf_target.load_state_dict(self.qf_behaviour.state_dict())