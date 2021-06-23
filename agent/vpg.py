from utils.buffer import ReplayBuffer, Transition
from agent import Agent
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

from utils.mlp import Network

class VPGAgent(Agent):

    """VPG/REINFORCE Agent Implementation"""

    def __init__(self, device, obs_dim, act_dim, gamma, alpha) -> None:
        super().__init__()

        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.state_list = []
        self.action_list = []
        self.reward_list = []

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = max(40, 2 * max(self.obs_dim, self.act_dim))

        # Policy Network
        self.policy_network = Network(self.obs_dim, self.hidden_dim,
                                    self.act_dim).to(self.device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.alpha)
    
    def _get_policy(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        policy_distribution = F.softmax(self.policy_network(state), dim=1)
        return policy_distribution.flatten().detach().cpu().numpy()
    
    def _get_returns(self):
        cumulative_rewards = [self.reward_list[-1]]
        for reward in reversed(self.reward_list[:-1]):
            cumulative_reward = self.gamma * cumulative_rewards[0] + reward
            cumulative_rewards.append(cumulative_reward)
        return cumulative_rewards

    def train(self, done):
        if done == False:
            # Episode not completed
            return
        
        states = torch.tensor(self.state_list, dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.action_list, dtype=torch.int64).to(self.device)
        cumulative_returns = torch.tensor(self._get_returns()).to(self.device)

        # print(states.shape)
        # print(actions.shape)
        # print(cumulative_returns.shape)

        scores = self.policy_network(states)
        # print(scores.shape)
        probs = F.softmax(scores, dim=1)
        log_probs = F.log_softmax(scores, dim=1)

        log_probs_action = torch.flatten(torch.gather(log_probs, 1, actions.unsqueeze(1)))
        loss = - torch.mean(log_probs_action * cumulative_returns)
        # print(loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def observe(self, state, action, next_state, reward):
        # Store rewards
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        pass

    def act(self, state):
        policy_distribution = self._get_policy(state)
        action = np.random.choice(self.act_dim, p=policy_distribution)
        return action

    def update(self, done):
        if done:
            self.state_list.clear()
            self.action_list.clear()
            self.reward_list.clear()