from agent import Agent
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils.mlp import Network

class VPGAgent(Agent):

    """VPG/REINFORCE Agent Implementation"""

    def __init__(self, device, obs_dim, act_dim, gamma, alpha, batch_update) -> None:
        super().__init__()

        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.state_list = []
        self.action_list = []
        self.reward_list = []
        self.state_batch = []
        self.action_batch = []
        self.returns = []
        self.batch_num = 0
        self.batch_update = batch_update

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
        return policy_distribution
    
    def _get_returns(self, normalized=False):
        returns = np.array([self.gamma ** i * r for i, r in enumerate(self.reward_list)])
        returns = returns[::-1].cumsum()[::-1]
        return returns

    def train(self, done):
        if not done or self.batch_num != self.batch_update:
            # Episode not completed
            return
        
        states = torch.tensor(self.state_batch, dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.action_batch, dtype=torch.int64).to(self.device)
        cumulative_returns = torch.tensor(self.returns).to(self.device)

        scores = self.policy_network(states)
        log_probs = F.log_softmax(scores, dim=1)

        log_probs_action = torch.flatten(torch.gather(log_probs, 1, actions.unsqueeze(1)))
        loss = - torch.mean(log_probs_action * cumulative_returns)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def observe(self, state, action, next_state, reward):
        # Store rewards
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)

    def act(self, state):
        policy_distribution = self._get_policy(state).squeeze(0).detach().cpu().numpy()
        action = np.random.choice(self.act_dim, p=policy_distribution)
        return action

    def update(self, done):
        if not done or self.batch_num != self.batch_update:
            return
        self.state_batch.clear()
        self.action_batch.clear()
        self.returns.clear()
        self.batch_num = 0
    
    def update_batch(self, done):
        if not done:
            return
        self.state_batch.extend(self.state_list)
        self.action_batch.extend(self.action_list)
        self.returns.extend(self._get_returns(normalized=True))
        self.state_list.clear()
        self.action_list.clear()
        self.reward_list.clear()
        self.batch_num += 1