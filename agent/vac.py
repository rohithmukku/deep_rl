from agent import Agent
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils.buffer import ReplayBuffer, Transition
from utils.mlp import Network

class VACAgent(Agent):

    """Vanilla Actor Critic Agent Implementation"""

    def __init__(self, device, obs_dim, act_dim, gamma, actor_lr, critic_lr, batch_update) -> None:
        super().__init__()

        self.device = device
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.state_list = []
        self.action_list = []
        self.reward_list = []
        self.next_state_list = []
        self.state_batch = []
        self.action_batch = []
        self.returns = []
        self.next_state_batch = []
        self.batch_num = 0
        self.batch_update = batch_update

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = max(40, 2 * max(self.obs_dim, self.act_dim))

        # Policy Network
        self.actor_network = Network(self.obs_dim, self.hidden_dim,
                                     self.act_dim).to(self.device)
        
        self.critic_network = Network(self.obs_dim, self.hidden_dim,
                                      self.act_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.critic_lr)
    
    def _get_policy(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        policy_distribution = F.softmax(self.actor_network(state), dim=1)
        return policy_distribution
    
    def _get_returns(self, normalized=False):
        returns = np.array([self.gamma ** i * r for i, r in enumerate(self.reward_list)])
        returns = returns[::-1].cumsum()[::-1]
        return returns

    def actor_train(self, states, actions, state_action_values):
        scores = self.actor_network(states)
        log_probs = F.log_softmax(scores, dim=1)

        log_probs_action = torch.flatten(torch.gather(log_probs, 1, actions.unsqueeze(1)))
        actor_loss = - torch.mean(log_probs_action * state_action_values)

        # Update the actor network parameters
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def critic_train(self, states, actions, state_action_values):
        rewards = torch.tensor(self.returns, dtype=torch.int64).to(self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            self.next_state_batch)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([torch.from_numpy(item).float() for item in self.next_state_batch
                                                if item is not None]).to(self.device)

        next_state_values = torch.empty(states.shape[0], device=self.device).fill_(0.)
        next_state_values[non_final_mask] = self.critic_network(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + rewards
        expected_state_action_values = expected_state_action_values.detach()

        critic_loss = F.mse_loss(state_action_values.squeeze(1), expected_state_action_values)

        # Update the critic network parameters
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def train(self, done):
        if not done or self.batch_num != self.batch_update:
            return
        states = torch.tensor(self.state_batch, dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.action_batch, dtype=torch.int64).to(self.device)
        state_action_values = Variable(self.critic_network(states).gather(1, actions.view(-1,1)), requires_grad=True)
        self.actor_train(states, actions, state_action_values)
        self.critic_train(states, actions, state_action_values)

    def observe(self, state, action, next_state, reward):
        if not isinstance(next_state, np.ndarray):
            reward = -1
        # Store rewards
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.next_state_list.append(next_state)

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
        self.next_state_batch.clear()
        self.batch_num = 0
    
    def update_batch(self, done):
        if not done:
            return
        self.state_batch.extend(self.state_list)
        self.action_batch.extend(self.action_list)
        self.returns.extend(self._get_returns(normalized=True))
        self.next_state_batch.extend(self.next_state_list)
        self.state_list.clear()
        self.action_list.clear()
        self.reward_list.clear()
        self.next_state_list.clear()
        self.batch_num += 1