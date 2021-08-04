from torch.functional import norm
from agent import Agent
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils.mlp import PPOActor, PPOCritic

class PPOAgent(Agent):

    """Proximal Policy Optimization Implementation"""

    def __init__(self, device, obs_dim, act_dim,
                 gamma, lam, actor_lr, critic_lr,
                 batch_update, batch_size, epsilon_clip,
                 k_epochs, target_kl, action_space,
                 writer: SummaryWriter()) -> None:
        super().__init__()

        self.device = device
        self.gamma = gamma
        self.lam = lam
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
        self.log_probs = []
        self.loss_list = []
        self.batch_num = 0
        self.batch_update = batch_update
        self.batch_size = batch_size
        self.epsilon_clip = epsilon_clip
        self.k_epochs = k_epochs
        self.action_space = action_space
        self.target_kl = target_kl
        self.steps = 0
        self.writer = writer

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = max(64, 2 * self.obs_dim)

        # Policy Network
        self.actor_network = PPOActor(self.obs_dim, self.hidden_dim,
                                      self.action_space).to(self.device)
        
        self.critic_network = PPOCritic(self.obs_dim, self.hidden_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.critic_lr)
    
    def _get_returns(self, normalized=False):
        print(self.reward_list)
        returns = np.array([self.gamma ** i * r for i, r in enumerate(self.reward_list)])
        returns = returns[::-1].cumsum()[::-1]
        if normalized:
            normalized_returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)
            print(normalized_returns)
            return normalized_returns
        return returns
    
    def _get_pi_loss(self, states, actions, old_log_probs_action, advantage):
        dist = self.actor_network(states)
        entropy = torch.mean(dist.entropy())
        log_probs_action = dist.log_prob(actions)
        ratio = torch.exp(log_probs_action - old_log_probs_action.detach())
        surrogate_loss1 = ratio * advantage
        surrogate_loss2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantage
        loss = - torch.mean(torch.min(surrogate_loss1, surrogate_loss2))
        return loss, entropy
    
    def _get_v_loss(self, state_values, rewards):
        return 0.5 * F.mse_loss(state_values.squeeze(1), rewards)

    def train(self, done):
        if not done or self.batch_num != self.batch_update:
            return
        states = torch.tensor(self.state_batch, dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.action_batch, dtype=torch.int64).to(self.device)
        returns = torch.tensor(self.returns, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(self.device)
        
        state_values = self.critic_network(states)
        advantage = returns - state_values.detach()
        actor_loss, entropy = self._get_pi_loss(states, actions, old_log_probs, advantage)
        critic_loss = self._get_v_loss(state_values, returns)
        self.writer.add_scalar("Loss/Actor", actor_loss.item(), global_step=self.steps)
        self.writer.add_scalar("Loss/Critic", critic_loss.item(), global_step=self.steps)
        self.writer.add_scalar("Actor/Entropy", entropy.item(), global_step=self.steps)
        self.loss_list.append(critic_loss.item())
        for epoch in range(self.k_epochs):
            actor_loss, entropy = self._get_pi_loss(states, actions, old_log_probs, advantage)
            state_values = self.critic_network(states)
            # Update the network parameters
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            critic_loss = self._get_v_loss(state_values, returns)
            total_loss = actor_loss +  0.5 * critic_loss - 0.001 * entropy
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def observe(self, state, action, next_state, reward, done):
        self.steps += 1
        # Store rewards
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.next_state_list.append(next_state)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            policy_distribution = self.actor_network(state)
        action = policy_distribution.sample()
        log_prob = policy_distribution.log_prob(action)
        self.log_probs.append(log_prob)
        return action

    def update(self, done):
        if not done or self.batch_num != self.batch_update:
            return
        self.state_batch.clear()
        self.action_batch.clear()
        self.returns.clear()
        self.next_state_batch.clear()
        self.log_probs.clear()
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
    
    def get_loss(self):
        loss_np = np.array(self.loss_list, dtype=np.float32)
        if loss_np.size == 0:
            return 0
        return np.mean(loss_np)