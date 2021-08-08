from agent import Agent
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from utils.buffer import DDPGBuffer
from utils.mlp import DDPGActorCritic

class DDPGAgent(Agent):

    """Deep Deterministic Policy Gradient Implementation"""

    def __init__(self, device, obs_dim, act_dim, gamma,
                 buffer_size, min_size, batch_size,
                 actor_lr, critic_lr, tau, noise_scale,
                 update_steps, explore_steps, action_space,
                 writer) -> None:
        super().__init__()

        self.device = device
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.min_size = min_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.loss_list = []
        self.tau = tau
        self.noise_scale = noise_scale
        self.steps = 0
        self.update_steps = update_steps
        self.explore_steps = explore_steps
        self.action_space = action_space
        self.act_limit = action_space.shape[0]
        self.writer = writer

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = 256

        self.ac = DDPGActorCritic(obs_dim, self.hidden_dim, self.action_space).to(self.device)
        self.ac_targ = deepcopy(self.ac)

        for p in self.ac_targ.parameters():
            p.requires_grad = False
        
        self.replay_buffer = DDPGBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.buffer_size)

        self.actor_optimizer = optim.Adam(self.ac.pi.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.ac.q.parameters(), lr=self.critic_lr)
    
    def compute_loss_q(self, data):
        states, actions, rewards, next_states, done = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = self.ac.q(states, actions)

        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(next_states, self.ac_targ.pi(next_states))
            backup = rewards + self.gamma * (1 - done) * q_pi_targ

        loss_q = ((q - backup)**2).mean()

        return loss_q
    
    def compute_loss_pi(self, data):
        states = data['obs']
        q_pi = self.ac.q(states, self.ac.pi(states))
        return -q_pi.mean()
    
    def update_parameters(self):
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.tau)
                p_targ.data.add_((1 - self.tau) * p.data)

    def train(self, done):
        if self.steps < self.min_size or self.steps % self.update_steps != 0:
            return
        
        for _ in range(self.update_steps):
            data = self.replay_buffer.sample_batch(self.batch_size, device=self.device)
            self.critic_optimizer.zero_grad()
            loss_q = self.compute_loss_q(data)
            loss_q.backward()
            self.critic_optimizer.step()

            for p in self.ac.q.parameters():
                p.requires_grad = False

            self.actor_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.actor_optimizer.step()

            for p in self.ac.q.parameters():
                p.requires_grad = True

            self.update_parameters()

        self.loss_list.append(loss_pi.item() + loss_q.item())

    def observe(self, state, action, next_state, reward, done):
        self.steps += 1
        self.replay_buffer.store(state, action, next_state, reward, done)

    def act(self, state):
        if self.steps <= self.explore_steps:
            return self.action_space.sample()
        action = self.ac.act(torch.as_tensor(state, dtype=torch.float32).to(self.device))
        action += self.noise_scale * np.random.randn(self.act_dim)
        return np.clip(action, -self.act_limit, self.act_limit)

    def update(self, done):
        return
    
    def update_batch(self, done):
        return
    
    def get_loss(self):
        loss_np = np.array(self.loss_list, dtype=np.float32)
        if loss_np.size == 0:
            return 0
        return np.mean(loss_np)