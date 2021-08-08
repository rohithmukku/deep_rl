from agent.ppo_spinningup import PPOBuffer
from agent import Agent
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils.mlp import PPOActor, PPOCritic
from utils.buffer import PPOBuffer

class PPOAgent(Agent):

    """Proximal Policy Optimization Implementation"""

    def __init__(self, device, obs_dim, act_dim,
                 gamma, lam, actor_lr, critic_lr,
                 steps_per_epoch, batch_size, epsilon_clip,
                 k_epochs, target_kl, action_space,
                 writer: SummaryWriter()) -> None:
        super().__init__()

        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lam = lam
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_num = 0
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.epsilon_clip = epsilon_clip
        self.k_epochs = k_epochs
        self.action_space = action_space
        self.target_kl = target_kl
        self.steps = 0
        self.writer = writer
        self.loss_list = []
        self.store_list = {}
        self.hidden_dim = 64
        self.buffer = PPOBuffer(self.obs_dim, self.act_dim, steps_per_epoch, self.gamma, self.lam)

        # Policy Network
        self.actor_network = PPOActor(self.obs_dim, self.hidden_dim,
                                      self.action_space).to(self.device)
        
        self.critic_network = PPOCritic(self.obs_dim, self.hidden_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.critic_lr)
    
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.actor_network.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.epsilon_clip) | ratio.lt(1 - self.epsilon_clip)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info
    
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.critic_network.v(obs) - ret)**2).mean()

    def train(self, done):
        if not done or self.steps % self.steps_per_epoch != 0:
            return
        data = self.buffer.get(device=self.device)

        pi_l_old, _ = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()
        
        # Train policy with multiple steps of gradient descent
        for _ in range(self.k_epochs):
            self.actor_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                break
            loss_pi.backward()
            self.actor_optimizer.step()
        
        # Value function learning
        for _ in range(self.k_epochs):
            self.critic_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.critic_optimizer.step()
        
        self.loss_list.append(pi_l_old + 0.5 * v_l_old)

    def observe(self, state, action, next_state, reward, done):
        self.steps += 1
        self.buffer.store(state, action, reward,
                          self.store_list['value'], self.store_list['logp'])

    def act(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        action, logp = self.actor_network.step(state)
        value = self.critic_network.step(state)
        self.store_list['action'] = action
        self.store_list['value'] = value
        self.store_list['logp'] = logp
        return action

    def update(self, done):
        if done:
            self.buffer.finish_path()
        elif self.steps % self.steps_per_epoch != 0:
            self.buffer.finish_path(self.store_list['value'])
    
    def update_batch(self, done):
        if not done:
            return
    
    def get_loss(self):
        loss_np = np.array(self.loss_list, dtype=np.float32)
        if loss_np.size == 0:
            return 0
        return np.mean(loss_np)