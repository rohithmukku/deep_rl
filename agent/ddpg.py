from agent import Agent
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils.buffer import ReplayBuffer, Transition
from utils.mlp import DDPGActor, DDPGCritic

class OUActionNoise:
    def __init__(self, mean, std_deviation=0.15, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class DDPGAgent(Agent):

    """Deep Deterministic Policy Gradient Implementation"""

    def __init__(self, device, obs_dim, act_dim, gamma,
                 buffer_size, min_size, batch_size,
                 actor_lr, critic_lr, tau, update_steps,
                 explore_steps, action_space,
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
        self.batch_num = 0
        self.tau = tau
        self.steps = 0
        self.update_steps = update_steps
        self.explore_steps = explore_steps
        self.act_limit = action_space.shape[0]
        self.writer = writer

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = max(40, 2 * max(self.obs_dim, self.act_dim))

        # Policy Network
        self.actor_network = DDPGActor(self.obs_dim, self.hidden_dim,
                                     self.act_dim, self.act_limit).to(self.device)
        self.target_actor_network = DDPGActor(self.obs_dim, self.hidden_dim,
                                     self.act_dim, self.act_limit).to(self.device)
        
        self.critic_network = DDPGCritic(self.obs_dim, self.hidden_dim,
                                      self.act_dim).to(device)
        self.target_critic_network = DDPGCritic(self.obs_dim, self.hidden_dim,
                                      self.act_dim).to(device)
        
        self.replay_buffer = ReplayBuffer(max_capacity=self.buffer_size, min_capacity=self.min_size)
        self.noise = OUActionNoise(mean=np.zeros(self.act_dim))

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.critic_lr)

        self.update_parameters(tau=1)
    
    def update_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        actor_param_dict = dict(self.actor_network.named_parameters())
        target_actor_param_dict = dict(self.target_actor_network.named_parameters())
        critic_param_dict = dict(self.critic_network.named_parameters())
        target_critic_param_dict = dict(self.target_critic_network.named_parameters())

        for layer in critic_param_dict:
            critic_param_dict[layer] = tau * critic_param_dict[layer].clone() + \
                                        (1 - tau) * target_critic_param_dict[layer].clone()

        for layer in actor_param_dict:
            actor_param_dict[layer] = tau * actor_param_dict[layer].clone() + \
                                        (1 - tau) * target_actor_param_dict[layer].clone()
        
        self.target_actor_network.load_state_dict(actor_param_dict)
        self.target_critic_network.load_state_dict(critic_param_dict)

    def train(self, done):
        self.steps += 1
        if self.steps < self.replay_buffer.min_capacity or self.steps % self.update_steps != 0:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack([torch.from_numpy(item).float() for item in batch.state]).to(self.device)
        action_batch = torch.tensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        done_batch = torch.tensor(batch.done).to(self.device)
        next_state_batch = torch.stack([torch.from_numpy(item).float() for item in batch.state]).to(self.device)
        
        with torch.no_grad():
            target_actions = self.target_actor_network(next_state_batch)
            target_state_action_values = self.target_critic_network(next_state_batch, target_actions).squeeze()
            expected_state_action_values = (1 - done_batch) * (target_state_action_values * self.gamma) + reward_batch

        state_action_values = self.critic_network(state_batch, action_batch)

        # Optimize the critic model
        self.critic_network.train()
        self.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(state_action_values.squeeze(1), expected_state_action_values.detach())
        critic_loss.backward()
        self.critic_optimizer.step()

        for layer in self.critic_network.parameters():
            layer.requires_grad = False

        # Optimize the critic model
        self.actor_optimizer.zero_grad()
        actions = self.actor_network(state_batch)
        actor_loss = - self.critic_network(state_batch, actions).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        for layer in self.critic_network.parameters():
            layer.requires_grad = True
        
        self.writer.add_scalar("Loss/Actor", actor_loss.item(), global_step=self.steps)
        self.writer.add_scalar("Loss/Critic", critic_loss.item(), global_step=self.steps)

        self.loss_list.append(actor_loss.item() + critic_loss.item())

        self.update_parameters()

    def observe(self, state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, reward, done)

    def act(self, state):
        if self.steps < self.explore_steps:
            return np.clip(np.random.randn(), -self.act_limit, self.act_limit)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu = self.actor_network(state)
        noise = torch.tensor(self.noise(), dtype=torch.float32).to(self.device)
        mu_prime = mu + noise
        action = mu_prime.flatten().cpu().detach().numpy()
        action = np.clip(action, -self.act_limit, self.act_limit)
        return action

    def update(self, done):
        return
    
    def update_batch(self, done):
        return
    
    def get_loss(self):
        loss_np = np.array(self.loss_list, dtype=np.float32)
        if loss_np.size == 0:
            return 0
        return np.mean(loss_np)