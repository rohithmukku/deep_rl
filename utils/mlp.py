import torch
import numpy as np
from gym.spaces import Box, Discrete
from torch import nn
from torch.distributions import Categorical, Normal, MultivariateNormal

class Network(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
            nn.Identity()
        )
    
    def forward(self, x):
        return self.network(x)

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.logits_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
            nn.Identity()
        )

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
            nn.Identity()
        )

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.v_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Identity()
        )

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)

class PPOActor(nn.Module):
    def __init__(self, in_dim, hidden_dim, action_space):
        super().__init__()

        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(in_dim, action_space.shape[0], hidden_dim)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(in_dim, action_space.n, hidden_dim)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
        return a.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

class PPOCritic(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.v = MLPCritic(in_dim, hidden_dim)
    
    def step(self, obs):
        with torch.no_grad():
            v = self.v(obs)
        return v.cpu().numpy()

class DDPGActor(nn.Module):

    def __init__(self, in_dim, hidden_dim, act_dim, act_limit):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()
        )
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.pi(obs)

class DDPGCritic(nn.Module):

    def __init__(self, obs_dim, hidden_dim, act_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Identity()
        )

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)

class DDPGActorCritic(nn.Module):

    def __init__(self, obs_dim, hidden_dim, action_space):
        super().__init__()

        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = DDPGActor(obs_dim, hidden_dim, act_dim, act_limit)
        self.q = DDPGCritic(obs_dim, hidden_dim, act_dim)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()

# class DDPGActor(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, act_limit):
#         super().__init__()

#         self.base = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ReLU()
#         )

#         self.mu = nn.Sequential(
#             nn.Linear(hidden_dim, out_dim),
#             nn.Tanh()
#         )

#         self.act_limit = act_limit

#     def forward(self, x):
#         base = self.base(x)
#         mu = self.mu(base)

#         return self.act_limit * mu

# class DDPGCritic(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super().__init__()

#         self.state_value = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ReLU()
#         )

#         self.action_value = nn.Sequential(
#             nn.Linear(out_dim, hidden_dim),
#             nn.ReLU()
#         )

#         self.state_action_value = nn.Sequential(
#             nn.Linear(hidden_dim, 1)
#         )
    
#     def forward(self, state, action):
#         state_value = self.state_value(state)
#         action_value = self.action_value(action)

#         x = state_value + action_value
#         state_action_value = self.state_action_value(x)

#         return state_action_value