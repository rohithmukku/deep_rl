import torch
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

class CategoricalNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Identity()
        )
    
    def forward(self, x):
        logits = self.network(x)
        dist = Categorical(logits=logits)
        return dist

class GaussianNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.cov_var = torch.full(size=(out_dim, ), fill_value=0.5)
        self.cov_mat = torch.nn.Parameter(torch.diag(self.cov_var))
        # log_std = -0.5 * torch.ones(out_dim, dtype=torch.float32)
        # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Identity()
        )
    
    def forward(self, x):
        mu = self.mu_network(x)
        # std = torch.exp(self.log_std)
        cov_mat = self.cov_mat.unsqueeze(0).repeat(mu.shape[0], 1, 1)
        dist = MultivariateNormal(mu, self.cov_mat)
        return dist

class PPOActor(nn.Module):
    def __init__(self, in_dim, hidden_dim, action_space):
        super().__init__()

        if isinstance(action_space, Box):
            self.network = GaussianNetwork(in_dim, hidden_dim, action_space.shape[0])
        else:
            self.network = CategoricalNetwork(in_dim, hidden_dim, action_space.n)

    def forward(self, x):
        return self.network(x)

class PPOCritic(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Identity()
        )
    
    def forward(self, x):
        return self.network(x)

class DDPGActor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, act_limit):
        super().__init__()

        self.base = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh()
        )

        self.act_limit = act_limit

    def forward(self, x):
        base = self.base(x)
        mu = self.mu(base)

        return self.act_limit * mu

class DDPGCritic(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.state_value = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )

        self.action_value = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU()
        )

        self.state_action_value = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        state_value = self.state_value(state)
        action_value = self.action_value(action)

        x = state_value + action_value
        state_action_value = self.state_action_value(x)

        return state_action_value