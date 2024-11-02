from . import layer_init, GaussianDistribution, SquashedGaussianDistribution

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, Categorical

class ContinuousMLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64, distribution_cls=GaussianDistribution):
        super(ContinuousMLPActorCritic, self).__init__()

        self.distribution_cls = distribution_cls
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        action_distribution = self.distribution_cls(action_mean, action_std)

        if action is None:
            action = action_distribution.sample()

        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()

        if entropy is None:
            entropy = -log_prob

        value = self.critic(x)
        return action, log_prob, entropy, value

    def get_action(self, x, deterministic=True):
        action_mean = self.actor_mean(x)
        if deterministic:
            action = self.distribution_cls.transform_action(action_mean)
        else:
            action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
            action_distribution = self.distribution_cls(action_mean, action_std)
            action = action_distribution.sample()
        return action 

