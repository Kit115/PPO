from . import layer_init, GaussianDistribution

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, Categorical


class ContinuousCNNActorCritic(nn.Module):
    def __init__(self, act_dim, num_frames=8, distribution_cls=GaussianDistribution):
        super().__init__()

        self.distribution_cls = distribution_cls

        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(num_frames, 8, kernel_size=3, stride=2)),
            nn.Tanh(),
            layer_init(nn.Conv2d(8, 16, kernel_size=3, stride=2)),
            nn.Tanh(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=2)),
            nn.Tanh(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2)),
            nn.Tanh(),
            layer_init(nn.Conv2d(64, 128, kernel_size=5)),
            nn.Tanh(),
            nn.Flatten(1)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(128, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 1), std=1.0)
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(128, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, act_dim), std=0.01)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, obs):
        obs = self.encoder(self._transform_obs(obs))
        return self.critic(obs)

    def get_action_and_value(self, observation, action=None):
        x = self.encoder(self._transform_obs(observation))
            
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

    def get_action(self, observation, deterministic=True):
        x = self.encoder(self._transform_obs(observation))
        action_mean = self.actor_mean(x)
        if deterministic:
            action = self.distribution_cls.transform_action(action_mean)
        else:
            action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
            action_distribution = self.distribution_cls(action_mean, action_std)
            action = action_distribution.sample()
        return action 

    def _transform_obs(self, obs):
        obs = obs.squeeze(-1).float()
        obs = (obs - obs.mean()) / obs.std()
        return obs








