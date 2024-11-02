import torch
from torch.distributions import Normal, Categorical
from abc import ABC, abstractmethod
from typing import Optional

class Distribution(ABC):
    @staticmethod
    @abstractmethod
    def transform_action(action):
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def entropy(self) -> Optional[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self) -> torch.Tensor:
        raise NotImplementedError



class GaussianDistribution(Distribution):
    def __init__(self, mean, std):
        self.dist = Normal(mean, std)

    @staticmethod
    def transform_action(action):
        return action

    def sample(self):
        return self.dist.sample()

    def entropy(self):
        return self.dist.entropy().sum(-1)

    def log_prob(self, action):
        return self.dist.log_prob(action).sum(-1)



class SquashedGaussianDistribution(Distribution):
    def __init__(self, mean, std):
        self.dist = Normal(mean, std)

    @staticmethod
    def _squash_action(unsquashed_action):
        return torch.tanh(unsquashed_action)

    @staticmethod
    def _unsquash_action(squashed_action, eps=1e-6):
        squashed_action = torch.clamp(squashed_action, min = -1. + eps, max = 1. - eps)
        return 0.5 * (squashed_action.log1p() - (-squashed_action).log1p())

    @staticmethod
    def transform_action(action):
        return SquashedGaussianDistribution._squash_action(action)

    def sample(self):
        unsquashed_action = self.dist.sample()
        action = SquashedGaussianDistribution._squash_action(unsquashed_action)
        return action

    def entropy(self):
        return None

    def log_prob(self, action):
        unsquashed_action = SquashedGaussianDistribution._unsquash_action(action)
        unsquashed_log_prob = self.dist.log_prob(unsquashed_action).sum(-1)

        squash_correction = torch.log(1 - torch.tanh(unsquashed_action) ** 2 + 1e-6).sum(dim=-1)
        log_prob = unsquashed_log_prob - squash_correction
        return log_prob



