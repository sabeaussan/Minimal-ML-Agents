import abc
from typing import List
from mlagents.torch_utils import torch, nn
import numpy as np
import math
from mlagents.trainers.torch_modules.layers import linear_layer, Initialization

EPSILON = 1e-7  # Small value to avoid divide by zero


class DistInstance(nn.Module, abc.ABC):
    @abc.abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Return a sample from this distribution.
        """
        pass

    @abc.abstractmethod
    def deterministic_sample(self) -> torch.Tensor:
        """
        Return the most probable sample from this distribution.
        """
        pass

    @abc.abstractmethod
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Returns the log probabilities of a particular value.
        :param value: A value sampled from the distribution.
        :returns: Log probabilities of the given value.
        """
        pass

    @abc.abstractmethod
    def entropy(self) -> torch.Tensor:
        """
        Returns the entropy of this distribution.
        """
        pass

    @abc.abstractmethod
    def exported_model_output(self) -> torch.Tensor:
        """
        Returns the tensor to be exported to ONNX for the distribution
        """
        pass


class GaussianDistInstance(DistInstance):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def sample(self):
        sample = self.mean + torch.randn_like(self.mean) * self.std
        return sample

    def deterministic_sample(self):
        return self.mean

    def log_prob(self, value):
        var = self.std ** 2
        log_scale = torch.log(self.std + EPSILON)
        return (
            -((value - self.mean) ** 2) / (2 * var + EPSILON)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )

    def pdf(self, value):
        log_prob = self.log_prob(value)
        return torch.exp(log_prob)

    def entropy(self):
        return torch.mean(
            0.5 * torch.log(2 * math.pi * math.e * self.std ** 2 + EPSILON),
            dim=1,
            keepdim=True,
        )  # Use equivalent behavior to TF

    def exported_model_output(self):
        return self.sample()


class TanhGaussianDistInstance(GaussianDistInstance):
    def __init__(self, mean, std):
        super().__init__(mean, std)
        self.transform = torch.distributions.transforms.TanhTransform(cache_size=1)

    def sample(self):
        unsquashed_sample = super().sample()
        squashed = self.transform(unsquashed_sample)
        return squashed

    def _inverse_tanh(self, value):
        capped_value = torch.clamp(value, -1 + EPSILON, 1 - EPSILON)
        return 0.5 * torch.log((1 + capped_value) / (1 - capped_value) + EPSILON)

    def log_prob(self, value):
        unsquashed = self.transform.inv(value)
        return super().log_prob(unsquashed) - self.transform.log_abs_det_jacobian(
            unsquashed, value
        )



class GaussianDistribution(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_outputs: int,
        tanh_squash: bool = False,
    ):
        super().__init__()
        self.mu = linear_layer(
            hidden_size,
            num_outputs,
            kernel_init=Initialization.KaimingHeNormal,
            kernel_gain=0.2,
            bias_init=Initialization.Zero,
        )
        self.tanh_squash = tanh_squash
        self.log_sigma = nn.Parameter(torch.zeros(1, num_outputs, requires_grad=True))

    def forward(self, inputs: torch.Tensor) -> List[DistInstance]:
        mu = self.mu(inputs)

        # Expand so that entropy matches batch size. Note that we're using
        # mu*0 here to get the batch size implicitly since Barracuda 1.2.1
        # throws error on runtime broadcasting due to unknown reason. We
        # use this to replace torch.expand() becuase it is not supported in
        # the verified version of Barracuda (1.0.X).
        log_sigma = mu * 0 + self.log_sigma
        if self.tanh_squash:
            return TanhGaussianDistInstance(mu, torch.exp(log_sigma))
        else:
            return GaussianDistInstance(mu, torch.exp(log_sigma))

