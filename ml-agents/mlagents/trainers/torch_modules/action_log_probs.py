from typing import List, Optional, NamedTuple
from mlagents.torch_utils import torch
import numpy as np

from mlagents.trainers.torch_modules.utils import ModelUtils
from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents_envs.base_env import _ActionTupleBase


class LogProbsTuple(_ActionTupleBase):
    """
    An object whose fields correspond to the log probs of actions of different types.
    Continuous and discrete are numpy arrays
    Dimensions are of (n_agents, continuous_size) and (n_agents, discrete_size),
    respectively. Note, this also holds when continuous or discrete size is
    zero.
    """


class ActionLogProbs(NamedTuple):
    """
    A NamedTuple containing the tensor for continuous log probs and list of tensors for
    discrete log probs of individual actions as well as all the log probs for an entire branch.
    Utility functions provide numpy <=> tensor conversions to be used by the optimizers.
    :param continuous_tensor: Torch tensor corresponding to log probs of continuous actions
    :param discrete_list: List of Torch tensors each corresponding to log probs of the discrete actions that were
    sampled.
    :param all_discrete_list: List of Torch tensors each corresponding to all log probs of
    a discrete action branch, even the discrete actions that were not sampled. all_discrete_list is a list of Tensors,
    each Tensor corresponds to one discrete branch log probabilities.
    """

    continuous_tensor: torch.Tensor



    def to_log_probs_tuple(self) -> LogProbsTuple:
        """
        Returns a LogProbsTuple. Only adds if tensor is not None. Otherwise,
        LogProbsTuple uses a default.
        """
        log_probs_tuple = LogProbsTuple()
        if self.continuous_tensor is not None:
            continuous = ModelUtils.to_numpy(self.continuous_tensor)
            log_probs_tuple.add_continuous(continuous)
        return log_probs_tuple

    def _to_tensor_list(self) -> List[torch.Tensor]:
        """
        Returns the tensors in the ActionLogProbs as a flat List of torch Tensors. This
        is private and serves as a utility for self.flatten()
        """
        tensor_list: List[torch.Tensor] = []
        if self.continuous_tensor is not None:
            tensor_list.append(self.continuous_tensor)
        return tensor_list

    def flatten(self) -> torch.Tensor:
        """
        A utility method that returns all log probs in ActionLogProbs as a flattened tensor.
        This is useful for algorithms like PPO which can treat all log probs in the same way.
        """
        return torch.cat(self._to_tensor_list(), dim=1)

    @staticmethod
    def from_buffer(buff: AgentBuffer) -> "ActionLogProbs":
        """
        A static method that accesses continuous and discrete log probs fields in an AgentBuffer
        and constructs the corresponding ActionLogProbs from the retrieved np arrays.
        """
        continuous: torch.Tensor = None

        if BufferKey.CONTINUOUS_LOG_PROBS in buff:
            continuous = ModelUtils.list_to_tensor(buff[BufferKey.CONTINUOUS_LOG_PROBS])
            
        return ActionLogProbs(continuous)
