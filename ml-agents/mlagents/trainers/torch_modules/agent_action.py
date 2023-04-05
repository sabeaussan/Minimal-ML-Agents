from typing import List, Optional, NamedTuple
import itertools
import numpy as np
from mlagents.torch_utils import torch

from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.torch_modules.utils import ModelUtils
from mlagents_envs.base_env import ActionTuple


class AgentAction(NamedTuple):
    """
    A NamedTuple containing the tensor for continuous actions and list of tensors for
    discrete actions. Utility functions provide numpy <=> tensor conversions to be
    sent as actions to the environment manager as well as used by the optimizers.
    :param continuous_tensor: Torch tensor corresponding to continuous actions
    :param discrete_list: List of Torch tensors each corresponding to discrete actions
    """

    continuous_tensor: torch.Tensor


    def slice(self, start: int, end: int) -> "AgentAction":
        """
        Returns an AgentAction with the continuous and discrete tensors slices
        from index start to index end.
        """
        _cont = None
        _disc_list = []
        if self.continuous_tensor is not None:
            _cont = self.continuous_tensor[start:end]
        return AgentAction(_cont, _disc_list)

    def to_action_tuple(self, clip: bool = False,clipping_value : int = 1) -> ActionTuple:
        """
        Returns an ActionTuple
        """
        action_tuple = ActionTuple()
        if self.continuous_tensor is not None:
            _continuous_tensor = self.continuous_tensor
            if clip:
                _continuous_tensor = (torch.clamp(_continuous_tensor, -3, 3) / 3 ) * clipping_value
            continuous = ModelUtils.to_numpy(_continuous_tensor)
            action_tuple.add_continuous(continuous)
        return action_tuple

    @staticmethod
    def from_buffer(buff: AgentBuffer) -> "AgentAction":
        """
        A static method that accesses continuous and discrete action fields in an AgentBuffer
        and constructs the corresponding AgentAction from the retrieved np arrays.
        """
        continuous: torch.Tensor = None
        if BufferKey.CONTINUOUS_ACTION in buff:
            continuous = ModelUtils.list_to_tensor(buff[BufferKey.CONTINUOUS_ACTION])
        return AgentAction(continuous)


