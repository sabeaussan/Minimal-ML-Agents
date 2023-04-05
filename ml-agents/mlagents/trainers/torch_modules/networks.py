from typing import Callable, List, Dict, Tuple, Optional, Union
import abc

from mlagents.torch_utils import torch, nn

from mlagents.trainers.torch_modules.action_model import ActionModel
from mlagents.trainers.torch_modules.agent_action import AgentAction
from mlagents.trainers.torch_modules.action_log_probs import ActionLogProbs
from mlagents.trainers.settings import NetworkSettings, ModuleSettings
from mlagents.trainers.torch_modules.utils import ModelUtils
from mlagents.trainers.torch_modules.decoders import ValueHeads
from mlagents.trainers.torch_modules.layers import  LinearEncoder
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.torch_modules.encoders import VectorInput

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]
EncoderFunction = Callable[
    [torch.Tensor, int, ActivationFunction, int, str, bool], torch.Tensor
]

EPSILON = 1e-7


class ObservationEncoder(nn.Module):

    def __init__(
        self,
        observation_size: int,
        normalize: bool = False,
    ):
        """
        Returns an ObservationEncoder that can process and encode a set of observations.
        Will use an RSA if needed for variable length observations.
        """
        super().__init__()
        self.processors = ModelUtils.create_input_processors(
            observation_size,
            normalize=normalize,
        )
        self.normalize = normalize
        self.observation_size = observation_size


    @property
    def total_enc_size(self) -> int:
        """
        Returns the total encoding size for this ObservationEncoder.
        """
        return self.observation_size


    def update_normalization(self, buffer: AgentBuffer) -> None:
        obs = ObsUtil.from_buffer(buffer, 1)
        processed_obs = torch.as_tensor(obs[0])
        for enc in self.processors:
            enc.update_normalization(processed_obs)

    def copy_normalization(self, other_encoder: "ObservationEncoder") -> None:
        if self.normalize:
            for n1, n2 in zip(self.processors, other_encoder.processors):
                if isinstance(n1, VectorInput) and isinstance(n2, VectorInput):
                    n1.copy_normalization(n2)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode observations using a list of processors and an RSA.
        :param inputs: List of Tensors corresponding to a set of obs.
        """
        processed_obs = inputs[0]
        for idx, processor in enumerate(self.processors):
            processed_obs = processor(processed_obs)

        return processed_obs


class NetworkBody(nn.Module):
    def __init__(
        self,
        observation_size: int,
        network_settings: NetworkSettings,
        module_settings : ModuleSettings,
    ):
        super().__init__()
        self.normalize = network_settings.normalize
        self.h_size = network_settings.hidden_units
        self.observation_encoder = ObservationEncoder(
            observation_size,
            self.normalize,
        )



        self._body_endoder = LinearEncoder(observation_size, network_settings.num_layers, self.h_size)

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.observation_encoder.update_normalization(buffer)

    def copy_normalization(self, other_network: "NetworkBody") -> None:
        self.observation_encoder.copy_normalization(other_network.observation_encoder)

    def forward(self,inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # mettre input base avant normalization
        encoded_self = self.observation_encoder(inputs)
        encoding = self._body_endoder(encoded_self)
        return encoding



class Critic(abc.ABC):
    @abc.abstractmethod
    def update_normalization(self, buffer: AgentBuffer) -> None:
        """
        Updates normalization of Actor based on the provided List of vector obs.
        :param vector_obs: A List of vector obs as tensors.
        """
        pass

    def critic_pass(self,inputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get value outputs for the given obs.
        :param inputs: List of inputs as tensors.
        :param memories: Tensor of memories, if using memory. Otherwise, None.
        :returns: Dict of reward stream to output tensor for values.
        """
        pass


class ValueNetwork(nn.Module, Critic):
    def __init__(
        self,
        stream_names: List[str],
        network_settings: NetworkSettings,
        module_settings : ModuleSettings,
        outputs_per_stream: int = 1,
    ):

        # This is not a typo, we want to call __init__ of nn.Module
        nn.Module.__init__(self)
        observation_size = module_settings.task_obs_dim + module_settings.state_dim
        self.network_body = NetworkBody(
            observation_size, network_settings, module_settings
        )
        encoding_size = network_settings.hidden_units
        self.value_heads = ValueHeads(stream_names, encoding_size, outputs_per_stream)

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    def critic_pass(self,inputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        value_outputs = self.forward(inputs)
        return value_outputs

    def forward(self,inputs: List[torch.Tensor],) -> Dict[str, torch.Tensor]:
        encoding = self.network_body(inputs)
        output = self.value_heads(encoding)
        return output


class Actor(abc.ABC):
    @abc.abstractmethod
    def update_normalization(self, buffer: AgentBuffer) -> None:
        """
        Updates normalization of Actor based on the provided List of vector obs.
        :param vector_obs: A List of vector obs as tensors.
        """
        pass

    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
    ) -> Tuple[AgentAction, ActionLogProbs, torch.Tensor, torch.Tensor]:
        """
        Returns sampled actions.
        If memory is enabled, return the memories as well.
        :param inputs: A List of inputs as tensors.
        :param masks: If using discrete actions, a Tensor of action masks.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        :return: A Tuple of AgentAction, ActionLogProbs, entropies, and memories.
            Memories will be None if not using memory.
        """
        pass

    def get_stats(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
    ) -> Tuple[ActionLogProbs, torch.Tensor]:
        """
        Returns log_probs for actions and entropies.
        If memory is enabled, return the memories as well.
        :param inputs: A List of inputs as tensors.
        :param actions: AgentAction of actions.
        :param masks: If using discrete actions, a Tensor of action masks.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        :return: A Tuple of AgentAction, ActionLogProbs, entropies, and memories.
            Memories will be None if not using memory.
        """

        pass



class SimpleActor(nn.Module, Actor):
    MODEL_EXPORT_VERSION = 3  # Corresponds to ModelApiVersion.MLAgents2_0

    def __init__(
        self,
        module_settings : ModuleSettings,
        action_size : int,
        network_settings: NetworkSettings,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__()
        observation_size = module_settings.task_obs_dim + module_settings.state_dim
        
        self.network_body = NetworkBody(observation_size, network_settings, module_settings)
        self.encoding_size = network_settings.hidden_units

        self.action_model = ActionModel(
            self.encoding_size,
            action_size,
            tanh_squash=tanh_squash,
            deterministic=network_settings.deterministic,
        )

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    def get_action_and_stats(self,inputs: List[torch.Tensor]) -> Tuple[AgentAction, ActionLogProbs, torch.Tensor]:
        encoding = self.network_body(inputs)
        action, log_probs, entropies = self.action_model(encoding)
        return action, log_probs, entropies

    def get_stats(self,inputs: List[torch.Tensor],actions: AgentAction) -> Tuple[ActionLogProbs, torch.Tensor]:
        encoding = self.network_body(inputs)
        log_probs, entropies = self.action_model.evaluate(encoding, actions)

        return log_probs, entropies



class GlobalSteps(nn.Module):
    def __init__(self):
        super().__init__()
        self.__global_step = nn.Parameter(
            torch.Tensor([0]).to(torch.int64), requires_grad=False
        )

    @property
    def current_step(self):
        return int(self.__global_step.item())

    @current_step.setter
    def current_step(self, value):
        self.__global_step[:] = value

    def increment(self, value):
        self.__global_step += value


