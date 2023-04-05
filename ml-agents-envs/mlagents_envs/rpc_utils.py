from mlagents_envs.base_env import (
    ActionSpec,
    ObservationSpec,
    DimensionProperty,
    BehaviorSpec,
    DecisionSteps,
    TerminalSteps,
    ObservationType,
)
from mlagents_envs.exception import UnityObservationException
from mlagents_envs.timers import hierarchical_timer, timed
from mlagents_envs.communicator_objects.agent_info_pb2 import AgentInfoProto
from mlagents_envs.communicator_objects.observation_pb2 import (
    ObservationProto,
    NONE as COMPRESSION_TYPE_NONE,
)
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
import numpy as np
import io
from typing import cast, List, Tuple, Collection, Optional, Iterable
from PIL import Image


PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def behavior_spec_from_proto(
    brain_param_proto: BrainParametersProto, agent_info: AgentInfoProto
) -> BehaviorSpec:
    """
    Converts brain parameter and agent info proto to BehaviorSpec object.
    :param brain_param_proto: protobuf object.
    :param agent_info: protobuf object.
    :return: BehaviorSpec object.
    """
    observation_specs = []
    for obs in agent_info.observations:
        observation_specs.append(
            ObservationSpec(
                name=obs.name,
                shape=tuple(obs.shape),
                observation_type=ObservationType(obs.observation_type),
                dimension_property=tuple(
                    DimensionProperty(dim) for dim in obs.dimension_properties
                )
                if len(obs.dimension_properties) > 0
                else (DimensionProperty.UNSPECIFIED,) * len(obs.shape),
            )
        )

    # proto from communicator < v1.3 does not set action spec, use deprecated fields instead
    if (
        brain_param_proto.action_spec.num_continuous_actions == 0
        and brain_param_proto.action_spec.num_discrete_actions == 0
    ):
        if brain_param_proto.vector_action_space_type_deprecated == 1:
            action_spec = ActionSpec(
                brain_param_proto.vector_action_size_deprecated[0], ()
            )
        else:
            action_spec = ActionSpec(
                0, tuple(brain_param_proto.vector_action_size_deprecated)
            )
    else:
        action_spec_proto = brain_param_proto.action_spec
        action_spec = ActionSpec(
            action_spec_proto.num_continuous_actions,
        )
    return BehaviorSpec(observation_specs, action_spec)


class OffsetBytesIO:
    """
    Simple file-like class that wraps a bytes, and allows moving its "start"
    position in the bytes. This is only used for reading concatenated PNGs,
    because Pillow always calls seek(0) at the start of reading.
    """

    __slots__ = ["fp", "offset"]

    def __init__(self, data: bytes):
        self.fp = io.BytesIO(data)
        self.offset = 0

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            res = self.fp.seek(offset + self.offset)
            return res - self.offset
        raise NotImplementedError()

    def tell(self) -> int:
        return self.fp.tell() - self.offset

    def read(self, size: int = -1) -> bytes:
        return self.fp.read(size)

    def original_tell(self) -> int:
        """
        Returns the offset into the original byte array
        """
        return self.fp.tell()




def _check_observations_match_spec(
    obs_index: int,
    observation_spec: ObservationSpec,
    agent_info_list: Collection[AgentInfoProto],
) -> None:
    """
    Check that all the observations match the expected size.
    This gives a nicer error than a cryptic numpy error later.
    """
    expected_obs_shape = tuple(observation_spec.shape)
    for agent_info in agent_info_list:
        agent_obs_shape = tuple(agent_info.observations[obs_index].shape)
        if expected_obs_shape != agent_obs_shape:
            raise UnityObservationException(
                f"Observation at index={obs_index} for agent with "
                f"id={agent_info.id} didn't match the ObservationSpec. "
                f"Expected shape {expected_obs_shape} but got {agent_obs_shape}."
            )




def _raise_on_nan_and_inf(data: np.array, source: str) -> np.array:
    # Check for NaNs or Infinite values in the observation or reward data.
    # If there's a NaN in the observations, the np.mean() result will be NaN
    # If there's an Infinite value (either sign) then the result will be Inf
    # See https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy for background
    # Note that a very large values (larger than sqrt(float_max)) will result in an Inf value here
    # Raise a Runtime error in the case that NaNs or Infinite values make it into the data.
    if data.size == 0:
        return data

    d = np.mean(data)
    has_nan = np.isnan(d)
    has_inf = not np.isfinite(d)

    if has_nan:
        raise RuntimeError(f"The {source} provided had NaN values.")
    if has_inf:
        raise RuntimeError(f"The {source} provided had Infinite values.")


@timed
def _process_rank_one_or_two_observation(
    obs_index: int,
    observation_spec: ObservationSpec,
    agent_info_list: Collection[AgentInfoProto],
) -> np.ndarray:
    if len(agent_info_list) == 0:
        return np.zeros((0,) + observation_spec.shape, dtype=np.float32)
    try:
        np_obs = np.array(
            [
                agent_obs.observations[obs_index].float_data.data
                for agent_obs in agent_info_list
            ],
            dtype=np.float32,
        ).reshape((len(agent_info_list),) + observation_spec.shape)
    except ValueError:
        # Try to get a more useful error message
        _check_observations_match_spec(obs_index, observation_spec, agent_info_list)
        # If that didn't raise anything, raise the original error
        raise
    _raise_on_nan_and_inf(np_obs, "observations")
    return np_obs


@timed
def steps_from_proto(
    agent_info_list: Collection[AgentInfoProto], behavior_spec: BehaviorSpec
) -> Tuple[DecisionSteps, TerminalSteps]:
    decision_agent_info_list = [
        agent_info for agent_info in agent_info_list if not agent_info.done
    ]
    terminal_agent_info_list = [
        agent_info for agent_info in agent_info_list if agent_info.done
    ]
    decision_obs_list: List[np.ndarray] = []
    terminal_obs_list: List[np.ndarray] = []
    for obs_index, observation_spec in enumerate(behavior_spec.observation_specs):
        decision_obs_list.append(
            _process_rank_one_or_two_observation(
                obs_index, observation_spec, decision_agent_info_list
            )
        )
        terminal_obs_list.append(
            _process_rank_one_or_two_observation(
                obs_index, observation_spec, terminal_agent_info_list
            )
        )
    decision_rewards = np.array(
        [agent_info.reward for agent_info in decision_agent_info_list], dtype=np.float32
    )
    terminal_rewards = np.array(
        [agent_info.reward for agent_info in terminal_agent_info_list], dtype=np.float32
    )

    decision_group_rewards = np.array(
        [agent_info.group_reward for agent_info in decision_agent_info_list],
        dtype=np.float32,
    )
    terminal_group_rewards = np.array(
        [agent_info.group_reward for agent_info in terminal_agent_info_list],
        dtype=np.float32,
    )

    _raise_on_nan_and_inf(decision_rewards, "rewards")
    _raise_on_nan_and_inf(terminal_rewards, "rewards")
    _raise_on_nan_and_inf(decision_group_rewards, "group_rewards")
    _raise_on_nan_and_inf(terminal_group_rewards, "group_rewards")

    decision_group_id = [agent_info.group_id for agent_info in decision_agent_info_list]
    terminal_group_id = [agent_info.group_id for agent_info in terminal_agent_info_list]

    max_step = np.array(
        [agent_info.max_step_reached for agent_info in terminal_agent_info_list],
        dtype=np.bool,
    )
    decision_agent_id = np.array(
        [agent_info.id for agent_info in decision_agent_info_list], dtype=np.int32
    )
    terminal_agent_id = np.array(
        [agent_info.id for agent_info in terminal_agent_info_list], dtype=np.int32
    )

    return (
        DecisionSteps(
            decision_obs_list,
            decision_rewards,
            decision_agent_id,
            decision_group_id,
            decision_group_rewards,
        ),
        TerminalSteps(
            terminal_obs_list,
            terminal_rewards,
            max_step,
            terminal_agent_id,
            terminal_group_id,
            terminal_group_rewards,
        ),
    )

