from typing import List, NamedTuple
import numpy as np

from mlagents.trainers.buffer import (
    AgentBuffer,
    ObservationKeyPrefix,
    AgentBufferKey,
    BufferKey,
)
from mlagents_envs.base_env import ActionTuple
from mlagents.trainers.torch_modules.action_log_probs import LogProbsTuple


class AgentStatus(NamedTuple):
    """
    Stores observation, action, and reward for an agent. Does not have additional
    fields that are present in AgentExperience.
    """

    obs: List[np.ndarray]
    reward: float
    action: ActionTuple
    done: bool


class AgentExperience(NamedTuple):
    """
    Stores the full amount of data for an agent in one timestep. Includes
    the status' of group mates and the group reward, as well as the probabilities
    outputted by the policy.
    """

    obs: List[np.ndarray]
    reward: float
    done: bool
    action: ActionTuple
    action_probs: LogProbsTuple
    interrupted: bool
    group_status: List[AgentStatus]
    group_reward: float


class ObsUtil:
    @staticmethod
    def get_name_at(index: int) -> AgentBufferKey:
        """
        returns the name of the observation given the index of the observation
        """
        return ObservationKeyPrefix.OBSERVATION, index

    @staticmethod
    def get_name_at_next(index: int) -> AgentBufferKey:
        """
        returns the name of the next observation given the index of the observation
        """
        return ObservationKeyPrefix.NEXT_OBSERVATION, index

    @staticmethod
    def from_buffer(batch: AgentBuffer, num_obs: int) -> List[np.array]:
        """
        Creates the list of observations from an AgentBuffer
        """
        result: List[np.array] = []
        for i in range(num_obs):
            result.append(batch[ObsUtil.get_name_at(i)])
        return result

    @staticmethod
    def from_buffer_next(batch: AgentBuffer, num_obs: int) -> List[np.array]:
        """
        Creates the list of next observations from an AgentBuffer
        """
        result = []
        for i in range(num_obs):
            result.append(batch[ObsUtil.get_name_at_next(i)])
        return result


class Trajectory(NamedTuple):
    steps: List[AgentExperience]
    next_obs: List[
        np.ndarray
    ]  # Observation following the trajectory, for bootstrapping
    next_group_obs: List[List[np.ndarray]]
    agent_id: str

    def to_agentbuffer(self) -> AgentBuffer:
        """
        Converts a Trajectory to an AgentBuffer
        :param trajectory: A Trajectory
        :returns: AgentBuffer. Note that the length of the AgentBuffer will be one
        less than the trajectory, as the next observation need to be populated from the last
        step of the trajectory.
        """
        agent_buffer_trajectory = AgentBuffer()
        obs = self.steps[0].obs
        for step, exp in enumerate(self.steps):
            is_last_step = step == len(self.steps) - 1
            if not is_last_step:
                next_obs = self.steps[step + 1].obs
            else:
                next_obs = self.next_obs

            num_obs = len(obs)
            for i in range(num_obs):
                agent_buffer_trajectory[ObsUtil.get_name_at(i)].append(obs[i])
                agent_buffer_trajectory[ObsUtil.get_name_at_next(i)].append(next_obs[i])



            agent_buffer_trajectory[BufferKey.DONE].append(exp.done)


            # Adds the log prob and action of continuous/discrete separately
            agent_buffer_trajectory[BufferKey.CONTINUOUS_ACTION].append(exp.action.continuous)

            if not is_last_step:
                next_action = self.steps[step + 1].action
                cont_next_actions = next_action.continuous
            else:
                cont_next_actions = np.zeros_like(exp.action.continuous)

            agent_buffer_trajectory[BufferKey.NEXT_CONT_ACTION].append(
                cont_next_actions
            )

            agent_buffer_trajectory[BufferKey.CONTINUOUS_LOG_PROBS].append(
                exp.action_probs.continuous
            )

            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS].append(exp.reward)

            # Store the next visual obs as the current
            obs = next_obs
        return agent_buffer_trajectory

    @property
    def done_reached(self) -> bool:
        """
        Returns true if trajectory is terminated with a Done.
        """
        return self.steps[-1].done

    @property
    def interrupted(self) -> bool:
        """
        Returns true if trajectory was terminated because max steps was reached.
        """
        return self.steps[-1].interrupted
