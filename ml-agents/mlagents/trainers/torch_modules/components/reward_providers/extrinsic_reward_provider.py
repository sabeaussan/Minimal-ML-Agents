import numpy as np
from typing import Dict

from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.torch_modules.components.reward_providers.base_reward_provider import (
    BaseRewardProvider,
)
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.settings import RewardSignalSettings


class ExtrinsicRewardProvider(BaseRewardProvider):
    """
    Evaluates extrinsic reward. For single-agent, this equals the individual reward
    given to the agent. For the POCA algorithm, we want not only the individual reward
    but also the team and the individual rewards of the other agents.
    """

    def __init__(self, specs: BehaviorSpec, settings: RewardSignalSettings) -> None:
        super().__init__(specs, settings)
        self.add_groupmate_rewards = False

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        indiv_rewards = np.array(
            mini_batch[BufferKey.ENVIRONMENT_REWARDS], dtype=np.float32
        )
        total_rewards = indiv_rewards
        return total_rewards

    def update(self, mini_batch: AgentBuffer) -> Dict[str, np.ndarray]:
        return {}
