from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from mlagents.torch_utils import torch, default_device
import copy

from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.behavior_id_utils import get_global_agent_id
from mlagents.trainers.policy import Policy
from mlagents_envs.base_env import DecisionSteps, BehaviorSpec
from mlagents_envs.timers import timed

from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.torch_modules.networks import SimpleActor, GlobalSteps

from mlagents.trainers.torch_modules.utils import ModelUtils
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.torch_modules.agent_action import AgentAction
from mlagents.trainers.torch_modules.action_log_probs import ActionLogProbs

from mlagents_envs.base_env import ActionTuple
EPSILON = 1e-7  # Small value to avoid divide by zero


class TorchPolicy(Policy):
    def __init__(
        self,
        seed: int,
        behavior_spec: BehaviorSpec,
        trainer_settings: TrainerSettings,
        tanh_squash: bool = False,              # check importance dans la convergence
        separate_critic: bool = True,
        condition_sigma_on_obs: bool = True,
    ):
        """
        Policy that uses a multilayer perceptron to map the observations to actions. Could
        also use a CNN to encode visual input prior to the MLP. Supports discrete and
        continuous actions, as well as recurrent networks.
        :param seed: Random seed.
        :param behavior_spec: Assigned BehaviorSpec object.
        :param trainer_settings: Defined training parameters.
        :param load: Whether a pre-trained model will be loaded or a new one created.
        :param tanh_squash: Whether to use a tanh function on the continuous output,
        or a clipped output.
        """
        super().__init__(
            seed, behavior_spec, trainer_settings, tanh_squash, condition_sigma_on_obs
        )
        self.global_step = (
            GlobalSteps()
        )  # could be much simpler if TorchPolicy is nn.Module
        self.grads = None

        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }
        print(behavior_spec[1])
        self.module_settings = trainer_settings.module_settings
        self.actor = SimpleActor(
            module_settings = trainer_settings.module_settings,
            action_size = behavior_spec[1].continuous_size,
            network_settings= trainer_settings.network_settings,
            conditional_sigma=self.condition_sigma_on_obs,
            tanh_squash=tanh_squash,
        )

        self.actor.to(default_device())
        self._clip_action = not tanh_squash

        print()
        print("=================== ACTOR NETWORK ====================")
        print(self.actor)
        print()


    def update_normalization(self, buffer: AgentBuffer) -> None:
        """
        If this policy normalizes vector observations, this will update the norm values in the graph.
        :param buffer: The buffer with the observations to add to the running estimate
        of the distribution.
        """

        if self.normalize:
            self.actor.update_normalization(buffer)

    @timed
    def sample_actions(self,obs: List[torch.Tensor]) -> Tuple[AgentAction, ActionLogProbs, torch.Tensor]:
        """
        :param obs: List of observations.
        :param masks: Loss masks for RNN, else None.
        :param memories: Input memories when using RNN, else None.
        :param seq_len: Sequence length when using RNN.
        :return: Tuple of AgentAction, ActionLogProbs, entropies, and output memories.
        """
        actions, log_probs, entropies = self.actor.get_action_and_stats(obs)
        return (actions, log_probs, entropies)

    def evaluate_actions(self,obs: List[torch.Tensor],actions: AgentAction,) -> Tuple[ActionLogProbs, torch.Tensor]:
        log_probs, entropies = self.actor.get_stats(obs, actions)
        return log_probs, entropies

    @timed
    def evaluate(self, decision_requests: DecisionSteps, global_agent_ids: List[str]) -> Dict[str, Any]:
        """
        Evaluates policy for the agent experiences provided.
        :param global_agent_ids:
        :param decision_requests: DecisionStep object containing inputs.
        :return: Outputs from network as defined by self.inference_dict.
        """
        obs = decision_requests.obs
        tensor_obs = [torch.as_tensor(np_ob) for np_ob in obs]
        #print(tensor_obs[0].shape)
        run_out = {}
        with torch.no_grad():
            action, log_probs, entropy = self.sample_actions(tensor_obs)
        action_tuple = action.to_action_tuple()
        run_out["action"] = action_tuple

        env_action_tuple = action.to_action_tuple(clip=self._clip_action)
        run_out["env_action"] = env_action_tuple 

        run_out["log_probs"] = log_probs.to_log_probs_tuple()
        run_out["entropy"] = ModelUtils.to_numpy(entropy)
        run_out["learning_rate"] = 0.0
        return run_out

    def get_action(self, decision_requests: DecisionSteps, worker_id: int = 0) -> ActionInfo:
        """
        Decides actions given observations information, and takes them in environment.
        :param worker_id:
        :param decision_requests: A dictionary of behavior names and DecisionSteps from environment.
        :return: an ActionInfo containing action, memories, values and an object
        to be passed to add experiences
        """
        if len(decision_requests) == 0:
            #ret = ActionInfo.empty().env_action
            return ActionInfo.empty()

        global_agent_ids = [
            get_global_agent_id(worker_id, int(agent_id))
            for agent_id in decision_requests.agent_id
        ]  # For 1-D array, the iterator order is correct.

        run_out = self.evaluate(decision_requests, global_agent_ids)
        self.check_nan_action(run_out.get("action"),decision_requests)
        return ActionInfo(
            action=run_out.get("action"),
            env_action=run_out.get("env_action"),
            outputs=run_out,
            agent_ids=list(decision_requests.agent_id),
        )

    def get_current_step(self):
        """
        Gets current model step.
        :return: current model step.
        """
        return self.global_step.current_step

    def set_step(self, step: int) -> int:
        """
        Sets current model step to step without creating additional ops.
        :param step: Step to set the current model step to.
        :return: The step the model was set to.
        """
        self.global_step.current_step = step
        return step

    def increment_step(self, n_steps):
        """
        Increments model step.
        """
        self.global_step.increment(n_steps)
        return self.get_current_step()

    def load_weights(self, values: List[np.ndarray]) -> None:
        self.actor.load_state_dict(values)

    def init_load_weights(self) -> None:
        pass

    def get_weights(self) -> List[np.ndarray]:
        return copy.deepcopy(self.actor.state_dict())

    def get_modules(self):
        return {"Policy": self.actor, "global_step": self.global_step}
