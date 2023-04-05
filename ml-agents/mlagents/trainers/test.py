from mlagents.trainers.torch_modules.encoders import VectorInput
from mlagents.trainers.torch_modules.layers import LinearEncoder
from mlagents.trainers.torch_modules.action_model import ActionModel
from mlagents.trainers.settings import ModuleSettings
from mlagents_envs.base_env import ActionSpec
from unity_env_gym import make_unity_env
import torch
import numpy as np
import glob
import re
import matplotlib.pyplot as plt

@torch.no_grad()
def test_agent(nb_test_episodes):
    rewards_episode = []
    reward_history = []
    reward_tracking = []
    for episode in range(nb_test_episodes):
        obs = env.reset()
        done = False 
        while not done: 
            env.render()
            with torch.no_grad(): 
                obs = torch.tensor(obs).float().to("cuda:0")
                norm_obs = vector_input(obs)
                encoding = body(norm_obs)
                action,_,_ = action_head(encoding)
                robot_act = (torch.clamp(action.continuous_tensor, -3, 3) / 3 ).cpu()
            next_obs, rew, done, _ = env.step(robot_act)
            rewards_episode.append(rew)
            obs = next_obs 
        print(np.sum(rewards_episode))
        reward_history.append(np.sum(rewards_episode))
        rewards_episode = []
    return np.mean(reward_history)




ENV = "CustomWalker"
STATE_DIM = 56
TASK_DIM = 18
ACTION_DIM = 28


HIDDEN_DIM = 512
LATENT_DIM = 6
ENV_NAME = f"envs/{ENV}.x86_64"

step = 10000027

RES_DIR_PATH = "results/test_walker/checkpoints/"


obs_dim = STATE_DIM + TASK_DIM

vector_input = VectorInput(input_size = obs_dim, normalize = True)
vector_input.load_state_dict(torch.load(RES_DIR_PATH+f"vector_input-{step}.pth"),strict = True)

body = LinearEncoder(
    input_size=obs_dim,
    num_layers=3,
    hidden_size=HIDDEN_DIM,
)
body.load_state_dict(torch.load(RES_DIR_PATH+f"body_endoder-{step}.pth"),strict = True)


action_head = ActionModel(
    HIDDEN_DIM,
    ACTION_DIM,
    tanh_squash=False,
    deterministic=False,
)
action_head.load_state_dict(torch.load(RES_DIR_PATH+f"action_model-{step}.pth"),strict = True)

env = make_unity_env(ENV_NAME, worker_id = 40, no_graphics = False, time_scale = 1.0)

test_agent(100)

env.close()