import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import torch

def test_env(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # env_cfg.terrain.num_rows = max(2, env_cfg.terrain.max_init_terrain_level+1)    #!
    # env_cfg.terrain.num_rows = 10
    # env_cfg.terrain.num_cols = 10
    env_cfg.terrain.num_rows = 10   #!
    env_cfg.terrain.num_cols = 1
    # env_cfg.env.num_envs = env_cfg.terrain.num_rpillar_xows * env_cfg.terrain.num_cols
    env_cfg.env.num_envs = 3
    env_cfg.terrain.max_init_terrain_level = 3
    
    env_cfg.terrain.border_size = 0.0
    env_cfg.terrain.terrain_dict = {
                        "step": 0.0, # proportions[0]
                        "gap": 0.0,  # proportions[1]
                        "slope": 0.0,
                        "stair": 0.0, 
                        "discrete": 1.0, 
                        "flat": 0.0,       # proportions[5]
                        "steppingstones": 0.0, # proportions[6]
                        "crawl": 0.0,     # proportions[7]
                        "log": 0.0,
                        "crack": 0.0,
                        "dual": 0.0,
                        "parkour": 0.0
                        }
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = True
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    for i in range(int(1000*env.max_episode_length)):
        actions = 0.*torch.ones(env.num_envs, env.num_actions, device=env.device)
        obs, _, rew, done, info = env.step(actions)
    print("Done")

if __name__ == '__main__':
    args = get_args()
    
    # choose task
    args.task = 'a1_v2'
    args.debug_viz = True
    
    test_env(args)
