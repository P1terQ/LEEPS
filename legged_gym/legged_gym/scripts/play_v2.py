from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
from legged_gym.utils import webviewer


def play(args):
    
    if args.web:
        web_viewer = webviewer.WebViewer()
    
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # env_cfg.env.num_envs = 10
    # env_cfg.terrain.num_rows = 3
    # env_cfg.terrain.num_cols = 10
    # env_cfg.terrain.terrain_dict = {
    #                     "step": 0.1, # proportions[0]
    #                     "gap": 0.1,  # proportions[1]
    #                     "slope": 0.1,
    #                     "stair": 0.1, 
    #                     "pillar": 0.1, 
    #                     "flat": 0.0,       # proportions[5]
    #                     "steppingstones": 0.1, # proportions[6]
    #                     "crawl": 0.1,     # proportions[7]
    #                     "log": 0.3,
    #                     "crack": 0.0,
    #                     "pyramid upstair": 0.0,
    #                     "pyramid gap": 0.0,
    #                     "simple_flat": 0.0
    #                     }   
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.terrain_dict = {
                        "step": 0.0, # proportions[0]
                        "gap": 0.0,  # proportions[1]
                        "slope": 0.0,
                        "stair": 0.0, 
                        "discrete": 0.0, 
                        "flat": 0.0,       # proportions[5]
                        "steppingstones": 1.0, # proportions[6]
                        "crawl": 0.0,     # proportions[7]
                        "log": 0.0,
                        "crack": 0.0,
                        "dual": 0.0,
                        "parkour": 0.0
                        }   
    env_cfg.terrain.curriculum = True
    
    # env_cfg.depth.camera_num_envs = 1
    # env_cfg.depth.camera_terrain_num_rows = 10
    # env_cfg.depth.camera_terrain_num_cols = 1
    
    #! 难度random,也可以在terrain中设置
    # env_cfg.terrain.num_rows = 1
    # env_cfg.terrain.num_cols = 1
    # env_cfg.env.num_envs = 1
    # env_cfg.terrain.border_size = 0.0
    # env_cfg.terrain.terrain_dict = {
    #                     "step": 0.0, # proportions[0]
    #                     "gap": 0.0,  # proportions[1]
    #                     "slope": 0.0,
    #                     "stair": 0.0, 
    #                     "pillar": 0.0, 
    #                     "flat": 0.0,       # proportions[5]
    #                     "steppingstones": 0.0, # proportions[6]
    #                     "crawl": 0.0,     # proportions[7]
    #                     "log": 1.0,
    #                     "crack": 0.0,
    #                     "pyramid upstair": 0.0,
    #                     }
    # env_cfg.terrain.curriculum = False

    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    if args.web:
        web_viewer.setup(env)
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner_rsl(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)


    for i in range(50*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)


if __name__ == '__main__':

    args = get_args()
    
    args.task = 'a1_v2'
    
    # args.debug_viz = True  
    # args.remote = True
    # args.use_camera = True

    # args.load_run = '/home/ustc/robot/projects/legged_locomotion/iros2024/weights/a1_v2/Nov28_10-39-59_001-69'
    
    play(args)
