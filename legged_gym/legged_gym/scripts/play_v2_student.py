from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
from loguru import logger

import time


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    env_cfg.depth.camera_num_envs = 1
    env_cfg.depth.camera_terrain_num_rows = 10
    env_cfg.depth.camera_terrain_num_cols = 1
    env_cfg.terrain.terrain_dict = {
                        "step": 0.0, # proportions[0]
                        "gap": 0.0,  # proportions[1]
                        "slope": 0.0,
                        "stair": 0.0, 
                        "pillar": 0.0, 
                        "flat": 0.0,       # proportions[5]
                        "steppingstones": 1.0, # proportions[6]
                        "crawl": 0.0,     # proportions[7]
                        "log": 0.0,
                        "crack": 0.0,
                        "pyramid upstair": 0.0,
                        "pyramid gap": 0.0,
                        "simple_flat": 0.0
                        }   
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 1
    env_cfg.terrain.simplify_grid = False
    # env_cfg.terrain.curriculum = False
       
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    # env_cfg.terrain.simplify_grid = False
    
    
        
    # prepare environment
    env: LeggedV2
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner_rsl(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)
    
    estimator = ppo_runner.get_estimator_inference_policy()
    
    terrain_depth_encoder = ppo_runner.get_terrain_depth_encoder_inference_policy()
    ceiling_depth_encoder = ppo_runner.get_ceiling_depth_encoder_inference_policy()
    
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None # [1, 58, 87]


    for i in range(50*int(env.max_episode_length)):
        
        time1= time.time()
        # print("time1: ", time1)
        
        if infos["depth"] is not None:
            obs_student = obs[:, :env.cfg.env.n_proprio].clone()
          
            terrain_depth_latent_estimated = terrain_depth_encoder(infos["depth"], obs_student)
            ceiling_depth_latent_estimated = ceiling_depth_encoder(infos["depth"], obs_student)
        # else:
        #     raise ValueError('depth is None')
        
        time2= time.time()
        # print("time2: ", time2)
                
        priv_selfstates_estimated = estimator(obs_student)
        obs[:, env.cfg.env.n_proprio+env.cfg.env.n_scan_terrain+env.cfg.env.n_scan_ceiling:env.cfg.env.n_proprio+env.cfg.env.n_scan_terrain+env.cfg.env.n_scan_ceiling+env.cfg.env.n_priv] = priv_selfstates_estimated
        
        if hasattr(ppo_runner.alg, "depth_actor"):
            actions = ppo_runner.alg.depth_actor(obs.detach(), 
                                                 use_historyestimate=True, 
                                                 terrain_scandots_latent=terrain_depth_latent_estimated, 
                                                 ceiling_scandots_latent=ceiling_depth_latent_estimated)
        else:
            raise ValueError('ppo_runner.alg.depth_actor is None')
        
        # time3= time.time()
        # print("time3: ", time3)
        # print("duration: ", time3-time1)        
        
        # actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    
    args.task = 'a1_v2'
    args.debug_viz = True  
    args.use_camera = True
    args.delay = True
    args.resume = True
    
    # args.remote = True
    
    # args.load_run = '/home/ustc/robot/projects/legged_locomotion/iros2024/weights/a1_v2/Dec05_10-35-14_002-09'
    
    if args.load_run == None:
        logger.warning(f'load_run must be specified when training student policy')
        raise ValueError('load_run must be specified')
    
    play(args)
