import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
# import torch
import wandb

def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    
    #! test
    # env_cfg.env.debug_viz = True
    # env_cfg.env.num_envs = 1
    # env_cfg.terrain.num_rows = max(2, env_cfg.terrain.max_init_terrain_level+1)    
    # env_cfg.terrain.num_cols = 2
    # env_cfg.terrain.max_init_terrain_level = 9
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    log_root = os.path.join(root_dir, 'weights', train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + args.run_name)
    os.makedirs(log_dir)
    
    wandb.init(project=args.task,
               name=args.run_name,
               entity="p1terq",
               group=args.run_name[:3],
               mode="online", 
               dir=log_dir,
               config={
                "ENV_CFG": vars(env_cfg),
                "TRAIN_CFG": vars(train_cfg)
               })
    
    
    wandb.save("/home/ustc/robot/learning/iros2024/legged_gym/legged_gym/envs/legged" + "/legged_v2_config.py", policy="now")
    wandb.save("/home/ustc/robot/learning/iros2024/legged_gym/legged_gym/envs/legged" + "/legged_v2.py", policy="now")
    
    env, env_cfg = task_registry.make_env(name=args.task, args=args)    
    ppo_runner, train_cfg = task_registry.make_alg_runner_rsl(env=env, name=args.task, args=args, log_dir_=log_dir)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    
    # args.headless = True
    args.task = 'a1_v2'    # task选择任务的文件
    
    # args.run_name = '000-00'    # run_name用来保存权重目录. 前3为表示group，后面为实验编号. 相同代码不同参数的实验为同一group
    
    train(args)
