import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
# import torch
import wandb
from loguru import logger

def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    
    # env_cfg.env.num_envs = 12
    # env_cfg.terrain.num_rows = max(2, env_cfg.terrain.max_init_terrain_level+1)    
    # env_cfg.terrain.num_cols = 2
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    log_root = os.path.join(root_dir, 'logs', train_cfg.runner.experiment_name)
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
    
    
    wandb.save("/home/ustc/robot/learning/iros2024/legged_gym/legged_gym/envs/terrainprimitive" + "/legged_vls_config.py", policy="now")
    wandb.save("/home/ustc/robot/learning/iros2024/legged_gym/legged_gym/envs/terrainprimitive" + "/legged_vls.py", policy="now")
    
    env, env_cfg = task_registry.make_env(name=args.task, args=args)    
    ppo_runner, train_cfg = task_registry.make_alg_runner_rsl(env=env, name=args.task, args=args, log_dir_=log_dir)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    
    # args.headless = True
    args.task = 'a1_v2'    # task选择任务的文件
    
    # args.run_name = '000-00'    # run_name用来保存权重目录. 前3为表示group，后面为实验编号. 相同代码不同参数的实验为同一group
    
    args.use_camera = True
    args.delay = True
    args.resume = True
    
    # gap
    # args.load_run = '/home/ustc/robot/learning/iros2024/logs/a1_v2/Nov24_17-24-59_001-43'
    args.load_run = '/home/ustc/robot/learning/iros2024/logs/a1_v2/Dec02_20-28-03_001-78'
    
    if args.load_run == None:
        logger.warning(f'load_run must be specified when training student policy')
        raise ValueError('load_run must be specified')

    train(args)
