import time
import os
from collections import deque
import statistics

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import datetime

from rsl_rl.algorithms import PPO,PPORMA
from rsl_rl.modules import *
from rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings

class OnPolicyRunnerDepth:
    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        self.runner_cfg = train_cfg["runner"] 
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]
        self.depth_encoder_cfg = train_cfg["depth_encoder"]
        self.device = device
        self.env = env
        
        actor_critic = ActorCriticRMA(num_prop = self.env.cfg.env.n_proprio,  # 本体感受
                                    num_scandots = self.env.cfg.env.n_scan, # 扫描点
                                    num_priv_info = self.env.cfg.env.n_priv_latent, # 环境的priv信息
                                    num_priv_explicit = self.env.cfg.env.n_priv,    # 本体的priv信息
                                    num_history = self.env.cfg.env.history_len, # 历史obs_buffer长度
                                    num_critic_obs = self.env.num_obs,
                                    num_actions = self.env.num_actions,
                                    scan_encoder_dims = self.policy_cfg["scan_encoder_dims"],
                                    actor_hidden_dims = self.policy_cfg["actor_hidden_dims"],
                                    critic_hidden_dims = self.policy_cfg["critic_hidden_dims"],
                                    priv_encoder_dims = self.policy_cfg["priv_encoder_dims"],   # 环境的priv信息encoder
                                    activation = self.policy_cfg["activation"],
                                    init_noise_std = self.policy_cfg["init_noise_std"]
                                    ).to(self.device)
        
        estimator = Estimator(input_dim=env.cfg.env.n_proprio,
                              output_dim=env.cfg.env.n_priv,
                              hidden_dims=self.estimator_cfg["hidden_dims"],
                              activation = self.estimator_cfg["activation"],
                              ).to(self.device)
        
        # 是否使用depth camera
        self.if_depth = self.depth_encoder_cfg["if_depth"]
        
        if self.if_depth:
            depth_backbone = DepthBackbone58x87(scan_output_dim=self.policy_cfg["scan_encoder_dims"][-1])
            depth_encder = RecurrentDepthBackbone(base_backbone=depth_backbone, 
                                                  n_proprio=self.env.cfg.env.n_proprio).to(self.device)
            depth_actor = deepcopy(actor_critic.actor)
        else:
            depth_encder = None
            depth_actor = None
            
        self.alg = PPORMA(actor_critic=actor_critic,
                       estimator=estimator,
                       estimator_params = self.estimator_cfg,
                       depth_encoder=depth_encder,
                       depth_encoder_paras = self.depth_encoder_cfg,
                       depth_actor=depth_actor,
                       device = self.device,
                       **self.alg_cfg)  # 剩下的rl参数通过alg_cfg传入
        
        self.num_steps_per_env = self.runner_cfg["num_steps_per_env"]
        self.save_interval = self.runner_cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]
        
        self.alg.init_storage(num_envs=self.env.num_envs,
                              num_transitions_per_env=self.num_steps_per_env,
                              actor_obs_shape=[self.env.num_obs],
                              critic_obs_shape=[self.env.num_privileged_obs],
                              action_shape=[self.env.num_actions])
        
        if self.if_depth:
            self.learn = self.learn_depth
        else:
            self.learn = self.learn_RL
            
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        
    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(input= self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        # get AC obs
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        
        infos = {}
        if self.if_depth:
            infos["depth"] = self.env.depth_buffer.clone().to(self.device)
        else:
            infos["depth"] = None
            
        # switch to train mode
        self.alg.actor_crtic.train()
        
        ep_infos = []
        rew_buffer = deque(maxlen=100)
        len_buffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        # 0 + num_learning_iterations
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)
        
        for it in range(self.current_learning_iteration, tot_iter):
            start_time = time.time()
            hist_encoding = it % self.dagger_update_freq == 0   # freq=20. 
            
            with torch.inference_mode():
                for i in range(self.num_steps_per_env): # 每24步更新一次
                    # update distribution and sample action
                    actions = self.alg.act(obs, critic_obs, hist_encoding=hist_encoding)
                    # interact
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = \
                        obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    total_rewards = self.alg.process_env_step(rewards, dones, infos)
        
                    if self.log_dir is not None:
                        if 'episode' in infos:  # true. 这个['episode']中都是reward的信息
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_rewards
                        cur_episode_length += 1
                        
                        new_ids = (dones>0).nonzero(as_tuple=False)
                        
                        rew_buffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        len_buffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        
                stop_time = time.time()
                collection_time = stop_time - start_time
                
                # learning step
                start_time = stop_time
                self.alg.compute_returns(critic_obs)    # compute  advantage and return
            mean_value_loss, mean_surrogate_loss, mean_estimator_loss, _, _, mean_priv_reg_loss, priv_reg_coef = self.alg.update()

            if hist_encoding:   # update history encoder
                mean_hist_latent_loss = self.alg.update_dagger()
                
            stop_time = time.time()
            learn_time = stop_time - start_time
            
            if self.log_dir is not None:
                self.log(locals())  # log infos
                
            if it < 2500:
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            elif it < 5000:
                if it % (2*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                if it % (5*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()    # 记录了reward信息，每次学习完清空
            
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))


    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs['collection_time'] + locs['learn_time']
        
        ep_string = f''
        if locs['ep_infos']:    # reward信息
            for key in locs['ep_infos'][0]: # 遍历每一项reward
                infotensor =torch.tensor([], device=self.device)    # 用于存储每一个环境的reward的容器
                # 遍历每一个环境的key reward
                for ep_info in locs['ep_infos']:
                    #handle scalar and zero-dim tensors
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                # 计算key reward在所有的环境中的平均值
                value = torch.mean(infotensor) 
                # add to tensorboard
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                # print string
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_crtic.action_std.mean()
        # 每个env_step消耗的时间
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/estimator', locs['mean_estimator_loss'], locs['it'])
        self.writer.add_scalar('Loss/priv_reg', locs['mean_priv_reg_loss'], locs['it'])
        self.writer.add_scalar('Loss/priv_reg_coef', locs['priv_reg_coef'], locs['it'])
        
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        
        if len(locs['rew_buffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rew_buffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rew_buffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['len_buffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['len_buffer']), self.tot_time)

        # 初始化print到terminal的信息
        # 记录当前的学习次数
        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rew_buffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rew_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
        
        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)
        

    def learn_depth(self, num_learning_iterations, init_at_random_ep_len=False):
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)
        
        ep_infos = []
        rew_buffer = deque(maxlen=100)
        len_buffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_epsisode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        obs = self.env.get_observations()
        infos = {}
        if self.if_depth:
            #todo 目前是只输入最后一帧的depth， 之后改成4帧的试试
            infos["depth"] = self.env.depth_buffer.clone().to(self.device)[:, -1]
        else:
            infos["depth"] = None
        self.alg.depth_encoder.train()
        self.alg.depth_actor.train()
        
        for it in range(self.current_learning_iteration, tot_iter):
            start_time = time.time()
            
            depth_latent_buffer = []
            scandots_latent_buffer = []
            actions_teacher_buffer = []
            actions_student_buffer = []
            
            for i in range(self.depth_encoder_cfg["num_steps_per_env"]): # 5 * 24
                if infos["depth"] != None:
                    
                    with torch.no_grad:
                        scandots_latent = self.alg.actor_crtic.actor.infer_scandots_latent(obs)
                    scandots_latent_buffer.append(scandots_latent)
                    
                    obs_prop = obs[:, :self.env.cfg.env.n_proprio].clone()
                    
                    depth_latent = self.alg.depth_encoder(infos["depth"].clone(), obs_prop)
                    depth_latent_buffer.append(depth_latent)
                    
                with torch.no_grad():
                    actions_teacher = self.alg.actor_crtic.act_inference(obs, use_historyestimate=True, scandots_latent=None)
                    actions_teacher_buffer.append(actions_teacher)
                    
                obs_student = obs.clone()
                actions_student = self.alg.depth_actor(obs_student, use_historyestimate=True, depth_latent=depth_latent)
                actions_student_buffer.append(actions_student)
                
                obs, privileged_obs, rewards, dones, infos = self.env.step(actions_student)
                
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                if self.log_dir is not None:
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_epsisode_length += 1
                    new_ids = (dones>0).nonzero(as_tuple=False)
                    rew_buffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    len_buffer.extend(cur_epsisode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_epsisode_length[new_ids] = 0

            stop_time = time.time()
            collection_time = stop_time - start_time
            start_time = stop_time
            
            actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
            actions_student_buffer = torch.cat(actions_student_buffer, dim=0)
            depth_actor_loss = self.alg.depth_actor.update(actions_teacher_buffer, actions_student_buffer)
            
            stop_time = time.time()
            learn_time = stop_time - start_time
            
            #? 不清楚这个什么意思
            self.alg.depth_encoder.detach_hidden_states()    
        
            if self.log_dir is not None:
                self.log_vision(locals())

            if (it-self.start_learning_iteration < 2500 and it % self.save_interval == 0) or \
               (it-self.start_learning_iteration < 5000 and it % (2*self.save_interval) == 0) or \
               (it-self.start_learning_iteration >= 5000 and it % (5*self.save_interval) == 0):
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                    
            ep_infos.clear()
            
            
    def log_vision(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs['collection_time'] + locs['learn_time']
        
        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                # 每项reward在所有env中的平均
                infotensor =torch.Tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    #handle scalar and zero-dim tensors
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                
                # write to tensorboard
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                # print terminal string
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        
        mean_std = self.alg.actor_crtic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        
        self.writer.add_scalar('Loss/depth_actor', locs['depth_actor_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rew_buffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rew_buffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rew_buffer']), self.tot_time)
            # self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['len_buffer']), locs['it'])
            # self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['len_buffer']), self.tot_time)
        
        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rew_buffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rew_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['len_buffer']):.2f}\n"""
                        #   f"""{'Depth encoder loss:':>{pad}} {locs['depth_encoder_loss']:.4f}\n"""
                          f"""{'Depth actor loss:':>{pad}} {locs['depth_actor_loss']:.4f}\n""")
                        #   f"""{'Yaw loss:':>{pad}} {locs['yaw_loss']:.4f}\n"""
                        #   f"""{'Delta yaw ok percentage:':>{pad}} {locs['delta_yaw_ok_percentage']:.4f}\n""")
        else:
            log_string = (f"""{'#' * width}\n""")      
                  
        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)
        
    
    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_crtic.state_dict(),
            'estimator_state_dict:': self.alg.estimator.state_dict(),
            'optimizer_state_dict': self.alg.AC_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos
        }
        if self.if_depth:
            state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
            state_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()

        torch.save(state_dict, path)
    
    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_crtic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        
        if self.if_depth:
            self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])
            self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
            
        if load_optimizer:
            self.alg.AC_optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            
        self.current_learning_iteration = loaded_dict['iter']
        
        print("*" * 80)
        return loaded_dict['infos']
    
    
    
        
        
        
        
        
        
        