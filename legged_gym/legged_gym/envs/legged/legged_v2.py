
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
# from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.legged.base_task import BaseTask
from legged_gym.utils.terrain_vls import Terrainvls

from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from scipy.spatial.transform import Rotation as R

from .legged_v2_config import LeggedV2Cfg

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import sys
from legged_gym.utils.gamepad_reader import Gamepad

import csv

class LeggedV2(BaseTask):
    def __init__(self, cfg: LeggedV2Cfg, sim_params, physics_engine, sim_device, headless):

        self.cfg = cfg
        self.sim_params = sim_params
        self.terrain_height_samples = None
        self.ceiling_height_samples = None
        self.record_now = False
        
        #! visualization
        self.debug_viz = self.cfg.env.debug_viz
        
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        
        # image resize matrix
        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        
        if self.cfg.env.remote:
            self.gamepad = Gamepad()
            self.gamepad_callback = self.gamepad.get_command
        
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
            
        self._init_buffers()
        self._prepare_reward_function()
        
        self.init_done = True
        self.global_counter = 0 # 一直是比common_step_counter少1

        self.reset_idx(torch.arange(self.num_envs, device=self.device)) # reset every env
        self.post_physics_step()
        
        #! 可视化环境中的x_edges和y_edges
        # self.gym.clear_lines(self.viewer)
        # x_edge_geom = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 0, 0))
        # # for i in range(self.x_edge_mask.shape[0]):
        #     # for j in range(self.x_edge_mask.shape[1]):
        # for i in range(self.terrain.heightsamples.shape[0]):
        #     for j in range(self.terrain.heightsamples.shape[1]):
        #         if self.x_edge_mask[i,j]:
        #             pose = gymapi.Transform(gymapi.Vec3(i * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size, j * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size, self.terrain.heightsamples[i,j] * self.cfg.terrain.vertical_scale), r=None)
        #             gymutil.draw_lines(x_edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
        # y_edge_geom = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 0, 0))
        # # for i in range(self.y_edge_mask.shape[0]):
        # #     for j in range(self.y_edge_mask.shape[1]):
        # for i in range(self.terrain.heightsamples.shape[0]):
        #     for j in range(self.terrain.heightsamples.shape[1]):
        #         if self.y_edge_mask[i,j]:
        #             pose = gymapi.Transform(gymapi.Vec3(i * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size, j * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size, self.terrain.heightsamples[i,j] * self.cfg.terrain.vertical_scale), r=None)
        #             gymutil.draw_lines(y_edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        # print("edge visulization done")

    def step(self, actions):

        actions = self.reindex(actions) # action从实际的顺序转换到sim的顺序

        actions.to(self.device)
        # push new action to buffer
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)   # sim中的顺序
        
        # action delay. 暂时还没用上
        if self.cfg.domain_rand.action_delay:   # true
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:   # 24 * 8000 = 192000
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
            if self.viewer:
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)    # 1
            # 花里胡哨的,反正delay都取[:,-2]就好了
            indices = -self.delay -1   
            actions = self.action_history_buf[:, indices.long()] # delay for 1/50=20ms


        self.global_counter += 1
        
        # clip action
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale  # clip/0.5
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        # 可视化更新
        self.render()

        # policy每次新采样出来的action都会被执行decimation次. 实际中会更多
        for _ in range(self.cfg.control.decimation):    
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            
        #! 
        self.post_physics_step()

        # clip observation
        clip_obs = self.cfg.normalization.clip_observations # 100
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        # update_interval=5. Policy是0.02s一次. depth是0.1s更新一次
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None
            
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_history_observations(self):
        # [num_env, history_len=10, num_obs]
        return self.proprio_obs_history_buf
    
    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image
    
    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        # 1.腐蚀
        depth_image = self.crop_depth_image(depth_image)
        # 2.加噪
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        # 3.clip depth to [-2, 0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        # 4. resize image from (106, 60) to (87, 58)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        # 5. 将depth归一化到[-0.5, 0.5]
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def update_depth_buffer(self):

        if (not self.cfg.depth.use_camera) or (self.global_counter % self.cfg.depth.update_interval != 0):
            return
        
        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            
            depth_image = gymtorch.wrap_tensor(depth_image_)
            # add randomization to depth image
            depth_image = self.process_depth_image(depth_image, i)

            init_flag = self.episode_length_buf <= 1    
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                # 把最新的depth image加到depth_buffer尾
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)

        self.gym.end_access_image_tensors(self.sim)


    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[self.robot_actor_idxs, 3:7]
        self.base_pos_world[:] = self.root_states[self.robot_actor_idxs, 0:3]
        self.base_lin_vel_world[:] = self.root_states[self.robot_actor_idxs, 7:10]
        self.base_ang_vel_world[:] = self.root_states[self.robot_actor_idxs, 10:13]

        # transformation robot status to base frame
        self.base_lin_vel_base[:] = quat_rotate_inverse(self.base_quat, self.base_lin_vel_world)
        self.base_ang_vel_base[:] = quat_rotate_inverse(self.base_quat, self.base_ang_vel_world)
        self.projected_gravity_base[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_forward[:] = quat_apply(self.base_quat, self.forward_vec)
        
        self.base_lin_acc_world = (self.base_lin_vel_world - self.last_root_vel[:, :3]) / self.dt

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        # compute contact forces and status
        contact = torch.norm(self.robot_contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) # [num_env, 4]
        self.last_contacts = contact
        
        # 记录contact
        self.episode_length_buf_s = self.episode_length_buf*self.dt
        # print("episode_length_buf_s: ", self.episode_length_buf_s)
        combined_tensor = torch.cat((self.episode_length_buf_s.unsqueeze(-1), contact.float()), dim=-1)
        # print("combined_tensor: ", combined_tensor)
        
        # combined_list = combined_tensor.tolist()
        # csv_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "tests", "data", "contact_218.csv")
        # with open(csv_file_path, 'a+', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerows(combined_list)
        
        

        # 遥控控制,override env_goal
        if self.cfg.env.remote:
            delta_x, delta_y, episode_s, e_stop = self.gamepad_callback()
            # print("episode_s: ", episode_s)

            # delta_goal = torch.tensor([delta_x, delta_y, 0], device=self.device)
            # delta_goal_base = quat_rotate_inverse(self.base_quat, delta_goal.unsqueeze(0))
            # self.env_goal[self.lookat_id][0] = self.base_pos_world[self.robot_actor_idxs[self.lookat_id]][0] + delta_goal_base[self.lookat_id][0]
            # self.env_goal[self.lookat_id][1] = self.base_pos_world[self.robot_actor_idxs[self.lookat_id]][1] + delta_goal_base[self.lookat_id][1]
            
            self.env_goal[self.lookat_id][0] = self.base_pos_world[self.robot_actor_idxs[self.lookat_id]][0] + torch.tensor(delta_x, device=self.device)
            self.env_goal[self.lookat_id][1] = self.base_pos_world[self.robot_actor_idxs[self.lookat_id]][1] + torch.tensor(delta_y, device=self.device)
            self.episode_length_buf[self.lookat_id] = np.ceil(episode_s / self.dt) 
            
            
            if e_stop:  #! 这样退出gym的UI还怪方便的
                sys.exit(0)

        # 计算target相对机器人的位置
        self.robot2target_world[:] = self.env_goal[:] - self.base_pos_world[:]
        self.targetpos_base[:] = quat_rotate_inverse(self.base_quat, self.robot2target_world)
        
        
        # 更新每个foot腾空相开始时的episode
        # refresh_flag = (self.contact_filt == True) 
        # self.foot_swing_timer[~refresh_flag] += 1
        # self.foot_swing_timer[refresh_flag] = 0
        
        #  2.get heightpoints 3.add randomization
        self._post_physics_step_callback() 

        self.check_termination()
            
        self.compute_reward()
        
        # reset terminated envs
        if not self.cfg.env.remote:
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            self.reset_idx(env_ids)

        # add new depth image to depth_buffer
        self.update_depth_buffer()

        self.compute_observations()

        # record last vals
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[self.robot_actor_idxs, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            
            if not self.cfg.depth.use_camera:
                # self._draw_height_samples()
                # self._draw_feet()
                # self._draw_coord()
                self._draw_goals()

            
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)
                
        # record vedio
        self.record_video()
                                                     

    # 不太理解为什么这样reindex
    def reindex_feet(self, vec):
        return vec[:, [1, 0, 3, 2]]

    def reindex(self, vec):
        return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]

    def check_termination(self):
        """ Check if environments need to be reset.termination conditions: roll, pitch, height, timeout
        """
        self.reset_buf = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        
        self.contact_termination_buf = torch.any(torch.norm(self.robot_contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = (self.episode_length_buf > self.max_episode_length) 

        self.roll_cutoff = torch.abs(self.roll) > 1.5
        self.pitch_cutoff = torch.abs(self.pitch) > 1.5
        self.height_cutoff = self.root_states[self.robot_actor_idxs, 2] < -0.1   # 掉坑里去了也要terminatin
        # self.height_cutoff = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)

        #! play的时候可以把contact_termination去掉
        self.reset_buf |= self.contact_termination_buf
        self.reset_buf |= self.roll_cutoff
        self.reset_buf |= self.pitch_cutoff
        self.reset_buf |= self.height_cutoff
        
        self.reset_buf |= self.time_out_buf

        
        
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # update terrain curriculum
        if self.cfg.terrain.curriculum: 
            # 更新地形的难度
            self._update_terrain_curriculum(env_ids)      
            
        # 每次在一整个episode结束后再更新command curriculum
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):    
            self.update_command_curriculum(env_ids) # not implemented


        
        
        #todo 
        if self.cfg.terrain.pyramid:
            # self.env_goal[env_ids] = 2*(torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device, requires_grad=False)-0.5)*torch.tensor([4.8, 4.8, 0], device=self.device) + self.env_origins[env_ids]
            rand_radius = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False) * 2 + 3
            rand_theta = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False) * 2 * math.pi
            rand_envgoal_x = rand_radius * torch.cos(rand_theta)
            rand_envgoal_y = rand_radius * torch.sin(rand_theta)
            self.env_goal[env_ids] = torch.stack((rand_envgoal_x, rand_envgoal_y, torch.zeros(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False)), dim=1) + self.env_origins[env_ids]        
        else:
            flat_env_flag = self.env_class[env_ids] == 6
            num_flatenv_reset = flat_env_flag.sum()
            if num_flatenv_reset != 0:
                self.env_goal[env_ids[flat_env_flag]] = 2*(torch.rand(num_flatenv_reset, 3, dtype=torch.float, device=self.device, requires_grad=False)-0.5)*torch.tensor([3.5, 1.5, 0], device=self.device) \
                                                            + self.env_origins[env_ids[flat_env_flag]]
                                                            
            self.env_goal[env_ids[~flat_env_flag]] = self.terrain_goals[self.terrain_levels, self.terrain_types][env_ids[~flat_env_flag]]
        
        #! pyramid test

        # reset robot states and cmd
        # self._resample_commands(env_ids)

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        
        if 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames = self.video_frames[:]   # 没有start_recording时
            self.video_frames = []
        
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        
        self.reset_buf[env_ids] = 1
        
        self.proprio_obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        # self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.

        # fill extras
        self.extras["episode"] = {}
        self.extras["step"] = {}
        for key in self.episode_sums.keys():
            # get environment average of every reward term in a given episode
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.extras["step"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.episode_length_buf[env_ids]
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        # log additional curriculum info
        if self.cfg.terrain.curriculum: # TRUE,同样也添加到了log中
            # get average terrain level of every environment in a given episode
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            
        if self.cfg.commands.curriculum:    # false
            # self.extras["episode"]["object2target_radius"] = self.command_ranges["object2target_radius"][1]
            pass
            
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:  # TRUE. 对timeout而不是termination做特殊处理
            self.extras["time_outs"] = self.time_out_buf
            
        # todo add more info log here, eg: success rate
        # 环境reset时机器人距离object的距离

        
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        raise NotImplementedError
            
        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.    # 每一次step每个env所有的reward_terms的和
        
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            
            self.rew_buf += rew # step所有reward term的和
            
            self.episode_sums[name] += rew  # 每一项reward
            
        if self.cfg.rewards.only_positive_rewards:  # true
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales: # FALSE
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            
    
    def compute_observations(self):
        """ 
        Computes observations
        """
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)   # [num_env,2]
        
        proprio_obs_buf = torch.cat(( #skill_vector, 
                            self.base_ang_vel_base  * self.obs_scales.ang_vel,   # 3
                            imu_obs,    # 2
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),  # 12
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),   # 12
                            self.reindex(self.action_history_buf[:, -1]),   # 12
                            self.reindex_feet(self.contact_filt.float()-0.5),   # 4
                            # self.targetpos_base[:,:2]
                            self.targetpos_base[:,:],   # 3
                            
                            ((self.max_episode_length_s - self.episode_length_buf * self.dt) / self.max_episode_length_s).unsqueeze(-1),
                            
                            ),dim=-1)
                
        # test =((self.max_episode_length_s - self.episode_length_buf * self.dt) / self.max_episode_length_s).unsqueeze(-1)
        # test_ =((self.max_episode_length_s - self.episode_length_buf * self.dt) / self.max_episode_length_s).unsqueeze(0)
        # print("targetpos_base: ", self.targetpos_base)
                
        # todo 加上object的params
        # 通过过去的状态估计机器人的速度
        priv_robotstate = torch.cat((self.base_lin_vel_base * self.obs_scales.lin_vel, # [num_envs.3]
                                   self.base_pos_world[:,2].unsqueeze(-1),
                                   ), dim=-1)
        
        # [num_env, 29]
        priv_envstate = torch.cat((
            self.mass_params_tensor,    # 4(包含了mass和com pos)
            self.friction_coeffs_tensor,    # 1
            self.motor_strength - 1, # 12
            self.motor_offsets, # 12
            self.Kp_factors - 1,    # 12
            self.Kd_factors - 1,    # 12
            self.gravities - 9.81,  # 3
        ), dim=-1)
        
        
        if self.cfg.terrain.measure_heights:    # 为什么运行play的时候，也会进这里
            # [num_env, 132]                # [num_env, 1]                                     -             [num_env, 132]
            terrain_heights_z_buffer = torch.clip(self.root_states[self.robot_actor_idxs, 2].unsqueeze(1) - 0.3 - self.measured_terrain_heights_z, -1, 1.)
            
            obstacle_heights_z_buffer = torch.clip(self.measured_obstacle_heights_z - self.root_states[self.robot_actor_idxs, 2].unsqueeze(1), -1, 1)
            
            self.obs_buf = torch.cat([proprio_obs_buf, terrain_heights_z_buffer, obstacle_heights_z_buffer, priv_robotstate, priv_envstate, self.proprio_obs_history_buf.view(self.num_envs, -1)], dim=-1)
            
        else:
            self.obs_buf = torch.cat([proprio_obs_buf, priv_robotstate, priv_envstate, self.proprio_obs_history_buf.view(self.num_envs, -1)], dim=-1)

        
        self.proprio_obs_history_buf = torch.where( # [num_env, history_len, num_obs]
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([proprio_obs_buf] * self.cfg.env.history_len, dim=1),   # 10
            #! 把最新的obs_buf加到proprio_obs_history_buff尾
            torch.cat([self.proprio_obs_history_buf[:, 1:], proprio_obs_buf.unsqueeze(1)], dim=1)   
        )
        
        
    def get_noisy_measurement(self, x, scale):
        if self.cfg.noise.add_noise:
            x = x + (2.0 * torch.rand_like(x) - 1) * scale * self.cfg.noise.noise_level
        return x


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.graphics_device_id = self.sim_device_id  # required in headless mode
            
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type  # trimesh
        
        start = time()
        print("*"*80)
        print("Start creating ground...")
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrainvls(self.cfg.terrain, self.num_envs, self.gym, self.sim) #! create different terrains
            
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._add_trimesh2sim()  #! add tremish to simulation
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")        
        
        
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*"*80)
        
        self._create_envs()


    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        
    def lookat(self, i):
        look_at_id = self.robot_actor_idxs[i]
        look_at_pos = self.root_states[look_at_id, :3].clone()
        cam_pos = look_at_pos + self.lookat_vec
        self.set_camera(cam_pos, look_at_pos)
        
        
    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
                
            if not self.free_cam:
                self.lookat(self.lookat_id)
                
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:  # "Q": quit
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:  # "V": toggle viewer sync
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "reset" and evt.value >0:
                    # self.reset_idx([self.lookat_id])
                    self.reset_idx(torch.tensor([self.lookat_id], device=self.device, dtype=torch.long, requires_grad=False))
                    
                
                # 只有不在fix_cam的状态下,这些键盘指令才有用
                if not self.free_cam:
                    for i in range(9):
                        if evt.action == "lookat" + str(i) and evt.value > 0:
                            self.lookat(i)
                            self.lookat_id = i
                    if evt.action == "prev_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id-1) % self.num_envs
                        self.lookat(self.lookat_id)
                    if evt.action == "next_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id+1) % self.num_envs
                        self.lookat(self.lookat_id)
                        
                    if evt.action == "vx_plus" and evt.value > 0:
                        self.commands[self.lookat_id, 0] += 0.2
                    if evt.action == "vx_minus" and evt.value > 0:
                        self.commands[self.lookat_id, 0] -= 0.2
                    if evt.action == "left_turn" and evt.value > 0:
                        self.commands[self.lookat_id, 3] += 0.5
                    if evt.action == "right_turn" and evt.value > 0:
                        self.commands[self.lookat_id, 3] -= 0.5
                
                # "F": swtich between free cam and fix cam
                if evt.action == "free_cam" and evt.value > 0:
                    self.free_cam = not self.free_cam
                    if self.free_cam:
                        self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
                
                if evt.action == "pause" and evt.value > 0:
                    self.pause = True
                    while self.pause:
                        time.sleep(0.1)
                        self.gym.draw_viewer(self.viewer, self.sim, True)
                        for evt in self.gym.query_viewer_action_events(self.viewer):
                            if evt.action == "pause" and evt.value > 0:
                                self.pause = False
                        if self.gym.query_viewer_has_closed(self.viewer):
                            sys.exit()

                    
            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            self.gym.poll_viewer_events(self.viewer)
            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
            
            if not self.free_cam:
                p = self.gym.get_viewer_camera_transform(self.viewer, None).p
                cam_trans = torch.tensor([p.x, p.y, p.z], requires_grad=False, device=self.device)
                look_at_pos = self.root_states[self.robot_actor_idxs[self.lookat_id], :3].clone()
                self.lookat_vec = cam_trans - look_at_pos
            

    #------------- Callbacks --------------
    def _process_robot_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction: # TRUE
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props
    

    def _process_dof_props(self, props, env_id):
        
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit   # 1
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props


    def _process_rigid_body_props(self, props, env_id):
        # No need to use tensors as only called upon env creation
        if self.cfg.domain_rand.randomize_base_mass:    # TRUE
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1, ))
            
        if self.cfg.domain_rand.randomize_base_com: # TRUE
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
            
        mass_params = np.concatenate([rand_mass, rand_com])
        return props, mass_params
    
    
    def _post_physics_step_callback(self):
        
        # self._teleport_robots(torch.arange(self.num_envs, device=self.device))
        
        # 每6s resample一次command
        # env_ids_resampleCMD = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0)
        # self._resample_commands(env_ids_resampleCMD.nonzero(as_tuple=False).flatten())
        
        # measure terrain heights at set intervals
        if self.cfg.terrain.measure_heights and (self.global_counter % self.cfg.depth.update_interval) == 0:
            self.measured_terrain_heights_z, self.measured_obstacle_heights_z = self._get_scan() # [num_envs，num_scandots]
               
               
        # add randomization

        self._push_robots() # add random vel
        
        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.dof_rand_interval) == 0).nonzero(as_tuple=False).flatten()
        self._randomize_dof_props(env_ids, self.cfg)

        # if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
        #     self._randomize_gravity()
        # if int(self.common_step_counter - self.cfg.domain_rand.gravity_rand_duration) % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
        #     self._randomize_gravity(torch.tensor([0, 0, 0]))  
        


    def _randomize_dof_props(self, env_ids, cfg):
        if self.cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = self.cfg.domain_rand.motor_strength_range
            self.motor_strength[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1) \
                * (max_strength - min_strength) + min_strength
                
        if self.cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) \
                * (max_offset - min_offset) + min_offset
                
        if self.cfg.domain_rand.randomize_Kp_factor:
            min_Kp_factor, max_Kp_factor = self.cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1) \
                * (max_Kp_factor - min_Kp_factor) + min_Kp_factor
                
        if self.cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = self.cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1) \
                * (max_Kd_factor - min_Kd_factor) + min_Kd_factor


    def _randomize_gravity(self, external_force = None):

        if external_force is not None:
            self.gravities[:, :] = external_force.unsqueeze(0)
            
        elif self.cfg.domain_rand.randomize_gravity:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device, requires_grad=False) * (max_gravity - min_gravity) + min_gravity
            self.gravities[:, :] = external_force.unsqueeze(0)

        sim_params = self.gym.get_sim_params(self.sim)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)

    def _resample_commands(self, env_ids):

        raise NotImplementedError
        

    def _compute_torques(self, actions):

        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (actions_scaled + self.default_dof_pos_all - self.dof_pos + self.motor_offsets) \
                     - self.d_gains * self.Kd_factors * self.dof_vel
            
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        torques = torques * self.motor_strength
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)


    def _reset_dofs(self, env_ids):

        # dof pos randomization
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(0., 0.9, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.
        
        env_ids_int32 = self.robot_actor_idxs[env_ids].to(device=self.device).to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), #  Buffer containing actor indices
                                              len(env_ids_int32))
        
        
    def _reset_root_states(self, env_ids):
    
        robot_env_ids = self.robot_actor_idxs[env_ids].to(device=self.device)
        
        # base position
        if self.custom_origins: # true
            self.root_states[robot_env_ids] = self.base_init_state
            
            self.root_states[robot_env_ids, :3] += self.env_origins[env_ids]
            
            self.root_states[robot_env_ids, 0:1] += torch_rand_float(-self.cfg.init_state.x_init_range, self.cfg.init_state.x_init_range, (len(robot_env_ids), 1), device=self.device)
            self.root_states[robot_env_ids, 1:2] += torch_rand_float(-self.cfg.init_state.y_init_range, self.cfg.init_state.y_init_range, (len(robot_env_ids), 1), device=self.device)
                            
        else:
            self.root_states[robot_env_ids] = self.base_init_state
            self.root_states[robot_env_ids, :3] += self.env_origins[env_ids]
         
        if self.cfg.terrain.pyramid:
            #! pyramid test
            random_yaw_angle = 2*(torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device, requires_grad=False)-0.5)*torch.tensor([0, 0, 3.14], device=self.device)
        else:
            random_yaw_angle = 2*(torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device, requires_grad=False)-0.5)*torch.tensor([0, 0, self.cfg.init_state.yaw_init_range], device=self.device)
            
            flat_env_flag = self.env_class[robot_env_ids] == 6
            num_flatenv_reset = flat_env_flag.sum()
            if num_flatenv_reset != 0:
                random_yaw_angle[flat_env_flag] = 2*(torch.rand(num_flatenv_reset, 3, dtype=torch.float, device=self.device, requires_grad=False)-0.5)*torch.tensor([0, 0, 3.14], device=self.device)

                            
        self.root_states[robot_env_ids,3:7] = quat_from_euler_xyz(random_yaw_angle[:,0], random_yaw_angle[:,1], random_yaw_angle[:,2])
        self.root_states[robot_env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(robot_env_ids), 6), device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
       
        env_ids_int32 = (robot_env_ids).to(dtype=torch.int32)        
        
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), 
                                                     len(env_ids_int32))

    def _push_robots(self):

        if self.cfg.domain_rand.push_robots:
            env_ids = torch.arange(self.num_envs, device=self.device)
            env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]

            max_vel = self.cfg.domain_rand.max_push_vel_xy
            self.root_states[self.robot_actor_idxs[env_ids], 7:9] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 2), device=self.device) # lin vel x/y
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))


    def _update_terrain_curriculum(self, env_ids):

        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
                
        #! 计算规则:
        # move_up = 0
        # move_down = 0
        move_up = torch.norm(self.robot2target_world[env_ids], dim=-1) < 0.5
        move_down = torch.norm(self.robot2target_world[env_ids], dim=-1) > 2.0
        
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        # reset env_origins and env_class
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

        
    def _teleport_robots(self, env_ids):
        """ Teleports any robots that are too close to the edge to the other side
        """
        robot_env_ids = self.robot_actor_idxs[env_ids].to(device=self.device)
        if self.cfg.terrain.teleport_robots:
            thresh = self.cfg.terrain.teleport_thresh

            # 不用offset会怎样
            # x_offset = int(cfg.terrain.x_offset * cfg.terrain.horizontal_scale)

            low_x_ids = robot_env_ids[self.root_states[robot_env_ids, 0] < thresh]
            self.root_states[low_x_ids, 0] += self.cfg.terrain.terrain_length * (self.cfg.terrain.num_rows - 1)

            high_x_ids = robot_env_ids[self.root_states[robot_env_ids, 0] > self.cfg.terrain.terrain_length * self.cfg.terrain.num_rows - thresh]
            self.root_states[high_x_ids, 0] -= self.cfg.terrain.terrain_length * (self.cfg.terrain.num_rows - 1)

            low_y_ids = robot_env_ids[self.root_states[robot_env_ids, 1] < thresh]
            self.root_states[low_y_ids, 1] += self.cfg.terrain.terrain_width * (self.cfg.terrain.num_cols - 1)

            high_y_ids = robot_env_ids[
                self.root_states[robot_env_ids, 1] > self.cfg.terrain.terrain_width * self.cfg.terrain.num_cols - thresh]
            self.root_states[high_y_ids, 1] -= self.cfg.terrain.terrain_width * (self.cfg.terrain.num_cols - 1)

            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.refresh_actor_root_state_tensor(self.sim)

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)   #! [12,13]->[24,13]
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # [24,2]
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)    #! [204,3]->[216,3]
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)    # [48,6]
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)    #! [204,13]->[216,13]
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
            
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)   # [num_env*2,13]
        self.base_quat = self.root_states[self.robot_actor_idxs, 3:7]   # [num_env,4]
        self.base_pos_world = self.root_states[self.robot_actor_idxs, :3] # [num_env,3]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor)  # 20
        self.robot_link_states = self.rigid_body_states.view(self.num_envs, -1, 13)[:,:17]    # [num_env,17,13]

        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) # [num_env*num_dof,2]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6) # for feet only, see create_env()
        
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)
        self.robot_contact_forces = self.contact_forces.view(self.num_envs, -1, 3)[:,:17]   # [num_env,17,3]
        

        self.motor_strength = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)      


        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[self.robot_actor_idxs, 7:13])

        
        if self.cfg.env.history_encoding:   # TRUE
            self.proprio_obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dofs, device=self.device, dtype=torch.float)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        # sample commands
        # self._resample_commands(torch.arange(self.num_envs, device=self.device, requires_grad=False))
        # self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        
        self.base_lin_vel_world = self.root_states[self.robot_actor_idxs, 7:10]
        self.base_ang_vel_world = self.root_states[self.robot_actor_idxs, 10:13]
        self.base_lin_vel_base = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 7:10])
        self.base_ang_vel_base = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 10:13])
        
        # 将gravity转换到base frame
        self.projected_gravity_base = quat_rotate_inverse(self.base_quat, self.gravity_vec)  # gravity projected on the base frame
        # 
        self.projected_forward = quat_rotate_inverse(self.base_quat, self.forward_vec)
        
        self.robot2target_world = self.env_goal - self.base_pos_world
        self.targetpos_base = quat_rotate_inverse(self.base_quat, self.robot2target_world)
                
        # self.rew_container_base_approach_object = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        self.rew_container_base_forward_vel = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        
        self.rew_container_task = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        self.rew_container_exploration = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        self.rew_container_stalling = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        self.rew_container_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        self.rew_container_crawl = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        self.rew_container_still = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        self.rew_container_norotation = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)
            
        
        self.foot_swing_timer = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.long, requires_grad=False)
        
        if self.cfg.terrain.measure_heights:
            self.terrain_scanpoints_xybase, self.ceiling_scanpoints_xybase = self._init_3dscan_points() # 初始化的scandots
                        
        self.measured_terrain_heights_z = 0
        self.measured_obstacle_heights_z = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # 每个env的default_dof_pos
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        print(self.default_dof_pos)

        self.default_dof_pos_all[:] = self.default_dof_pos[0]

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,  
                                            self.cfg.depth.buffer_len,  # LEN = 2
                                            self.cfg.depth.resized[1], 
                                            self.cfg.depth.resized[0]).to(self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
                
        # prepare list of  reward functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums. 每个env中每项reward的episode sum
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border
        hf_params.transform.p.y = -self.terrain.border
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.flatten(order='C'), hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _add_trimesh2sim(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        print("Adding trimesh to simulation...")
        # 地面tremish
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)          
        
        #! 天花板tremish. 这个只有在discrete地形中才生效
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices_ceiling.shape[0]
        tm_params.nb_triangles = self.terrain.triangles_ceiling.shape[0]
        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        # self.gym.add_triangle_mesh(self.sim, self.terrain.vertices_ceiling.flatten(order='C'), self.terrain.triangles_ceiling.flatten(order='C'), tm_params)  
        
        print("Trimesh added")
        
        self.terrain_height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.ceiling_height_samples = torch.tensor(self.terrain.ceilingsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

        
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.y_edge_mask = torch.tensor(self.terrain.y_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]
            camera_props.height = self.cfg.depth.original[1]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov 
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)
            
            local_transform = gymapi.Transform()
            
            camera_position = np.copy(config.position)
            camera_angle = np.random.uniform(config.angle[0], config.angle[1])
            
            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
            
            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        robot_asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        robot_asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        robot_asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        robot_asset_options.fix_base_link = self.cfg.asset.fix_base_link
        robot_asset_options.density = self.cfg.asset.density
        robot_asset_options.angular_damping = self.cfg.asset.angular_damping
        robot_asset_options.linear_damping = self.cfg.asset.linear_damping
        robot_asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        robot_asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        robot_asset_options.armature = self.cfg.asset.armature
        robot_asset_options.thickness = self.cfg.asset.thickness
        robot_asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, robot_asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        robot_dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        robot_rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        # create force sensors on rigid body#! create force sensors on foot
        for s in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)  
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:    # thigh calf base
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on: # base
            termination_contact_names.extend([s for s in body_names if name in s])  

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
    
        # set env distribution for the first time
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.robot_actor_idxs = []
        
        print("Creating env...")
        for i in tqdm(range(self.num_envs)):    # tqdm 在长循环中加入进度条
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            
            pos = self.env_origins[i].clone()
            pos[0:1] += torch_rand_float(-self.cfg.init_state.x_init_range, self.cfg.init_state.x_init_range, (1, 1), device=self.device).squeeze(1)
            pos[1:2] += torch_rand_float(-self.cfg.init_state.y_init_range, self.cfg.init_state.y_init_range, (1, 1), device=self.device).squeeze(1)
            
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            robot_rigid_shape_props = self._process_robot_rigid_shape_props(robot_rigid_shape_props_asset, i) # friction rand
            self.gym.set_asset_rigid_shape_properties(robot_asset, robot_rigid_shape_props)
            
            robot_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "robot", i, self.cfg.asset.self_collisions, 0)
            
            robot_dof_props = self._process_dof_props(robot_dof_props_asset, i) # read dof pos/vel/torque limits
            self.gym.set_actor_dof_properties(env_handle, robot_handle, robot_dof_props)
            
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i) # mass, com pos rand
            self.gym.set_actor_rigid_body_properties(env_handle, robot_handle, body_props, recomputeInertia=True)


            # attach camera to robot
            self.attach_camera(i, env_handle, robot_handle) 
               
            # privileged info(mass params)
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)   

            
            # append handles
            self.envs.append(env_handle)
            self.actor_handles.append(robot_handle)
            
            # collect robot actor id and object actor id
            self.robot_actor_idxs.append(self.gym.get_actor_index(env_handle, robot_handle, gymapi.DOMAIN_SIM))
            
            
        self.robot_actor_idxs = torch.Tensor(self.robot_actor_idxs).to(device=self.device,dtype=torch.long)

            
        if self.cfg.domain_rand.randomize_friction: # TRUE
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
            
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)    # get hip index
            
        thigh_names = ["FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint"]
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(thigh_names):
            self.thigh_indices[i] = self.dof_names.index(name)  # get thigh index
            
        calf_names = ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(calf_names):
            self.calf_indices[i] = self.dof_names.index(name)   # get calf index
        
        # record vedio preparationrendering_camera
        from legged_gym.sensors.floating_camera_sensor import FloatingCameraSensor
        self.recording_camera = FloatingCameraSensor(self)
        self.video_frames = []
        self.complete_video_frames = []
        
    def record_video(self): # 对env[0]进行record
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            bx, by, bz = self.root_states[self.robot_actor_idxs[0], 0], self.root_states[self.robot_actor_idxs[0], 1], self.root_states[self.robot_actor_idxs[0], 2]
            target_loc = [bx, by , bz]
            cam_distance = [0, -1.0, 1.0]
            self.recording_camera.set_position(target_loc, cam_distance)
            self.video_frame = self.recording_camera.get_observation()
            self.video_frames.append(self.video_frame)
            
    def start_recording(self):
        self.complete_video_frames = None
        self.record_now = True
        # self.debug_viz = True
        self.lookat_id = 0
        
    def finish_recording(self):
        self.complete_video_frames = []
        self.vedio_frames = []
        self.record_now = False
        # self.debug_viz = False
        
    def get_complete_video_frames(self):
        if self.complete_video_frames is None:
            return []
        return self.complete_video_frames
        
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            
            #! 如果使用terrain curriculum,就把机器人放在行数不超过5的地方。不使用terrain curriculum,就在所有行中随机分配
            max_init_level = self.cfg.terrain.max_init_terrain_level    # 5
            if not self.cfg.terrain.curriculum: 
                max_init_level = self.cfg.terrain.num_rows - 1
            
            #                                  min      max               shape
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)  # 行随机分布
            # self.terrain_levels = torch.arange(self.num_envs, device=self.device) % (max_init_level) 
            # print("terrain_levels: ", self.terrain_levels)
            
            # types则是按照顺序分配到每一列
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types] #! 分配env_origins

            self.terrain_goals = torch.from_numpy(self.terrain.goal).to(self.device).to(torch.float)  # [cfg.num_rows, cfg.num_cols, 3]
            self.env_goal = self.terrain_goals[self.terrain_levels, self.terrain_types] 
            
            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types] #! 不同env对应的terrain type
            

        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt  #! 4 * 0.005 = 0.2
        self.obs_scales = self.cfg.normalization.obs_scales
        
        self.reward_scales = class_to_dict(self.cfg.rewards.scales) # convert reward scale class to dict 
        reward_norm_factor = 1#np.sum(list(self.reward_scales.values()))
        
        for rew in self.reward_scales:  # 所有的reward_scales除以norm_factor.不过目前norm_factor都是一
            self.reward_scales[rew] = self.reward_scales[rew] / reward_norm_factor
        
        if self.cfg.commands.curriculum:    # FALSE
            self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        else:
            self.command_ranges = class_to_dict(self.cfg.commands.default_ranges)
            
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        
        self.task_episode_length_s = self.cfg.env.task_episode_length_s
        self.task_episode_length = np.ceil(self.task_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.dof_rand_interval = np.ceil(self.cfg.domain_rand.dof_rand_interval_s / self.dt)
        self.cfg.domain_rand.gravity_rand_interval = np.ceil(self.cfg.domain_rand.gravity_rand_interval_s / self.dt)
        self.cfg.domain_rand.gravity_rand_duration = np.ceil(self.cfg.domain_rand.gravity_rand_interval * self.cfg.domain_rand.gravity_impulse_duration)

    def _draw_height_samples(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        i = self.lookat_id

        base_pos = (self.root_states[self.robot_actor_idxs[i], :3]).cpu().numpy()
        
        #! terrain heightpoints
        terrain_sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        terrain_sphere_geom = gymutil.WireframeSphereGeometry(0.015, 8, 8, None, color=(1, 1, 0))
        terrain_heights_z = self.measured_terrain_heights_z[i].cpu().numpy()    
        terrain_height_points_xyyaw = quat_apply_yaw(self.base_quat[i].repeat(terrain_heights_z.shape[0]), self.terrain_scanpoints_xybase[i]).cpu().numpy() # [132,3]
        for j in range(terrain_heights_z.shape[0]):
            x = terrain_height_points_xyyaw[j, 0] + base_pos[0]
            y = terrain_height_points_xyyaw[j, 1] + base_pos[1]
            z = terrain_heights_z[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(terrain_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        
        #! apex heightpoints
        # obstacle_sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 1))
        obstacle_sphere_geom = gymutil.WireframeSphereGeometry(0.02, 10, 10, None, color=(0, 0, 1))
        obstacle_heights_z = self.measured_obstacle_heights_z[i].cpu().numpy()
        obstacle_height_points_xyyaw = quat_apply_yaw(self.base_quat[i].repeat(obstacle_heights_z.shape[0]), self.ceiling_scanpoints_xybase[i]).cpu().numpy() # [132,3]
        for j in range(obstacle_heights_z.shape[0]):
            x = obstacle_height_points_xyyaw[j, 0] + base_pos[0]
            y = obstacle_height_points_xyyaw[j, 1] + base_pos[1]
            z = obstacle_heights_z[j]
            
            if z == 0:
                z = 0.5
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(obstacle_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

            
            # if z != 0:
            #     gymutil.draw_lines(obstacle_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
            
            
        #! 地形的边缘点，这个加上太卡了
        # x_edge_geom = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(0, 1, 0))
        # for i in range(self.x_edge_mask.shape[0]):
        #     for j in range(self.x_edge_mask.shape[1]):
        #         if self.x_edge_mask[i,j]:
        #             pose = gymapi.Transform(gymapi.Vec3(i * self.cfg.terrain.horizontal_scale, j * self.cfg.terrain.horizontal_scale, self.terrain.heightsamples[i,j] * self.cfg.terrain.vertical_scale), r=None)
        #             gymutil.draw_lines(x_edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
        # y_edge_geom = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 0, 0))
        # for i in range(self.y_edge_mask.shape[0]):
        #     for j in range(self.y_edge_mask.shape[1]):
        #         if self.y_edge_mask[i,j]:
        #             pose = gymapi.Transform(gymapi.Vec3(i * self.cfg.terrain.horizontal_scale, j * self.cfg.terrain.horizontal_scale, self.terrain.heightsamples[i,j] * self.cfg.terrain.vertical_scale), r=None)
        #             gymutil.draw_lines(y_edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            
                
    
    def _draw_goals(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1, 0, 0))

        # env goal
        goal = self.env_goal[self.lookat_id].cpu().numpy()        
        goal_xy = goal[:2] + self.terrain.cfg.border_size
        pts = (goal_xy/self.terrain.cfg.horizontal_scale).astype(int)
        goal_z = self.terrain_height_samples[pts[0], pts[1]].cpu().item() * self.terrain.cfg.vertical_scale
        pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

        
        #! robot2target_world
        # sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 0, 1))
    
        # robot2target_vec = self.env_goal[self.lookat_id, :3] - self.root_states[self.robot_actor_idxs[self.lookat_id], :3]
        # norm = torch.norm(robot2target_vec, dim=-1, keepdim=True)
        # target_vec_norm = robot2target_vec / (norm + 1e-5)
        
        # for i in range(10):            
        #     pose_arrow = self.root_states[self.robot_actor_idxs[self.lookat_id], :3] + (norm/10)*(i+1) * target_vec_norm[:3]
        #     pose_arrow=pose_arrow.cpu().numpy()
        #     pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_arrow[2]), r=None)
        #     gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            
            
        #! robotvel_world
        # sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 1))
        
        # robotvel_vec = self.base_lin_vel_world[self.lookat_id, :3]  # torch.Size([3])
        # norm = torch.norm(robotvel_vec, dim=-1, keepdim=True)
        # target_vec_norm = robotvel_vec / (norm + 1e-5)
        # for i in range(10):

        #     # pose_arrow = self.root_states[self.robot_actor_idxs[self.lookat_id], :2] + 0.4*(i+1) * target_vec_norm[:2]
        #     pose_arrow = self.root_states[self.robot_actor_idxs[self.lookat_id], :3] + norm/20*(i+1) * target_vec_norm[:3]
        #     pose_arrow=pose_arrow.cpu().numpy()
        #     pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_arrow[2]), r=None)
        #     gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
        #! targetvel
        # target_velocity = self.robot2target_world[:,:2] / (torch.norm(self.robot2target_world[:,:2], dim=-1, keepdim=True) + 1e-5) * 0.5
        # sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0.3, 0.6, 0.3))
        
        # targetvel_vec = torch.cat([target_velocity[self.lookat_id, ], torch.zeros(1).to(self.device)], dim=0)
        
        # norm = torch.norm(targetvel_vec, dim=-1, keepdim=True)
        # target_vec_norm = targetvel_vec / (norm + 1e-5)
        # for i in range(10):

        #     pose_arrow = self.root_states[self.robot_actor_idxs[self.lookat_id], :3] + norm/20*(i+1) * target_vec_norm[:3]
        #     pose_arrow=pose_arrow.cpu().numpy()
        #     pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_arrow[2]), r=None)
        #     gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
    def _draw_feet(self):
        #! draw feet at edge
    #     non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))    # green
    #     edge_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 0, 1))    # 紫红色

        feet_pos = self.robot_link_states[:, self.feet_indices, :3] # [num_env, 4, 3]
        
        ff_geom = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 0, 0))
        feet_force = self.robot_contact_forces[:, self.feet_indices, :3] # [num_env, 4, 3]
        
        # for i in range(4):
    #         pose = gymapi.Transform(gymapi.Vec3(feet_pos[self.lookat_id, i, 0], feet_pos[self.lookat_id, i, 1], feet_pos[self.lookat_id, i, 2]), r=None)
            
    #         # draw if foot at edge
    #         if self.feet_at_edge[self.lookat_id, i]:
    #             # gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[i], pose)
    #             gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    #         else:
    #             # gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, self.envs[i], pose)
    #             gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
            # draw foot force
            # ff_norm = (torch.norm(feet_force[self.lookat_id, i, :3]) + 1e-5)
            # ff_vec = feet_force[self.lookat_id, i, :3] / ff_norm
            # for j in range(15):
            #     pose_arrow = ff_norm/10000 * (j+1) * ff_vec + feet_pos[self.lookat_id, i, :3]
            #     pose_arrow = pose_arrow.cpu().numpy()
            #     pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_arrow[2]), r=None)
            #     gymutil.draw_lines(ff_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
                    
        #! draw feet contact
        # non_contact_geom = gymutil.WireframeSphereGeometry(0.025, 16, 16, None, color=(0, 1, 0))
        # contact_geom = gymutil.WireframeSphereGeometry(0.025, 16, 16, None, color=(1, 0, 0)) 
        non_contact_geom = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(0, 1, 0))
        contact_geom = gymutil.WireframeSphereGeometry(0.03, 16, 16, None, color=(1, 0, 0))         
        # feet_pos = self.robot_link_states[:, self.feet_indices, :3] # [num_env, 4, 3]
                    
        for i in range(4):
            pose = gymapi.Transform(gymapi.Vec3(feet_pos[self.lookat_id, i, 0], feet_pos[self.lookat_id, i, 1], feet_pos[self.lookat_id, i, 2]), r=None)

            if self.contact_filt[self.lookat_id, i]:
                gymutil.draw_lines(contact_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            else:
                gymutil.draw_lines(non_contact_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

            
            
    def _draw_coord(self):
        vec_x = np.array([1,0,0])
        vec_y = np.array([0,1,0])
        vec_z = np.array([0,0,1])
        vec_x_world_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(1, 0, 0))
        vec_y_world_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(0, 1, 0))
        vec_z_world_geom = gymutil.WireframeSphereGeometry(0.04, 16, 16, None, color=(0, 0, 1))
        
        #! 绘制世界坐标系的 xyz 轴
        for i in range(5):
            pose_arrow = 0.1 * (i+1) * vec_x + self.env_origins[self.lookat_id].cpu().numpy()
            pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_arrow[2]), r=None)
            gymutil.draw_lines(vec_x_world_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        for i in range(5):
            pose_arrow = 0.1 * (i+1) * vec_y + self.env_origins[self.lookat_id].cpu().numpy()
            pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_arrow[2]), r=None)
            gymutil.draw_lines(vec_y_world_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        for i in range(5):
            pose_arrow = 0.1 * (i+1) * vec_z + self.env_origins[self.lookat_id].cpu().numpy()
            pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_arrow[2]), r=None)
            gymutil.draw_lines(vec_z_world_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    
        #! 绘制机器人坐标系的 xyz 轴
        base_quat = self.base_quat[self.lookat_id].cpu().numpy()
        projected_vecx = quat_apply_np(base_quat, vec_x)
        projected_vecy = quat_apply_np(base_quat, vec_y)
        projected_vecz = quat_apply_np(base_quat, vec_z)
        vec_x_base_geom = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 0, 0))
        vec_y_base_geom = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(0, 1, 0))
        vec_z_base_geom = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(0, 0, 1))
        for i in range(5):
            pose_arrow = 0.1 * (i+1) * projected_vecx + self.base_pos_world[self.lookat_id].cpu().numpy()
            pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_arrow[2]), r=None)
            gymutil.draw_lines(vec_x_base_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        for i in range(5):
            pose_arrow = 0.1 * (i+1) * projected_vecy + self.base_pos_world[self.lookat_id].cpu().numpy()
            pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_arrow[2]), r=None)
            gymutil.draw_lines(vec_y_base_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        for i in range(5):
            pose_arrow = 0.1 * (i+1) * projected_vecz + self.base_pos_world[self.lookat_id].cpu().numpy()
            pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_arrow[2]), r=None)
            gymutil.draw_lines(vec_z_base_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    
    
    def _init_3dscan_points(self):

        terrain_y = torch.tensor(self.cfg.terrain.terrain_measured_points_y, device=self.device, requires_grad=False)   
        terrain_x = torch.tensor(self.cfg.terrain.terrain_measured_points_x, device=self.device, requires_grad=False)   
        terrain_grid_x, terrain_grid_y = torch.meshgrid(terrain_x, terrain_y)   # [dim(terrain_x), dim(terrain_y)]

        self.num_terrainscan_points = terrain_grid_x.numel() # 返回number of elements = dim(terrain_x) * dim(terrain_y)
        terrainscan_points = torch.zeros(self.num_envs, self.num_terrainscan_points, 3, device=self.device, requires_grad=False)

        for i in range(self.num_envs):
            #todo noise 是 0
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_terrainscan_points,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_terrainscan_points,2), device=self.device).squeeze() + offset
            terrainscan_points[i, :, 0] = terrain_grid_x.flatten() + xy_noise[:, 0]
            terrainscan_points[i, :, 1] = terrain_grid_y.flatten() + xy_noise[:, 1]
            
            
        ceiling_y = torch.tensor(self.cfg.terrain.ceiling_measured_points_y, device=self.device, requires_grad=False)
        ceiling_x = torch.tensor(self.cfg.terrain.ceiling_measured_points_x, device=self.device, requires_grad=False)
        ceiling_grid_x, ceiling_grid_y = torch.meshgrid(ceiling_x, ceiling_y)
        
        self.num_ceilingscan_points = ceiling_grid_x.numel()
        ceiling_points = torch.zeros(self.num_envs, self.num_ceilingscan_points, 3, device=self.device, requires_grad=False)
        
        for i in range(self.num_envs):
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_ceilingscan_points,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_ceilingscan_points,2), device=self.device).squeeze() + offset
            ceiling_points[i, :, 0] = ceiling_grid_x.flatten() + xy_noise[:, 0]
            ceiling_points[i, :, 1] = ceiling_grid_y.flatten() + xy_noise[:, 1]
            
        return terrainscan_points, ceiling_points
    

    # def _get_heights(self, env_ids=None):
    def _get_scan(self):

        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_terrainscan_points, device=self.device, requires_grad=False), torch.zeros(self.num_envs, self.num_ceiling_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        # if env_ids:
        #     points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[self.robot_actor_idxs[env_ids], :3]).unsqueeze(1)
        # else:
        #     points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[self.robot_actor_idxs, :3]).unsqueeze(1)
            
        # 根据base的位姿对scandots进行平移旋转
        terrain_scan_points_world = quat_apply_yaw(self.base_quat.repeat(1, self.num_terrainscan_points), self.terrain_scanpoints_xybase) + (self.root_states[self.robot_actor_idxs, :3]).unsqueeze(1) 
        
        terrain_scan_points_world += self.terrain.cfg.border_size 
        terrain_scan_points_world = (terrain_scan_points_world/self.terrain.cfg.horizontal_scale).long()
        terrain_px = terrain_scan_points_world[:, :, 0].view(-1)   # 把横坐标都取出来
        terrain_py = terrain_scan_points_world[:, :, 1].view(-1)   # 把纵坐标都取出来
        terrain_px = torch.clip(terrain_px, 0, self.terrain_height_samples.shape[0]-2)  
        terrain_py = torch.clip(terrain_py, 0, self.terrain_height_samples.shape[1]-2)  

        terrain_heights1 = self.terrain_height_samples[terrain_px, terrain_py]  # 264
        terrain_heights2 = self.terrain_height_samples[terrain_px+1, terrain_py]
        terrain_heights3 = self.terrain_height_samples[terrain_px, terrain_py+1]
        terrain_heights = torch.min(terrain_heights1, terrain_heights2)
        terrain_heights = torch.min(terrain_heights, terrain_heights3)
        
        ceiling_scan_points_world = quat_apply_yaw(self.base_quat.repeat(1, self.num_ceilingscan_points), self.ceiling_scanpoints_xybase) + (self.root_states[self.robot_actor_idxs, :3]).unsqueeze(1)
        
        ceiling_scan_points_world += self.terrain.cfg.border_size
        ceiling_scan_points_world = (ceiling_scan_points_world/self.terrain.cfg.horizontal_scale).long()
        ceiling_px = ceiling_scan_points_world[:, :, 0].view(-1)
        ceiling_py = ceiling_scan_points_world[:, :, 1].view(-1)
        ceiling_px = torch.clip(ceiling_px, 0, self.ceiling_height_samples.shape[0]-2)  
        ceiling_py = torch.clip(ceiling_py, 0, self.ceiling_height_samples.shape[1]-2)  
        
        ceiling_heights1 = self.ceiling_height_samples[ceiling_px, ceiling_py]  # 264
        ceiling_heights2 = self.ceiling_height_samples[ceiling_px+1, ceiling_py]
        ceiling_heights3 = self.ceiling_height_samples[ceiling_px, ceiling_py+1]
        ceiling_heights = torch.min(ceiling_heights1, ceiling_heights2)
        ceiling_heights = torch.min(ceiling_heights, ceiling_heights3)

        return terrain_heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale, ceiling_heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale


    ################## parkour rewards ##################

    #! task rewards
    
    def _reward_task_distance(self):
        #! 根据episode length判断。 只有在episode的最后一秒才有效
        take_effect_flag = self.episode_length_buf > self.task_episode_length
        
        self.rew_container_task[~take_effect_flag] = 0.
        
        # temp = 1 / (1 + torch.norm(self.robot2target_world, dim=-1))\  只需要xy坐标上接近就行了
        temp = 1 / (1 + torch.norm(self.robot2target_world[:,:2], dim=-1))
        self.rew_container_task[take_effect_flag] = \
            temp[take_effect_flag] / (self.max_episode_length_s - self.task_episode_length_s)   # 1 / (1 + robot2target_world + [] )
        
        return self.rew_container_task
    
    def _reward_vel_tracking(self):
    
        # 定义一个目标速度target_velocity，大小是0.5m/s,方向是从机器人到目标点
        # 离目标点远的时候，target_velocity大小是0.85，方向是从机器人到目标点；离目标点近的时候，target_velocity大小是0
        a = torch.norm(self.robot2target_world[:,:2], dim=-1) < 0.3
        # print("a.shape: ", a.shape)
        b = torch.zeros_like(self.base_lin_vel_world[:,:2])
        # print("b.shape: ", b.shape)
        c = self.robot2target_world[:,:2] / (torch.norm(self.robot2target_world[:,:2], dim=-1, keepdim=True) + 1e-5) * 0.85
        # print("c.shape: ", c.shape)
        # 定义一个target_velocity，使用条件判断a，如果a为true,那么target_velocity=b,否则target_velocity=c
        target_velocity = torch.where(a.unsqueeze(1), b, c)
        
        
        # target_velocity = torch.where(torch.norm(self.robot2target_world[:,:2], dim=-1) < 0.3, torch.zeros_like(self.base_lin_vel_world[:,:2]), self.robot2target_world[:,:2] / (torch.norm(self.robot2target_world[:,:2], dim=-1, keepdim=True) + 1e-5) * 0.85)
        
        lin_vel_error = torch.sum(torch.square(self.base_lin_vel_world[:,:2] - target_velocity), dim=-1)
        
        # print("target_velocity: ", target_velocity)
        # print("current velocity: ", self.base_lin_vel_world[:,:2] )
        # print("distance: ", torch.norm(self.robot2target_world[:,:2], dim=-1))
        # print("vel_tracking_reward: ", torch.where(torch.norm(self.robot2target_world[:,:2], dim=-1) < 0.3, torch.zeros_like(lin_vel_error), torch.exp(-lin_vel_error/0.25)))
        
        # 如果机器人和目标点的距离小于0.3，return 0；否则return vel_error
        # return torch.where(torch.norm(self.robot2target_world[:,:2], dim=-1) < 0.3, torch.ones_like(lin_vel_error)*1.5, torch.exp(-lin_vel_error/0.25))
        return torch.where(torch.norm(self.robot2target_world[:,:2], dim=-1) < 0.3, torch.exp(-lin_vel_error/0.25)*2.0, torch.exp(-lin_vel_error/0.25))


    #! auxiliary terms
    
    def _reward_exploration_vel(self):

        # 当r_task到达其最大值的一半后，r_exploration失效
        # take_effect_flag = self.rew_container_task > 0.5
        
        # 距离目标点比较近的时候，把exploration_vel detach掉
        #! 满足任意一个条件时，exploration_vel失效
        take_effect_flag = (torch.norm(self.robot2target_world[:,:2], dim=-1) < self.cfg.rewards.goal_threshold) | (self.episode_length_buf > self.task_episode_length)
        self.rew_container_exploration[take_effect_flag] = 0.
        
        norm = torch.norm(self.robot2target_world[:,:2], dim=-1, keepdim=True)
        target_vec_norm = self.robot2target_world[:,:2] / (norm + 1e-5)
        
        cur_vel = self.base_lin_vel_world[:,:2] # [num_env, 2]
        # print("vel_norm: ", torch.norm(cur_vel, dim=-1))
        
        temp = torch.sum(target_vec_norm * cur_vel, dim=-1)
        
        # 阈值的最大速度是1m/s 现在环境变短了，且完成任务的时间变长了，没必要阈值给1.0，太快了
        # shreshold = torch.ones_like(temp, dtype=torch.float, device=self.device, rerquires_grad=False) * 0.7
        shreshold = torch.ones_like(temp, dtype=torch.float, device=self.device) * 0.7
        temp = torch.minimum(temp, shreshold)
        
        self.rew_container_exploration[~take_effect_flag] = temp[~take_effect_flag]
        # print(self.rew_container_exploration)
        
        return self.rew_container_exploration
    
    def _reward_stalling(self):
        # 当离目标点远且速度小于0.6时，给予惩罚
        take_effect_flag = (torch.norm(self.base_lin_vel_world[:,:2], dim=-1) < 0.7) & \
                            (torch.norm(self.robot2target_world[:,:2], dim=-1) > self.cfg.rewards.goal_threshold)
                            
        # print("vel_norm: ", torch.norm(self.base_lin_vel_world[:,:2], dim=-1))
                            
        self.rew_container_stalling[take_effect_flag] = -1
        self.rew_container_stalling[~take_effect_flag] = 0

        return self.rew_container_stalling
    
    def _reward_facing_target(self):
        # 机器人面向target运动。因为摄像头在正面
        norm = torch.norm(self.robot2target_world[:,:2], dim=-1, keepdim=True)
        target_vec_norm = self.robot2target_world[:,:2] / (norm + 1e-5)
        # print("facingtarget_reward: ", torch.sum(target_vec_norm * self.projected_forward[:,:2], dim=-1))
        return torch.sum(target_vec_norm * self.projected_forward[:,:2], dim=-1)
    
    def _reward_early_termination(self):
        # 如果early termination，就给一个比较大的惩罚
        return self.height_cutoff | self.contact_termination_buf | self.roll_cutoff | self.pitch_cutoff
    
    def _reward_staystill_atgoal(self):
        # 机器人到达goal后，应该保持静止站立
        
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

        contact_error = torch.sum(~self.contact_filt)
        yawvel_error = torch.norm(self.base_ang_vel_base[:,2], dim=-1)
        
        sum = dof_error + 0.4*contact_error + 0.5*yawvel_error

        reward = 1 / (1 + sum) * (torch.norm(self.robot2target_world[:,:2], dim=-1) < self.cfg.rewards.goal_threshold) 
        
        return reward
    
    
    #! gait shaping reward
    def _reward_feetair_awaygoal(self):
        first_contact = (self.feet_air_time > 0.) * self.contact_filt   # 第一次contact [num_env, 4]
        
        self.feet_air_time += self.dt   # 4条腿的air_time [num_env, 4]
        
        # 给self.feet_air_time设置上限为0.6
        self.feet_air_time_clip = torch.minimum(self.feet_air_time, torch.ones_like(self.feet_air_time, dtype=torch.float, device=self.device, requires_grad=False) * 0.6)
        
        # feet_air_time如果大于0.5s，有reward
        # rew_airtime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  
        # 当feet_air_time小于0.5时，是负的奖励
        rew_airtime = torch.sum((self.feet_air_time_clip - 0.3) * first_contact, dim=1)  # 计算上一回抬腿的时间，如果大于0.3s，reward
        # print(rew_airtime)
        
        # 只有当距离目标点比较远时，才有效
        rew_airtime *= torch.norm(self.robot2target_world[:,:2], dim=-1) > self.cfg.rewards.goal_threshold
        
        self.feet_air_time *= ~self.contact_filt    # 当feet是contact时，feet_air_time为0

        # if rew_airtime != 0:
        #     print("rew_airtime: ", rew_airtime)

        return rew_airtime
            
    def _reward_feet_floating(self):
        # 如果后腿超过0.8s不着地，加上惩罚
        # take_effect_flag = (self.contact_filt==True) & (self.last_contacts==False)
        
        # 检查是否存在超过0.8s没有着地的腿
        # float_id = torch.any(self.feet_air_time > 0.6, dim=1)
        
        self.feet_air_time_minus = self.feet_air_time - 0.6
        self.feet_float = torch.maximum(self.feet_air_time_minus, torch.zeros_like(self.feet_air_time_minus, dtype=torch.float, device=self.device, requires_grad=False))
        reward = torch.sum(self.feet_float, dim=1)
        # print("feet_float: ", self.feet_float)
        # print("feet_air_time: ", self.feet_air_time)
        # print(reward)
        
        return reward
            
        
    def _reward_feet_contact_forces(self):
        # 惩罚过大的接触力 100-> 80
        return torch.sum( (torch.norm(self.robot_contact_forces[:, self.feet_indices, :], dim=-1) -  80.).clip(min=0.), dim=1)
    
    def _reward_feet_slip(self):
        # 惩罚脚拖着走
        feet_vel = torch.square(torch.norm(self.robot_link_states[:, self.feet_indices, 7:10], dim=2).view(self.num_envs, -1))  # [num_env, 4]
        rew_slip = torch.sum(self.contact_filt * feet_vel, dim=1)
        
        return rew_slip
    
    
    #! parkour reward
    
    def _reward_feet_stumble(self):
        # 惩罚脚在xy轴上的力
        rew = torch.any(torch.norm(self.robot_contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.robot_contact_forces[:, self.feet_indices, 2]), dim=1)        
        return rew.float()

    def _reward_feet_edge(self):
        # 惩罚脚在边缘
        feet_pos_xy = ((self.robot_link_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)   # 不要超出范围
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edgex = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
        feet_at_edgey = self.y_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
        feet_at_edge = feet_at_edgex | feet_at_edgey
    
        self.feet_at_edge = self.contact_filt & feet_at_edge    # [num_env, 4]
        # rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        
        # 所有地形难度应该都加
        rew = torch.sum(self.feet_at_edge, dim=-1)
        
        return rew

        
    #! normalization reward
    
    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel_base[:, 2])
        return rew
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel_base[:, :2]), dim=1)
     
    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.projected_gravity_base[:, :2]), dim=1)
        return rew

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        return torch.sum(1.*(torch.norm(self.robot_contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    
    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error
    



    # def _reward_contact_sequence(self):
    #     # 每个foot每0.5s至少进行一次触地。惩罚过长的腾空相
    #     # take_effect_flag = self.foot_swing_timer > 0.5/self.dt  #[num_env,4]
            
    #     self.rew_container_contact[:] = torch.sum(self.foot_swing_timer[self.foot_swing_timer > 1.0/self.dt], dim=-1)
        
    #     # 刚开始的1s内，机器人reset，此时不会受到penalty
    #     self.rew_container_contact[self.episode_length_buf < 1.0 / self.dt] = 0.
                
    #     return self.rew_container_contact


    # def _reward_norotation_neargoal(self):
    #     # 当机器人距离goal小于1m时，就不应该旋转了
    #     take_effect_flag = (torch.norm(self.robot2target_world[:,:2], dim=-1) < 1.0) 
        
    #     self.rew_container_norotation[take_effect_flag] = torch.norm(self.base_ang_vel_base[:,:2], dim=-1)[take_effect_flag]

    #     self.rew_container_norotation[~take_effect_flag] = 0.
        
    #     return self.rew_container_norotation
    

    

    
    


