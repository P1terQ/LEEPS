from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
# from .legged_robot_config import LeggedRobotCfg
from .legged_tracker_config import LeggedTrackerCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedTrackerCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        
        self._init_command_distribution()

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
            
        self._init_buffers()
        
        self._prepare_reward_function()
        
        self.init_done = True
        self.record_now = False
        self.record_eval_now = False
        
        
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        # self.prev_base_pos = self.base_pos.clone()
        # self.prev_base_quat = self.base_quat.clone()
        # self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        
        # step physics and render each frame
        self.render()
        
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            
            # if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
            
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        if self.record_now:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:self.num_envs, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        
        # Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        
        self.reset_buf |= self.time_out_buf

        if self.cfg.rewards.use_terminal_body_height:
            self.body_height_buf = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) \
                                   < self.cfg.rewards.terminal_body_height
                                   
            self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)

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
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            
            self.episode_sums[name] += rew
            
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew   # 不知道为啥这两项还要加上reward_scales
            else:
                self.command_sums[name] += rew
            
        if self.cfg.rewards.only_positive_rewards:
            # self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)
            
        self.episode_sums["total"] += self.rew_buf
            
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            
        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1
        
    
    def compute_observations(self):
        """ Computes observations
        """
        # todo cssss
        self.obs_buf = torch.cat((self.projected_gravity,   
                                  self.commands * self.commands_scale,
                                  (self.dof_pos[:, :12] - self.default_dof_pos[:,:12]) * self.obs_scales.dof_pos,
                                  self.dof_vel[:, :12] * self.obs_scales.dof_vel,
                                  self.actions,
                                  self.last_actions,
                                  self.clock_inputs
                                  ), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        
        self._create_envs()
        

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _randomize_gravity(self, external_force = None):

        if external_force is not None:
            self.gravities[:, :] = external_force.unsqueeze(0)
        elif self.cfg.domain_rand.randomize_gravity:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range   # [-1, 1]
            external_force = torch.rand(3, dtype=torch.float, device=self.device, requires_grad=False) \
                * (max_gravity - min_gravity) + min_gravity

            self.gravities[:, :] = external_force.unsqueeze(0) 

        sim_params = self.gym.get_sim_params(self.sim)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device) # add external force to gravity
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)   # set gravity
    
    
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        # if self.cfg.domain_rand.randomize_friction:
        #     if env_id==0:
        #         # prepare friction randomization
        #         friction_range = self.cfg.domain_rand.friction_range
        #         num_buckets = 64
        #         bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
        #         friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
        #         self.friction_coeffs = friction_buckets[bucket_ids]

        for s in range(len(props)):
            props[s].friction = self.friction_coeffs[env_id]
            props[s].restitution = self.restitutions[env_id, 0]
            
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
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
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props
    
    
    def _randomize_rigid_body_props(self, env_ids):
        if self.cfg.domain_rand.randomize_base_mass: # true
            min_payload, max_payload = self.cfg.domain_rand.added_mass_range
            self.payloads[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False) \
                * (max_payload - min_payload) + min_payload
                
        if self.cfg.domain_rand.randomize_com_displacement:  # false
            min_com_displacement, max_com_displacement = self.cfg.domain_rand.com_displacement_range
            self.com_displacements[env_ids,:] = torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device, requires_grad=False) \
                * (max_com_displacement - min_com_displacement) + min_com_displacement

        if self.cfg.domain_rand.randomize_friction:  # true
            min_friction, max_friction = self.cfg.domain_rand.friction_range
            self.friction_coeffs[env_ids,:] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device, requires_grad=False) \
                * (max_friction - min_friction) + min_friction

        if self.cfg.domain_rand.randomize_restitution:   # true
            min_restitution, max_restitution = self.cfg.domain_rand.restitution_range
            self.restitutions[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,requires_grad=False) \
                * (max_restitution - min_restitution) + min_restitution
             
             
    def refresh_actor_rigid_shape_props(self, env_ids):
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(self.num_dof):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitutions[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)
            
             
    def _randomize_dof_props(self, env_ids):
        if self.cfg.domain_rand.randomize_motor_strength:    # true
            min_strength, max_strength = self.cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids,:] = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1) \
                * (max_strength - min_strength) + min_strength
                
        if self.cfg.domain_rand.randomize_motor_offset:  # true
            min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids,:] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) \
                * (max_offset - min_offset) + min_offset
                
        if self.cfg.domain_rand.randomize_Kp_factor: # false
            min_Kp_factor, max_Kp_factor = self.cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids,:] = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1) \
                * (max_Kp_factor - min_Kp_factor) + min_Kp_factor
        
        if self.cfg.domain_rand.randomize_Kd_factor: # false
            min_Kd_factor, max_Kd_factor = self.cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids,:] = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1) \
                * (max_Kd_factor - min_Kd_factor) + min_Kd_factor
                                                    

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        # if self.cfg.domain_rand.randomize_base_mass:
        #     rng = self.cfg.domain_rand.added_mass_range
        #     props[0].mass += np.random.uniform(rng[0], rng[1])
            
        self.default_body_mass = props[0].mass

        props[0].mass = self.default_body_mass + self.payloads[env_id]
        props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1], self.com_displacements[env_id, 2])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 当机器人太靠近一边的时候,将其传送到另外一边
        self._teleport_robots()
        
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % sample_interval==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)    # resample different cmd vals
        self._step_contact_targets()    # compute gait phase and normalized contact states(for reward computation)
        
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            
        self._push_robots()
            
        # 持续gravity_rand_duration进行gravity rand.之后就恢复正常
        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if int(self.common_step_counter - self.cfg.domain_rand.gravity_rand_duration) % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity(torch.tensor([0, 0, 0]))
            
        # randomize env ids
        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero(as_tuple=False).flatten()
        self._randomize_dof_props(env_ids)
        
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._randomize_rigid_body_props(env_ids)
            self.refresh_actor_rigid_shape_props(env_ids)


    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        
        if len(env_ids) == 0:
            return
        
        timesteps = int(self.cfg.commands.resampling_time / self.dt)    # 每500个step resample一次
        ep_len = min(self.cfg.env.max_episode_length, timesteps)    
        
        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)): 
            
            # 取出属于这个category的env ids
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:  # false, len(env_ids_in_category) = 2048
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue            

            env_ids_in_category = env_ids[env_ids_in_category]  # 把[true,true,true...] -> [0,1,2,...]
            
            task_rewards, success_thresholds = [], []
            
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force", "tracking_contacts_shaped_vel"]:
                if key in self.command_sums.keys(): # 计算 4种奖励的 平均每一步的reward
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)   # 4 种任务在不同category下每步的奖励
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])    # 定值

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]

            if len(success_thresholds) > 0:     # true
                curriculum.update(old_bins,     # 增加old_bins和其adjacent的weights    
                                  task_rewards, 
                                  success_thresholds,
                                  local_range=np.array( # 定义了一个curriculum的adjacent的范围
                                      [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        # assign resampled environments to new categories
        random_env_floats = torch.rand(len(env_ids), device=self.device)    # 2048个 0-1之间的随机数
        probability_per_category = 1. / len(self.category_names)    # 1 / 4
        category_env_ids = [env_ids[torch.logical_and(probability_per_category * i <= random_env_floats,    # 2048个环境，每个环境对应的category(步态类别)
                                                      random_env_floats < probability_per_category * (i + 1))] for i in range(len(self.category_names))]

        for i, (category, env_ids_in_category, curriculum) in enumerate(zip(self.category_names, category_env_ids, self.curricula)):

            batch_size = len(env_ids_in_category)
            if batch_size == 0: 
                continue

            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)   # sample15个命令的bin_idex和对应的值
            #更新 cmd bin id
            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            #更新 category
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i
            #记录command
            self.commands[env_ids_in_category, :] = torch.Tensor(new_commands[:, :self.cfg.commands.num_commands]).to(self.device)

        if self.cfg.commands.num_commands > 5:  # true
            
            #! 我也不知道这个具体的作用是啥。对步态的相位做一个normalization
            if self.cfg.commands.gaitwise_curricula:    # true
                for i, (category, env_ids_in_category) in enumerate(zip(self.category_names, category_env_ids)):
                    if category == "pronk":  # pronking
                        self.commands[env_ids_in_category, 5] = (self.commands[env_ids_in_category, 5] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 6] = (self.commands[env_ids_in_category, 6] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 7] = (self.commands[env_ids_in_category, 7] / 2 - 0.25) % 1
                    elif category == "trot":  # trotting
                        self.commands[env_ids_in_category, 5] = self.commands[env_ids_in_category, 5] / 2 + 0.25
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "pace":  # pacing
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = self.commands[env_ids_in_category, 6] / 2 + 0.25
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "bound":  # bounding
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = self.commands[env_ids_in_category, 7] / 2 + 0.25

            elif self.cfg.commands.exclusive_phase_offset:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                trotting_envs = env_ids[random_env_floats < 0.34]
                pacing_envs = env_ids[torch.logical_and(0.34 <= random_env_floats, random_env_floats < 0.67)]
                bounding_envs = env_ids[0.67 <= random_env_floats]
                self.commands[pacing_envs, 5] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[trotting_envs, 6] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 7] = 0

            elif self.cfg.commands.balance_gait_distribution:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                pronking_envs = env_ids[random_env_floats <= 0.25]
                trotting_envs = env_ids[torch.logical_and(0.25 <= random_env_floats, random_env_floats < 0.50)]
                pacing_envs = env_ids[torch.logical_and(0.50 <= random_env_floats, random_env_floats < 0.75)]
                bounding_envs = env_ids[0.75 <= random_env_floats]
                self.commands[pronking_envs, 5] = (self.commands[pronking_envs, 5] / 2 - 0.25) % 1
                self.commands[pronking_envs, 6] = (self.commands[pronking_envs, 6] / 2 - 0.25) % 1
                self.commands[pronking_envs, 7] = (self.commands[pronking_envs, 7] / 2 - 0.25) % 1
                self.commands[trotting_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 5] = 0
                self.commands[pacing_envs, 7] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 5] = self.commands[trotting_envs, 5] / 2 + 0.25
                self.commands[pacing_envs, 6] = self.commands[pacing_envs, 6] / 2 + 0.25
                self.commands[bounding_envs, 7] = self.commands[bounding_envs, 7] / 2 + 0.25

            if self.cfg.commands.binary_phases:
                self.commands[env_ids, 5] = (torch.round(2 * self.commands[env_ids, 5])) / 2.0 % 1
                self.commands[env_ids, 6] = (torch.round(2 * self.commands[env_ids, 6])) / 2.0 % 1
                self.commands[env_ids, 7] = (torch.round(2 * self.commands[env_ids, 7])) / 2.0 % 1
                
        # setting the smaller commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
                
        # reset command sums to zero. Sum every resample CMD
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0. 
            
            
    def _step_contact_targets(self):

        if self.cfg.env.observe_gait_commands:

            frequencies = self.commands[:, 4]
            
            # phase offset
            phases = self.commands[:, 5]    
            offsets = self.commands[:, 6]   
            bounds = self.commands[:, 7]    
            
            durations = self.commands[:, 8] # stance在整个gait的占比
            
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0) # dt * frequencies

            if self.cfg.commands.pacing_offset:
                # [FR, FL, RR, RL] 是每个foot的phase
                foot_indices = [self.gait_indices + phases + offsets + bounds,  # 0.5
                                self.gait_indices + bounds, # 0
                                self.gait_indices + offsets, # 0
                                self.gait_indices + phases] # 0.5
            else: # true
                # add gait phase offset 
                foot_indices = [self.gait_indices + phases + offsets + bounds,
                                self.gait_indices + offsets,
                                self.gait_indices + bounds,
                                self.gait_indices + phases]
            
            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

            # for idxs in foot_indices:
            #     # choose swing or stance 
            #     stance_idxs = torch.remainder(idxs, 1) < durations  
            #     swing_idxs = torch.remainder(idxs, 1) > durations

            #     idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            #     idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (0.5 / (1 - durations[swing_idxs]))

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            # self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
            # self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
            # self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
            # self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

            # self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
            # self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
            # self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
            # self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

            # von mises distribution
            kappa = self.cfg.rewards.kappa_gait_probs   # 0.07
            
            # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2
            smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf 

            #! 定义了一种映射，当为0和0.5的时候为0.5. 当0~0.5时，为接近1的值，当0.5~1.0时，为接近0的值
            smoothing_multiplier_FL = ( smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * 
                    (1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) 
                                       +
                    smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) 
                    * (1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
            
            smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
            
            smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
            
            smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        if self.cfg.commands.num_commands > 9:
            self.desired_footswing_height = self.commands[:, 9]
              

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range
        
        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]   # add actions_scaled to the end of lag_buffer
            self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
        else:
            self.joint_pos_target = actions_scaled + self.default_dof_pos
        
        control_type = self.cfg.control.control_type
        
        if control_type=="P":
            # torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
            # add kp,kd,offset randomization
            torques = self.p_gains * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.d_gains * self.Kd_factors * self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        if self.cfg.domain_rand.push_robots:
            env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]

            max_vel = self.cfg.domain_rand.max_push_vel_xy
            self.root_states[env_ids, 7:9] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 2), device=self.device)  # lin vel x/y
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            

    def _teleport_robots(self):
        """ Teleports any robots that are too close to the edge to the other side
        """
        if self.cfg.terrain.teleport_robots:
            thresh = self.cfg.terrain.teleport_thresh

            # x_offset = int(self.cfg.terrain.x_offset * self.cfg.terrain.horizontal_scale)
            env_ids = torch.arange(self.num_envs, device=self.device, requires_grad=False)

            low_x_ids = env_ids[self.root_states[:, 0] < thresh]
            self.root_states[low_x_ids, 0] += self.cfg.terrain.terrain_length * (self.cfg.terrain.num_rows - 1)

            high_x_ids = env_ids[self.root_states[env_ids, 0] > self.cfg.terrain.terrain_length * self.cfg.terrain.num_rows - thresh]
            self.root_states[high_x_ids, 0] -= self.cfg.terrain.terrain_length * (self.cfg.terrain.num_rows - 1)

            low_y_ids = env_ids[self.root_states[env_ids, 1] < thresh]
            self.root_states[low_y_ids, 1] += self.cfg.terrain.terrain_width * (self.cfg.terrain.num_cols - 1)

            high_y_ids = env_ids[self.root_states[env_ids, 1] > self.cfg.terrain.terrain_width * self.cfg.terrain.num_cols - thresh]
            self.root_states[high_y_ids, 1] -= self.cfg.terrain.terrain_width * (self.cfg.terrain.num_cols - 1)

            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.refresh_actor_root_state_tensor(self.sim)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """        
        
        noise_vec = torch.zeros_like(self.obs_buf[0])
        
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,
                                torch.zeros(self.cfg.commands.num_commands), # cat cmd here， 15
                                torch.ones(12) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                                torch.ones(12) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                                torch.zeros(self.num_actions),
                                torch.zeros(self.num_actions),  # prev action
                                torch.zeros(4), # clock inputs
                                ), dim=0)
            
        noise_vec = noise_vec.to(self.device)
            
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        # self.prev_base_pos = self.base_pos.clone()
        
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies, :]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.prev_foot_velocities = self.foot_velocities.clone()
        
        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        
        
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)  # prepare noise vec for observation
        
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        
        # self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_value = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)    # [2048,15]
        self.commands = torch.zeros_like(self.commands_value)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, 
                                            self.obs_scales.lin_vel, 
                                            self.obs_scales.ang_vel,
                                            self.obs_scales.body_height_cmd, 
                                            
                                            self.obs_scales.gait_freq_cmd,
                                            self.obs_scales.gait_phase_cmd, 
                                            self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.gait_phase_cmd, 
                                            self.obs_scales.gait_phase_cmd,
                                            
                                            self.obs_scales.footswing_height_cmd, 
                                            self.obs_scales.body_pitch_cmd,
                                            self.obs_scales.body_roll_cmd, 
                                            self.obs_scales.stance_width_cmd,   #?
                                            self.obs_scales.stance_length_cmd,
                                            
                                            self.obs_scales.aux_reward_cmd
                                            ], device=self.device, requires_grad=False,)[:self.cfg.commands.num_commands]
        
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False, )
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
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
        
        
    def _init_custom_buffers__(self):
        # domain randomization properties
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))  #! 0，0，-1

        # if custom initialization values were passed in, set them here
        # dynamics_params = ["friction_coeffs", "restitutions", "payloads", "com_displacements", "motor_strengths",
        #                    "Kp_factors", "Kd_factors"]
        # if self.initial_dynamics_dict is not None:  # none
        #     for k, v in self.initial_dynamics_dict.items():
        #         if k in dynamics_params:
        #             setattr(self, k, v.to(self.device))

        # some observation related terms
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        # self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        # self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        
    # 925 
    def _init_command_distribution(self):
        
        self.category_names = ['nominal']
        if self.cfg.commands.gaitwise_curricula:    
            self.category_names = ['pronk', 'trot', 'pace', 'bound'] 
            
        if self.cfg.commands.curriculum_type == "RewardThresholdCurriculum": 
            from legged_gym.envs.tracker.curriculum import RewardThresholdCurriculum
            CurriculumClass = RewardThresholdCurriculum
            
        self.curricula = []
        for category in self.category_names:
            self.curricula += [CurriculumClass(seed=self.cfg.commands.curriculum_seed,
                                               # key ranges
                                               x_vel=(self.cfg.commands.limit_vel_x[0], self.cfg.commands.limit_vel_x[1], self.cfg.commands.num_bins_vel_x),
                                               y_vel=(self.cfg.commands.limit_vel_y[0], self.cfg.commands.limit_vel_y[1], self.cfg.commands.num_bins_vel_y),
                                               yaw_vel=(self.cfg.commands.limit_vel_yaw[0], self.cfg.commands.limit_vel_yaw[1], self.cfg.commands.num_bins_vel_yaw),
                                               body_height=(self.cfg.commands.limit_body_height[0], self.cfg.commands.limit_body_height[1], self.cfg.commands.num_bins_body_height),   
                                               gait_frequency=(self.cfg.commands.limit_gait_frequency[0], self.cfg.commands.limit_gait_frequency[1], self.cfg.commands.num_bins_gait_frequency),
                                               gait_phase=(self.cfg.commands.limit_gait_phase[0], self.cfg.commands.limit_gait_phase[1], self.cfg.commands.num_bins_gait_phase),
                                               gait_offset=(self.cfg.commands.limit_gait_offset[0], self.cfg.commands.limit_gait_offset[1], self.cfg.commands.num_bins_gait_offset),
                                               gait_bounds=(self.cfg.commands.limit_gait_bound[0], self.cfg.commands.limit_gait_bound[1], self.cfg.commands.num_bins_gait_bound),
                                               gait_duration=(self.cfg.commands.limit_gait_duration[0], self.cfg.commands.limit_gait_duration[1], self.cfg.commands.num_bins_gait_duration),
                                               footswing_height=(self.cfg.commands.limit_footswing_height[0], self.cfg.commands.limit_footswing_height[1], self.cfg.commands.num_bins_footswing_height),
                                               body_pitch=(self.cfg.commands.limit_body_pitch[0], self.cfg.commands.limit_body_pitch[1], self.cfg.commands.num_bins_body_pitch),
                                               body_roll=(self.cfg.commands.limit_body_roll[0], self.cfg.commands.limit_body_roll[1], self.cfg.commands.num_bins_body_roll),
                                               stance_width=(self.cfg.commands.limit_stance_width[0], self.cfg.commands.limit_stance_width[1], self.cfg.commands.num_bins_stance_width),
                                               stance_length=(self.cfg.commands.limit_stance_length[0], self.cfg.commands.limit_stance_length[1], self.cfg.commands.num_bins_stance_length),
                                               aux_reward_coef=(self.cfg.commands.limit_aux_reward_coef[0], self.cfg.commands.limit_aux_reward_coef[1], self.cfg.commands.num_bins_aux_reward_coef),
                                               )]
            
        if self.cfg.commands.curriculum_type == "LipschitzCurriculum":  # false
            for curriculum in self.curricula:
                curriculum.set_params(lipschitz_threshold=self.cfg.commands.lipschitz_threshold, binary_phases=self.cfg.commands.binary_phases)
        
        self.env_command_bins = np.zeros(len(self.num_envs), dtype=np.int)    # 2048
        self.env_command_categories = np.zeros(len(self.num_envs), dtype=np.int)  # 2048
        
        low = np.array(
            [self.cfg.commands.ranges.lin_vel_x[0], self.cfg.commands.lin_vel_y[0],
             self.cfg.commands.ranges.ang_vel_yaw[0], self.cfg.commands.body_height_cmd[0],
             self.cfg.commands.ranges.gait_frequency_cmd_range[0],
             self.cfg.commands.ranges.gait_phase_cmd_range[0], self.cfg.commands.gait_offset_cmd_range[0],
             self.cfg.commands.ranges.gait_bound_cmd_range[0], self.cfg.commands.gait_duration_cmd_range[0],
             self.cfg.commands.ranges.footswing_height_range[0], self.cfg.commands.body_pitch_range[0],
             self.cfg.commands.ranges.body_roll_range[0],self.cfg.commands.stance_width_range[0],
             self.cfg.commands.ranges.stance_length_range[0], self.cfg.commands.aux_reward_coef_range[0], ])
        high = np.array(
            [self.cfg.commands.ranges.lin_vel_x[1], self.cfg.commands.lin_vel_y[1],
             self.cfg.commands.ranges.ang_vel_yaw[1], self.cfg.commands.body_height_cmd[1],
             self.cfg.commands.ranges.gait_frequency_cmd_range[1],
             self.cfg.commands.ranges.gait_phase_cmd_range[1], self.cfg.commands.gait_offset_cmd_range[1],
             self.cfg.commands.ranges.gait_bound_cmd_range[1], self.cfg.commands.gait_duration_cmd_range[1],
             self.cfg.commands.ranges.footswing_height_range[1], self.cfg.commands.body_pitch_range[1],
             self.cfg.commands.ranges.body_roll_range[1],self.cfg.commands.stance_width_range[1],
             self.cfg.commands.ranges.stance_length_range[1], self.cfg.commands.aux_reward_coef_range[1], ])
        
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)
        

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
                
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums. create episode_sums for every reward term
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                                for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.episode_sums_eval = {name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) 
                                for name in self.reward_scales.keys()}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.command_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual", 
                                               "ep_timesteps"]}
        

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
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
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
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

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

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)   # 4*4+1=17
        self.num_dofs = len(self.dof_names) # 3*4=12
        
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self._get_env_origins() # fill self.env_origins
        
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        self.default_friction = rigid_shape_props_asset[1].friction # 1.0
        self.default_restitution = rigid_shape_props_asset[1].restitution   # 0.0
        
        self._init_custom_buffers__()   # randomization terms and clock values
        self._randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device))  # init randomization buffer
        self._randomize_gravity()
        
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            
            # add randomization to the initial position in x and y direction
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            # randomize friction and restitution
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            
            # set dof pos/vel/torque limits
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            
            # add mass and com randomization
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            
            self.envs.append(env_handle)    # env handles
            self.actor_handles.append(actor_handle) # actor handles

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):    # 4,8,12,16
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):    # [ 2,  6, 10, 14,  3,  7, 11, 15]
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)): # 0
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        # newly add funtion: recording vedio
        if self.cfg.env.record_vedio:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 360
            self.camera_props.height = 240
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], 
                                            gymapi.Vec3(1.5, 1, 3.0),   # location 
                                            gymapi.Vec3(0, 0, 0))   # target
        self.vedio_writer = None
        self.vedio_frames = []
        self.complete_vedio_frames = []


    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            min_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: 
                max_init_level = self.cfg.terrain.num_rows - 1
            if not self.cfg.terrain.curriculum: min_init_level = 0
                
            if self.cfg.terrain.center_robots:   # 刚开始的时候把所有机器人放在中间(的难度和种类)
                min_terrain_level = self.cfg.terrain.num_rows // 2 - self.cfg.terrain.center_span # 15 - 4 = 11
                max_terrain_level = self.cfg.terrain.num_rows // 2 + self.cfg.terrain.center_span - 1 # 15 + 4 - 1 = 18
                min_terrain_type = self.cfg.terrain.num_cols // 2 - self.cfg.terrain.center_span  # 15 - 4 = 11
                max_terrain_type = self.cfg.terrain.num_cols // 2 + self.cfg.terrain.center_span - 1  # 15 + 4 - 1 = 18
                self.terrain_levels = torch.randint(min_terrain_level, max_terrain_level + 1, (self.num_envs,), device=self.device)
                self.terrain_types = torch.randint(min_terrain_type, max_terrain_type + 1, (self.num_envs,), device=self.device)
            else:   # 一开始随机分布机器人位置
                self.terrain_levels = torch.randint(min_init_level, max_init_level+1, (self.num_envs,), device=self.device)
                self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
                
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)   # 已经在terrain中初始化好了的env_origins
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            
        else:   # flat ground平均【分配
            self.custom_origins = False

            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.curriculum_thresholds = vars(self.cfg.curriculum_thresholds)
        
        # self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.command_ranges = vars(self.cfg.commands.ranges)    # vars : 对象object的属性和属性值的字典对象
        
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
            
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)  # 751
        self.cfg.domain_rand.rand_interval = np.ceil(self.cfg.domain_rand.rand_interval_s / self.dt)  
        self.cfg.domain_rand.gravity_rand_interval = np.ceil(self.cfg.domain_rand.gravity_rand_interval_s / self.dt)
        self.cfg.domain_rand.gravity_rand_duration = np.ceil(self.cfg.domain_rand.gravity_rand_interval * self.cfg.domain_rand.gravity_impulse_duration)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids: # 只针对某几个env进行yaw transformation，还是所有的环境
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]  #! sample from height_samples
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    #! task rewards
    def _reward_tracking_lin_vel(self): # linear vel tracking 
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma_lin)  #! 对x和y方向的linear vel tracking error惩罚
    
    def _reward_tracking_ang_vel(self): # angular vel tracking
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma_yaw)  #! 对角速度进行惩罚

    #! fixed auxiliary reward(for better sim2real transfer)
    def _reward_lin_vel_z(self):    # z velocity penalty
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])    #! 对z方向的linear vel进行惩罚
    
    def _reward_ang_vel_xy(self):   # roll pitch velocity penalty
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) #! 对x和y方向的angular vel进行惩罚

    def _reward_orientation(self):  
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)    #! 对pitch和yaw进行惩罚(但不是要追踪pitch和yaw吗？)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)     #! 惩罚关节力矩
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1) #! 惩罚关节的加速度
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1) #! 惩罚两次action的变化大小
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),  dim=1)    #! 惩罚指定关节上的碰撞力大小
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)    # upper limit
        return torch.sum(out_of_limits, dim=1)  #! 这个有点疑惑，难道不会是一个常数吗？
    
    def _reward_jump(self):
        reference_heights = 0
        body_height = self.base_pos[:, 2] - reference_heights
        jump_height_target = self.commands[:, 3] + self.cfg.rewards.base_height_target
        reward = - torch.square(body_height - jump_height_target)
        return reward
    
    def _reward_tracking_contacts_shaped_force(self):       #! 为了让foot在摆动的时候不要发生触碰
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))
        return reward / 4
    
    def _reward_tracking_contacts_shaped_vel(self):     #! 为了让foot在支撑的时候不发生滑动
        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma)))
        return reward / 4
    
    def _reward_dof_pos(self):  #! 接近默认的关节位置
        # Penalize dof positions
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)  
    
    def _reward_dof_vel(self):  #! 惩罚大的关节速度
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_action_smoothness_1(self):  #! 惩罚target_joint_pos的变化
        # Penalize changes in actions
        diff = torch.square(self.joint_pos_target[:, :12] - self.last_joint_pos_target[:, :12])
        diff = diff * (self.last_actions[:, 12] != 0)  # ignore first step
        return torch.sum(diff, dim=1)
    
    def _reward_action_smoothness_2(self):  #! 惩罚target_joint_pos的变化
        # Penalize changes in actions
        diff = torch.square(self.joint_pos_target[:, 12] - 2 * self.last_joint_pos_target[:, 12] + self.last_last_joint_pos_target[:, 12])
        diff = diff * (self.last_actions[:, 12] != 0)  # ignore first step
        diff = diff * (self.last_last_actions[:, 12] != 0)  # ignore second step
        return torch.sum(diff, dim=1)
    
    def _reward_feet_slip(self):    #! 惩罚支撑相的滑动
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip
    
    def _reward_feet_contact_vel(self):     #! 惩罚摆动相的触地速度（尽可能慢的触地）
        reference_heights = 0   #todo 之后reference_height修改为感知点得到的值
        near_ground = self.foot_positions[:, :, 2] - reference_heights < 0.03
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:3], dim=2).view(self.num_envs, -1))
        rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)
        return rew_contact_vel
    
    def _reward_feet_contact_forces(self):  #! 惩罚支撑相的触地力（尽可能小的触地力）
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    
    #! augemented auxiliary rewards
    def _reward_feet_clearance_cmd_linear(self):    #! 摆动时追踪foot的高度
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1)# - reference_heights
        target_height = self.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
        
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)
    
    def _reward_feet_impact_vel(self):  # stance phase tracking #! 惩罚触地时的速度
        prev_foot_velocities = self.prev_foot_velocities[:, :, 2].view(self.num_envs, -1)
        contact_states = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0

        rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))

        return torch.sum(rew_foot_impact_vel, dim=1)
    
    def _reward_collision(self):    #! 惩罚指定link上出现的碰撞力
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_orientation_control(self):  # pitch tracking
        # Penalize non flat base orientation
        #! 应该是同时有track pitch 和 yaw 的
        roll_pitch_commands = self.commands[:, 10:12]   
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1], torch.tensor([1, 0, 0], device=self.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0], torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.gravity_vec)

        return torch.sum(torch.square(self.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)
    
    def _reward_raibert_heuristic(self):    # foot hold tracking
        
        # 计算foot在base坐标系下
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1) 
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat), cur_footsteps_translated[:, i, :])    

        # nominal positions: [FR, FL, RR, RL]   #! nominal foot position由stance width和stance length决定 
        desired_stance_width = self.commands[:, 12:13]  # y是左右方向上
        desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)

        desired_stance_length = self.commands[:, 13:14] # x上前后方向上
        desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        
        frequencies = self.commands[:, 4]
        x_vel_des = self.commands[:, 0:1]
        yaw_vel_des = self.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        
        #? 没看懂这里的offset是怎么算的
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
    