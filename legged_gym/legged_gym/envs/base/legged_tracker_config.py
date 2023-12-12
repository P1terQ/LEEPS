from .base_config import BaseConfig

class LeggedTrackerCfg(BaseConfig):
    class env:
        
        num_envs = 4096
        
        num_observations = 235
        num_scalar_observations = 42
        # if not None a privilige_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_privileged_obs = 18
        privileged_future_horizon = 1
        num_actions = 12
        num_observation_history = 15
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        # observe_vel = True
        # observe_only_ang_vel = False
        # observe_only_lin_vel = False
        # observe_yaw = False
        # observe_contact_states = False
        # observe_command = True
        # observe_height_command = False
        # observe_gait_commands = False
        # observe_timing_parameter = False
        # observe_clock_inputs = False
        # observe_two_prev_actions = False
        # observe_imu = False
        # record_video = True
        # recording_width_px = 360
        # recording_height_px = 240
        # recording_mode = "COLOR"
        # num_recording_envs = 1
        # debug_viz = False
        # all_agents_share = False

        # priv_observe_friction = True
        # priv_observe_friction_indep = True
        # priv_observe_ground_friction = False
        # priv_observe_ground_friction_per_foot = False
        # priv_observe_restitution = True
        # priv_observe_base_mass = True
        # priv_observe_com_displacement = True
        # priv_observe_motor_strength = False
        # priv_observe_motor_offset = False
        # priv_observe_joint_friction = True
        # priv_observe_Kp_factor = True
        # priv_observe_Kd_factor = True
        # priv_observe_contact_forces = False
        # priv_observe_contact_states = False
        # priv_observe_body_velocity = False
        # priv_observe_foot_height = False
        # priv_observe_body_height = False
        # priv_observe_gravity = False
        # priv_observe_terrain_type = False
        # priv_observe_clock_inputs = False
        # priv_observe_doubletime_clock_inputs = False
        # priv_observe_halftime_clock_inputs = False
        # priv_observe_desired_contact_states = False
        # priv_observe_dummy_variable = False
        
    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 0  # 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        # terrain_noise_magnitude = 0.1
        # # rough terrain only:
        # terrain_smoothness = 0.005
        measure_heights = True  # measure heigtfield around the robot
        # # 1mx1.6m rectangle (without center line)
        # measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # selected = False  # select a unique terrain type and pass all arguments
        # terrain_kwargs = None  # Dict of arguments for selected terrain
        
        min_init_terrain_level = 0  # starting curriculum state
        max_init_terrain_level = 5  
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # # trimesh only:
        # slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces
        # difficulty_scale = 1.
        # x_init_range = 1.
        # y_init_range = 1.
        # yaw_init_range = 0.
        # x_init_offset = 0.
        # y_init_offset = 0.
        
        teleport_robots = True  # 当机器人太靠近边缘的时候，将其teleport到另外一边
        teleport_thresh = 2.0
        
        # max_platform_height = 0.2
        
        center_robots = False   # 一开始将所有机器人放在中间位置的环境中
        center_span = 5 # 中间环境的范围
        
    class commands:
        
        class ranges:   # cmd limits
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            body_height_cmd = [-0.05, 0.05]
            gait_frequency_cmd_range = [2.0, 2.01]
            gait_phase_cmd_range = [0.0, 0.01]
            gait_offset_cmd_range = [0.0, 0.01]
            gait_bound_cmd_range = [0.0, 0.01]
            gait_duration_cmd_range = [0.49, 0.5]
            footswing_height_range = [0.06, 0.061]
            body_pitch_range = [0.0, 0.01]
            body_roll_range = [0.0, 0.01]
            stance_width_range = [0.0, 0.01]
            stance_length_range = [0.0, 0.01]
            aux_reward_coef_range = [0.0, 0.01]
        
        num_commands = 15
        resampling_time = 10.   # resample command
            
        curriculum_type = "RewardThresholdCurriculum" # command curriculum type
        curriculum_seed = 100   # command curriculum seed
        # cucrriculum limits
        limit_vel_x = [-10.0, 10.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-10.0, 10.0]
        limit_body_height = [-0.05, 0.05]
        limit_gait_phase = [0, 0.01]
        limit_gait_offset = [0, 0.01]
        limit_gait_bound = [0, 0.01]
        limit_gait_frequency = [2.0, 2.01]
        limit_gait_duration = [0.49, 0.5]
        limit_footswing_height = [0.06, 0.061]
        limit_body_pitch = [0.0, 0.01]
        limit_body_roll = [0.0, 0.01]
        limit_aux_reward_coef = [0.0, 0.01]
        limit_compliance = [0.0, 0.01]
        limit_stance_width = [0.0, 0.01]
        limit_stance_length = [0.0, 0.01]
        # curriculum bins
        num_bins_vel_x = 25
        num_bins_vel_y = 3
        num_bins_vel_yaw = 25
        num_bins_body_height = 1
        num_bins_gait_frequency = 11
        num_bins_gait_phase = 11
        num_bins_gait_offset = 2
        num_bins_gait_bound = 2
        num_bins_gait_duration = 3
        num_bins_footswing_height = 1
        num_bins_body_pitch = 1
        num_bins_body_roll = 1
        num_bins_aux_reward_coef = 1
        num_bins_compliance = 1
        num_bins_compliance = 1
        num_bins_stance_width = 1
        num_bins_stance_length = 1
        
        gaitwise_curricula = True   # 再command distribution中加入多种步态
        
    class init_state:
        pos = [0.0, 0.0, 1.]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # target angles when action = 0.0
        default_joint_angles = {"joint_a": 0., "joint_b": 0.}

    class control:        
        control_type = 'P' #'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        hip_scale_reduction = 1.0   # used in compute torques
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        name = "robot" # actor name in gym
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        # # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # # replace collision cylinders with capsules, leads to faster/more stable simulation
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01
        
    class domain_rand:
        rand_interval_s = 10    # domain rand interval(dof,rigid body, actor rigid shape)
        randomize_rigids_after_start = True # if add rand to rigid body & actor rigid shape
        
        # randomize rigid body props
        randomize_friction = True
        friction_range = [0.5, 1.25]  # increase range
        randomize_restitution = False
        restitution_range = [0, 1.0]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        randomize_com_displacement = False
        com_displacement_range = [-0.15, 0.15]
        
        # randomize dof props
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        randomize_Kp_factor = False
        Kp_factor_range = [0.8, 1.3]
        randomize_Kd_factor = False
        Kd_factor_range = [0.5, 1.5]
        
        gravity_rand_interval_s = 7
        gravity_impulse_duration = 1.0
        randomize_gravity = False
        gravity_range = [-1.0, 1.0]
        
        push_robots = True  # random push during training
        push_interval_s = 15    
        max_push_vel_xy = 1.
        
        randomize_lag_timesteps = True  # add lag when compute torques
        lag_timesteps = 6   # init lag_buffer
               
    class rewards:
        
        class scales:
            termination = -0.0
            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 0.5
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            # orientation = -0.
            # torques = -0.00001
            # dof_vel = -0.
            # dof_acc = -2.5e-7
            # base_height = -0.
            # feet_air_time = 1.0
            # collision = -1.
            # feet_stumble = -0.0
            # action_rate = -0.01
            # stand_still = -0.
            # tracking_lin_vel_lat = 0.
            # tracking_lin_vel_long = 0.
            # tracking_contacts = 0.
            # tracking_contacts_shaped = 0.
            # tracking_contacts_shaped_force = 0.
            # tracking_contacts_shaped_vel = 0.
            # jump = 0.0
            # energy = 0.0
            # energy_expenditure = 0.0
            # survival = 0.0
            # dof_pos_limits = 0.0
            # feet_contact_forces = 0.
            # feet_slip = 0.
            # feet_clearance_cmd_linear = 0.
            # dof_pos = 0.
            # action_smoothness_1 = 0.
            # action_smoothness_2 = 0.
            # base_motion = 0.
            # feet_impact_vel = 0.0
            # raibert_heuristic = 0.0
        
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        # only_positive_rewards_ji22_style = False
        sigma_rew_neg = 5
        
        tracking_sigma_lin = 0.25  # tracking reward = exp(-error^2/sigma)
        # tracking_sigma_lat = 0.25  # tracking reward = exp(-error^2/sigma)
        # tracking_sigma_long = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 0.25  # tracking reward = exp(-error^2/sigma)
        # soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        # soft_dof_vel_limit = 1.
        # soft_torque_limit = 1.
        base_height_target = 1. # reward jump
        max_contact_force = 100.  # forces above this value are penalized
        use_terminal_body_height = False    # 是否将body heigth作为termination条件
        terminal_body_height = 0.20
        # use_terminal_foot_height = False
        # terminal_foot_height = -0.005
        # use_terminal_roll_pitch = False
        # terminal_body_ori = 0.5
        # kappa_gait_probs = 0.07
        gait_force_sigma = 50.  # _reward_tracking_contacts_shaped_force
        gait_vel_sigma = 0.5  # _reward_tracking_contacts_shaped_vel
        # footswing_height = 0.09
        
    class normalization:
        
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            # imu = 0.1
            # height_measurements = 5.0
            # friction_measurements = 1.0
            body_height_cmd = 2.0
            gait_phase_cmd = 1.0
            gait_freq_cmd = 1.0
            footswing_height_cmd = 0.15
            body_pitch_cmd = 0.3
            body_roll_cmd = 0.3
            aux_reward_cmd = 1.0
            # compliance_cmd = 1.0
            stance_width_cmd = 1.0
            stance_length_cmd = 1.0
            # segmentation_image = 1.0
            # rgb_image = 1.0
            # depth_image = 1.0


        clip_observations = 100.
        clip_actions = 100.

        # friction_range = [0.05, 4.5]
        # ground_friction_range = [0.05, 4.5]
        # restitution_range = [0, 1.0]
        # added_mass_range = [-1., 3.]
        # com_displacement_range = [-0.1, 0.1]
        # motor_strength_range = [0.9, 1.1]
        # motor_offset_range = [-0.05, 0.05]
        # Kp_factor_range = [0.8, 1.3]
        # Kd_factor_range = [0.5, 1.5]
        # joint_friction_range = [0.0, 0.7]
        # contact_force_range = [0.0, 50.0]
        # contact_state_range = [0.0, 1.0]
        # body_velocity_range = [-6.0, 6.0]
        # foot_height_range = [0.0, 0.15]
        # body_height_range = [0.0, 0.60]
        # gravity_range = [-1.0, 1.0]
        # motion = [-0.01, 0.01]

    class noise:
        add_noise = True    # if add noise vec to observation
        noise_level = 1.0  # scales other values
        
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            # lin_vel = 0.1
            # ang_vel = 0.2
            # imu = 0.1
            gravity = 0.05
            # contact_states = 0.05
            # height_measurements = 0.1
            # friction_measurements = 0.0
            # segmentation_image = 0.0
            # rgb_image = 0.0
            # depth_image = 0.0

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]
        
    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        use_gpu_pipeline = True
        
        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)





