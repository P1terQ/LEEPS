
from posixpath import relpath
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
from .base_config import BaseConfig
import torch.nn as nn

class LeggedV2Cfg(BaseConfig):
    class play:
        load_student_config = False
        mask_priv_obs = False
        
    class env:
        num_envs = 4096 # 4096     #! 为啥是这个奇怪的值 6144

        n_scan_terrain = 725
        n_scan_ceiling = 70
        # n_scan_terrain = 132
        # n_scan_ceiling = 132
        
        n_priv = 4 # priv robot state(lihear vel + com_height_baseframe)
        n_priv_latent = 4 + 1 + 48 + 3
        # n_proprio = 3 + 2 + 4 + 36 + 2  # 3 + 2 + 4 + 36 + 4 = 49
        n_proprio = 3 + 2 + 4 + 36 + 3 + 1 #+ 1
        history_len = 10

        # num_observations = n_proprio + n_scan + history_len*nproprio + n_priv_latent + n_priv # critic obs
        num_observations = n_proprio + n_scan_terrain + n_scan_ceiling + history_len*n_proprio + n_priv_latent + n_priv 

        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        
        # 可以对timeouts的情况做特殊处理
        #! 论文提示不用timeout bootstrapping更好
        send_timeouts = False # send time out information to the algorithm
        
        #! 从起点x=5到终点x=8，不算远距离
        # episode_length_s = 6 # episode length in seconds   
        episode_length_s = 8     
        history_encoding = True # add obs history to buffer
        
        # task_episode_length_s = 5
        # task_episode_length_s = 7
        task_episode_length_s = 6   #!最后的2s需要足够长，才能使机器人以一个稳定的形式停下来
        
        target_radius = [2,3]
        
        debug_viz = False
        remote = False


    class depth:
        use_camera = False  # 默认情况下不使用camera,先用scandots
        
        position = [0.27, 0, 0.03]  # front camera
        angle = [-5, 5]  # positive pitch down

        # policy每运行5次，就更新一次depth_buffer
        update_interval = 5  # 5 works without retraining, 8 worse

        original = (106, 60)    # 一开始获得的depth image的大小
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2  # depth_buffer len
        
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        
        #// scale = 1
        #//invert = True
        #! 使用camera训练depth encoder时的环境设置
        camera_num_envs = 192   
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20
        
    class normalization:
        class obs_scales:
            # obs normalization factor
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            #// height_measurements = 5.0
            
        clip_observations = 100.
        clip_actions = 1.2
        
    class noise:    # not used 
        add_noise = False
        noise_level = 1.0 # scales other values
        
        # quantize_height = True
        # class noise_scales:
        #     rotation = 0.0
        #     dof_pos = 0.01
        #     dof_vel = 0.05
        #     lin_vel = 0.05
        #     ang_vel = 0.05
        #     gravity = 0.02
        #     height_measurements = 0.02

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast
        y_range = [-0.4, 0.4]   # 生成parkour terrain时需要
        
        # 使用camera训练encoder时的参数
        simplify_grid = False
        max_error_camera = 2
        horizontal_scale_camera = 0.05 #0.1   # train student就把这个缩小一点，可以节省一点computation time?
        
        # terrain params
        edge_width_thresh = 0.05
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        vertical_scale = 0.005 # [m]
        
        # 平地上实验，给了4.rough情况下感觉不给也可以
        border_size = 4 # [m]
        # border_size = 0 # [m]
        
        height = [0.02, 0.06]   # terrain roughness height
        downsampled_scale = 0.075

        
        #! 在平地上test的时候可以不用curriculum，在复杂地形上还是需要
        # curriculum =  False   # terrain curriculum
        curriculum =  True
        
        selected = False # select a unique terrain type and pass all arguments
        
        # triangle mesh params
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        
        # scandot params
        measure_heights = True
        
        # [25, 29] 1.6m x 2.0m rectangle  725
        terrain_measured_points_x = [-0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                                     0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.45, 1.6]
        terrain_measured_points_y = [-0.8, -0.65, -0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 
                                     0.35, 0.4, 0.45, 0.5, 0.65, 0.8]
        # [7, 10] 0.9 x 1.35 rectangle 70
        ceiling_measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
        ceiling_measured_points_y = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45]
        
        # terrain_measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
        # terrain_measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        # ceiling_measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
        # ceiling_measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        
        measure_horizontal_noise = 0.0
        
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 2 # starting curriculum state
        
        #! subterrain的长和宽
        terrain_length = 7. #7. #8.  
        terrain_width = 4.
        pyramid = False
        
        #! 训练时subterrain的数目
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 10 # number of terrain cols (types)
        
        terrain_dict = {
                        "step": 0.0, # proportions[0]
                        "gap": 1.0,  # proportions[1]
                        "slope": 0.0,
                        "stair": 0.0, 
                        "discrete": 0.0, 
                        "flat": 0.0,       # proportions[5]
                        "steppingstones": 0.0, # proportions[6]
                        "crawl": 0.0,     # proportions[7]
                        "log": 0.0,
                        "crack": 0.0,
                        "dual": 0.0,
                        "parkour": 0.0
                        }
        terrain_proportions = list(terrain_dict.values())
        
        # trimesh only:
        #! 这个之所以改小,是因为1.5的时候训练stairs的时候都变成slope了
        slope_treshold = 1.5#1.5# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True    # 将env_origin_z都设置为0，而不是地面高度


        teleport_robots = True # 当机器人靠近环境边缘时，reset到另一边
        teleport_thresh = 2.0

    # 跳箱子这个task下完全不用commands
    class commands: 
        curriculum = False
        # max_curriculum = 1.
        num_commands = 1 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 6. # time before command are changed[s]
        
        
        # if use command curriculum
        class ranges:
            # lin_vel_x = [0., 1.5] # min max [m/s]
            # lin_vel_y = [0.0, 0.0]   # min max [m/s]
            # ang_vel_yaw = [0, 0]    # min max [rad/s]
            # heading = [0, 0]
            
            object2target_radius = [0.3, 2]

        # if not use command curriculum
        class default_ranges:
            object2target_radius = [0.5, 1.0]


    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}
        
        x_init_range = 0.1
        y_init_range = 0.2
        
        # 这个还是改回来比较好，目前的难点不是这个
        yaw_init_range = 3.14/6
        # yaw_init_range = 3.14
        

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:       
        
        randomize_friction = True
        friction_range = [0.6, 2.]
        
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.]
        
        randomize_base_com = True
        added_com_range = [-0.2, 0.2]
        
        dof_rand_interval_s = 10
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        
        randomize_motor_offset = True
        motor_offset_range = [-0.002, 0.002]
        
        randomize_Kp_factor = True
        Kp_factor_range = [0.8, 1.3]
        
        randomize_Kd_factor = True
        Kd_factor_range = [0.5, 1.5]
        
        randomize_gravity = False
        gravity_rand_interval_s = 7
        gravity_impulse_duration = 1.0
        gravity_range = [-1.0, 1.0]
        
        # push暂时先不用
        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 0.5

        delay_update_global_steps = 24 * 8000   # action buffer delay.其实不需要
        #! 这个标志位会在args中重置，若需要传入--delay
        action_delay = False
        action_curr_step = [1, 1]
        # action_curr_step_scratch = [0, 1]
        action_delay_view = 1
        action_buf_len = 8  # action_history_buf 长度
        

        
    class rewards:
        class scales:                       
            # main task term
            task_distance = 10.0    # 5.0
            
            #! 新加的task term，用来做消融实验，这个起作用的时候task_distance和exploration_terms应注释
            # vel_tracking = 2.0
            
            # exploration terms
            exploration_vel = 1.5 #2.0 #1.2
            stalling = 1
            
            facing_target = 0.3
            early_termination = -200 
            staystill_atgoal = 1000 #250 # 200 #8 #2


            # gait shaping terms
            feet_contact_forces = -0.01
            feet_slip = -0.08 #-0.04
            dof_error = -0.08   # -0.04
            hip_pos = -0.5
            feetair_awaygoal = 1.5
            feet_floating = -8.0#-2.0    
            feet_stumble = -1


            # parkour terms
            feet_edge = -1

            
            # normalization terms
            #todo 把lin_vel_z和orientation去掉之后gait特别奇怪,加回来做对比实验看下
            lin_vel_z = -1.0 # 3维空间运动不需要这个惩罚了
            ang_vel_xy = -0.05
            orientation = -1.
            dof_acc = -1.5e-7 #-2.5e-7
            collision = -10.
            action_rate = -0.05 #-0.1
            delta_torques = -1.0e-7
            torques = -0.00001
            # dof_error = -0.08
            
            
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        
        goal_threshold = 0.1
        
        # tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        # soft_dof_vel_limit = 1
        # soft_torque_limit = 0.4
        # base_height_target = 1.
        # max_contact_force = 40. # forces above this value are penalized
        

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedV2CfgPPO(BaseConfig):
    seed = 3
    runner_class_name = 'OnPolicyRunner'
 
    class policy:
        init_noise_std = 1.0
        continue_from_last_std = True
        
        terrain_scan_encoder_dims = [512, 256, 128] # todo 这个mlp的维度需要测试
        ceiling_scan_encoder_dims = [64, 32, 16]

        # todo 新的那一篇还是3层的，是不是3层就够了
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        
        priv_encoder_dims = [64, 20]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        tanh_encoder_output = False
    
    class algorithm:    # ppo params
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99    # Discount factor
        lam = 0.95  # GAE discount factor
        desired_kl = 0.01
        max_grad_norm = 1.
        
        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]
    
    class depth_encoder:
        if_depth = LeggedV2Cfg.depth.use_camera
        depth_shape = LeggedV2Cfg.depth.resized
        buffer_len = LeggedV2Cfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = LeggedV2Cfg.depth.update_interval * 24  

    class estimator:
        train_with_scandots = LeggedV2Cfg.terrain.measure_heights
        
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        activation = 'elu'
        priv_states_dim = LeggedV2Cfg.env.n_priv
        num_prop = LeggedV2Cfg.env.n_proprio
        num_scan_terrain = LeggedV2Cfg.env.n_scan_terrain
        num_scan_ceiling = LeggedV2Cfg.env.n_scan_ceiling

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 48 # 24 # per iteration
        max_iterations = 50000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'give me a name'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
        
        


