
from isaacgym import gymapi, gymutil
from legged_gym.utils.barrier_track import BarrierTrack
from legged_gym.envs.a1.a1_config import A1RoughCfg
import numpy as np

class terrain(A1RoughCfg.terrain):
    mesh_type = "trimesh" # Don't change
    num_rows = 1
    num_cols = 1  
    
    selected = "BarrierTrack" # "BarrierTrack" or "TerrainPerlin", "TerrainPerlin" can be used for training a walk policy.
    
    max_init_terrain_level = 0
    border_size = 0
    slope_treshold = 20.

    curriculum = False # for walk
    
    horizontal_scale = 0.025 # [m]
    # vertical_scale = 1. # [m] does not change the value in hightfield
    pad_unavailable_info = True

    #! track的参数
    BarrierTrack_kwargs = dict(
        options= [
            "climb",
            "stairs",
            "stairs_down",
            "slope",
            "slope_down",
            "crawl",
            "tilt",
            "leap",
            "steppingstone",
            "flat"
        ], # each race track will permute all the options
        
        #! 一般都是true如果randomize_obstacle_order=False.最好只生成一个环境。里边包含所有的obstacle
        randomize_obstacle_order = False,
        n_obstacles_per_track = 10 + 1,  # +1是start track
        
        track_width= 1.6,
        track_block_length= 2., # the x-axis distance from the env origin point
        
        # 每个cols之间的wall
        wall_thickness= (0.04, 0.2), # [m]
        wall_height= -0.05, 
        # wall_thickness= 1.0,
        # wall_height= 1,
        
        climb= dict(
            height= (0.2, 0.6),
            depth= (0.1, 0.8), # size along the forward axis
            fake_offset= 0.0, # [m] an offset that make the robot easier to get into the obstacle
            climb_down_prob= 0.0,
        ),
        
        crawl= dict(
            height= (0.25, 0.5),
            depth= (0.1, 0.6), # size along the forward axis
            wall_height= 0.6,
            no_perlin_at_obstacle= False,
        ),
        
        tilt= dict(
            width= (0.24, 0.32),
            depth= (0.4, 1.), # size along the forward axis
            opening_angle= 0.0, # [rad] an opening that make the robot easier to get into the obstacle
            wall_height= 0.5,
        ),
        
        leap= dict(
            length= (0.2, 1.0),
            depth= (0.4, 0.8),
            height= 0.2,
        ),
        
        slope = dict(
            gradient = (0.2, 0.4),
        ),
        
        slope_down = dict(
            gradient = (-0.4, -0.2),
        ),
        
        stairs = dict(
            step_width = (0.3, 0.5),
            step_height = (0.1, 0.2)
        ),
        
        stairs_down = dict(
            step_width = (0.3, 0.5),
            step_height = (-0.2, -0.1)
        ),
        
        steppingstone = dict(
            stone_size = (0.2, 0.3),
            stone_distance = (0.1, 0.2),
            max_height = (0.05, 0.1),
        ),
        
        add_perlin_noise= True, #! add noise
        # add_perlin_noise= False,
        
        border_perlin_noise= True,
        
        border_height= 0.,
        virtual_terrain= False,
        draw_virtual_terrain= True,
        engaging_next_threshold= 1.2,
        curriculum_perlin= False,
        no_perlin_threshold= 0.0,
    )

    TerrainPerlin_kwargs = dict(
        zScale= [0.05, 0.1],
        # zScale= 0.1, # Use a constant zScale for training a walk policy
        frequency= 10,
    )


if __name__ == '__main__':
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments()
    
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    args.physics_engine = gymapi.SIM_PHYSX
    sim_params.substeps = 2
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    
    
    terrain_cfg = terrain()
    
    terrain_ = BarrierTrack(terrain_cfg, num_envs=0)
    
    terrain_.add_terrain_to_sim(gym, sim, device= "cpu")
    
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    
    cam_initpos = gymapi.Vec3(0, -2, 2.5)
    cam_inittarget = gymapi.Vec3(0, 0, 1)
    cam_direction = np.array(cam_inittarget) - np.array(cam_initpos)
    
    gym.viewer_camera_look_at(viewer, None, cam_initpos, cam_inittarget)
    
    dt = sim_params.dt
    cam_vel = np.array([1., 0., 0.])
    cam_dpos = np.array([0. ,0. ,0. ])

    while not gym.query_viewer_has_closed(viewer):
        # cam_dpos += cam_vel * dt
        # cam_pos = gymapi.Vec3(0+cam_dpos[0], -2+cam_dpos[1], 2.5+cam_dpos[2])
        # cam_target = gymapi.Vec3(0+cam_dpos[0], 0+cam_dpos[1], 1+cam_dpos[2])
        # gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
        