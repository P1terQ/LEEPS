import numpy as np
from copy import copy

from isaacgym import gymapi, gymutil
from isaacgym.terrain_utils import convert_heightfield_to_trimesh
from legged_gym.utils import trimesh
# import legged_gym.utils.trimesh
from legged_gym.utils.perlin import TerrainPerlin
# import legged_gym.utils.perlin.TerrainPerlin
from legged_gym.utils.console import colorize
import torch


class BarrierTrack:
    
    # default_kwargs 
    track_kwargs = dict(
        options = ["climb", "crawl", "tilt", "leap", "slope", "stairs", "steppingstone", "slope_down", "stairs_down", "flat"],
        
        #! randomize_obstacle_order=True, n_blocks_per_track = num_options + 1, else n_blocks_per_track = n_obstacles_per_track + 1
        randomize_obstacle_order = True,
        # randomize_obstacle_order = False, # if True, will randomize the order of the obstacles instead of the order in options
        n_obstacles_per_track= 1, # number of obstacles per track, only used when randomize_obstacle_order is True
        
        track_width= 1.6,
        track_block_length= 1.2, # the x-axis distance from the env origin point
        
        climb= dict(
            height= 0.3,
            depth= 0.04, # size along the forward axis
            fake_offset= 0.0, # [m] fake offset will make climb's height info greater than its physical height.
            fake_height= 0.0, # [m] offset/height only one of them can be non-zero
            climb_down_prob= 0.0, # if > 0, will have a chance to climb down from the obstacle
        ),
        crawl= dict(
            height= 0.32,
            depth= 0.04, # size along the forward axis
            wall_height= 0.8,
            no_perlin_at_obstacle= False, # if True, will set the heightfield to zero at the obstacle
        ),
        tilt= dict(
            width= 0.18,
            depth= 0.04, # size along the forward axis
            opening_angle= 0.3, # [rad] an opening that make the robot easier to get into the obstacle
            wall_height= 0.8,
        ),
        leap= dict(
            length= 0.8,
            depth= 0.5,
            height= 0.1, # expected leap height over the gap
        ),
        
        add_perlin_noise= False,
        border_perlin_noise= False,
        curriculum_perlin= True, # If True, perlin noise scale will be depends on the difficulty if possible.
        no_perlin_threshold= 0.02, # If the perlin noise is too small, clip it to zero.
        
        border_height= 0., # Incase we want the surrounding plane to be lower than the track
        
        virtual_terrain= False,
        draw_virtual_terrain= False, # set True for visualization
    )
    max_track_options = 4 # ("tilt", "crawl", "climb", "dynamic") at most
    track_options_id_dict = {
        "tilt": 1,
        "crawl": 2,
        "climb": 3,
        "leap": 4,
        "slope": 5,
        "stairs": 6,
        "steppingstone": 7,
        "slope_down": 8,
        "stairs_down": 9,
        "flat": 10,
    } # track_id are aranged in this order
    
    def __init__(self, cfg, num_envs: int) -> None:
        self.cfg = cfg
        self.num_envs = num_envs

        assert self.cfg.mesh_type == "trimesh", "Not implemented for mesh_type other than trimesh, get {}".format(self.cfg.mesh_type)   # 地形必须要是teimesh
        assert getattr(self.cfg, "BarrierTrack_kwargs", None) is not None, "Must provide BarrierTrack_kwargs in cfg.terrain"    # 检查cfg是否有BarrierTrack_kwargs
        
        self.track_kwargs.update(self.cfg.BarrierTrack_kwargs) # update the default kwargs with the kwargs in cfg.terrain.BarrierTrack_kwargs
        
        #! 就不用perlin_noise试试
        # if self.track_kwargs["add_perlin_noise"] and not hasattr(self.cfg, "TerrainPerlin_kwargs"): # false, 有TerrainPerlin_kwargs
        #     print(colorize(
        #         "Warning: Please provide cfg.terrain.TerrainPerlin to configure perlin noise for all surface to step on.",
        #         color= "yellow",
        #     ))

        self.env_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3), dtype= np.float32)
        
    def add_terrain_to_sim(self, gym, sim, device= "cpu"):  
        self.gym = gym
        self.sim = sim
        self.device = device
        
        # init length,witdh,resolution params
        self.initialize_track()
        
        # init map size and heightfield_raw
        self.build_heightfield_raw()
        
        # init track_info_map and track_width_map
        self.initialize_track_info_buffer()
        
        self.track_origins_px = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3), dtype= int) # 每个track的origin(x,y,z).以pixel的形式存储
        
        for col_idx in range(self.cfg.num_cols):    # 遍历每个track
            starting_height_px = 0  # 以col为单位初始化，每条col的起始高度都是0
            
            for row_idx in range(self.cfg.num_rows):
                
                self.track_origins_px[row_idx, col_idx] = [int(row_idx * self.track_resolution[0]) + self.border, int(col_idx * self.track_resolution[1]) + self.border, starting_height_px, ]
    
                starting_height_px = self.add_track_to_sim(self.track_origins_px[row_idx, col_idx], row_idx= row_idx, col_idx= col_idx, )
                
        # 而且貌似不加上这个plane也可以
        # self.add_plane_to_sim(starting_height_px)   # 传入的starting_height_px也没用上
        
        for i in range(self.cfg.num_rows):
            for j in range(self.cfg.num_cols):
                # refresh env_origins
                self.env_origins[i, j, 0] = self.track_origins_px[i, j, 0] * self.cfg.horizontal_scale
                self.env_origins[i, j, 1] = self.track_origins_px[i, j, 1] * self.cfg.horizontal_scale
                self.env_origins[i, j, 2] = self.track_origins_px[i, j, 2] * self.cfg.vertical_scale
                self.env_origins[i, j, 1] += self.track_kwargs["track_width"] / 2
                
        self.env_origins_pyt = torch.from_numpy(self.env_origins).to(self.device)   # 之后用来计算info的时候会用到
    
    
    def initialize_track(self):
        self.track_block_resolution = ( # track_block的分辨率大小
            np.ceil(self.track_kwargs["track_block_length"] / self.cfg.horizontal_scale).astype(int), np.ceil(self.track_kwargs["track_width"] / self.cfg.horizontal_scale).astype(int), )    
        
        self.n_blocks_per_track = (self.track_kwargs["n_obstacles_per_track"] + 1) if self.track_kwargs["randomize_obstacle_order"] else (len(self.track_kwargs["options"]) + 1)
        
        self.track_resolution = (   # 整个track的大小,相当于track_block大小乘以个数. a track consist of a connected track_blocks     
            # np.ceil(self.track_kwargs["track_block_length"] * self.n_blocks_per_track / self.cfg.horizontal_scale).astype(int), #todo fixme
            np.rint(self.track_kwargs["track_block_length"] * self.n_blocks_per_track / self.cfg.horizontal_scale).astype(int), 
            np.ceil(self.track_kwargs["track_width"] / self.cfg.horizontal_scale).astype(int),
        ) 
        
        # 用来计算info
        self.env_block_length = self.track_kwargs["track_block_length"] # 2.0m
        
        # 用来计算noise
        self.env_length = self.track_kwargs["track_block_length"] * self.n_blocks_per_track # 4.0
        self.env_width = self.track_kwargs["track_width"]   # 1.6
        
    
    def build_heightfield_raw(self):
        
        self.border = int(self.cfg.border_size / self.cfg.horizontal_scale) # 5 / 0.025 = 200

        # 整个地图的大小
        map_x_size = int(self.cfg.num_rows * self.track_resolution[0]) + 2 * self.border 
        map_y_size = int(self.cfg.num_cols * self.track_resolution[1]) + 2 * self.border   
    
        # build heightfield
        self.heightfield_raw = np.zeros((map_x_size, map_y_size), dtype= np.float32)  
        
        if self.track_kwargs["add_perlin_noise"] and self.track_kwargs["border_perlin_noise"]:  # true 
            
            #? 不明白这个的含义
            TerrainPerlin_kwargs = self.cfg.TerrainPerlin_kwargs
            for k, v in self.cfg.TerrainPerlin_kwargs.items():
                if isinstance(v, (tuple, list)):
                    if TerrainPerlin_kwargs is self.cfg.TerrainPerlin_kwargs:
                        TerrainPerlin_kwargs = copy(self.cfg.TerrainPerlin_kwargs)
                    TerrainPerlin_kwargs[k] = v[0]
                    
            heightfield_noise = TerrainPerlin.generate_fractal_noise_2d(    # 产生z方向上的噪音
                xSize= self.env_length * self.cfg.num_rows + 2 * self.cfg.border_size,  # 每env的长度 * 地图的个数。 map的大小
                ySize= self.env_width * self.cfg.num_cols + 2 * self.cfg.border_size,
                xSamples= map_x_size,   # 以分辨率为单位.map的大小
                ySamples= map_y_size,
                **TerrainPerlin_kwargs,
            ) / self.cfg.vertical_scale
            
            # 在平地上加入噪声
            self.heightfield_raw += heightfield_noise 
            # border上的噪声还是0
            self.heightfield_raw[self.border:-self.border, self.border:-self.border] = 0. 
            
            # 一般来说border_height都是0
            if self.track_kwargs["border_height"] != 0.:    
                self.heightfield_raw[:, :self.border] += self.track_kwargs["border_height"] / self.cfg.vertical_scale
                self.heightfield_raw[:, -self.border:] += self.track_kwargs["border_height"] / self.cfg.vertical_scale
                    
        
    def initialize_track_info_buffer(self):
        #! For each `track` block (n_options + 1 in total), 3 parameters are enabled:
        # - track_id: int, starting track is 0, other numbers depends on the options order.
        # - obstacle_depth: float,
        # - obstacle_critical_params: e.g. tilt width, crawl height, climb height

        self.track_info_map = torch.zeros((self.cfg.num_rows + 1, self.cfg.num_cols, self.n_blocks_per_track, 3),
            dtype= torch.float32, device= self.device, )
        self.track_width_map = torch.zeros((self.cfg.num_rows, self.cfg.num_cols),
            dtype= torch.float32, device= self.device, )
        
        
    def add_track_to_sim(self, track_origin_px, row_idx= None, col_idx= None):

        if self.track_kwargs["randomize_obstacle_order"] and len(self.track_kwargs["options"]) > 0:     # true
            obstacle_order = np.random.choice(len(self.track_kwargs["options"]), size = self.track_kwargs.get("n_obstacles_per_track", 1), replace= True, )     #! choose different obstacles
        else:
            obstacle_order = np.arange(len(self.track_kwargs["options"]))

        difficulties = self.get_difficulty(row_idx, col_idx)    # 判断是否curriculum.是的话根据row_idx来增加难度
        difficulty, virtual_track = difficulties[:2]  
        
        if self.track_kwargs["add_perlin_noise"]:   
            TerrainPerlin_kwargs = self.cfg.TerrainPerlin_kwargs
            
            for k, v in self.cfg.TerrainPerlin_kwargs.items():  #! zScale, frequency
                if isinstance(v, (tuple, list)):
                    
                    #? 这一步不知道在干啥，不是显然的吗
                    if TerrainPerlin_kwargs is self.cfg.TerrainPerlin_kwargs:   
                        TerrainPerlin_kwargs = copy(self.cfg.TerrainPerlin_kwargs)
                        
                    # curriculum noise or random noise 
                    if difficulty is None or (not self.track_kwargs["curriculum_perlin"]):  
                        TerrainPerlin_kwargs[k] = np.random.uniform(*v)    
                    else:
                        TerrainPerlin_kwargs[k] = v[0] * (1 - difficulty) + v[1] * difficulty
                    
                    # noise minimum threshold
                    if self.track_kwargs["no_perlin_threshold"] > TerrainPerlin_kwargs[k]:      
                        TerrainPerlin_kwargs[k] = 0.
                        
            heightfield_noise = TerrainPerlin.generate_fractal_noise_2d(    #! generate z-axis noise
                xSize= self.env_length,
                ySize= self.env_width,
                xSamples= self.track_resolution[0],
                ySamples= self.track_resolution[1],
                **TerrainPerlin_kwargs, #! 根据采样的z_scale和frequency生成noise
            ) / self.cfg.vertical_scale
        
        
        block_starting_height_px = track_origin_px[2]   # origin的起始高度

        # 随机生存wall的厚度
        wall_thickness = np.random.uniform(*self.track_kwargs["wall_thickness"]) if isinstance(self.track_kwargs["wall_thickness"], (tuple, list)) else self.track_kwargs["wall_thickness"]

        #! 生成starting_track
        starting_trimesh, starting_heightfield, block_info, height_offset_px = self.get_starting_track(wall_thickness)  #! start track中的block_info是[0,0]

        self.heightfield_raw[track_origin_px[0]: track_origin_px[0] + self.track_block_resolution[0], track_origin_px[1]: track_origin_px[1] + self.track_block_resolution[1], ] = starting_heightfield
        
        # add noise to starting track
        if "heightfield_noise" in locals():
            self.heightfield_raw[track_origin_px[0]: track_origin_px[0] + self.track_block_resolution[0], track_origin_px[1]: track_origin_px[1] + self.track_block_resolution[1], ] += heightfield_noise[:self.track_block_resolution[0]]
            
            starting_trimesh_noised = convert_heightfield_to_trimesh(
                self.fill_heightfield_to_scale(self.heightfield_raw[track_origin_px[0]: track_origin_px[0] + self.track_block_resolution[0], track_origin_px[1]: track_origin_px[1] + self.track_block_resolution[1], ]),
                self.cfg.horizontal_scale, self.cfg.vertical_scale, self.cfg.slope_treshold, )
            
            self.add_trimesh_to_sim(starting_trimesh_noised,
                np.array([
                    track_origin_px[0] * self.cfg.horizontal_scale,
                    track_origin_px[1] * self.cfg.horizontal_scale,
                    block_starting_height_px * self.cfg.vertical_scale,
                ]))
        else:    
            self.add_trimesh_to_sim(starting_trimesh, np.array([track_origin_px[0] * self.cfg.horizontal_scale, track_origin_px[1] * self.cfg.horizontal_scale, block_starting_height_px * self.cfg.vertical_scale, ]))
        
        # track的第一个block.
        self.track_info_map[row_idx, col_idx, 0, 0] = 0 # id = 0
        self.track_info_map[row_idx, col_idx, 0, 1:] = block_info   # obstacle_depth and obstacle_critical_params
        self.track_width_map[row_idx, col_idx] = self.env_width - wall_thickness * 2    # 之后用来获取info的
        
        block_starting_height_px += height_offset_px # 加上height offset(虽然=0)   
        
        #! 生成不同obstacle
        for obstacle_idx, obstacle_selection in enumerate(obstacle_order):  # 一开始都是0, 0
            
            obstacle_name = self.track_kwargs["options"][obstacle_selection]    # 选取obstable name，一般只有一个
            obstacle_id = self.track_options_id_dict[obstacle_name]     # name 对应的id
            
            track_trimesh, track_heightfield, block_info, height_offset_px = getattr(self, "get_" + obstacle_name + "_track")(  #! create climb track here
                wall_thickness,
                starting_trimesh,   # trimesh template
                starting_heightfield,   # heightfield template
                difficulty = difficulty,
                heightfield_noise= heightfield_noise[self.track_block_resolution[0] * (obstacle_idx + 1): self.track_block_resolution[0] * (obstacle_idx + 2)] if "heightfield_noise" in locals() else None,
                # heightfield_noise = None,
                virtual = virtual_track,
            )
            
            heightfield_x0 = track_origin_px[0] + self.track_block_resolution[0] * (obstacle_idx + 1)   # x是长度，随着obstacle数量是递增的
            heightfield_x1 = track_origin_px[0] + self.track_block_resolution[0] * (obstacle_idx + 2)
            heightfield_y0 = track_origin_px[1] # y是宽度。一直都是这个值
            heightfield_y1 = track_origin_px[1] + self.track_block_resolution[1]
            
            #! add track_height to heightfield_raw
            self.heightfield_raw[heightfield_x0: heightfield_x1, heightfield_y0: heightfield_y1, ] = track_heightfield
            
            self.add_trimesh_to_sim(
                track_trimesh,
                np.array([
                    heightfield_x0 * self.cfg.horizontal_scale,
                    heightfield_y0 * self.cfg.horizontal_scale,
                    block_starting_height_px * self.cfg.vertical_scale,
                ])
            )
            
            # fill info
            self.track_info_map[row_idx, col_idx, obstacle_idx + 1, 0] = obstacle_id
            self.track_info_map[row_idx, col_idx, obstacle_idx + 1, 1:] = block_info
            
            block_starting_height_px += height_offset_px    # 加上新生成地形的高度，为了让每段地形连续

        return block_starting_height_px

        
    def add_plane_to_sim(self, final_height_px= 0.):
        # if self.track_kwargs["add_perlin_noise"] and self.track_kwargs["border_perlin_noise"]:
        
        plane_size_x = self.heightfield_raw.shape[0] * self.cfg.horizontal_scale
        plane_size_y = self.heightfield_raw.shape[1] * self.cfg.horizontal_scale
        
        plane_box_size = np.array([plane_size_x, plane_size_y, 0.02])
        plane_trimesh = trimesh.box_trimesh(plane_box_size, plane_box_size / 2)
        self.add_trimesh_to_sim(plane_trimesh, np.zeros(3))
        
        
        
    '''生成各种track'''
    def get_starting_track(self, wall_thickness):
        
        # ["track_block_length" / self.cfg.horizontal_scale, "track_width" / self.cfg.horizontal_scale]
        track_heighfield_template = np.zeros(self.track_block_resolution, dtype= np.float32)
        
        # add wall height to both side of the track
        track_heighfield_template[:, :np.ceil(wall_thickness / self.cfg.horizontal_scale).astype(int)] += (np.random.uniform(*self.track_kwargs["wall_height"])     # if random sample wall_height
            if isinstance(self.track_kwargs["wall_height"], (tuple, list)) else self.track_kwargs["wall_height"]) / self.cfg.vertical_scale
        track_heighfield_template[:, -np.ceil( wall_thickness / self.cfg.horizontal_scale).astype(int):] += (np.random.uniform(*self.track_kwargs["wall_height"]) 
            if isinstance(self.track_kwargs["wall_height"], (tuple, list)) else self.track_kwargs["wall_height"]) / self.cfg.vertical_scale

        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heighfield_template),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        
        track_heightfield = track_heighfield_template
        
        block_info = torch.tensor([ # obstacle info
            0., # obstacle depth (along x-axis)
            0., # critical parameter for each obstacle
        ], dtype= torch.float32, device= self.device)
        
        height_offset_px = 0    # 平地，offset = 0
        
        return track_trimesh, track_heightfield, block_info, height_offset_px
    
    
    def get_climb_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):
        if isinstance(self.track_kwargs["climb"]["depth"], (tuple, list)):
            if not virtual:
                climb_depth = min(*self.track_kwargs["climb"]["depth"])
            else:
                climb_depth = np.random.uniform(*self.track_kwargs["climb"]["depth"])
        else:
            climb_depth = self.track_kwargs["climb"]["depth"]
            
        if isinstance(self.track_kwargs["climb"]["height"], (tuple, list)):
            if difficulty is None:
                climb_height = np.random.uniform(*self.track_kwargs["climb"]["height"])
            else:
                climb_height = (1-difficulty) * self.track_kwargs["climb"]["height"][0] + difficulty * self.track_kwargs["climb"]["height"][1]
        else:
            climb_height = self.track_kwargs["climb"]["height"]
            
        if self.track_kwargs["climb"].get("climb_down_prob", 0.) > 0.:
            if np.random.uniform() < self.track_kwargs["climb"]["climb_down_prob"]:
                climb_height = -climb_height
                
        depth_px = int(climb_depth / self.cfg.horizontal_scale)
        height_value = climb_height / self.cfg.vertical_scale
        wall_thickness_px = int(wall_thickness / self.cfg.horizontal_scale) + 1
        
        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()
            
        if not virtual and height_value > 0.:
            track_heightfield[1:, wall_thickness_px: -wall_thickness_px, ] += height_value  #! 生成climb height
            
        if height_value < 0.:
            track_heightfield[depth_px:, max(0, wall_thickness_px-1): min(-1, -wall_thickness_px+1), ] += height_value
            
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        
        assert not (self.track_kwargs["climb"].get("fake_offset", 0.) != 0. and self.track_kwargs["climb"].get("fake_height", 0.) != 0.), "fake_offset and fake_height cannot be both non-zero"
        
        climb_height_ = climb_height + (self.track_kwargs["climb"].get("fake_offset", 0.) if self.track_kwargs["climb"].get("fake_offset", 0.) == 0. else self.track_kwargs["climb"].get("fake_height", 0.))
        
        block_info = torch.tensor([
            climb_depth,
            climb_height_,
        ], dtype= torch.float32, device= self.device)
        
        height_offset_px = height_value if not virtual else min(height_value, 0)
        
        return track_trimesh, track_heightfield, block_info, height_offset_px
    
    
    def get_tilt_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):
        tilt_depth = np.random.uniform(*self.track_kwargs["tilt"]["depth"]) if isinstance(self.track_kwargs["tilt"]["depth"], (tuple, list)) else self.track_kwargs["tilt"]["depth"]
        
        tilt_wall_height = np.random.uniform(*self.track_kwargs["tilt"]["wall_height"]) if isinstance(self.track_kwargs["tilt"]["wall_height"], (tuple, list)) else self.track_kwargs["tilt"]["wall_height"]
        
        tilt_opening_angle = np.random.uniform(*self.track_kwargs["tilt"]["opening_angle"]) if isinstance(self.track_kwargs["tilt"].get("opening_angle", 0.), (tuple, list)) else self.track_kwargs["tilt"].get("opening_angle", 0.)
        
        if isinstance(self.track_kwargs["tilt"]["width"], (tuple, list)):
            if difficulty is None:
                tilt_width = np.random.uniform(*self.track_kwargs["tilt"]["width"])
            else:
                tilt_width = difficulty * self.track_kwargs["tilt"]["width"][0] + (1-difficulty) * self.track_kwargs["tilt"]["width"][1]
        else:
            tilt_width = self.track_kwargs["tilt"]["width"]
        
        depth_px = int(tilt_depth / self.cfg.horizontal_scale)
        height_value = tilt_wall_height / self.cfg.vertical_scale
        wall_thickness_px = int(wall_thickness / self.cfg.horizontal_scale) + 1
        wall_gap_px = int(tilt_width / self.cfg.horizontal_scale / 2)

        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()
        
        # If index out of limit error occured, it might because of the too large tilt width
        if virtual:
            pass # no modification on the heightfield (so as the trimesh)
        elif tilt_opening_angle == 0:
            track_heightfield[1: depth_px+1, wall_thickness_px: int(self.track_block_resolution[1] / 2 - wall_gap_px), ] = height_value
            track_heightfield[1: depth_px+1, int(self.track_block_resolution[1] / 2 + wall_gap_px): -wall_thickness_px, ] = height_value
        else:
            for depth_i in range(1, depth_px + 1):
                wall_gap_px_row = wall_gap_px + (depth_px - depth_i) * np.tan(tilt_opening_angle)
                track_heightfield[depth_i, wall_thickness_px: int(self.track_block_resolution[1] / 2 - wall_gap_px_row), ] = height_value
                track_heightfield[depth_i, int(self.track_block_resolution[1] / 2 + wall_gap_px_row): -wall_thickness_px, ] = height_value
                
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        
        block_info = torch.tensor([tilt_depth, tilt_width, ], dtype= torch.float32, device= self.device)
        height_offset_px = 0
        
        return track_trimesh, track_heightfield, block_info, height_offset_px
    
    def get_crawl_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):
        crawl_depth = np.random.uniform(*self.track_kwargs["crawl"]["depth"]) if isinstance(self.track_kwargs["crawl"]["depth"], (tuple, list)) else self.track_kwargs["crawl"]["depth"]
        
        if isinstance(self.track_kwargs["crawl"]["height"], (tuple, list)):
            if difficulty is None:
                crawl_height = np.random.uniform(*self.track_kwargs["crawl"]["height"])
            else:
                crawl_height = difficulty * self.track_kwargs["crawl"]["height"][0] + (1-difficulty) * self.track_kwargs["crawl"]["height"][1]
        else:
            crawl_height = self.track_kwargs["crawl"]["height"]
        
        crawl_wall_height = np.random.uniform(*self.track_kwargs["crawl"]["wall_height"]) if isinstance(self.track_kwargs["crawl"]["wall_height"], (tuple, list)) else self.track_kwargs["crawl"]["wall_height"]
        
        if not heightfield_noise is None:
            if self.track_kwargs["crawl"].get("no_perlin_at_obstacle", False):
                depth_px = int(crawl_depth / self.cfg.horizontal_scale)
                wall_thickness_px = int(wall_thickness / self.cfg.horizontal_scale) + 1
                
                heightfield_template = heightfield_template.copy()
                heightfield_template[1: depth_px+1, :wall_thickness_px, ] += heightfield_noise[1: depth_px+1, :wall_thickness_px]
                heightfield_template[1: depth_px+1, -max(wall_thickness_px, 1):, ] += heightfield_noise[1: depth_px+1, -max(wall_thickness_px, 1):]
                heightfield_template[depth_px+1:, ] += heightfield_noise[depth_px+1:]
            else:
                heightfield_template = heightfield_template + heightfield_noise
                
            trimesh_template = convert_heightfield_to_trimesh(
                self.fill_heightfield_to_scale(heightfield_template),
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold,
            )
        
        upper_bar_trimesh = trimesh.box_trimesh(np.array([crawl_depth, self.track_kwargs["track_width"] - wall_thickness*2, crawl_wall_height, ], dtype= np.float32),
            np.array([crawl_depth / 2, self.track_kwargs["track_width"] / 2, crawl_height + crawl_wall_height / 2, ], dtype= np.float32), )
        
        if not virtual:
            track_trimesh = trimesh.combine_trimeshes(trimesh_template, upper_bar_trimesh, )
        else:
            track_trimesh = trimesh_template
            
        block_info = torch.tensor([
            crawl_depth if self.track_kwargs["crawl"].get("fake_depth", 0.) <= 0 else self.track_kwargs["crawl"]["fake_depth"],
            crawl_height,
        ], dtype= torch.float32, device= self.device)
        height_offset_px = 0
        
        return track_trimesh, heightfield_template, block_info, height_offset_px
    
    def get_leap_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):
        leap_depth = np.random.uniform(*self.track_kwargs["leap"]["depth"]) if isinstance(self.track_kwargs["leap"]["depth"], (tuple, list)) else self.track_kwargs["leap"]["depth"]
        
        if isinstance(self.track_kwargs["leap"]["length"], (tuple, list)):
            if difficulty is None:
                leap_length = np.random.uniform(*self.track_kwargs["leap"]["length"])
            else:
                leap_length = (1-difficulty) * self.track_kwargs["leap"]["length"][0] + difficulty * self.track_kwargs["leap"]["length"][1]
        else:
            leap_length = self.track_kwargs["leap"]["length"]
            
        length_px = int(leap_length / self.cfg.horizontal_scale)
        depth_value = leap_depth / self.cfg.vertical_scale
        wall_thickness_px = int(wall_thickness / self.cfg.horizontal_scale) + 1

        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()
            
        # There is no difference between virtual/non-virtual environment.
        track_heightfield[1: length_px+1, max(0, wall_thickness_px-1): min(-1, -wall_thickness_px+1), ] -= depth_value
        
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        
        block_info = torch.tensor([
            leap_length + self.track_kwargs["leap"].get("fake_offset", 0.), # along x(forward)-axis
            leap_depth, # along z(downward)-axis
        ], dtype= torch.float32, device= self.device)
        height_offset_px = 0
        
        return track_trimesh, track_heightfield, block_info, height_offset_px
    
    def get_slope_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):    
        
        assert virtual == False, "virtual has to be false" 
    
        if isinstance(self.track_kwargs["slope"]["gradient"], (tuple, list)):
            slope_gradient = np.random.uniform(*self.track_kwargs["slope"]["gradient"])
        else:
            slope_gradient = self.track_kwargs["slope"]["gradient"]
            
        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()
        
        length = self.track_block_resolution[0] # track_block_length.80
        width = self.track_block_resolution[1]  # track_width.64
        x = np.arange(0, width)  
        y = np.arange(0, length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        yy = yy.reshape(length, 1)  # （80， 1）  
        max_height = int(slope_gradient * (self.cfg.horizontal_scale / self.cfg.vertical_scale) * length)
        
        track_heightfield[np.arange(length), :] += (max_height * yy / length)
        
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        
        block_info = torch.tensor([
            0,
            slope_gradient,
        ], dtype= torch.float32, device= self.device)
        
        height_offset_px = max_height
        
        return track_trimesh, track_heightfield, block_info, height_offset_px
   
    def get_flat_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):    
        
        assert virtual == False, "virtual has to be false" 
            
        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()
        
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        
        block_info = torch.tensor([
            0,
            0,
        ], dtype= torch.float32, device= self.device)
        
        height_offset_px = 0
        
        return track_trimesh, track_heightfield, block_info, height_offset_px
   
        
    def get_slope_down_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):    
        
        assert virtual == False, "virtual has to be false" 
    
        if isinstance(self.track_kwargs["slope_down"]["gradient"], (tuple, list)):
            slope_gradient = np.random.uniform(*self.track_kwargs["slope_down"]["gradient"])
        else:
            slope_gradient = self.track_kwargs["slope_down"]["gradient"]
            
        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()
        
        length = self.track_block_resolution[0] # track_block_length.80
        width = self.track_block_resolution[1]  # track_width.64
        x = np.arange(0, width)  
        y = np.arange(0, length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        yy = yy.reshape(length, 1)  # （80， 1）  
        max_height = int(slope_gradient * (self.cfg.horizontal_scale / self.cfg.vertical_scale) * length)
        
        track_heightfield[np.arange(length), :] += (max_height * yy / length)
        
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        
        block_info = torch.tensor([
            0,
            slope_gradient,
        ], dtype= torch.float32, device= self.device)
        
        height_offset_px = max_height
        
        return track_trimesh, track_heightfield, block_info, height_offset_px
        
    def get_stairs_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):    
        
        assert virtual == False, "virtual has to be false" 
    
        if isinstance(self.track_kwargs["stairs"]["step_width"], (tuple, list)):
            stairs_step_width = np.random.uniform(*self.track_kwargs["stairs"]["step_width"])
        else:
            stairs_step_width = self.track_kwargs["stairs"]["step_width"]
            
        if isinstance(self.track_kwargs["stairs"]["step_height"], (tuple, list)):
            stairs_step_height = np.random.uniform(*self.track_kwargs["stairs"]["step_height"])
        else:
            stairs_step_height = self.track_kwargs["stairs"]["step_height"]
            
        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()
        
        step_width = int(stairs_step_width / self.cfg.horizontal_scale)
        step_height = int(stairs_step_height / self.cfg.vertical_scale)
        
        num_steps = self.track_block_resolution[0] // step_width  # 对商向下取整
        height =  step_height
        for i in range(num_steps):
            track_heightfield[i * step_width: (i + 1) * step_width, :] += height
            height += step_height
        track_heightfield[num_steps * step_width: self.track_block_resolution[0], :] += height
        
        
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        
        block_info = torch.tensor([
            step_width,
            step_height,
        ], dtype= torch.float32, device= self.device)
        
        height_offset_px = height
        
        return track_trimesh, track_heightfield, block_info, height_offset_px
    
    def get_stairs_down_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):    
        
        assert virtual == False, "virtual has to be false" 
    
        if isinstance(self.track_kwargs["stairs_down"]["step_width"], (tuple, list)):
            stairs_step_width = np.random.uniform(*self.track_kwargs["stairs_down"]["step_width"])
        else:
            stairs_step_width = self.track_kwargs["stairs_down"]["step_width"]
            
        if isinstance(self.track_kwargs["stairs_down"]["step_height"], (tuple, list)):
            stairs_step_height = np.random.uniform(*self.track_kwargs["stairs_down"]["step_height"])
        else:
            stairs_step_height = self.track_kwargs["stairs_down"]["step_height"]
            
        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()
        
        step_width = int(stairs_step_width / self.cfg.horizontal_scale)
        step_height = int(stairs_step_height / self.cfg.vertical_scale)
        
        num_steps = self.track_block_resolution[0] // step_width  # 对商向下取整
        height =  step_height
        for i in range(num_steps):
            track_heightfield[i * step_width: (i + 1) * step_width, :] += height
            height += step_height
        track_heightfield[num_steps * step_width: self.track_block_resolution[0], :] += height
        
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        
        block_info = torch.tensor([
            step_width,
            step_height,
        ], dtype= torch.float32, device= self.device)
        
        height_offset_px = height
        
        return track_trimesh, track_heightfield, block_info, height_offset_px
    
    def get_steppingstone_track(self,
            wall_thickness,
            trimesh_template,
            heightfield_template,
            difficulty= None,
            heightfield_noise= None,
            virtual= False,
        ):    
        
        assert virtual == False, "virtual has to be false" 
    
        if isinstance(self.track_kwargs["steppingstone"]["stone_size"], (tuple, list)):
            steppingstone_stone_size = np.random.uniform(*self.track_kwargs["steppingstone"]["stone_size"])
        else:
            steppingstone_stone_size = self.track_kwargs["steppingstone"]["stone_size"]
            
        if isinstance(self.track_kwargs["steppingstone"]["stone_distance"], (tuple, list)):
            steppingstone_stone_distance = np.random.uniform(*self.track_kwargs["steppingstone"]["stone_distance"])
        else:
            steppingstone_stone_distance = self.track_kwargs["steppingstone"]["stone_distance"]
            
        if isinstance(self.track_kwargs["steppingstone"]["max_height"], (tuple, list)):
            steppingstone_max_height = np.random.uniform(*self.track_kwargs["steppingstone"]["max_height"])
        else:
            steppingstone_max_height = self.track_kwargs["steppingstone"]["max_height"]
            
            
        if not heightfield_noise is None:
            track_heightfield = heightfield_template + heightfield_noise
        else:
            track_heightfield = heightfield_template.copy()
        
        stone_size = int(steppingstone_stone_size / self.cfg.horizontal_scale)
        stone_distance = int(steppingstone_stone_distance / self.cfg.horizontal_scale)
        max_height = int(steppingstone_max_height / self.cfg.vertical_scale)
        height_range = np.arange(-max_height-1, max_height, step=1)
        
        start_x = 0
        start_y = 0
        # track_heightfield[:, :] = int(-5 / self.cfg.vertical_scale) 
        track_heightfield[:, :] -= int(5 / self.cfg.vertical_scale) 
        
        block_length = self.track_block_resolution[0]
        block_width = self.track_block_resolution[1]
        
        while start_y < block_length: 
            stop_y = min(start_y + stone_size, block_length)
            start_x = np.random.randint(0, stone_size)
            stop_x = max(0, start_x - stone_distance)
            
            # track_heightfield[start_y: stop_y, 0: stop_x] = np.random.choice(height_range)
            track_heightfield[start_y: stop_y, 0: stop_x] += np.random.choice(height_range) + int(5 / self.cfg.vertical_scale)

            while start_x < block_width:
                stop_x = min(start_x + stone_size, block_width)
                # track_heightfield[start_y: stop_y, start_x: stop_x] = np.random.choice(height_range)
                track_heightfield[start_y: stop_y, start_x: stop_x] += np.random.choice(height_range) + int(5 / self.cfg.vertical_scale)
                start_x += stone_size + stone_distance
        
            start_y += stone_size + stone_distance
        
        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(track_heightfield),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        
        block_info = torch.tensor([
            stone_size,
            stone_distance,
        ], dtype= torch.float32, device= self.device)
        
        height_offset_px = 0
        
        return track_trimesh, track_heightfield, block_info, height_offset_px
        

    def get_difficulty(self, env_row_idx, env_col_idx):
        difficulty = env_row_idx / (self.cfg.num_rows - 1) if self.cfg.curriculum else None
        virtual_terrain = self.track_kwargs["virtual_terrain"]
        return difficulty, virtual_terrain
        
    def fill_heightfield_to_scale(self, heightfield):
        """ Due to the rasterization of the heightfield, the trimesh size does not match the 
        heightfield_resolution * horizontal_scale, so we need to fill enlarge heightfield to
        meet this scale.
        """
        assert len(heightfield.shape) == 2, "heightfield must be 2D"
        heightfield_x_fill = np.concatenate([heightfield, heightfield[-2:, :], ], axis= 0)
        heightfield_y_fill = np.concatenate([heightfield_x_fill, heightfield_x_fill[:, -2:], ], axis= 1)
        return heightfield_y_fill
        
    def add_trimesh_to_sim(self, trimesh, trimesh_origin):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = trimesh[0].shape[0]
        tm_params.nb_triangles = trimesh[1].shape[0]
        tm_params.transform.p.x = trimesh_origin[0]
        tm_params.transform.p.y = trimesh_origin[1]
        tm_params.transform.p.z = trimesh_origin[2]
        tm_params.static_friction = self.cfg.static_friction
        tm_params.dynamic_friction = self.cfg.dynamic_friction
        tm_params.restitution = self.cfg.restitution
        self.gym.add_triangle_mesh(self.sim, trimesh[0].flatten(order= "C"), trimesh[1].flatten(order= "C"), tm_params, )
        
        