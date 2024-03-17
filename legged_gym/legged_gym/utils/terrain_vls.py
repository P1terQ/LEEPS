# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from scipy.ndimage import binary_dilation
import pyfqmr
from isaacgym import gymutil, gymapi

from isaacgym import terrain_utils
from legged_gym.envs.legged.legged_v2_config import LeggedV2Cfg
from .trimesh import box_trimesh, combine_trimeshes

class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)
        
        #! dualobs
        # self.ceiling_height_raw = np.ones((self.width, self.length), dtype=np.int16) / vertical_scale
        self.ceiling_height_raw = np.zeros((self.width, self.length), dtype=np.int16)

class Terrainvls:
    def __init__(self, cfg: LeggedV2Cfg.terrain, num_robots, gym, sim) -> None:
        self.gym = gym
        self.sim = sim

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        
        if self.type in ["none", 'plane']:
            return
        
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        
        # 将比例加起来用于判断
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]     

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)  # 80
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)    # 360

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.goal = np.zeros((cfg.num_rows, cfg.num_cols, 3)) 

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        
        #! dual obs        
        self.ceiling_height_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        # self.ceiling_height_raw = np.ones((self.tot_rows , self.tot_cols), dtype=np.int16) / self.cfg.vertical_scale

        
        if cfg.curriculum:  # curriculum会按照行数来增加难度
            self.curiculum()
            
        elif cfg.selected:
            self.selected_terrain()
            
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        self.ceilingsamples = self.ceiling_height_raw   # int16
        self.ceilingsamples = self.ceilingsamples.astype(np.int16)
        
        #todo 之后用vision based student policy的时候加上simplified trimesh
        if self.type=="trimesh":
            print("Heightmap creation finished. Converting heightmap to trimesh...")
            
            # self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(   self.height_field_raw,  # 整个地图的height field
            #                                                                                 self.cfg.horizontal_scale,
            #                                                                                 self.cfg.vertical_scale,
            #                                                                                 self.cfg.slope_treshold)            
            # half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
            # structure = np.ones((half_edge_width*2+1, 1))
            # self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)   #! x方向上的边缘(x方向上height的变化大于slope_threshold)
            
            # 地面height field
            self.vertices, self.triangles, self.x_edge_mask, self.y_edge_mask = convert_heightfield_to_trimesh(   
                                                                                            self.height_field_raw,  # 整个地图的height field
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)            
            half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
            structure_x = np.ones((half_edge_width*2+1, 1))   # 用于膨胀的结构
            self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure_x) 
            structure_y = np.ones((1, half_edge_width*2+1))
            self.y_edge_mask = binary_dilation(self.y_edge_mask, structure=structure_y)
            # print(self.x_edge_mask.sum())
            # print(self.y_edge_mask.sum())
            
            # 天花板height_field
            self.vertices_ceiling, self.triangles_ceiling, self.x_edge_mask_ceiling, self.y_edge_mask_ceiling = convert_heightfield_to_trimesh(   
                                                                                            self.ceiling_height_raw,  # 整个地图的height field
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)  
            
           
            if self.cfg.simplify_grid:
                mesh_simplifier = pyfqmr.Simplify()
                mesh_simplifier.setMesh(self.vertices, self.triangles)
                mesh_simplifier.simplify_mesh(target_count = int(0.05*self.triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10)

                self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                self.vertices = self.vertices.astype(np.float32)
                self.triangles = self.triangles.astype(np.uint32)
            
            
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
        self.gym.add_triangle_mesh(
            self.sim,
            trimesh[0].flatten(order= "C"),
            trimesh[1].flatten(order= "C"),
            tm_params,
        )
            
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])
            difficulty = 0.5
            terrain = self.make_terrain(choice, difficulty, i, j)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        # dif_set = [0.1, 0.5, 1.0]
        
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                # difficulty = dif_set[i]
                
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty, i, j)   # num_row num_col
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        if self.cfg.num_sub_terrains == 1:
            terrain_type = self.cfg.terrain_kwargs.pop('type')
            for k in range(self.cfg.num_sub_terrains):
                # Env coordinates in the world
                (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

                terrain = SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.vertical_scale,
                                horizontal_scale=self.horizontal_scale)

                eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
                self.add_terrain_to_map(terrain, i, j)
        else:
            terrain_list = self.cfg.terrain_kwargs
            for k in range(self.cfg.num_sub_terrains):
                assert len(terrain_list) == self.cfg.num_sub_terrains 
                
                # Env coordinates in the world
                (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

                terrain = SubTerrain("terrain",
                                                   width=self.width_per_env_pixels,
                                                   length=self.length_per_env_pixels,
                                                   vertical_scale=self.cfg.vertical_scale,
                                                   horizontal_scale=self.cfg.horizontal_scale)
 
                terrain_type = terrain_list[int(k)].pop('type')
                eval(terrain_type)(terrain, **terrain_list[int(k)])

                self.add_terrain_to_map(terrain, i, j)
                
    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)

    
    def make_terrain(self, choice, difficulty, num_row_, num_col_):
        
        terrain = SubTerrain("terrain",
                            width=self.length_per_env_pixels,
                            length=self.width_per_env_pixels,
                            vertical_scale=self.cfg.vertical_scale,
                            horizontal_scale=self.cfg.horizontal_scale)
        
        #! test mode: set all subterrain difficulty to 1
        # difficulty = 1.0
        
        if choice < self.proportions[0]:
            idx = 1
            
            # if difficulty < 0.5:
            #     num_stones = 1
            # else:
            #     num_stones = 2

            self.step_terrain_goal(terrain,
                        num_stones=1,
                        # step_height= 0.5*difficulty,    # 之前是0.5再加一个难度
                        step_height= 0.7*difficulty,
                        # step_height= 0.5,  
                        x_range=[1.0,2.0],
                        )
            # terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)

            
        elif choice < self.proportions[1]:
            idx = 2
            
            if difficulty == 0:
                num_gaP = 0
            elif difficulty<0.5:
                num_gaP = 1
            else:
                num_gaP = 2
  
            #! 训完这个的depth policy感觉platform_length应该大一点，不然第一个gap来的太快，根本看不见
            self.gap_terrain_goal(terrain,
                            num_gaps=num_gaP,
                            # gap_size=0.1 + 0.9 * difficulty,    # 降低初始的gap_size
                            gap_size=0.1 + 1.0 * difficulty,
                            # gap_size=0.8,
                            gap_depth=[0.5, 1],
                            pad_height=0,
                            x_range=[0.8, 1.5],
                            # y_range=[-0.4*difficulty, 0.4*difficulty],    # distill depth的时候terrain horizonal scale是0.1,所以这边要改一下
                            y_range=[-0.3*difficulty - 0.1, 0.3*difficulty + 0.1],
                            half_valid_width=[1.0-0.6*difficulty, 1.2-0.6*difficulty],  # 减少平台的宽度，强调落点的准确性
                             )


        elif choice < self.proportions[2]:
            idx = 3
            self.slope_terrain_goal(terrain,
                          slope=1.3*difficulty)
            
        elif choice < self.proportions[3]:
            idx = 4
            self.stairs_terrain_goal(terrain,
                           step_width=0.5-0.25*difficulty,
                            step_height=0.12*difficulty,
            )
            

        elif choice < self.proportions[4]:
            idx = 5
            
            self.discrete_terrain_goal(terrain,
                                    max_height=0.15 * difficulty,
                                    min_size=1.,
                                    max_size=3.,
                                    num_rects=20*difficulty
                                )
            self.add_roughness(terrain, difficulty=difficulty)

        elif choice < self.proportions[5]:
            idx = 6
            
            self.flat_terrain_goal(terrain)
            self.add_roughness(terrain, difficulty=difficulty)
            

        elif choice < self.proportions[6]:
            idx = 7
            
            flat_ = False
            if difficulty == 0:
                flat_ = True
            
            self.stepping_stones_terrain_goal(terrain, 
                                              stone_size=0.8-0.6*difficulty, 
                                              stone_distance=0.05+0.15*difficulty,
                                            #   stone_distance=0.05+0.05*difficulty,
                                              max_height=0.02*difficulty,
                                              if_flat=flat_)
            
        elif choice < self.proportions[7]: 
            idx = 8

            flat_ = False
            if difficulty == 0:
                flat_ = True
                
            self.crawl_terrain_goal(terrain,
                                    num_row=num_row_,
                                    num_col=num_col_,
                                    crawl_length = 0.5 + 2.5*difficulty,
                                    # crawl_height = 0.45 - 0.25*difficulty,
                                    # crawl_height=0.2,
                                    crawl_height = 0.42 - 0.25*difficulty,  # 0.17
                                    flat=flat_
                               )
            
            
        elif choice < self.proportions[8]:  #! use this for push box
            idx = 9
            
            if_flat = False
            if difficulty == 0:
                if_flat = True
            
            self.log_terrain_goal(terrain,
                            log_length_range = [0.8+1.5*difficulty, 1.3+1.5*difficulty],
                            # log_width = 0.2,
                            log_width = 0.6 - 0.4 * difficulty,
                            flat=if_flat
                             )
            

        elif choice < self.proportions[9]:  #! use this for push box
            idx = 10
            
            if_flat = False
            if difficulty == 0:
                if_flat = True
            
            self.crack_terrain_goal(terrain,
                            log_length_range = [1.0+2.0*difficulty, 1.5+2.0*difficulty],
                            log_width = 0.4 - 0.1*difficulty,
                            flat=if_flat
                             )


        elif choice < self.proportions[10]:
            idx = 11
            
            if_flat = False
            if difficulty == 0:
                if_flat = True
            
            self.dualobs_terrain_goal(terrain, 
                                    num_row=num_row_,
                                    num_col=num_col_,
                                    terrain_stone_distance = [0.8-0.5*difficulty, 1.3-0.5*difficulty],
                                    air_stone_distance = [1.0-0.5*difficulty, 2.0-0.5*difficulty],
                                    if_flat_=if_flat
                                       )
            # self.add_roughness(terrain, difficulty=0.1)
            
        elif choice < self.proportions[11]:
            idx = 12
            if_flat = False
            if difficulty == 0:
                if_flat = True
                
            # self.parkour_terrain_goal(terrain,
            #                           step_height=0.5*difficulty,
            #                           log_width=0.6 - 0.35 * difficulty,
            #                           steppingstone_width=0.8-0.6*difficulty,
            #                           steppingstone_apart=0.15*difficulty,
            #                           steppingstone_height=0.02*difficulty,
            #                           slope_incliniation = -0.4*difficulty,
            #                           gap_width=0.8*difficulty,
            #                           if_flat_=if_flat)
            self.parkour_terrain_goal(terrain,
                                      step_height=0.7*difficulty,
                                      log_width=0.6 - 0.4 * difficulty,
                                      steppingstone_width=0.8-0.6*difficulty,
                                      steppingstone_apart=0.2*difficulty,
                                      steppingstone_height=0.02*difficulty,
                                      slope_incliniation = -0.4*difficulty,
                                      gap_width=1.0*difficulty,
                                      if_flat_=if_flat)
            
            
        terrain.idx = idx
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw
        
        # self.ceiling_height_raw[start_x: end_x, start_y:end_y] = terrain.ceiling_height_raw
        # self.ceiling_height_raw[i * self.length_per_env_pixels: (i + 1) * self.length_per_env_pixels, j * self.width_per_env_pixels:(j + 1) * self.width_per_env_pixels] = terrain.ceiling_height_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale        

        if terrain.idx == 6: # flat
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z] 
        else:
            self.env_origins[i, j] = [i * self.env_length + 0.7, (j + 0.5) * self.env_width, env_origin_z]  # 0.7
            self.goal[i,j,0] = terrain.goal[0] + i * self.env_length
            self.goal[i,j,1] = terrain.goal[1] + j * self.env_width
        
        self.terrain_type[i, j] = terrain.idx
        
        
    def dualobs_terrain_goal(self, 
                            terrain, 
                            num_row,
                            num_col,
                            terrain_stone_distance,
                            air_stone_distance,
                            # height_range = [0.2, 0.6],
                            terrain_stone_size = [0.2,0.4],
                            terrain_height_range = [0.1, 0.25],   # 障碍物的高度
                            air_stone_size = [0.35,0.55],  # 每个障碍物的size
                            ceiling_height_range = [0.25, 0.35],   # 障碍物的高度
                            platform_size = 2.0,
                            if_flat_ = False
                            ):

        if not if_flat_:

            stone_size_min = int(terrain_stone_size[0] / terrain.horizontal_scale)
            stone_size_max = int(terrain_stone_size[1] / terrain.horizontal_scale)
            # stone_distance = [0.5,1.5]  # 两个障碍物之间的距离
            stone_distance_min = int(terrain_stone_distance[0] / terrain.horizontal_scale)
            stone_distance_max = int(terrain_stone_distance[1] / terrain.horizontal_scale)
            
            height_range_min = int(terrain_height_range[0] / terrain.vertical_scale)
            height_range_max = int(terrain_height_range[1] / terrain.vertical_scale)
            
            x_rand = [-0.1, 0.1]
            x_rand_min = round( x_rand[0] / terrain.horizontal_scale)
            x_rand_max = round( x_rand[1] / terrain.horizontal_scale)
            
            platform_size = int(platform_size / terrain.horizontal_scale)
            
            #! 生成terrain obstacle

            start_x = platform_size # 上方obstacle开始位置
            obstacle_length = round(3 / terrain.horizontal_scale)   # obstacle的总长
            end_x = start_x + obstacle_length   # 上方obstacle结束位置
            start_y = np.random.randint(stone_distance_min, stone_distance_max)
            
            while start_x < end_x:
                stone_size = np.random.randint(stone_size_min, stone_size_max)
                stone_distance = np.random.randint(stone_distance_min, stone_distance_max)
                height = np.random.randint(height_range_min, height_range_max)
                
                stop_x = min(terrain.width, start_x + stone_size)
                start_y = stone_distance
                stop_y = start_y + stone_size
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
                start_y = stop_y
                
                while start_y < terrain.length - (stone_size_max+stone_distance):    # 固定x,递增y
                    stone_size = np.random.randint(stone_size_min, stone_size_max)
                    stone_distance = np.random.randint(stone_distance_min, stone_distance_max)
                    height = np.random.randint(height_range_min, height_range_max)
                    
                    obstacle_rand_x = np.random.randint(x_rand_min, x_rand_max)
                    obstacle_start_x = start_x + obstacle_rand_x
                    obstacle_end_x = obstacle_start_x + stone_size
                    
                    start_y += stone_distance
                    stop_y = start_y + stone_size
                    terrain.height_field_raw[obstacle_start_x: obstacle_end_x, start_y: stop_y] = height
                    start_y = stop_y
                    
                    
                start_x += stone_size + stone_distance  # 再加到下一行
            
            #! 生层ceiling obstacle  
            height_range_min = int(ceiling_height_range[0] / terrain.vertical_scale)
            height_range_max = int(ceiling_height_range[1] / terrain.vertical_scale)
            
            stone_size_min = int(air_stone_size[0] / terrain.horizontal_scale)
            stone_size_max = int(air_stone_size[1] / terrain.horizontal_scale)
            
            stone_distance_min = int(air_stone_distance[0] / terrain.horizontal_scale)
            stone_distance_max = int(air_stone_distance[1] / terrain.horizontal_scale)
            
            # subterrain的坐标
            env_start_x = num_row * self.length_per_env_pixels
            # end_x = (num_row + 1) * self.length_per_env_pixels
            env_start_y = num_col * self.width_per_env_pixels
            env_end_y = (num_col + 1) * self.width_per_env_pixels
            
            obs_start_x = platform_size # 上方obstacle开始位置
            obstacle_length = round(3 / terrain.horizontal_scale)
            obs_end_x = obs_start_x + obstacle_length   # 上方obstacle结束位置
            obs_start_y = np.random.randint(stone_distance_min, stone_distance_max)
            
            while obs_start_x < obs_end_x:
                stone_size = np.random.randint(stone_size_min, stone_size_max)
                stone_distance = np.random.randint(stone_distance_min, stone_distance_max)
                obs_height = np.random.randint(height_range_min, height_range_max)
                
                obs_stop_x = min(terrain.width, obs_start_x + stone_size)
                obs_start_y = np.random.randint(0,stone_distance_min)
                obs_stop_y = obs_start_y + stone_size
                
                #! 把trimsh加到sim
                # terrain.ceiling_height_raw[start_x: stop_x, start_y: stop_y] = height
                upper_trimesh = box_trimesh(
                    np.array([stone_size*terrain.horizontal_scale, stone_size*terrain.horizontal_scale, stone_size*terrain.horizontal_scale], dtype=np.float32),    # size
                    np.array([(obs_start_x+obs_stop_x)/2*terrain.horizontal_scale, (obs_start_y+obs_stop_y)/2*terrain.horizontal_scale, obs_height*terrain.vertical_scale+stone_size/2*terrain.horizontal_scale], dtype=np.float32)  # pos
                )
                self.add_trimesh_to_sim(
                    upper_trimesh,
                    np.array([env_start_x * terrain.horizontal_scale,env_start_y * terrain.horizontal_scale,0]))
                
                #! add to map
                record_start_x = int(self.border + env_start_x + obs_start_x)
                record_stop_x = int(self.border + env_start_x + obs_stop_x)
                record_start_y = int(self.border + env_start_y + obs_start_y)
                record_stop_y = int(self.border + env_start_y + obs_stop_y)
                self.ceiling_height_raw[ record_start_x: record_stop_x, record_start_y:record_stop_y] = obs_height
                
                obs_start_y = obs_stop_y
                
                while obs_start_y < terrain.length - stone_distance_max:    #! 固定obs_start_x， 递增obs_start_y
                # while start_y < terrain.length - (stone_size_max+stone_distance):
                # while obs_start_y < terrain.length:
                    stone_size = np.random.randint(stone_size_min, stone_size_max)
                    stone_distance = np.random.randint(stone_distance_min, stone_distance_max)
                    obs_height = np.random.randint(height_range_min, height_range_max)
                    
                    obstacle_rand_x = np.random.randint(x_rand_min, x_rand_max)
                    obstacle_start_x = obs_start_x + obstacle_rand_x
                    obstacle_stop_x = obstacle_start_x + stone_size
                    
                    obs_start_y += stone_distance
                    obs_stop_y = obs_start_y + stone_size
                    # terrain.ceiling_height_raw[obstacle_start_x: obstacle_end_x, start_y: stop_y] = height

                    upper_trimesh = box_trimesh(
                        np.array([stone_size*terrain.horizontal_scale, stone_size*terrain.horizontal_scale, stone_size*terrain.horizontal_scale], dtype=np.float32),    # size
                        np.array([(obstacle_start_x+obstacle_stop_x)/2*terrain.horizontal_scale, (obs_start_y+obs_stop_y)/2*terrain.horizontal_scale, obs_height*terrain.vertical_scale+stone_size/2*terrain.horizontal_scale], dtype=np.float32)  # pos
                    )
                    self.add_trimesh_to_sim(
                        upper_trimesh,
                        np.array([env_start_x * terrain.horizontal_scale,env_start_y * terrain.horizontal_scale,0]))
                    
                    #! add to map
                    record_start_x = int(self.border + env_start_x + obstacle_start_x)
                    record_stop_x = int(self.border + env_start_x + obstacle_stop_x)
                    record_start_y = int(self.border + env_start_y + obs_start_y)
                    record_stop_y = int(self.border + env_start_y + obs_stop_y)
                    self.ceiling_height_raw[ record_start_x: record_stop_x, record_start_y:record_stop_y] = obs_height
                    
                    obs_start_y = obs_stop_y
                    
                obs_start_x += stone_size + stone_distance  # 再加到下一行
            
            
        goal = [terrain.width * terrain.horizontal_scale - 1.0 , terrain.length * 1/2 * terrain.horizontal_scale]
        terrain.goal = goal
        
        return terrain
        
        
    def crawl_terrain_goal(self,
                        terrain,
                        num_row,
                        num_col,
                        crawl_length = 2.0,
                        crawl_width_proportion = [0.7, 1.0],
                        crawl_height = 0.4,
                        crawl_wall_height = [0.3, 1.0],
                        platform_length = 1.0,
                        flat = False
                        ):   
        
        if not flat:   
            
            x_range=[0.5, 1.0]
            xrand_min = round( x_range[0] / terrain.horizontal_scale)
            xrand_max = round( x_range[1] / terrain.horizontal_scale)
            x_rand = np.random.randint(xrand_min, xrand_max)
              
            start_x = num_row * self.length_per_env_pixels
            # end_x = (num_row + 1) * self.length_per_env_pixels
            start_y = num_col * self.width_per_env_pixels
            end_y = (num_col + 1) * self.width_per_env_pixels
            
            crawl_width_proportion = np.random.uniform(crawl_width_proportion[0], crawl_width_proportion[1])
            crawl_wall_height_ = np.random.uniform(crawl_wall_height[0], crawl_wall_height[1])

            crawl_width = round(terrain.length*terrain.horizontal_scale*crawl_width_proportion)

            upper_trimesh = box_trimesh(
                np.array([crawl_length, crawl_width, crawl_wall_height_], dtype=np.float32),    # size
                np.array([platform_length + x_rand*terrain.horizontal_scale + crawl_length/2, 0, crawl_height + crawl_wall_height_/2], dtype=np.float32)  # pos
            )
            
            self.add_trimesh_to_sim(
                upper_trimesh,
                np.array([start_x * terrain.horizontal_scale,(start_y+end_y)/2 * terrain.horizontal_scale,0]))
            
            obstacle_start_x = int(self.border + start_x + (platform_length)/terrain.horizontal_scale + x_rand) # 单位是像素
            obstacle_end_x = int(self.border + start_x + (platform_length)/terrain.horizontal_scale + x_rand + crawl_length/terrain.horizontal_scale)
            obstacle_start_y = int(self.border + (start_y+end_y)/2 - crawl_width/2/terrain.horizontal_scale)
            obstacle_end_y = int(self.border + (start_y+end_y)/2 + crawl_width/2/terrain.horizontal_scale)
            
            self.ceiling_height_raw[ obstacle_start_x: obstacle_end_x, 
                                obstacle_start_y:obstacle_end_y] = crawl_height / terrain.vertical_scale
            # self.ceiling_height_raw[ obstacle_start_x: obstacle_end_x+3, 
            #                     obstacle_start_y:obstacle_end_y] = crawl_height / terrain.vertical_scale            
            
            # upper_trimesh = box_trimesh(
            #     np.array([crawl_length, crawl_width, crawl_wall_height_], dtype=np.float32),    # size
            #     np.array([platform_length + crawl_length/2, 0, (crawl_height+0.1) + crawl_wall_height_/2], dtype=np.float32)  # pos
            # )
            
            # self.add_trimesh_to_sim(
            #     upper_trimesh,
            #     np.array([start_x * terrain.horizontal_scale,(start_y+end_y)/2 * terrain.horizontal_scale,0]))
            
            # obstacle_start_x = int(self.border + start_x + platform_length/terrain.horizontal_scale)
            # obstacle_end_x = int(self.border + start_x + platform_length/terrain.horizontal_scale + crawl_length/terrain.horizontal_scale)
            # obstacle_start_y = int(self.border + (start_y+end_y)/2 - crawl_width/2/terrain.horizontal_scale)
            # obstacle_end_y = int(self.border + (start_y+end_y)/2 + crawl_width/2/terrain.horizontal_scale)
            
            # self.ceiling_height_raw[ obstacle_start_x: obstacle_end_x, 
            #                     obstacle_start_y:obstacle_end_y] = (crawl_height+0.1) / terrain.vertical_scale
            
            # another_upper_trimesh = box_trimesh(
            #     np.array([crawl_length/2, crawl_width, crawl_wall_height_], dtype=np.float32),    # size
            #     np.array([platform_length + crawl_length + crawl_length/4 , 0, crawl_height+0.05 + crawl_wall_height_/2], dtype=np.float32)  # pos
            # )
            
            # self.add_trimesh_to_sim(
            #     another_upper_trimesh,
            #     np.array([start_x * terrain.horizontal_scale,(start_y+end_y)/2 * terrain.horizontal_scale,0]))
            
            # obstacle_start_x = int(self.border + start_x + platform_length/terrain.horizontal_scale + crawl_length/terrain.horizontal_scale)
            # obstacle_end_x = int(self.border + start_x + platform_length/terrain.horizontal_scale + round(1.5 * crawl_length/terrain.horizontal_scale))
            # obstacle_start_y = int(self.border + (start_y+end_y)/2 - crawl_width/2/terrain.horizontal_scale)
            # obstacle_end_y = int(self.border + (start_y+end_y)/2 + crawl_width/2/terrain.horizontal_scale)
            
            # self.ceiling_height_raw[ obstacle_start_x: obstacle_end_x, 
            #                     obstacle_start_y:obstacle_end_y] = (crawl_height+0.05) / terrain.vertical_scale
            
        goal = [terrain.width * terrain.horizontal_scale - 1.0 , terrain.length * 1/2 * terrain.horizontal_scale]
        terrain.goal = goal
            
        #* pad edges
        pad_width = int(0.1 // terrain.horizontal_scale)
        # pad_height = int(0.5 // terrain.vertical_scale)
        terrain.height_field_raw[:, :pad_width] = 1000
        # terrain.height_field_raw[:, -pad_width:] = 1000
        # terrain.height_field_raw[:pad_width, :] = pad_height
        # terrain.height_field_raw[-pad_width:, :] = 1000
        
        return terrain
    
    
    def parkour_terrain_goal(self,
                            terrain,
                            step_height,
                            log_width,
                            steppingstone_width,
                            steppingstone_apart,
                            steppingstone_height,
                            gap_width,
                            slope_incliniation,
                            if_flat_=False,
                            platform_len = 1.5
                            
                            
                            ):
        
        step_height_ = int(step_height / terrain.vertical_scale)
        half_log_width_ = round(log_width / 2 / terrain.horizontal_scale)
        stone_size = int(steppingstone_width / terrain.horizontal_scale)
        stone_distance = int(steppingstone_apart / terrain.horizontal_scale)
        stone_height = int(steppingstone_height / terrain.vertical_scale)
        gap_width_ = int(gap_width / terrain.horizontal_scale)
        
        platform_len_ = int(platform_len / terrain.horizontal_scale)
        
        #! 生成step 
        step_width_ = int(0.7 / terrain.horizontal_scale)
        terrain.height_field_raw[platform_len_:platform_len_+step_width_,] = step_height_
        x_ = platform_len_+step_width_
        
        #! 生成独木桥
        mid_y = terrain.length // 2 
        log_length_ = int(1.0/ terrain.horizontal_scale)
        
        terrain.height_field_raw[x_:x_+log_length_,:mid_y-half_log_width_] = -200
        terrain.height_field_raw[x_:x_+log_length_,mid_y+half_log_width_:] = -200
        terrain.height_field_raw[x_:x_+log_length_, mid_y-half_log_width_:mid_y+half_log_width_] = step_height_
        x_ = x_+log_length_
        
        #! 生成平台
        # table_length_ = int(0.1 / terrain.horizontal_scale)
        # terrain.height_field_raw[x_:x_+table_length_,] = step_height_
        # x_ = x_+table_length_
        
        #! 梅花桩
        height_range = np.arange(-stone_height-1, stone_height, step=1) + step_height_
        start_x = x_
        end_x = start_x + int(0.9 / terrain.horizontal_scale)
        start_y = 0
        terrain.height_field_raw[start_x:end_x, :] = int(-10 / terrain.vertical_scale)
        while start_x < end_x:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
        
            while start_y < terrain.length: # 先填一行
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance  # next stepping stone
            start_x += stone_size + stone_distance  # 再加到下一行
            
        x_ = end_x
        
        #! 下坡
        start_x = x_
        end_x = start_x + int(0.6 / terrain.horizontal_scale)
        length = end_x - start_x
        
        x = np.arange(0, length)
        y = np.arange(0, terrain.length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(length, 1)
        
        max_height = int(slope_incliniation * (terrain.horizontal_scale / terrain.vertical_scale) * length) 
        # terrain.height_field_raw[start_x:end_x,:] =step_height_ + (max_height * xx).astype(terrain.height_field_raw.dtype)
        terrain.height_field_raw[start_x:end_x,:] = (max_height * xx / length).astype(terrain.height_field_raw.dtype) + step_height_
        x_ = end_x
        
        #! 把两边的高度都消掉
        half_width_ = int(0.7 / terrain.horizontal_scale)
        terrain.height_field_raw[platform_len_:x_,:mid_y-half_width_] = -200
        terrain.height_field_raw[platform_len_:x_,mid_y+half_width_:] = -200
        
        #! gap
        # gap_width_
        terrain.height_field_raw[x_:x_+gap_width_,] = -200
        
        
        
        
        goal = [terrain.width * terrain.horizontal_scale - 1.0 , terrain.length * 1/2 * terrain.horizontal_scale]
        terrain.goal = goal
        


    def stepping_stones_terrain_goal(self, 
                                terrain, 
                                stone_size, 
                                stone_distance, 
                                max_height, 
                                platform_size=1.5, 
                                depth=-10,
                                if_flat=False):

        # switch parameters to discrete units
        stone_size = int(stone_size / terrain.horizontal_scale)
        stone_distance = int(stone_distance / terrain.horizontal_scale)
        max_height = int(max_height / terrain.vertical_scale)
        height_range = np.arange(-max_height-1, max_height, step=1)
        platform_size = int(platform_size / terrain.horizontal_scale)
        
        x_range=[0.5, 1.0]
        xrand_min = round( x_range[0] / terrain.horizontal_scale)
        xrand_max = round( x_range[1] / terrain.horizontal_scale)
        
        obstacle_length = [1.8,2.5]
        obstacle_length_min = round(obstacle_length[0] / terrain.horizontal_scale)
        obstacle_length_max = round(obstacle_length[1] / terrain.horizontal_scale)

        if not if_flat:

            start_x = platform_size + np.random.randint(xrand_min, xrand_max)
            end_x = start_x + np.random.randint(obstacle_length_min, obstacle_length_max)
            start_y = 0
            terrain.height_field_raw[start_x:end_x, :] = int(depth / terrain.vertical_scale)    # 一开始是形成洼地
        
            # while start_x < terrain.width - platform_size:
            while start_x < end_x:
                
                stop_x = min(terrain.width, start_x + stone_size)
                start_y = np.random.randint(0, stone_size)
                # fill first hole
                stop_y = max(0, start_y - stone_distance)
                terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
                
                while start_y < terrain.length: # 先填一行
                    stop_y = min(terrain.length, start_y + stone_size)
                    terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                    start_y += stone_size + stone_distance  # next stepping stone
                start_x += stone_size + stone_distance  # 再加到下一行


            # terrain.height_field_raw[:start_x, :] = 0
            # terrain.height_field_raw[-platform_size:, :] = 0
            # terrain.height_field_raw[end_x:, :] = 0
        
        goal = [terrain.width * terrain.horizontal_scale - 1.0 , terrain.length * 1/2 * terrain.horizontal_scale]
        terrain.goal = goal
            
        #* pad edges
        # pad_width = int(0.1 // terrain.horizontal_scale)
        # pad_height = int(0.5 // terrain.vertical_scale)
        # terrain.height_field_raw[:, :pad_width] = pad_height
        # terrain.height_field_raw[:, -pad_width:] = pad_height
        # terrain.height_field_raw[:pad_width, :] = pad_height
        # terrain.height_field_raw[-pad_width:, :] = pad_height
        
        return terrain

    def log_terrain_goal(self,
                        terrain,
                        platform_len=2.0, 

                        log_length_range=[1.0, 1.5],   
                        log_width = 0.2,
                        y_rand_range=[-0.5, 0.5],

                        gap_depth=-200,
                        
                        pad_width=0.1,
                        pad_height=0.5,
                        flat=False):

        if not flat:

            # 中间位置
            mid_y = terrain.length // 2  # length is actually y width
            platform_len = round(platform_len / terrain.horizontal_scale)
            
            log_length_min = round(log_length_range[0] / terrain.horizontal_scale)
            log_length_max = round(log_length_range[1] / terrain.horizontal_scale)
            half_log_width = round(log_width / 2 / terrain.horizontal_scale)
            y_rand_min = round(y_rand_range[0] / terrain.horizontal_scale)
            y_rand_max = round(y_rand_range[1] / terrain.horizontal_scale)
            
            x_range=[0.5, 1.5]
            xrand_min = round( x_range[0] / terrain.horizontal_scale)
            xrand_max = round( x_range[1] / terrain.horizontal_scale)
            x_rand = np.random.randint(xrand_min, xrand_max)
            
            log_length_ = np.random.randint(log_length_min, log_length_max)
            y_rand_ = np.random.randint(y_rand_min, y_rand_max)
            
            terrain.height_field_raw[platform_len+x_rand:platform_len+x_rand+log_length_,:mid_y+y_rand_-half_log_width] = gap_depth
            terrain.height_field_raw[platform_len+x_rand:platform_len+x_rand+log_length_,mid_y+y_rand_+half_log_width:] = gap_depth
            
            # platform
            # terrain.height_field_raw[0:platform_len, :] = 0
            # terrain.height_field_raw[-platform_len:, :] = 0
        
            
        goal = np.zeros((1,2))
        goal = [terrain.width * terrain.horizontal_scale - 1.0 , terrain.length * 1/2 * terrain.horizontal_scale]
        terrain.goal = goal 
        
        
        # pad edges, 把边缘(width=0.1)的地方都填充为pad_height
        # pad_width = int(pad_width // terrain.horizontal_scale)
        # pad_height = int(pad_height // terrain.vertical_scale)
        # terrain.height_field_raw[:, :pad_width] = pad_height
        # terrain.height_field_raw[:, -pad_width:] = pad_height
        # terrain.height_field_raw[:pad_width, :] = pad_height
        # terrain.height_field_raw[-pad_width:, :] = pad_height


    def discrete_terrain_goal(self,
                            terrain,
                            max_height,
                            min_size,
                            max_size,
                            num_rects,
                            platform_size=1.5):   

            
        max_height = int(max_height / terrain.vertical_scale)
        min_size = int(min_size / terrain.horizontal_scale)
        max_size = int(max_size / terrain.horizontal_scale)

        (i, j) = terrain.height_field_raw.shape
        height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
        width_range = range(min_size, max_size, 4)
        length_range = range(min_size, max_size, 4)

        for _ in range(int(num_rects)):
            width = np.random.choice(width_range)
            length = np.random.choice(length_range)
            start_i = np.random.choice(range(0, i-width, 4))
            start_j = np.random.choice(range(0, j-length, 4))
            terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)            

        goal = [terrain.width * terrain.horizontal_scale - 1.0 , terrain.length * 1/2 * terrain.horizontal_scale]
        terrain.goal = goal
        
        platform_len = round(platform_size / terrain.horizontal_scale)
        terrain.height_field_raw[0:platform_len, :] = 0
        terrain.height_field_raw[-platform_len:, :] = 0
        #* pad edges
        # pad_width = int(0.1 // terrain.horizontal_scale)
        # pad_height = int(0.5 // terrain.vertical_scale)
        # terrain.height_field_raw[:, :pad_width] = pad_height
        # terrain.height_field_raw[:, -pad_width:] = pad_height
        # terrain.height_field_raw[:pad_width, :] = pad_height
        # terrain.height_field_raw[-pad_width:, :] = pad_height
        
        return terrain

    def gap_terrain_goal(self,
                        terrain,
                        #platform_len=0.6, 
                        platform_len=1.5,
                        platform_height=0., 
                        num_gaps=8,
                        gap_size=0.3,
                        x_range=[0.6, 1.0],     # gap长度范围
                        y_range=[-1.2, 1.2],    # 横向的随机化
                        half_valid_width=[0.6, 1.2],    # 高度随机化
                        gap_depth=-200,
                        pad_width=0.1,
                        # pad_width=0.2,
                        pad_height=0.5,
                        flat=False):
        
        if num_gaps != 0:
        
            # 中间位置
            mid_y = terrain.length // 2  # length is actually y width

            # y方向上的偏置
            dis_y_min = round(y_range[0] / terrain.horizontal_scale)
            dis_y_max = round(y_range[1] / terrain.horizontal_scale)

            # 起点平台的长度和高度(必须是0)
            platform_len = round(platform_len / terrain.horizontal_scale)
            platform_height = round(platform_height / terrain.vertical_scale)
            
            # 凹陷的深度
            gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)
            
            # 抬起平台的宽度
            half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

            # 两个隆起平台之间的距离
            gap_size = round(gap_size / terrain.horizontal_scale)
            
            dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
            dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

            #! dis_x表示初始化的时候的x坐标，last_dis_x表示上一个隆起平台的x坐标
            dis_x = platform_len
            last_dis_x = dis_x
            
            for i in range(num_gaps):   # 生成num_gaps个隆起平台
                rand_x = np.random.randint(dis_x_min, dis_x_max)
                dis_x += rand_x
                rand_y = np.random.randint(dis_y_min, dis_y_max)
                
                if not flat:    # gap挖空
                    terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

                # 台子两边区域挖空
                terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth
                terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
                
                last_dis_x = dis_x

            final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
            
            # 防止最后一个隆起平台超出地图,暂时没用到
            if final_dis_x > terrain.width:
                final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
                
            # 起点平台的高度置0
            terrain.height_field_raw[0:platform_len, :] = platform_height
            terrain.height_field_raw[-platform_len:, :] = platform_height
            
            
        goal = np.zeros((1,2))
        goal = [terrain.width * terrain.horizontal_scale - 1.0 , terrain.length * 1/2 * terrain.horizontal_scale]
        terrain.goal = goal 
        
        
        # pad edges, 把边缘(width=0.1)的地方都填充为pad_height
        pad_width = int(pad_width // terrain.horizontal_scale)
        pad_height = int(pad_height // terrain.vertical_scale)
        terrain.height_field_raw[:, :pad_width] = pad_height
        terrain.height_field_raw[:, -pad_width:] = pad_height
        terrain.height_field_raw[:pad_width, :] = pad_height
        terrain.height_field_raw[-pad_width:, :] = pad_height
        
    def step_terrain_goal(self,
                         terrain,
                    platform_len=1.0, 
                    platform_height=0., 
                    num_stones=0,
                    x_range=[0.5, 2.5],
                    step_height = 0.2,
                    ):

        step_x_min = round( x_range[0] / terrain.horizontal_scale)
        step_x_max = round( x_range[1] / terrain.horizontal_scale)

        step_height = round(step_height / terrain.vertical_scale)

        platform_len = round(platform_len / terrain.horizontal_scale)
        platform_height = round(platform_height / terrain.vertical_scale)
        
        terrain.height_field_raw[:platform_len, :] = platform_height

        rand_x = np.random.randint(step_x_min, step_x_max)
        dis_x = platform_len + rand_x
        for i in range(num_stones):
            
            rand_x = np.random.randint(step_x_min, step_x_max)
            
            terrain.height_field_raw[dis_x:dis_x+rand_x, ] = step_height
            
            dis_x += max(rand_x*2,rand_x + int(1.5/terrain.horizontal_scale))
            # dis_x += rand_x + int(1.5/terrain.horizontal_scale)
            
        
        goal = np.zeros((1,2))
        goal = [terrain.width * terrain.horizontal_scale - 1.0 , terrain.length * 1/2 * terrain.horizontal_scale]
        terrain.goal = goal 
        
        terrain.height_field_raw[-platform_len:, :] = platform_height

        
        # pad edges
        # pad_width = int(0.1 // terrain.horizontal_scale)
        # pad_height = int(0.5 // terrain.vertical_scale)
        # terrain.height_field_raw[:, :pad_width] = pad_height
        # terrain.height_field_raw[:, -pad_width:] = pad_height
        # terrain.height_field_raw[:pad_width, :] = pad_height
        # terrain.height_field_raw[-pad_width:, :] = pad_height

    def slope_terrain_goal(self,
                           terrain, 
                    slope, 
                    platform_length=1.5):
        
        if slope != 0:
            slope += 0.2

            start_x = int(platform_length/terrain.horizontal_scale) # 30
            stop_x = int(terrain.width - platform_length/terrain.horizontal_scale)  # 170
            length = stop_x - start_x   # 140
            
            # x = np.arange(0, terrain.width)
            x = np.arange(start_x, stop_x)  # [140,]
            y = np.arange(0, terrain.length)    # [80,]
            center_x = int(terrain.width / 2)   # 100
            
            xx, yy = np.meshgrid(x, y, sparse=True) # [1,140]
            xx = (center_x - np.abs(center_x-xx) - start_x ) / center_x   # [1,140]
            xx = xx.reshape(length, 1)  # [140,1]
            max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * length/2)
            
            terrain.height_field_raw[start_x:stop_x,:] += (max_height * xx).astype(terrain.height_field_raw.dtype)
            
            half_length = int(0.3/terrain.horizontal_scale)
            terrain.height_field_raw[center_x - half_length:center_x + half_length,:] = \
                terrain.height_field_raw[center_x - half_length:center_x + half_length,:].min()
        
        goal = np.zeros((1,2))
        goal = [terrain.width * terrain.horizontal_scale - 1.0 , terrain.length * 1/2 * terrain.horizontal_scale]
        terrain.goal = goal 
        # goal = [terrain.width * terrain.horizontal_scale - 1.0 , terrain.length * 1/2 * terrain.horizontal_scale]
        # terrain.goal = goal
        
        return terrain

    def stairs_terrain_goal(self,
                            terrain, 
                        step_width, 
                        step_height, 
                        platform_length=1.5):

        # switch parameters to discrete units
        step_width = int(step_width / terrain.horizontal_scale)
        

        if step_height!=0:

            step_height = int((step_height+0.03) / terrain.vertical_scale)
            height = 0
            start_x = int(platform_length/terrain.horizontal_scale)
            stop_x = int(terrain.width - platform_length/terrain.horizontal_scale)
                
            while (stop_x - start_x) > 0.3/terrain.horizontal_scale:
                start_x += step_width
                stop_x -= step_width
                height += step_height
                terrain.height_field_raw[start_x: stop_x, :] = height
                
            # 前后平台
            terrain.height_field_raw[:round(platform_length/terrain.horizontal_scale), :] = 0
            terrain.height_field_raw[-round(platform_length/terrain.horizontal_scale):, :] = 0
            
        goal = [terrain.width * terrain.horizontal_scale - 1.0 , terrain.length * 1/2 * terrain.horizontal_scale]
        terrain.goal = goal
            
        return terrain

    def flat_terrain_goal(self,
                          terrain):
        #! terrain中width和length的属性和env的width和length正好是反的
        # goal = [terrain.width * 4/5 * terrain.horizontal_scale , terrain.length * 1/2 * terrain.horizontal_scale]
        goal = [terrain.width * terrain.horizontal_scale - 1.0 , terrain.length * 1/2 * terrain.horizontal_scale]
        terrain.goal = goal
        
        # pad edges
        # pad_width = int(0.1 // terrain.horizontal_scale)
        # pad_height = int(0.5 // terrain.vertical_scale)
        # terrain.height_field_raw[:, :pad_width] = pad_height
        # terrain.height_field_raw[:, -pad_width:] = pad_height
        # terrain.height_field_raw[:pad_width, :] = pad_height
        # terrain.height_field_raw[-pad_width:, :] = pad_height
        
        return terrain


    def crack_terrain_goal(self,
                        terrain,
                        platform_len=2.5, 

                        log_length_range=[1.0, 1.5],   
                        log_width = 0.2,
                        y_rand_range=[-0.5, 0.5],

                        gap_depth=200,
                        
                        pad_width=0.1,
                        pad_height=0.5,
                        flat=False):

    
        # 中间位置
        # mid_y = terrain.length // 2  # length is actually y width
        # platform_len = round(platform_len / terrain.horizontal_scale)
        
        # log_length_min = round(log_length_range[0] / terrain.horizontal_scale)
        # log_length_max = round(log_length_range[1] / terrain.horizontal_scale)
        # half_log_width = round(log_width / 2 / terrain.horizontal_scale)
        # y_rand_min = round(y_rand_range[0] / terrain.horizontal_scale)
        # y_rand_max = round(y_rand_range[1] / terrain.horizontal_scale)
        
        
        # log_length_ = np.random.randint(log_length_min, log_length_max)
        # y_rand_ = np.random.randint(y_rand_min, y_rand_max)
        
        # if not flat:
        #     terrain.height_field_raw[platform_len:platform_len+log_length_,:mid_y+y_rand_-half_log_width] = gap_depth
        #     terrain.height_field_raw[platform_len:platform_len+log_length_,mid_y+y_rand_+half_log_width:] = gap_depth
            
        # # platform
        # terrain.height_field_raw[0:platform_len, :] = 0
        # terrain.height_field_raw[-platform_len:, :] = 0
        
        terrain.height_field_raw = 100
        
            
        goal = np.zeros((1,2))
        goal = [terrain.width * terrain.horizontal_scale - 1.0 , terrain.length * 1/2 * terrain.horizontal_scale]
        terrain.goal = goal 
        
        
        # pad edges, 把边缘(width=0.1)的地方都填充为pad_height
        # pad_width = int(pad_width // terrain.horizontal_scale)
        # pad_height = int(pad_height // terrain.vertical_scale)
        # terrain.height_field_raw[:, :pad_width] = pad_height
        # terrain.height_field_raw[:, -pad_width:] = pad_height
        # terrain.height_field_raw[:pad_width, :] = pad_height
        # terrain.height_field_raw[-pad_width:, :] = pad_height
        


    

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)  #?

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)    # height(next row) - height(this row)>slope_threshold 上坡 
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)     # height(this row) - height(next row)>slope_threshold 下坡
        
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)    # 与x方向上判断同理
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0, move_y != 0








