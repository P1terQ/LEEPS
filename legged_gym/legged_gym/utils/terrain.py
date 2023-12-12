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

from isaacgym import terrain_utils
# from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    # def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
    def __init__(self, cfg, num_robots) -> None:

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

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        
        if cfg.curriculum:  # curriculum会按照行数来增加难度
            self.curiculum()
            
        elif cfg.selected:
            self.selected_terrain()
            
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        if self.cfg.num_sub_terrains == 1:
            terrain_type = self.cfg.terrain_kwargs.pop('type')
            for k in range(self.cfg.num_sub_terrains):
                # Env coordinates in the world
                (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

                terrain = terrain_utils.SubTerrain("terrain",
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

                terrain = terrain_utils.SubTerrain("terrain",
                                                   width=self.width_per_env_pixels,
                                                   length=self.length_per_env_pixels,
                                                   vertical_scale=self.cfg.vertical_scale,
                                                   horizontal_scale=self.cfg.horizontal_scale)
 
                terrain_type = terrain_list[int(k)].pop('type')
                eval(terrain_type)(terrain, **terrain_list[int(k)])

                self.add_terrain_to_map(terrain, i, j)
    
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
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

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

'''
Generate Terrain Functions
'''
def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
    
def sloped_terrain(terrain, slope=1):
    """
    Generate a sloped terrain

    Parameters:
        terrain (SubTerrain): the terrain
        slope (int): positive or negative slope
    Returns:
        terrain (SubTerrain): update terrain
    """

    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(terrain.width, 1)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.width)
    terrain.height_field_raw[:, np.arange(terrain.length)] += (max_height * xx / terrain.width).astype(terrain.height_field_raw.dtype)
    return terrain

def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
    return terrain


def pyramid_sloped_terrain(terrain, slope=1, platform_size=1.):
    """
    Generate a sloped terrain

    Parameters:
        terrain (terrain): the terrain
        slope (int): positive or negative slope
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    center_x = int(terrain.width / 2)
    center_y = int(terrain.length / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x-xx)) / center_x
    yy = (center_y - np.abs(center_y-yy)) / center_y
    xx = xx.reshape(terrain.width, 1)
    yy = yy.reshape(1, terrain.length)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.width // 2 - platform_size
    x2 = terrain.width // 2 + platform_size
    y1 = terrain.length // 2 - platform_size
    y2 = terrain.length // 2 + platform_size

    min_h = min(terrain.height_field_raw[x1, y1], 0)
    max_h = max(terrain.height_field_raw[x1, y1], 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
    return terrain

def discrete_obstacles_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size=1.):
    """
    Generate a terrain with gaps

    Parameters:
        terrain (terrain): the terrain
        max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
        min_size (float): minimum size of a rectangle obstacle [meters]
        max_size (float): maximum size of a rectangle obstacle [meters]
        num_rects (int): number of randomly generated obstacles
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    max_height = int(max_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    (i, j) = terrain.height_field_raw.shape
    height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, i-width, 4))
        start_j = np.random.choice(range(0, j-length, 4))
        terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain

def wave_terrain(terrain, num_waves=1, amplitude=1.):
    """
    Generate a wavy terrain

    Parameters:
        terrain (terrain): the terrain
        num_waves (int): number of sine waves across the terrain length
    Returns:
        terrain (SubTerrain): update terrain
    """
    amplitude = int(0.5*amplitude / terrain.vertical_scale)
    if num_waves > 0:
        div = terrain.length / (num_waves * np.pi * 2)
        x = np.arange(0, terrain.width)
        y = np.arange(0, terrain.length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(terrain.width, 1)
        yy = yy.reshape(1, terrain.length)
        terrain.height_field_raw += (amplitude*np.cos(yy / div) + amplitude*np.sin(xx / div)).astype(
            terrain.height_field_raw.dtype)
    return terrain

def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-10):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)    # 一开始是形成洼地
    
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance)
            terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance
            start_y += stone_size + stone_distance
            
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain

def random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None,):
    """
    Generate a uniform noise terrain

    Parameters
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)

    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale

    # switch parameters to discrete units
    min_height = int(min_height / terrain.vertical_scale)   # 0
    max_height = int(max_height / terrain.vertical_scale)   # 0
    step = int(step / terrain.vertical_scale)   # 1

    heights_range = np.arange(min_height, max_height + step, step)  # ([-40,0,40])
    
    #! terrain.width * terrain.horizontal_scale / downsampled_scale=24. 创建一个24*24的值随机为-40,0,40的矩阵
    height_field_downsampled = np.random.choice(heights_range, (int(terrain.width * terrain.horizontal_scale / downsampled_scale), int(
        terrain.length * terrain.horizontal_scale / downsampled_scale)))

    x = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[0])     # 创建一个 height_field_downsampled.shape[0]的向量，值从0到terrain.width * terrain.horizontal_scale线性(随机)分布
    y = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[1])

    f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear') # 生成一个二维的函数

    x_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
    y_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))  # rint四舍五入

    terrain.height_field_raw += z_upsampled.astype(np.int16)    #! terrain就是加上这些高度
    return terrain

def sloped_terrain(terrain, slope=1):
    """
    Generate a sloped terrain

    Parameters:
        terrain (SubTerrain): the terrain
        slope (int): positive or negative slope
    Returns:
        terrain (SubTerrain): update terrain
    """

    x = np.arange(0, terrain.width) # [0, 1, 2, ..., width-1]
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(terrain.width, 1)
    
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.width)
    
    # 逐步加
    terrain.height_field_raw[:, np.arange(terrain.length)] += (max_height * xx / terrain.width).astype(terrain.height_field_raw.dtype)
    return terrain

def stairs_terrain(terrain, step_width, step_height):
    """
    Generate a stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float):  the height of the step [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)

    num_steps = terrain.width // step_width
    height = step_height
    for i in range(num_steps):
        terrain.height_field_raw[i * step_width: (i + 1) * step_width, :] += height
        height += step_height
    return terrain