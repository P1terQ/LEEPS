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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO

from .base.pc_legged_robot import PC_LeggedRobot
from .a1.pc_a1_config import PC_A1RoughCfg, PC_A1RoughCfgPPO

from .base.legged_robot_addfoottraj import LeggedRobotAddFootTraj

from .terrainprimitive.a1.a1primitive_config import A1PrimitiveCfg, A1PrimitiveCfgPPO
from .terrainprimitive.legged_primitive import LeggedPrimitive

from .terrainprimitive.a1.a1box_config import A1BoxCfg, A1BoxCfgPPO
from .terrainprimitive.legged_box import LeggedBox

from .terrainprimitive.a1.a1parkour_config import A1ParkourCfg, A1ParkourCfgPPO
from .terrainprimitive.legged_parkour import LeggedParkour

from .terrainprimitive.a1.a1push_config import A1PushCfg, A1PushCfgPPO
from .terrainprimitive.legged_push import LeggedPush

from .terrainprimitive.a1.a1ptp_config import A1PTPCfg, A1PTPCfgPPO
from .terrainprimitive.legged_ptp import LeggedPTP

from .terrainprimitive.a1.a1vls_config import A1VLSCfg, A1VLSCfgPPO
from .terrainprimitive.legged_vls import LeggedVLS

from .terrainprimitive.a1.a1v2_config import A1V2Cfg, A1V2CfgPPO
from .terrainprimitive.legged_v2 import LeggedV2

from .terrainprimitive.a1.a1vel_config import A1VelCfg, A1VelCfgPPO
from .terrainprimitive.legged_vel import LeggedVel

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )

task_registry.register( "pc_a1", PC_LeggedRobot, PC_A1RoughCfg(), PC_A1RoughCfgPPO() )
task_registry.register( "a1_footraj", LeggedRobotAddFootTraj, A1RoughCfg(), A1RoughCfgPPO() )

task_registry.register( "a1_pri", LeggedPrimitive, A1PrimitiveCfg(), A1PrimitiveCfgPPO() )
task_registry.register( "a1_box", LeggedBox, A1BoxCfg(), A1BoxCfgPPO() )
task_registry.register( "a1_parkour", LeggedParkour, A1ParkourCfg(), A1ParkourCfgPPO() )
task_registry.register( "a1_push", LeggedPush, A1PushCfg(), A1PushCfgPPO() )
task_registry.register( "a1_ptp", LeggedPTP, A1PTPCfg(), A1PTPCfgPPO() )
task_registry.register( "a1_vls", LeggedVLS, A1VLSCfg(), A1VLSCfgPPO() )
task_registry.register( "a1_v2", LeggedV2, A1V2Cfg(), A1V2CfgPPO() )
task_registry.register( "a1_vel", LeggedVel, A1VelCfg(), A1VelCfgPPO() )
