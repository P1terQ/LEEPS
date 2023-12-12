from absl import app
from absl import flags
from inputs import get_gamepad
import threading
import time
import numpy as np

FLAGS = flags.FLAGS
MAX_ABS_RX = 32768  # 最大键盘值
MAX_ABS_RY = 32768


def _interpolate(raw_reading, max_raw_reading, new_scale):
    #! 用于将键盘值转化为-1到1之间的值 * new_scale
    return raw_reading / max_raw_reading * new_scale

# class Gamepad:
#     def __init__(self, vel_scale_x=-1.0, vel_scale_y=-1.0, vel_scale_rot=1.0):
        
#         # keys and flags
#         self._vel_scale_x = vel_scale_x
#         self._vel_scale_y = vel_scale_y
#         self._vel_scale_rot = vel_scale_rot
        
#         self._lb_pressed = False
#         self._rb_pressed = False
        
#         # control values
#         self.vx, self.vy, self.wz = 0., 0., 0.
#         self.estop_flag = False
#         self.is_running = True
        
#         self.read_thread = threading.Thread(target=self.read_loop)
#         self.read_thread.start()
        
        
#     def read_loop(self):
#         while self.is_running and not self.estop_flag:
#             events = get_gamepad()
            
#             for event in events:
#                 if event.ev_type == 'Absolute' and event.code == 'ABS_Y':
#                     self.vx = _interpolate(event.state, MAX_ABS_RY, self._vel_scale_x)
#                 elif event.ev_type == 'Absolute' and event.code == 'ABS_X':
#                     self.vy = _interpolate(event.state, MAX_ABS_RX, self._vel_scale_y)
#                 elif event.ev_type == 'Absolute' and event.code == 'ABS_RX':
#                     self.wz = _interpolate(event.state, MAX_ABS_RX, self._vel_scale_rot)
                    
#                 elif event.ev_type == 'Key' and event.code == 'BTN_TL':
#                     self._lb_pressed = event.state == 1
#                 elif event.ev_type == 'Key' and event.code == 'BTN_TR':
#                     self._rb_pressed = event.state == 1

#             if self._lb_pressed and self._rb_pressed:
#                 self.estop_flag = True
#                 self.vx, self.vy, self.wz = 0., 0., 0.
                
#     def get_command(self):
#         # del time_since_reset  # unused
#         return (1.0 * self.vx, 0.5 * self.vy, 0), -1.5 * self.wz, self.estop_flag

#     def gamepad_stop(self):
#         self.is_running = False

class Gamepad:
    def __init__(self, goal_delta_scale_x=-4.0, goal_delta_scale_y=-1.0):
        
        # keys and flags
        self._goal_delta_scale_x = goal_delta_scale_x
        self._goal_delta_scale_y = goal_delta_scale_y
        
        self._lb_pressed = False
        self._rb_pressed = False
        
        # control values
        self.delta_x, self.delta_y = 0., 0.
        self.episode_length_s = 0.
        self.episode_param = 0.7
        
        self.estop_flag = False
        self.is_running = True
        
        self.read_thread = threading.Thread(target=self.read_loop)
        self.read_thread.start()
        
        
    def read_loop(self):
        while self.is_running and not self.estop_flag:
            events = get_gamepad()
            

            
            for event in events:
                
                # print("event.ev_type: ", event.ev_type)
                # print("event.code: ", event.code)
                # print("event.state: ", event.state)
                
                if event.ev_type == 'Absolute' and event.code == 'ABS_Y':
                    self.delta_x = _interpolate(event.state, MAX_ABS_RY, self._goal_delta_scale_x)
                elif event.ev_type == 'Absolute' and event.code == 'ABS_X':
                    self.delta_y = _interpolate(event.state, MAX_ABS_RX, self._goal_delta_scale_y)
                    
                elif event.ev_type == 'Absolute' and event.code == 'ABS_HAT0Y':
                    if event.state == -1:
                        self.episode_param += 0.05
                    elif event.state == 1:
                        self.episode_param -= 0.05
                    self.episode_param = np.clip(self.episode_param, 0.6, 1.0)
                    
                elif event.ev_type == 'Key' and event.code == 'BTN_TL':
                    self._lb_pressed = event.state == 1
                elif event.ev_type == 'Key' and event.code == 'BTN_TR':
                    self._rb_pressed = event.state == 1
                    
                # print("delta_x: ", self.delta_x)
                episode_percentage = _interpolate(np.square(self.delta_x) + np.square(self.delta_y),
                                                     np.square(self._goal_delta_scale_x)+np.square(self._goal_delta_scale_y),1.0)
                # print("episode_percentage: ", episode_percentage)
                self.episode_length_s = 1 + episode_percentage * 5.0
                    

            if self._lb_pressed and self._rb_pressed:
                self.estop_flag = True
                self.delta_x, self.delta_y = 0., 0.
                
    def get_command(self):
        # del time_since_reset  # unused
        return 1.0 * self.delta_x, 1.0 * self.delta_y, self.episode_param * self.episode_length_s, self.estop_flag

    def gamepad_stop(self):
        self.is_running = False
