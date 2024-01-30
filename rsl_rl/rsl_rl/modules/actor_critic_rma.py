import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

class StateHistoryEncoder(nn.Module):
    def __init__(self,
                 activation_fn,
                 input_size,    # num_prop
                 history_tsteps,
                 output_size):
        
        super(StateHistoryEncoder, self).__init__()
        
        self.activation_fn = activation_fn
        self.history_tsteps = history_tsteps

        channel_size = 10 #? 10 channels for depth, 3 for proprio。 不清楚为什么是这个channel size
        
        self.encoder = nn.Sequential(nn.Linear(input_size, 3*channel_size), self.activation_fn)
        
        # 一维卷积操作，用于处理一维序列数据
        if self.history_tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif self.history_tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        elif self.history_tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))
        
        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )
        
    def forward(self, obs):
        n_envs = obs.shape[0]
        # 将 proprioception投影到30维的latent  Project n_proprio -> 30
        projection = self.encoder(obs.reshape([n_envs * self.history_tsteps, -1])) # projection_shape: [num_env * tsteps, 30].
        
        # 将投影后的数据reshape成三维的数据，然后进行一维卷积操作，最后flatten成二维数据。 permute的作用是把第二维和第三维交换位置，高度和宽度互换
        mid = self.conv_layers(projection.reshape([n_envs, self.history_tsteps, -1]).permute(0, 2, 1)) # mid_shape: [num_env,  30]
        
        # 最后投影到输出的维度
        output = self.linear_output(mid) # output_shape: [num_env, 20]
        return output
        
        
class Actor_RMA(nn.Module):
    def __init__(self,
                 num_prop,
                 num_scandots_terrain,
                 num_scandots_ceiling,
                 num_actions,
                 
                 priv_terrainscan_encoder_dims,
                 priv_ceilingscan_encoder_dims,
                 
                 actor_hidden_dims,
                 priv_envinfo_encoder_dims,
                 num_priv_env,
                 num_priv_self,
                 num_history,
                 activation_fn):
        
        super(Actor_RMA, self).__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        
        self.num_prop = num_prop
        self.num_scan_terrain = num_scandots_terrain
        self.num_scan_ceiling = num_scandots_ceiling
        self.num_hist = num_history
        self.num_actions = num_actions
        
        self.num_priv_env = num_priv_env  # privileged env info 的dim
        self.num_priv_self = num_priv_self  # 人为指定的cmd
        
        # 将priv env obs encode为一个latent
        if len(priv_envinfo_encoder_dims) > 0:  # [64, 20]
            priv_envinfo_encoder_layers = []
            priv_envinfo_encoder_layers.append(nn.Linear(num_priv_env, priv_envinfo_encoder_dims[0]))
            priv_envinfo_encoder_layers.append(activation_fn)
            for l in range(len(priv_envinfo_encoder_dims) - 1):
                priv_envinfo_encoder_layers.append(nn.Linear(priv_envinfo_encoder_dims[l], priv_envinfo_encoder_dims[l + 1]))
                priv_envinfo_encoder_layers.append(activation_fn)
            
            self.priv_envinfo_encoder = nn.Sequential(*priv_envinfo_encoder_layers)
            priv_envinfo_encoder_output_dim = priv_envinfo_encoder_dims[-1]  
        else:
            self.priv_envinfo_encoder = nn.Identity()
            priv_envinfo_encoder_output_dim = num_priv_env
            
        # 从历史的prop中还原出priv_info_latent
        self.history_encoder = StateHistoryEncoder(activation_fn, num_prop, num_history, priv_envinfo_encoder_output_dim)
        print(self.history_encoder)
        # StateHistoryEncoder(
        #   (activation_fn): ELU(alpha=1.0)
        #   (encoder): Sequential(
        #     (0): Linear(in_features=49, out_features=30, bias=True)
        #     (1): ELU(alpha=1.0)
        #   )
        #   (conv_layers): Sequential(
        #     (0): Conv1d(30, 20, kernel_size=(4,), stride=(2,))
        #     (1): ELU(alpha=1.0)
        #     (2): Conv1d(20, 10, kernel_size=(2,), stride=(1,))
        #     (3): ELU(alpha=1.0)
        #     (4): Flatten(start_dim=1, end_dim=-1)
        #   )
        #   (linear_output): Sequential(
        #     (0): Linear(in_features=30, out_features=20, bias=True)
        #     (1): ELU(alpha=1.0)
        #   )
        # )
            
        # terrain scandots
        if len(priv_terrainscan_encoder_dims) > 0:  # [128, 64, 32]
            priv_terrainscan_encoder_layers = []
            priv_terrainscan_encoder_layers.append(nn.Linear(num_scandots_terrain, priv_terrainscan_encoder_dims[0]))
            priv_terrainscan_encoder_layers.append(activation_fn)
            for l in range(len(priv_terrainscan_encoder_dims) - 1):
                if l == len(priv_terrainscan_encoder_dims) - 2:
                    priv_terrainscan_encoder_layers.append(nn.Linear(priv_terrainscan_encoder_dims[l], priv_terrainscan_encoder_dims[l+1]))
                    priv_terrainscan_encoder_layers.append(nn.Tanh())
                else:
                    priv_terrainscan_encoder_layers.append(nn.Linear(priv_terrainscan_encoder_dims[l], priv_terrainscan_encoder_dims[l + 1]))
                    priv_terrainscan_encoder_layers.append(activation_fn)
            
            self.priv_terrainscan_encoder = nn.Sequential(*priv_terrainscan_encoder_layers)
            priv_terrainscan_encoder_output_dim = priv_terrainscan_encoder_dims[-1]
        else:
            self.priv_terrainscan_encoder = nn.Identity()
            priv_terrainscan_encoder_output_dim = num_scandots_terrain
        
        # ceiling scandots
        if len(priv_ceilingscan_encoder_dims) > 0:  # [128, 64, 32]
            priv_ceilingscan_encoder_layers = []
            priv_ceilingscan_encoder_layers.append(nn.Linear(num_scandots_ceiling, priv_ceilingscan_encoder_dims[0]))
            priv_ceilingscan_encoder_layers.append(activation_fn)
            for l in range(len(priv_ceilingscan_encoder_dims) - 1):
                if l == len(priv_ceilingscan_encoder_dims) - 2:
                    priv_ceilingscan_encoder_layers.append(nn.Linear(priv_ceilingscan_encoder_dims[l], priv_ceilingscan_encoder_dims[l+1]))
                    priv_ceilingscan_encoder_layers.append(nn.Tanh())
                else:
                    priv_ceilingscan_encoder_layers.append(nn.Linear(priv_ceilingscan_encoder_dims[l], priv_ceilingscan_encoder_dims[l + 1]))
                    priv_ceilingscan_encoder_layers.append(activation_fn)
            
            self.priv_ceilingscan_encoder = nn.Sequential(*priv_ceilingscan_encoder_layers)
            priv_ceilingscan_encoder_output_dim = priv_ceilingscan_encoder_dims[-1]
        else:
            self.priv_ceilingscan_encoder = nn.Identity()
            priv_ceilingscan_encoder_output_dim = num_scandots_ceiling
            
        
        actor_layers = []
        # actor input: prop, terrainscan_latent, ceilingscan_latent, priv_self(自身), priv_envinfo_latent(环境)
        actor_layers.append(nn.Linear(num_prop+ # 49
                                      priv_terrainscan_encoder_output_dim+  # 32
                                      priv_ceilingscan_encoder_output_dim+  # 32
                                      num_priv_self+    # 4
                                      priv_envinfo_encoder_output_dim,  # 20
                                      actor_hidden_dims[0]))    
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims)): # [512, 256, 128]
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor_backbone = nn.Sequential(*actor_layers)
        
    # 使用privinfo_encoder将priv_info encode 为 latent
    def infer_priv_envinfo_latent(self, obs):  #! 环境的priv info latent
        priv_envinfo = obs[:, self.num_prop+self.num_scan_terrain+self.num_scan_ceiling+self.num_priv_self: self.num_prop+self.num_scan_terrain+self.num_scan_ceiling+self.num_priv_self+self.num_priv_env]
        return self.priv_envinfo_encoder(priv_envinfo)
        
    # 使用history_encoder输出priv_latent
    def infer_hist_latent(self, obs):   #! 使用过去的prop估计环境的priv_info latent
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))    # view[num_envs, num_hist, num_prop]

    def infer_terrain_scandots_latent(self, obs):  
        scan = obs[:, self.num_prop:self.num_prop+self.num_scan_terrain]
        return self.priv_terrainscan_encoder(scan)

    def infer_ceiling_scandots_latent(self, obs):  
        scan = obs[:, self.num_prop+self.num_scan_terrain:self.num_prop+self.num_scan_terrain+self.num_scan_ceiling]
        return self.priv_ceilingscan_encoder(scan)
    
    def forward(self, obs, use_historyestimate: bool, terrain_scandots_latent=None, ceiling_scandots_latent=None):
        
        #! 当训练teacher policy的时候,scandots_latent=None. 训练student policy的时候,scandots_latent是depth_encoder的输出
        
        obs_terrainscan = obs [:,self.num_prop:self.num_prop+self.num_scan_terrain]
        if terrain_scandots_latent is None:      
            # true  
            terrain_scan_latent = self.priv_terrainscan_encoder(obs_terrainscan)   # 将输入的scandots encode为一个latent
        else:
            terrain_scan_latent = terrain_scandots_latent
            
        obs_ceilingscan = obs [:,self.num_prop+self.num_scan_terrain:self.num_prop+self.num_scan_terrain+self.num_scan_ceiling]
        if ceiling_scandots_latent is None:      
            # true  
            ceiling_scan_latent = self.priv_ceilingscan_encoder(obs_ceilingscan)   # 将输入的scandots encode为一个latent
        else:
            ceiling_scan_latent = ceiling_scandots_latent
        
        obs_prop_scan = torch.cat([obs[:,:self.num_prop], terrain_scan_latent, ceiling_scan_latent], dim=1)
        
        obs_priv_self = obs[:, self.num_prop+self.num_scan_terrain+self.num_scan_ceiling:self.num_prop+self.num_scan_terrain+self.num_scan_ceiling+self.num_priv_self]
        
        if use_historyestimate: #! use history priop to get priv env info latent
            # update dagger
            privinfo_latent = self.infer_hist_latent(obs)   # [num_env, history_encoder_outputdim] use prop history to estimate privinfo_latent
        else:
            privinfo_latent = self.infer_priv_envinfo_latent(obs)
            
        backbone_input = torch.cat([obs_prop_scan, obs_priv_self, privinfo_latent], dim=1)
        
        backbone_output = self.actor_backbone(backbone_input)   # get [num_env, 12] action
        
        return backbone_output
    
    
class ActorCriticRMA(nn.Module):
    is_recurrent = False
    def __init__(self,
                 num_prop,
                 num_scandots_terrain,
                 num_scandots_ceiling,
                 num_priv_env,
                 num_priv_self,
                 num_history,
                 num_critic_obs,
                 num_actions,
                 terrain_scan_encoder_dims=[256, 256, 256],
                 ceiling_scan_encoder_dims=[256, 256, 256],
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 priv_encoder_dims=[64, 20],
                 activation='elu',
                 init_noise_std=1.0
                 ): # kwargs的作用是会自动使用传入的kwargs来初始化形参的值。
            
        super(ActorCriticRMA, self).__init__() 
        
        activation_func = get_activation(activation)
        
        # ACTOR
        self.actor = Actor_RMA(num_prop, 
                               num_scandots_terrain, 
                               num_scandots_ceiling,
                               num_actions, 
                               terrain_scan_encoder_dims, 
                               ceiling_scan_encoder_dims,
                               actor_hidden_dims, 
                               priv_encoder_dims, 
                               num_priv_env, 
                               num_priv_self, 
                               num_history, 
                               activation_func)
        
        
        # CRITIC
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))  # 863 -> hidden_layers -> 1
        critic_layers.append(activation_func)
        for l in range(len(critic_hidden_dims)):  # [512, 256, 128]
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_func)
        
        self.critic = nn.Sequential(*critic_layers)
        
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        

        # Action noise
        self.action_std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.action_distribution = None
        
        # disable args validation for speedup
        Normal.set_default_validate_args = False
    
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)
    
    # not used
    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in enumerate(sequential) if isinstance(module, nn.Linear)]
    
    def reset(self, dones=None):
        pass
    
    def forward(self):
        raise NotImplementedError
    
    @property
    def get_action_mean(self):
        return self.action_distribution.mean
    
    @property
    def get_action_std(self):
        return self.action_distribution.stddev
    
    @property
    def get_action_entropy(self):
        return self.action_distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations, use_historyestimate):
        mean = self.actor(observations, use_historyestimate)
        self.action_distribution = Normal(mean, mean*0. + self.action_std)
        
    # used in training
    def act(self, observations, use_historyestimate = False):   
        self.update_distribution(observations, use_historyestimate)
        return self.action_distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.action_distribution.log_prob(actions).sum(dim=-1)
    
    # used in play
    def act_inference(self, observations, use_historyestimate = False, scandots_latent=None):
        actions_mean = self.actor(observations, use_historyestimate, scandots_latent)
        return actions_mean
    
    def evaluate(self, critic_observations):
        value = self.critic(critic_observations)
        return value
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.action_std.data = new_std.data
        
        
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

class Estimator(nn.Module):
    def __init__(self,  input_dim,
                        output_dim,
                        hidden_dims=[256, 128, 64],
                        activation="elu"):
        
        super(Estimator, self).__init__()

        self.input_dim = input_dim  # num_prop
        self.output_dim = output_dim    # num_priv_self
        
        activation = get_activation(activation)
        estimator_layers = []
        estimator_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        estimator_layers.append(activation)
        
        for l in range(len(hidden_dims)):   # [128, 64]
            if l == len(hidden_dims) - 1:
                estimator_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                estimator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                estimator_layers.append(activation)
        # estimator_layers.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator_layers)
    
    def forward(self, input):
        return self.estimator(input)
    
    def inference(self, input):
        with torch.no_grad():
            return self.estimator(input)
        