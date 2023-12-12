import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

class StateHistoryEncoder_Blind(nn.Module):
    def __init__(self,
                 activation_fn,
                 input_size,
                 history_tsteps,
                 output_size):
        
        super(StateHistoryEncoder_Blind, self).__init__()
        
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
        
        

class Actor_RMA_Blind(nn.Module):
    def __init__(self,
                 num_prop,
                 num_actions,
                 actor_hidden_dims,
                 privinfo_encoder_dims,
                 num_priv_info,
                 num_priv_explicit,
                 num_history,
                 activation_fn):
        
        super(Actor_RMA_Blind, self).__init__()
        
        self.num_prop = num_prop
        self.num_hist = num_history
        self.num_actions = num_actions
        self.num_priv_info = num_priv_info  # 外界的priv info
        self.num_priv_explicit = num_priv_explicit  # 机器人的priv info
        
        if len(privinfo_encoder_dims) > 0:  # [64, 20]
            priv_encoder_layers = []
            priv_encoder_layers.append(nn.Linear(num_priv_info, privinfo_encoder_dims[0]))
            priv_encoder_layers.append(activation_fn)
            for l in range(len(privinfo_encoder_dims) - 1):
                priv_encoder_layers.append(nn.Linear(privinfo_encoder_dims[l], privinfo_encoder_dims[l + 1]))
                priv_encoder_layers.append(activation_fn)
            
            self.privinfo_encoder = nn.Sequential(*priv_encoder_layers)
            privinfo_encoder_output_dim = privinfo_encoder_dims[-1]  
        else:
            self.privinfo_encoder = nn.Identity()
            privinfo_encoder_output_dim = num_priv_info
            
        # 从历史的prop中还原出priv_info_latent. 用history_encoder模仿学习privinfo_encoder
        self.history_encoder = StateHistoryEncoder_Blind(activation_fn, num_prop, num_history, privinfo_encoder_output_dim)            
        
        actor_layers = []
        # actor input: prop, priv_explicit, privinfo_latent
        actor_layers.append(nn.Linear(num_prop + num_priv_explicit + privinfo_encoder_output_dim, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims)): # [512, 256, 128]
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))   # 最后一层
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor_backbone = nn.Sequential(*actor_layers)
        
    # 使用privinfo_encoder将priv_info encode 为 latent
    def infer_privinfo_latent(self, obs):  
        privinfo = obs[:, self.num_prop+self.num_priv_explicit: self.num_prop+self.num_priv_explicit+self.num_priv_info]
        return self.privinfo_encoder(privinfo)
        
    # 使用history_encoder输出priv_latent
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))    # view[num_envs, num_hist, num_prop]

    def infer_scandots_latent(self, obs):
        raise NotImplementedError("infer_scandots_latent not implemented in Actor_RMA_Blind")


    def forward(self, obs, use_historyestimate: bool):
                
        # use historyencoder to output priv env info latent        
        if use_historyestimate:
            privinfo_latent = self.infer_hist_latent(obs)   # shape: [num_envs, 20]
        else:
            privinfo_latent = self.infer_privinfo_latent(obs)
            
        backbone_input = torch.cat([obs[:,:self.num_prop+self.num_priv_explicit], privinfo_latent], dim=1)
        
        backbone_output = self.actor_backbone(backbone_input)
        
        return backbone_output
    
    
class ActorCriticRMA_Blind(nn.Module):
    is_recurrent = False
    def __init__(self,
                 num_prop,
                 num_priv_info,
                 num_priv_explicit,
                 num_history,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 priv_encoder_dims=[64, 20],
                 activation='elu',
                 init_noise_std=1.0
                 ): # kwargs的作用是会自动使用传入的kwargs来初始化形参的值。
            
        super(ActorCriticRMA_Blind, self).__init__() 
        
        activation_func = get_activation(activation)
        
        # ACTOR
        self.actor = Actor_RMA_Blind(num_prop = num_prop, 
                                    num_actions = num_actions, 
                                    actor_hidden_dims = actor_hidden_dims, 
                                    privinfo_encoder_dims = priv_encoder_dims, 
                                    num_priv_info = num_priv_info, 
                                    num_priv_explicit = num_priv_explicit, 
                                    num_history = num_history, 
                                    activation_fn = activation_func)
        
        # CRITIC
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
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
        # actions_mean = self.actor(observations, use_historyestimate, scandots_latent)
        # return actions_mean
        raise NotImplementedError("act_inference not implemented in ActorCriticRMA_Blind")
    
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

class Estimator_Blind(nn.Module):
    def __init__(self,  input_dim,
                        output_dim,
                        hidden_dims=[256, 128, 64],
                        activation="elu"):
        
        super(Estimator_Blind, self).__init__()

        self.input_dim = input_dim  # 53
        self.output_dim = output_dim    # 9
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
        