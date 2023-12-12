import torch
import torch.nn as nn
from torch.distributions import Normal

def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
  if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
    nn.init.orthogonal_(module.weight.data, gain)
    nn.init.constant_(module.bias.data, 0)
  return module

class DepthEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 token_dim):
        super(DepthEncoder, self).__init__()
        
        self.layers = nn.Sequential = [
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=token_dim, kernel_size = 1)
        ]
        
        self.apply(orthogonal_init)
        
    def forward(self, x):
        x = self.layers(x)
        
        return x
    
class StateEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 state_input_dim,
                 token_dim):
        super(StateEncoder, self).__init__()
        
        self.in_channels = in_channels
        
        self.layers = nn.Sequential(
            nn.Linear(state_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, token_dim),
            nn.ReLU()
        )
        
        self.apply(orthogonal_init) # todo 不知道初始化的方式是否会有影响
        
    def forward(self, x):
        x = self.layers(x)
        
        return x
    
    

class MultiModalEncoder(nn.Module):
    def __init__(self, 
                 in_channels = 4,
                 obs_dim = 0,
                 state_dim = 0,
                 token_dim = 64,
                 ):
        super(MultiModalEncoder, self).__init__()
        
        assert obs_dim > 0, "obs_dim must be greater than 0"
        assert state_dim > 0, "state_dim must be greater than 0"
        
        self.depth_encoder = DepthEncoder(in_channels, token_dim)
        self.state_encoder = StateEncoder(in_channels, state_dim, token_dim)

        
    def forward(self, visual_x, state_x):
        visual_x = self.depth_encoder(visual_x)
        state_x = self.state_encoder(state_x)
        
        return visual_x, state_x        
        
    
class LocoTransformer(nn.Module):
    def __init__(self,
                 encoder,
                 token_dim = 64,
                 transformer_params=[[1, 256], [1, 256]],
                 append_hidden_shapes = [256, 256],
                 output_shape = 6
                ):
        self.encoder = encoder
        
        self.visual_append_layers = nn.ModuleList()
        for n_head, dim_feedforward in transformer_params:
            visual_att_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=n_head, dim_feedforward=dim_feedforward,dropout=0)
            self.visual_append_layers.append(visual_att_layer)

        self.visual_append_fcs = []
        for next_shape in append_hidden_shapes: # [256, 256]
            visual_fc = nn.Linear(visual_append_input_shape, next_shape)
            self.visual_append_fcs.append(visual_fc)
            self.visual_append_fcs.append(nn.ReLU())
            visual_append_input_shape = next_shape
        visual_last = nn.Linear(visual_append_input_shape, output_shape)
        self.visual_append_fcs.append(visual_last)
        self.visual_seq_append_fcs = nn.Sequential(*self.visual_append_fcs)
        
    def forward(self, visual_x, state_x):
        
        visual_out, state_out = self.encoder(visual_x, state_x)
        
        for att_layer in self.visual_append_layers:
            out = att_layer(visual_out)
        
        

class ActorCriticTransformer(nn.Module):
    def __init__(self, 
                 in_channels = 4,
                 obs_dim = 0,
                 state_dim = 0,
                 token_dim = 64,
                 transformer_params=[[1, 256], [1, 256]],
                 append_hidden_shapes = [256, 256],
                 output_shape = 6,
                 init_noise_std = 1.0,
                 num_actions = 12
                ):
        super(ActorCriticTransformer, self).__init__()
        
        self.multi_modal_encoder = MultiModalEncoder(in_channels, 
                                                     obs_dim, 
                                                     state_dim, 
                                                     token_dim)
        self.transformer = LocoTransformer(self.multi_modal_encoder, 
                                           token_dim, 
                                           transformer_params, 
                                           append_hidden_shapes, 
                                           output_shape)
        
        self.actor = self.transformer.copy()
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        self.crtic = self.transformer.copy()
        
        print(f"Actor Net: {self.actor}")
        print(f"Critic Net: {self.critic}")
        
        
    def forward(self, visual_x, state_x):
        None
        
        
# GaussianContPolicyLocoTransformer(

#     ENCODER
#   (encoder): LocoTransformerEncoder
#   (
#     (depth_visual_base): NatureEncoder
#     (
#       (layers): Sequential(
#         (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
#         (1): ReLU()
#         (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
#         (3): ReLU()
#         (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
#         (5): ReLU()
#       )
#     )
#     (depth_up_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))

#     (base): MLPBase
#     (
#       (seq_fcs): Sequential(
#         (0): Linear(in_features=84, out_features=256, bias=True)
#         (1): ReLU()
#         (2): Linear(in_features=256, out_features=256, bias=True)
#         (3): ReLU()
#       )
#     )
#     (state_projector): RLProjection
#     (
#       (projection): Sequential(
#         (0): Linear(in_features=256, out_features=64, bias=True)
#         (1): ReLU()
#       )
#     )

#     (flatten_layer): Flatten()
#   )

#     TRANSFORMER
#   (visual_append_layers): ModuleList
#   (
#     (0-1): 2 x TransformerEncoderLayer(
#       (self_attn): MultiheadAttention(
#         (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
#       )
#       (linear1): Linear(in_features=64, out_features=256, bias=True)
#       (dropout): Dropout(p=0, inplace=False)
#       (linear2): Linear(in_features=256, out_features=64, bias=True)
#       (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#       (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#       (dropout1): Dropout(p=0, inplace=False)
#       (dropout2): Dropout(p=0, inplace=False)
#     )
#   )
#   (visual_seq_append_fcs): Sequential
#   (
#     (0): Linear(in_features=128, out_features=256, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=256, out_features=256, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=256, out_features=6, bias=True)
#   )

# )

# LocoTransformer
# (
#   (encoder): LocoTransformerEncoder
#   (
#     (depth_visual_base): NatureEncoder(
#       (layers): Sequential(
#         (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
#         (1): ReLU()
#         (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
#         (3): ReLU()
#         (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
#         (5): ReLU()
#       )
#     )
#     (depth_up_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
#     (base): MLPBase(
#       (seq_fcs): Sequential(
#         (0): Linear(in_features=84, out_features=256, bias=True)
#         (1): ReLU()
#         (2): Linear(in_features=256, out_features=256, bias=True)
#         (3): ReLU()
#       )
#     )
#     (state_projector): RLProjection(
#       (projection): Sequential(
#         (0): Linear(in_features=256, out_features=64, bias=True)
#         (1): ReLU()
#       )
#     )
#     (flatten_layer): Flatten()
#   )

#   (visual_append_layers): ModuleList
#   (
#     (0-1): 2 x TransformerEncoderLayer(
#       (self_attn): MultiheadAttention(
#         (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
#       )
#       (linear1): Linear(in_features=64, out_features=256, bias=True)
#       (dropout): Dropout(p=0, inplace=False)
#       (linear2): Linear(in_features=256, out_features=64, bias=True)
#       (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#       (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#       (dropout1): Dropout(p=0, inplace=False)
#       (dropout2): Dropout(p=0, inplace=False)
#     )
#   )
#   (visual_seq_append_fcs): Sequential
#   (
#     (0): Linear(in_features=128, out_features=256, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=256, out_features=256, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=256, out_features=1, bias=True)
#   )
# )
        