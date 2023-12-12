import torch
import torch.nn as nn

class DepthBackbone58x87(nn.Module):
    def __init__(self, scan_output_dim, num_frames=1):
        super().__init__()
        
        # todo 尝试把num_frames改成4试试看
        self.num_frames = num_frames
        activation_func = nn.ELU()
        
        self.image_compression = nn.Sequential(
            nn.Conv2d(in_channels=num_frames, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation_func,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation_func,
            nn.Flatten(),
            nn.Linear(64*25*39, 128),
            activation_func,
            nn.Linear(128, scan_output_dim)
        )
        self.output_activation = activation_func
        
    def forward(self, images: torch.Tensor):    # [16, 58, 87]
        images_compressed = self.image_compression(images.unsqueeze(self.num_frames
                                                                    )) # [16, 32]. 这里unsqueeze(1)是因为nn.Conv2d的输入是[batch_size, channel, height, width]， channel为帧数=1
        latent = self.output_activation(images_compressed)

        return latent

class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, n_proprio, n_scan_encoder_outputdim) -> None:
        super().__init__()
        
        activation_func = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        
        self.combination_mlp = nn.Sequential(
                                    nn.Linear(n_scan_encoder_outputdim + n_proprio, 128), #! cat depth and proprio 
                                    activation_func,
                                    nn.Linear(128, n_scan_encoder_outputdim)
                                )
        
        self.rnn = nn.GRU(input_size=n_scan_encoder_outputdim, hidden_size=512, batch_first=True)

        #! 把输出层维度从34改成32，不输出yaw了
        self.output_mlp = nn.Sequential(nn.Linear(512, n_scan_encoder_outputdim), last_activation)
        
        self.hidden_states = None
        
    def forward(self, depth_image, proprioception):
        depth_conv = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat([depth_conv, proprioception], dim=-1))
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)   # None是在中间加一个维度
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent
    
    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()




