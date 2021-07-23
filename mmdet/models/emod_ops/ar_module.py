import torch
from torch import nn
from mmcv.cnn.utils import constant_init, kaiming_init

class SimAttention(nn.Module):
    def __init__(self, in_channels):
        super(SimAttention, self).__init__()
        
        self.conv_attn = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        
        kaiming_init(self.conv_attn, mode='fan_in')
        self.conv_attn.inited = True
        
    def forward(self, x):
        b, c, h, w = x.size()

        x_in = x
        x_in = x_in.view(b, c, h * w)
        x_in = x_in.unsqueeze(1)
        
        x_attn = self.conv_attn(x)
        x_attn = x_attn.view(b, 1, h * w)
        x_attn = self.softmax(x_attn)
        x_attn = x_attn.unsqueeze(-1)

        x_out = torch.matmul(x_in, x_attn)
        x_out = x_out.view(b, c, 1, 1)
        
        return x_out

class SimRelation(nn.Module):
    def __init__(self, in_channels, ratio, act=False):
        super(SimRelation, self).__init__()
        
        self.planes = int(in_channels * ratio)
        self.act = act
        
        self.mlp = nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=self.planes),
                nn.LayerNorm([self.planes]),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=self.planes, out_features=in_channels))
        constant_init(self.mlp[-1], val=0)
        
        if self.act:
            self.activate = nn.Sigmoid()
        
    def forward(self, x):
        x_in = x
        x_in = x_in.view(x.size(0), -1)
        
        x_out = self.mlp(x_in)
        if self.act:
            x_out = self.activate(x_out)
        x_out = x_out.view(x.size(0), x.size(1), 1, 1)
        
        return x_out

class ARModule(nn.Module):
    """AR Module for EMOD."""

    def __init__(self,
                 in_channels,
                 ratio,
                 fusion_type='add'):
        super(ARModule, self).__init__()
        assert fusion_type in ['add', 'mul'], 'fusion_type should be add or mul.'
        self.fusion_type = fusion_type
        
        # attention
        self.sim_attention = SimAttention(in_channels)
        
        # relation
        if self.fusion_type == 'add':
            self.sim_relation = SimRelation(in_channels, ratio, act=False)
        else:
            self.sim_relation = SimRelation(in_channels, ratio, act=True)

    def forward(self, x):
        x_attn = self.sim_attention(x)

        out = x
        if self.fusion_type == 'add':
            x_rel = self.sim_relation(x_attn)
            out = out + x_rel
        else:
            x_rel = self.sim_relation(x_attn)
            out = out * x_rel
            
        return out
