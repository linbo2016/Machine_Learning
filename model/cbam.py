import torch
import torch.nn as nn
import torch.nn.functional as F

class SAM(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x, return_attention=False):
        max_out = torch.max(x, dim=1)[0].unsqueeze(1)
        avg_out = torch.mean(x, dim=1).unsqueeze(1)
        concat = torch.cat((max_out, avg_out), dim=1)
        attention = self.conv(concat)
        attention_map = torch.sigmoid(attention)
        output = attention_map * x
        
        if return_attention:
            return output, attention_map
        else:
            return output

class CAM(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True)
        )

    def forward(self, x, return_attention=False):
        max_out = F.adaptive_max_pool2d(x, output_size=1)
        avg_out = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        max_out = self.shared_mlp(max_out.view(b, c)).view(b, c, 1, 1)
        avg_out = self.shared_mlp(avg_out.view(b, c)).view(b, c, 1, 1)
        attention = max_out + avg_out
        attention_map = torch.sigmoid(attention)
        output = attention_map * x
        
        if return_attention:
            return output, attention_map
        else:
            return output

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.cam = CAM(channels=self.channels, r=self.r)
        self.sam = SAM(bias=False)

    def forward(self, x, return_attention=False):
        # 第一步：通道注意力
        channel_out, channel_attn = self.cam(x, return_attention=True)
        
        # 第二步：空間注意力
        spatial_out, spatial_attn = self.sam(channel_out, return_attention=True)
        
        if return_attention:
            return spatial_out + x, {
                'channel_attention': channel_attn,
                'spatial_attention': spatial_attn
            }
        else:
            return spatial_out + x