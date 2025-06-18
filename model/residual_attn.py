import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualAttentionModule(nn.Module):
    """Residual Attention Module"""
    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super(ResidualAttentionModule, self).__init__()
        # 主幹卷積塊
        self.trunk_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # 注意力掩碼分支
        self.mask_branch = nn.Sequential(
            # 下採樣路徑
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(out_channels, out_channels, p),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(out_channels, out_channels, t),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 底部處理
            self._make_layer(out_channels, out_channels, r),
            
            # 上採樣路徑
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            self._make_layer(out_channels, out_channels, t),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            self._make_layer(out_channels, out_channels, p),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # 輸出處理
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)
    
    def forward(self, x, return_attention=False):
        # 主幹分支
        trunk_output = self.trunk_branch(x)
        
        # 掩碼分支
        mask_features = self.mask_branch(x)
        
        # 處理尺寸不匹配問題
        if mask_features.size() != trunk_output.size():
            mask_features = F.interpolate(
                mask_features, 
                size=(trunk_output.size(2), trunk_output.size(3)), 
                mode='bilinear', 
                align_corners=True
            )
        
        # 注意力機制
        output = trunk_output * mask_features + trunk_output
        
        if return_attention:
            return output, mask_features
        else:
            return output