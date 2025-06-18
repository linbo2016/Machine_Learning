import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cbam import CBAM
from model.residual_attn import ResidualAttentionModule

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class VGG19_CBAM(nn.Module):
    """VGG19 with CBAM attention modules"""
    def __init__(self, in_channels, out_channels):
        super(VGG19_CBAM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            conv_block(64, 64),
            CBAM(64, r=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            conv_block(128, 128),
            CBAM(128, r=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            *[conv_block(256, 256) for _ in range(3)],
            CBAM(256, r=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            *[conv_block(512, 512) for _ in range(3)],
            CBAM(512, r=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block5 = nn.Sequential(
            *[conv_block(512, 512) for _ in range(4)],
            CBAM(512, r=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=7*7*512, out_features=4096, bias=True),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=self.out_channels, bias=True)
        )

    def forward(self, x, return_attention=False):
        attention_maps = []
        
        # 修改每個模塊以收集注意力圖
        if return_attention:
            x, attn1 = self.conv_block1[4](x, return_attention=True)  # CBAM 模塊
            attention_maps.append(('block1', attn1))
            x = self.conv_block1[0:4](x)  # 之前的層
            x = self.conv_block1[5](x)    # 池化層
            
            x, attn2 = self.conv_block2[4](x, return_attention=True)
            attention_maps.append(('block2', attn2))
            x = self.conv_block2[0:4](x)
            x = self.conv_block2[5](x)
            
            x, attn3 = self.conv_block3[5](x, return_attention=True)
            attention_maps.append(('block3', attn3))
            x = self.conv_block3[0:5](x)
            x = self.conv_block3[6](x)
            
            x, attn4 = self.conv_block4[5](x, return_attention=True)
            attention_maps.append(('block4', attn4))
            x = self.conv_block4[0:5](x)
            x = self.conv_block4[6](x)
            
            x, attn5 = self.conv_block5[4](x, return_attention=True)
            attention_maps.append(('block5', attn5))
            x = self.conv_block5[0:4](x)
            x = self.conv_block5[5](x)
            
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            
            return x, attention_maps
        else:
            # 原始前向傳播
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.conv_block4(x)
            x = self.conv_block5(x)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

class ResidualAttentionNetwork(nn.Module):
    """Residual Attention Network"""
    def __init__(self, in_channels, out_channels):
        super(ResidualAttentionNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 第一個卷積層
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 特徵提取層 - 改進確保尺寸匹配
        self.stage1 = ResidualAttentionModule(64, 64, p=1, t=2, r=1)
        
        # 輸入尺寸會改變，使用自適應層確保尺寸匹配
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage2 = ResidualAttentionModule(128, 128, p=1, t=2, r=1)
        
        self.downsample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.stage3 = ResidualAttentionModule(256, 256, p=1, t=2, r=1)
        
        self.downsample3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.stage4 = ResidualAttentionModule(512, 512, p=1, t=2, r=1)

        # 分類頭
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=self.out_channels)

    def forward(self, x, return_attention=False):
        attention_maps = []
        
        x = self.conv1(x)
        
        if return_attention:
            x, attn1 = self.stage1(x, return_attention=True)
            attention_maps.append(('stage1', attn1))
            
            x = self.downsample1(x)
            x, attn2 = self.stage2(x, return_attention=True)
            attention_maps.append(('stage2', attn2))
            
            x = self.downsample2(x)
            x, attn3 = self.stage3(x, return_attention=True)
            attention_maps.append(('stage3', attn3))
            
            x = self.downsample3(x)
            x, attn4 = self.stage4(x, return_attention=True)
            attention_maps.append(('stage4', attn4))
            
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            
            return x, attention_maps
        else:
            # 原始前向傳播
            x = self.stage1(x)
            x = self.downsample1(x)
            x = self.stage2(x)
            x = self.downsample2(x)
            x = self.stage3(x)
            x = self.downsample3(x)
            x = self.stage4(x)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x