"""
DeepLab-v3+ implementation with modifications to improve performance on small lesions like microaneurysms.
Based on the paper findings, DeepLab-v3+ performed better than HED but still had significant
room for improvement, especially for microaneurysms (MA).

Enhancements:
1. Multi-scale feature fusion for better detection of varying lesion sizes
2. Attention mechanisms to focus on small lesions
3. Deep supervision for more precise boundary detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import config

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module for capturing multi-scale context.
    Enhanced to better handle small lesions.
    """
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()
        
        self.convs = nn.ModuleList()
        # 1x1 convolution branch
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        
        # Dilated convolution branches
        for rate in rates:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
        # Global features branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output 1x1 convolution
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        size = x.size()
        features = []
        
        for conv in self.convs:
            features.append(conv(x))
            
        # Global features
        global_features = self.global_avg_pool(x)
        global_features = F.interpolate(
            global_features, size=(size[2], size[3]), mode='bilinear', align_corners=True
        )
        features.append(global_features)
        
        # Concatenate features
        features = torch.cat(features, dim=1)
        
        # Process concatenated features
        return self.output_conv(features)

class ChannelAttention(nn.Module):
    """
    Channel attention module to focus on informative channels.
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on important spatial locations.
    Especially useful for detecting small lesions like microaneurysms.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Average and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and process
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        
        # Apply attention
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Combines channel and spatial attention.
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class DeepLabV3Plus(nn.Module):
    """
    DeepLab-v3+ architecture enhanced for DR lesion segmentation.
    """
    def __init__(self, num_classes, backbone='resnet50', pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            low_level_channels = 256  # After first residual block
            high_level_channels = 2048  # Final features
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            low_level_channels = 256
            high_level_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the classification head
        self.backbone.fc = nn.Identity()
        
        # DeepLab v3+ components
        aspp_rates = [6, 12, 18]
        self.aspp = ASPP(high_level_channels, 256, aspp_rates)
        
        # Low-level feature processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Attention modules
        self.low_att = CBAM(48)
        self.high_att = CBAM(256)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        
        # Deep supervision branches for better gradient flow
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(high_level_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, x):
        input_size = x.size()
        
        # Extract features
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # Low-level features
        x = self.backbone.layer1(x)
        low_level_features = x
        
        # High-level features
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        high_level_features = self.backbone.layer4(x)  # Store layer4 output
        
        # Apply ASPP to high-level features
        x = self.aspp(high_level_features)
        
        # Apply attention to high-level features
        x = self.high_att(x)
        
        # Deep supervision branch - use high_level_features directly
        aux_output = self.aux_classifier(high_level_features)
        aux_output = F.interpolate(
            aux_output, size=input_size[2:], mode='bilinear', align_corners=True
        )
        
        # Process low-level features
        low_level_features = self.low_level_conv(low_level_features)
        low_level_features = self.low_att(low_level_features)
        
        # Upsample high-level features
        x = F.interpolate(
            x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True
        )
        
        # Concatenate features
        x = torch.cat((x, low_level_features), dim=1)
        
        # Decoder
        x = self.decoder(x)
        
        # Upsample to input size
        x = F.interpolate(
            x, size=input_size[2:], mode='bilinear', align_corners=True
        )
        
        if self.training:
            return x, aux_output
        else:
            return x

def get_deeplab_model(num_classes=len(config.LESION_TYPES), backbone='resnet50', pretrained=True):
    """
    Factory function to create a DeepLab-v3+ model.
    """
    model = DeepLabV3Plus(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained
    )
    return model