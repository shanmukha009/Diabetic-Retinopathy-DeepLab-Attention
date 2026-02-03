"""
Standard U-Net implementation for DR lesion segmentation.
Based on the original U-Net paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class DoubleConv(nn.Module):
    """
    Double convolution block: (conv => BN => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Output convolution layer
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    Standard U-Net architecture
    """
    def __init__(self, num_classes, in_channels=3, bilinear=True, base_channels=64):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.bilinear = bilinear

        # Encoder (contracting path)
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # Decoder (expansive path)
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # Output layer
        self.outc = OutConv(base_channels, num_classes)
        
        # Optional deep supervision outputs for better training
        self.deep_supervision = True
        if self.deep_supervision:
            self.dsv1 = nn.Conv2d(base_channels * 8 // factor, num_classes, kernel_size=1)
            self.dsv2 = nn.Conv2d(base_channels * 4 // factor, num_classes, kernel_size=1)
            self.dsv3 = nn.Conv2d(base_channels * 2 // factor, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        if self.deep_supervision and self.training:
            dsv1 = self.dsv1(x)
            dsv1 = F.interpolate(dsv1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        x = self.up2(x, x3)
        if self.deep_supervision and self.training:
            dsv2 = self.dsv2(x)
            dsv2 = F.interpolate(dsv2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        x = self.up3(x, x2)
        if self.deep_supervision and self.training:
            dsv3 = self.dsv3(x)
            dsv3 = F.interpolate(dsv3, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        x = self.up4(x, x1)
        
        # Final output
        output = self.outc(x)
        
        if self.training and self.deep_supervision:
            return output, (dsv1, dsv2, dsv3)
        else:
            return output

class UNetWithDropout(nn.Module):
    """
    U-Net with dropout for regularization
    """
    def __init__(self, num_classes, in_channels=3, bilinear=True, base_channels=64, dropout_rate=0.1):
        super(UNetWithDropout, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.bilinear = bilinear
        self.dropout_rate = dropout_rate

        # Encoder (contracting path)
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # Dropout layers
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        # Decoder (expansive path)
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # Output layer
        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x1 = self.dropout1(x1)
        
        x2 = self.down1(x1)
        x2 = self.dropout2(x2)
        
        x3 = self.down2(x2)
        x3 = self.dropout3(x3)
        
        x4 = self.down3(x3)
        x4 = self.dropout4(x4)
        
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final output
        output = self.outc(x)
        
        return output

def get_unet_model(num_classes=len(config.LESION_TYPES), in_channels=3, bilinear=True, 
                   base_channels=64, with_dropout=False, dropout_rate=0.1):
    """
    Factory function to create a U-Net model.
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels (typically 3 for RGB)
        bilinear: Whether to use bilinear upsampling (True) or transposed convolutions (False)
        base_channels: Base number of channels (will be multiplied in deeper layers)
        with_dropout: Whether to include dropout layers
        dropout_rate: Dropout rate if with_dropout is True
        
    Returns:
        U-Net model instance
    """
    if with_dropout:
        model = UNetWithDropout(
            num_classes=num_classes,
            in_channels=in_channels,
            bilinear=bilinear,
            base_channels=base_channels,
            dropout_rate=dropout_rate
        )
    else:
        model = UNet(
            num_classes=num_classes,
            in_channels=in_channels,
            bilinear=bilinear,
            base_channels=base_channels
        )
    
    return model

# For compatibility with existing codebase
def get_standard_unet(num_classes=len(config.LESION_TYPES), pretrained=False):
    """
    Get a standard U-Net model for compatibility with existing training scripts.
    Note: pretrained parameter is included for API compatibility but ignored 
    since standard U-Net doesn't have pretrained weights.
    """
    return get_unet_model(num_classes=num_classes)

if __name__ == "__main__":
    # Test the model
    model = get_unet_model(num_classes=4)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
    # Test training mode
    model.train()
    output = model(x)
    if isinstance(output, tuple):
        main_output, aux_outputs = output
        print(f"Training mode - Main output: {main_output.shape}")
        print(f"Training mode - Auxiliary outputs: {len(aux_outputs)}")
    else:
        print(f"Training mode - Output shape: {output.shape}")