# modules/unet3d_new.py
# 3D U-Net with BatchNorm, Dropout & Residual Connections
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=3, base_filters=32):
        super(UNet3D, self).__init__()
        
        self.encoder1 = self.conv_block(in_channels, base_filters)
        self.encoder2 = self.conv_block(base_filters, base_filters * 2)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = self.conv_block(base_filters * 2, base_filters * 4)
        
        self.upconv2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(base_filters * 4, base_filters * 2)
        self.upconv1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(base_filters * 2, base_filters)
        
        self.output = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.2)  # Regularization
        )

    def forward(self, pre, post):
        x = torch.cat([pre, post], dim=1)
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.decoder2(torch.cat([self.upconv2(b), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e1], dim=1))
        flow = self.output(d1) * 10
        return flow
