import torch
import torch.nn as nn

class ConvBlockND(nn.Module):
    def __init__(self, in_c, out_c, dim):
        super().__init__()
        conv_cls = nn.Conv2d if dim == 2 else nn.Conv3d
        
        self.conv = nn.Sequential(
            conv_cls(in_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.GELU(),
            conv_cls(out_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.GELU()
        )
    def forward(self, x):
        return self.conv(x)

class FlexibleCVFUNet(nn.Module):
    def __init__(self, in_channels=1, base_filters=16, dim=3):
        super().__init__()
        self.dim = dim
        
        pool_cls = nn.MaxPool2d if dim == 2 else nn.MaxPool3d
        trans_cls = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
        conv_cls = nn.Conv2d if dim == 2 else nn.Conv3d
        
        # Encoder
        self.e1 = ConvBlockND(in_channels, base_filters, dim)
        self.pool1 = pool_cls(2)
        self.e2 = ConvBlockND(base_filters, base_filters*2, dim)
        self.pool2 = pool_cls(2)
        self.e3 = ConvBlockND(base_filters*2, base_filters*4, dim)
        
        # Decoder
        self.up2 = trans_cls(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.d2 = ConvBlockND(base_filters*4, base_filters*2, dim)
        self.up1 = trans_cls(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.d1 = ConvBlockND(base_filters*2, base_filters, dim)
        
        # Output Channels: 4 for 2D (EDT, Vx, Vy, visibility), 4 for 3D (EDT, Vx, Vy, Vz)
        out_channels = 4 if dim == 2 else 1 + dim
        self.out = conv_cls(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool1(e1))
        e3 = self.e3(self.pool2(e2))
        
        d2 = self.up2(e3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.d2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.d1(d1)
        
        return self.out(d1)
