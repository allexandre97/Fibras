import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.GELU(),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.GELU()
        )
    def forward(self, x):
        return self.conv(x)

class CVFUNet(nn.Module):
    def __init__(self, in_channels=1, base_filters=16):
        super().__init__()
        # Encoder
        self.e1 = ConvBlock3D(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)
        self.e2 = ConvBlock3D(base_filters, base_filters*2)
        self.pool2 = nn.MaxPool3d(2)
        self.e3 = ConvBlock3D(base_filters*2, base_filters*4)
        
        # Decoder
        self.up2 = nn.ConvTranspose3d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.d2 = ConvBlock3D(base_filters*4, base_filters*2)
        self.up1 = nn.ConvTranspose3d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.d1 = ConvBlock3D(base_filters*2, base_filters)
        
        # 4 Output Channels: [0] = EDT, [1, 2, 3] = Vx, Vy, Vz
        self.out = nn.Conv3d(base_filters, 4, kernel_size=1)

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