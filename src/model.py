import torch
import torch.nn as nn


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8, dilation=1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class ASPPBottleneck2D(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        branch_channels = max(channels // 4, 16)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, branch_channels, kernel_size=1),
                nn.GroupNorm(min(groups, branch_channels), branch_channels),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(channels, branch_channels, kernel_size=3, padding=2, dilation=2),
                nn.GroupNorm(min(groups, branch_channels), branch_channels),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(channels, branch_channels, kernel_size=3, padding=4, dilation=4),
                nn.GroupNorm(min(groups, branch_channels), branch_channels),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(channels, branch_channels, kernel_size=3, padding=8, dilation=8),
                nn.GroupNorm(min(groups, branch_channels), branch_channels),
                nn.GELU(),
            ),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * len(self.branches), channels, kernel_size=1),
            nn.GroupNorm(min(groups, channels), channels),
            nn.GELU(),
            ResidualBlock2D(channels, channels, groups=groups),
        )

    def forward(self, x):
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class STEDResUNet2D(nn.Module):
    """Residual 2D U-Net for structural STED fiber reconstruction.

    Output channels are centerline logits, cos(2theta), sin(2theta),
    traceability logits, normalized radius logits, and normalized bundle-count logits.
    """

    def __init__(self, in_channels=1, base_filters=32, out_channels=6, groups=8):
        super().__init__()
        widths = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8, base_filters * 12]

        self.e1 = ResidualBlock2D(in_channels, widths[0], groups=groups)
        self.e2 = ResidualBlock2D(widths[0], widths[1], groups=groups)
        self.e3 = ResidualBlock2D(widths[1], widths[2], groups=groups)
        self.e4 = ResidualBlock2D(widths[2], widths[3], groups=groups)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ResidualBlock2D(widths[3], widths[4], groups=groups),
            ASPPBottleneck2D(widths[4], groups=groups),
        )

        self.up4 = nn.ConvTranspose2d(widths[4], widths[3], kernel_size=2, stride=2)
        self.d4 = ResidualBlock2D(widths[3] + widths[3], widths[3], groups=groups)
        self.up3 = nn.ConvTranspose2d(widths[3], widths[2], kernel_size=2, stride=2)
        self.d3 = ResidualBlock2D(widths[2] + widths[2], widths[2], groups=groups)
        self.up2 = nn.ConvTranspose2d(widths[2], widths[1], kernel_size=2, stride=2)
        self.d2 = ResidualBlock2D(widths[1] + widths[1], widths[1], groups=groups)
        self.up1 = nn.ConvTranspose2d(widths[1], widths[0], kernel_size=2, stride=2)
        self.d1 = ResidualBlock2D(widths[0] + widths[0], widths[0], groups=groups)

        self.centerline_head = nn.Conv2d(widths[0], 1, kernel_size=1)
        self.orientation_head = nn.Conv2d(widths[0], 2, kernel_size=1)
        self.traceability_head = nn.Conv2d(widths[0], 1, kernel_size=1)
        self.radius_head = nn.Conv2d(widths[0], 1, kernel_size=1)
        self.bundle_count_head = nn.Conv2d(widths[0], 1, kernel_size=1)
        self.out_channels = out_channels

    @staticmethod
    def _concat_skip(skip, upsampled):
        if skip.shape[-2:] != upsampled.shape[-2:]:
            upsampled = torch.nn.functional.interpolate(
                upsampled,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.cat([skip, upsampled], dim=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.d4(self._concat_skip(e4, self.up4(b)))
        d3 = self.d3(self._concat_skip(e3, self.up3(d4)))
        d2 = self.d2(self._concat_skip(e2, self.up2(d3)))
        d1 = self.d1(self._concat_skip(e1, self.up1(d2)))

        return torch.cat(
            [
                self.centerline_head(d1),
                self.orientation_head(d1),
                self.traceability_head(d1),
                self.radius_head(d1),
                self.bundle_count_head(d1),
            ],
            dim=1,
        )
