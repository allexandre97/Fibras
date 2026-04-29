from numbers import Integral

import torch
import torch.nn as nn


DEFAULT_ASPP_DILATIONS = (1, 2, 4)
LEGACY_ASPP_DILATIONS = (2, 4, 8)
DEFAULT_UNET_DEPTH = 3
LEGACY_UNET_DEPTH = 4
PREDICTION_HEAD_TYPE = "shallow_3x3_gn_gelu"


def normalize_aspp_dilations(aspp_dilations):
    if isinstance(aspp_dilations, str):
        raise ValueError("aspp_dilations must be an iterable of positive integers, not a string.")
    try:
        values = tuple(aspp_dilations)
    except TypeError as error:
        raise ValueError("aspp_dilations must be an iterable of positive integers.") from error
    if not values:
        raise ValueError("aspp_dilations must contain at least one dilation.")

    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("aspp_dilations must contain only positive integers.")
        value = int(value)
        if value <= 0:
            raise ValueError("aspp_dilations must contain only positive integers.")
        normalized.append(value)
    return tuple(normalized)


def normalize_unet_depth(unet_depth):
    if isinstance(unet_depth, bool) or not isinstance(unet_depth, Integral):
        raise ValueError("unet_depth must be an integer in [3, 4].")
    unet_depth = int(unet_depth)
    if unet_depth not in (3, 4):
        raise ValueError("unet_depth must be an integer in [3, 4].")
    return unet_depth


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
    def __init__(self, channels, groups=8, aspp_dilations=DEFAULT_ASPP_DILATIONS):
        super().__init__()
        self.aspp_dilations = normalize_aspp_dilations(aspp_dilations)
        branch_channels = max(channels // (len(self.aspp_dilations) + 1), 16)
        branches = [
            nn.Sequential(
                nn.Conv2d(channels, branch_channels, kernel_size=1),
                nn.GroupNorm(min(groups, branch_channels), branch_channels),
                nn.GELU(),
            )
        ]
        for dilation in self.aspp_dilations:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.GroupNorm(min(groups, branch_channels), branch_channels),
                    nn.GELU(),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * len(self.branches), channels, kernel_size=1),
            nn.GroupNorm(min(groups, channels), channels),
            nn.GELU(),
            ResidualBlock2D(channels, channels, groups=groups),
        )

    def forward(self, x):
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class ShallowPredictionHead2D(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8, hidden_channels=None):
        super().__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.spatial = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(min(groups, hidden_channels), hidden_channels)
        self.act = nn.GELU()
        self.project = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.act(self.norm(self.spatial(x)))
        return self.project(x)


class STEDResUNet2D(nn.Module):
    """Residual 2D U-Net for structural STED fiber reconstruction.

    Output channels are centerline logits, cos(2theta), sin(2theta),
    traceability logits, normalized radius logits, and normalized bundle-count logits.
    """

    def __init__(
        self,
        in_channels=1,
        base_filters=32,
        out_channels=6,
        groups=8,
        aspp_dilations=DEFAULT_ASPP_DILATIONS,
        unet_depth=DEFAULT_UNET_DEPTH,
    ):
        super().__init__()
        self.unet_depth = normalize_unet_depth(unet_depth)
        width_multipliers = [1, 2, 4, 8] if self.unet_depth == 3 else [1, 2, 4, 8, 12]
        widths = [base_filters * multiplier for multiplier in width_multipliers]

        encoder_blocks = []
        prev_channels = in_channels
        for width in widths[:-1]:
            encoder_blocks.append(ResidualBlock2D(prev_channels, width, groups=groups))
            prev_channels = width
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ResidualBlock2D(widths[-2], widths[-1], groups=groups),
            ASPPBottleneck2D(widths[-1], groups=groups, aspp_dilations=aspp_dilations),
        )

        upsamplers = []
        decoder_blocks = []
        current_width = widths[-1]
        for skip_width in reversed(widths[:-1]):
            upsamplers.append(nn.ConvTranspose2d(current_width, skip_width, kernel_size=2, stride=2))
            decoder_blocks.append(ResidualBlock2D(skip_width + skip_width, skip_width, groups=groups))
            current_width = skip_width
        self.upsamplers = nn.ModuleList(upsamplers)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.head_refinement = ResidualBlock2D(widths[0], widths[0], groups=groups)

        self.centerline_head = ShallowPredictionHead2D(widths[0], 1, groups=groups)
        self.orientation_head = ShallowPredictionHead2D(widths[0], 2, groups=groups)
        self.traceability_head = ShallowPredictionHead2D(widths[0], 1, groups=groups)
        self.radius_head = ShallowPredictionHead2D(widths[0], 1, groups=groups)
        self.bundle_count_head = ShallowPredictionHead2D(widths[0], 1, groups=groups)
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
        skips = []
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for skip, upsample, decoder in zip(reversed(skips), self.upsamplers, self.decoder_blocks):
            x = decoder(self._concat_skip(skip, upsample(x)))
        refined = self.head_refinement(x)

        return torch.cat(
            [
                self.centerline_head(refined),
                self.orientation_head(refined),
                self.traceability_head(refined),
                self.radius_head(refined),
                self.bundle_count_head(refined),
            ],
            dim=1,
        )
