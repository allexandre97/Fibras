# Current and Proposed Model Architecture Report

This report describes the current model architecture implemented in
`src/model.py` and explains how every component works, why it is present, and
how it fits the STED fiber skeletonization task. It also proposes architecture
changes that may be better aligned with thin curvilinear structures in STED
images.

The current model class is `STEDResUNet2D`. It is a 2D residual U-Net with an
ASPP bottleneck and five output heads. The model maps a single-channel STED
image crop to six dense prediction channels:

1. Centerline logits.
2. `cos(2theta)` orientation channel.
3. `sin(2theta)` orientation channel.
4. Traceability logits.
5. Normalized radius logits.
6. Normalized bundle-count logits.

The model does not directly output a final binary skeleton. Instead, it predicts
continuous fields that are later decoded into a skeleton graph by
`CenterlineGraphDecoder` in `src/decoder.py`.

## 1. Task Context

The project is trying to skeletonize fibers visible in STED microscopy images.
That task is more specific than ordinary semantic segmentation.

In ordinary segmentation, the model usually needs to decide whether each pixel
belongs to an object region. For fiber skeletonization, the desired output is a
thin centerline representation of curvilinear structures. This has several
consequences:

- The target is spatially precise. A centerline shifted by one pixel can be a
  meaningful error even if the surrounding object region is still detected.
- The target is sparse. Most pixels are background, while only a small fraction
  lie on or near a fiber centerline.
- Local image evidence matters. Fibers are detected from local ridge-like
  intensity patterns, blur, contrast, thickness, and noise.
- Longer-range continuity also matters. A fiber may be locally faint or
  interrupted, but its direction and nearby context can indicate that it should
  remain connected.
- The model needs more than foreground probability. The downstream decoder uses
  centerline confidence, orientation, traceability, and support information to
  perform ridge non-maximum suppression, hysteresis thresholding, endpoint
  bridging, pruning, and graph extraction.

The current architecture reflects this by using a U-Net backbone for dense
pixelwise prediction and several specialized output heads for structural
quantities.

## 2. High-Level Architecture Summary

At a high level, `STEDResUNet2D` has the following structure:

```text
Input image
  |
  v
Encoder stage 1: ResidualBlock2D, width 32
  |
  v
MaxPool2d
  |
  v
Encoder stage 2: ResidualBlock2D, width 64
  |
  v
MaxPool2d
  |
  v
Encoder stage 3: ResidualBlock2D, width 128
  |
  v
MaxPool2d
  |
  v
Encoder stage 4: ResidualBlock2D, width 256
  |
  v
MaxPool2d
  |
  v
Bottleneck:
  ResidualBlock2D, width 384
  ASPPBottleneck2D, width 384
  |
  v
Decoder stage 4:
  ConvTranspose2d 384 -> 256
  concatenate encoder stage 4 skip
  ResidualBlock2D 512 -> 256
  |
  v
Decoder stage 3:
  ConvTranspose2d 256 -> 128
  concatenate encoder stage 3 skip
  ResidualBlock2D 256 -> 128
  |
  v
Decoder stage 2:
  ConvTranspose2d 128 -> 64
  concatenate encoder stage 2 skip
  ResidualBlock2D 128 -> 64
  |
  v
Decoder stage 1:
  ConvTranspose2d 64 -> 32
  concatenate encoder stage 1 skip
  ResidualBlock2D 64 -> 32
  |
  v
Five 1x1 output heads:
  centerline: 1 channel
  orientation: 2 channels
  traceability: 1 channel
  radius: 1 channel
  bundle count: 1 channel
  |
  v
Concatenated 6-channel output tensor
```

With the default `base_filters=32`, the channel widths are:

```text
[32, 64, 128, 256, 384]
```

For an input tensor of shape `(B, 1, H, W)`, the output tensor has shape:

```text
(B, 6, H, W)
```

For example, a `(1, 1, 64, 64)` input produces a `(1, 6, 64, 64)` output.

The default model has 10,461,862 trainable parameters.

## 3. Input and Output Contract

### 3.1 Input Tensor

The model expects a 2D image tensor:

```text
(batch, channels, height, width)
```

The default input channel count is 1:

```python
STEDResUNet2D(in_channels=1)
```

This matches grayscale STED image crops. The architecture is strictly 2D. It
uses `Conv2d`, `MaxPool2d`, and `ConvTranspose2d`, not 3D operations.

### 3.2 Output Tensor

The model returns a tensor with six channels. These channels are concatenated in
this exact order:

```python
[
    centerline_head(d1),
    orientation_head(d1),
    traceability_head(d1),
    radius_head(d1),
    bundle_count_head(d1),
]
```

The final tensor layout is:

```text
channel 0: centerline logits
channel 1: orientation cos(2theta)
channel 2: orientation sin(2theta)
channel 3: traceability logits
channel 4: radius logits
channel 5: bundle-count logits
```

The model itself does not apply sigmoid to the centerline, traceability, radius,
or bundle-count channels. It returns logits for those quantities. Sigmoid is
applied later in the loss and inference code.

This is intentional:

- `BCEWithLogitsLoss` style losses are numerically more stable when given raw
  logits rather than probabilities.
- Logits allow the network to express confidence without saturating too early.
- Inference can clip logits before sigmoid to avoid numerical overflow.

The orientation channels are different. They are not logits. They are raw
continuous values that are interpreted as a double-angle orientation vector.
During training, the predicted orientation is normalized before computing the
orientation error. During inference, the two channels are converted back into a
unit tangent vector.

### 3.3 The `out_channels` Argument

The constructor has this signature:

```python
def __init__(self, in_channels=1, base_filters=32, out_channels=6, groups=8):
```

However, `out_channels` is only stored as:

```python
self.out_channels = out_channels
```

It does not control the number of output channels. The architecture always
creates the same five heads and always returns six channels. This is not
behaviorally harmful for the current pipeline, because the dataset and inference
code expect six channels, but the argument is misleading. If someone passes
`out_channels=4`, the model still returns six channels.

## 4. ResidualBlock2D

`ResidualBlock2D` is the basic building block used throughout the encoder,
bottleneck, and decoder.

### 4.1 Code Structure

The block contains:

```text
main path:
  Conv2d
  GroupNorm
  GELU
  Conv2d
  GroupNorm

skip path:
  Identity, if input channels == output channels
  1x1 Conv2d, otherwise

merge:
  main path + skip path
  GELU
```

The forward pass is:

```python
residual = self.skip(x)
x = self.act(self.norm1(self.conv1(x)))
x = self.norm2(self.conv2(x))
return self.act(x + residual)
```

### 4.2 First 3x3 Convolution

The first convolution is:

```python
nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=3,
    padding=dilation,
    dilation=dilation,
)
```

By default, `dilation=1`, so this is a standard 3x3 convolution with padding 1.

A 3x3 convolution is a natural choice for image processing because it captures
local spatial patterns while keeping the parameter count manageable. In the
context of STED fiber skeletonization, the first 3x3 convolution in a block can
learn local ridge-like evidence such as:

- bright line center versus darker sides,
- local contrast,
- short oriented intensity changes,
- small gaps or noise patterns,
- local thickness cues.

For a convolution with kernel size 3, stride 1, dilation `d`, and padding `d`,
the output height and width are preserved. The general convolution output size
formula is:

```text
output = floor((input + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
```

For `kernel_size=3`, `stride=1`, and `padding=dilation`, this becomes:

```text
output = input
```

This matters because the task is dense prediction. The output must remain
aligned pixel-by-pixel with the input image and target maps.

### 4.3 Padding Choice

The block sets:

```python
padding = dilation
```

For a 3x3 kernel, this creates "same-size" convolution for stride 1. It ensures
that feature maps keep the same spatial resolution inside a stage.

Why this is useful:

- The U-Net skip tensors can be concatenated cleanly with decoder tensors.
- The final output can be aligned with target centerline maps.
- Pixelwise losses can compare predictions and targets at the same coordinates.

One caveat is that PyTorch `Conv2d` uses zero padding by default. Zero padding
adds artificial dark values around crop boundaries. For STED images, this can
produce edge artifacts if fibers approach the crop boundary. That does not make
the architecture invalid, but it is a point worth revisiting.

### 4.4 Dilation Argument

The block supports dilation:

```python
ResidualBlock2D(..., dilation=1)
```

The default is 1, and all uses in `STEDResUNet2D` currently rely on the default.
The block is written to support larger dilation, but the main model does not
pass a larger dilation to any `ResidualBlock2D` directly. The only dilated
convolutions in the current architecture are inside `ASPPBottleneck2D`.

For a 3x3 convolution, dilation changes the effective kernel size:

```text
dilation 1 -> effective 3x3 footprint
dilation 2 -> effective 5x5 footprint
dilation 4 -> effective 9x9 footprint
dilation 8 -> effective 17x17 footprint
```

Dilation increases spatial context without increasing the number of learned
kernel weights. The tradeoff is that high dilation samples pixels with gaps
between them, which can miss fine local details.

### 4.5 GroupNorm

After each convolution, the block applies:

```python
nn.GroupNorm(min(groups, out_channels), out_channels)
```

With the default `groups=8`, the normal case is 8 groups. For example:

```text
32 channels  -> 8 groups, 4 channels per group
64 channels  -> 8 groups, 8 channels per group
128 channels -> 8 groups, 16 channels per group
256 channels -> 8 groups, 32 channels per group
384 channels -> 8 groups, 48 channels per group
```

GroupNorm normalizes activations across groups of channels within each sample.
It does not depend on batch statistics.

This is a sensible choice for microscopy training because batch sizes are often
small due to image size and GPU memory. BatchNorm can become unstable or noisy
with small batches because its statistics are estimated across the batch.
GroupNorm avoids that dependency.

The `min(groups, out_channels)` guard prevents requesting more groups than
channels. For the current default widths this guard does not change anything,
but it makes the block more robust if very small widths are used.

### 4.6 GELU Activation

The activation function is:

```python
nn.GELU()
```

GELU is a smooth nonlinearity. Unlike ReLU, which abruptly clamps all negative
values to zero, GELU softly gates values based on their magnitude. This can be
useful for image restoration and dense prediction tasks where weak negative or
low-contrast evidence may still carry information.

In this model, GELU appears:

- after the first convolution and normalization in each residual block,
- after the residual addition at the end of each residual block,
- after each ASPP branch normalization,
- after the ASPP projection normalization.

The final output heads do not apply GELU. They are linear projections from
features to prediction channels.

### 4.7 Second 3x3 Convolution

The second convolution has the same output channel count as the block output:

```python
nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
```

This second convolution lets the block perform a two-step local transformation.
A single 3x3 convolution sees a 3x3 neighborhood. Two stacked 3x3 convolutions
with dilation 1 give a 5x5 effective receptive field while using fewer
parameters than a single 5x5 convolution.

For skeletonization, this lets each block combine low-level ridge cues into
slightly more structured local evidence, such as a short line segment or local
centerline confidence.

### 4.8 Skip Path

The skip path is:

```python
if in_channels == out_channels:
    self.skip = nn.Identity()
else:
    self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
```

If the number of input and output channels is the same, the skip path simply
passes the input through unchanged.

If the number of channels changes, the skip path uses a 1x1 convolution to
project the input tensor to the correct number of output channels.

This skip path is important for residual learning. Instead of forcing the block
to learn a completely new representation, the block can learn a correction to
the input representation:

```text
output = activation(transform(input) + projected_input)
```

Why this helps:

- It improves gradient flow through deep networks.
- It makes optimization easier.
- It lets the network preserve useful features when no large transformation is
  needed.
- It reduces the risk that deeper layers destroy fine spatial details.

For fiber skeletonization, preserving fine detail is important because the final
target is thin and spatially precise.

## 5. ASPPBottleneck2D

`ASPPBottleneck2D` is the multi-scale context module used at the deepest point
of the U-Net.

ASPP means Atrous Spatial Pyramid Pooling. "Atrous" is another term for dilated
convolution. The idea is to process the same feature map with several receptive
field sizes in parallel and then merge the results.

### 5.1 Input and Output

The ASPP bottleneck receives a tensor with `channels` channels and returns a
tensor with the same number of channels.

With the default model:

```text
channels = 384
```

The ASPP module is shape-preserving:

```text
(B, 384, H/16, W/16) -> (B, 384, H/16, W/16)
```

For a 64x64 input, the bottleneck spatial size is 4x4.

### 5.2 Branch Channel Count

The branch width is:

```python
branch_channels = max(channels // 4, 16)
```

With `channels=384`, this gives:

```text
branch_channels = 96
```

There are four branches, so after concatenation:

```text
4 * 96 = 384 channels
```

This is deliberate. The concatenated branch output has the same channel count as
the input. The following projection layer can therefore merge all branches back
into a 384-channel feature representation without changing the overall
bottleneck width.

The `max(..., 16)` guard prevents the ASPP branches from becoming too narrow if
the model is configured with a small base width.

### 5.3 Branch 1: 1x1 Convolution

The first branch is:

```python
nn.Conv2d(channels, branch_channels, kernel_size=1)
GroupNorm
GELU
```

A 1x1 convolution mixes channel information at each spatial position. It does
not look at neighboring pixels. It is included because not every useful
bottleneck feature needs extra spatial context. Some information should pass
through with only a channel transformation.

In an ASPP module, the 1x1 branch acts as the local or identity-scale branch.
It preserves the information available at each bottleneck location before the
larger dilated branches add context.

### 5.4 Branch 2: 3x3 Convolution With Dilation 2

The second branch is:

```python
nn.Conv2d(channels, branch_channels, kernel_size=3, padding=2, dilation=2)
GroupNorm
GELU
```

A 3x3 convolution with dilation 2 has an effective 5x5 footprint on the
bottleneck feature grid.

Because this is applied after four rounds of downsampling, each bottleneck grid
step corresponds to roughly 16 input pixels. That means dilation 2 at the
bottleneck samples a fairly broad region in the original image.

This branch is intended to capture medium-range structure, such as:

- local continuation of a fiber through a faint region,
- context around nearby parallel fibers,
- larger blur patterns caused by defocus,
- broader image texture around the line.

### 5.5 Branch 3: 3x3 Convolution With Dilation 4

The third branch is:

```python
nn.Conv2d(channels, branch_channels, kernel_size=3, padding=4, dilation=4)
GroupNorm
GELU
```

A 3x3 convolution with dilation 4 has an effective 9x9 footprint on the
bottleneck feature grid.

This is a much larger contextual view. For small 64x64 crops, the bottleneck is
only 4x4, so a 9x9 effective footprint is larger than the bottleneck feature
map itself. Padding allows the convolution to run, but much of the large
footprint interacts with padded values near crop boundaries.

The reason for including this branch is to provide broad context. The risk is
that it may be broader than necessary for precise centerline localization.

### 5.6 Branch 4: 3x3 Convolution With Dilation 8

The fourth branch is:

```python
nn.Conv2d(channels, branch_channels, kernel_size=3, padding=8, dilation=8)
GroupNorm
GELU
```

A 3x3 convolution with dilation 8 has an effective 17x17 footprint on the
bottleneck feature grid.

At the deepest U-Net resolution, this is extremely broad. For a 64x64 input
with a 4x4 bottleneck feature map, this branch is effectively operating with a
near-global or over-global context. It can help the model understand global
fiber layout, but it may also dilute the model's focus on exact local ridge
positions.

For thin skeleton prediction, this branch is the most questionable part of the
current architecture. It is not wrong, but it may be too coarse for the target
scale.

### 5.7 Concatenation

The four branch outputs are concatenated along the channel dimension:

```python
torch.cat([branch(x) for branch in self.branches], dim=1)
```

If each branch outputs 96 channels, the concatenated tensor has 384 channels:

```text
(B, 96, h, w)
(B, 96, h, w)
(B, 96, h, w)
(B, 96, h, w)
        |
        v
(B, 384, h, w)
```

Concatenation preserves the information from every scale. The next projection
layer learns how to combine those scales.

### 5.8 Projection Layer

The projection is:

```python
nn.Conv2d(branch_channels * len(self.branches), channels, kernel_size=1)
GroupNorm
GELU
ResidualBlock2D(channels, channels)
```

The 1x1 projection mixes the four branch outputs channel-by-channel. It learns
which scales are useful for each feature.

The following `ResidualBlock2D` further refines the merged multi-scale features.
This matters because simple concatenation and 1x1 mixing do not give the model a
chance to spatially smooth or reconcile the branch outputs. The residual block
adds two local 3x3 convolutions after the multi-scale merge.

### 5.9 Why ASPP Is Present

Fibers can be continuous over long distances, and local image evidence may be
ambiguous. ASPP gives the bottleneck access to multiple context scales:

- local bottleneck features from the 1x1 branch,
- medium context from dilation 2,
- broader context from dilation 4,
- very broad context from dilation 8.

This is useful when deciding whether a faint local ridge belongs to a larger
fiber structure.

The main tradeoff is localization. Skeletonization cares about precise ridge
location. Very broad bottleneck context can improve continuity, but it cannot
recover sub-pixel or one-pixel details by itself. Those details must come mostly
from high-resolution encoder skips and decoder refinement.

## 6. STEDResUNet2D Main Model

`STEDResUNet2D` assembles the residual blocks, pooling, ASPP bottleneck,
upsampling layers, skip concatenations, and output heads.

### 6.1 Constructor Arguments

The constructor arguments are:

```python
in_channels=1
base_filters=32
out_channels=6
groups=8
```

#### `in_channels`

This controls how many input image channels the first encoder block expects.
The default is 1, which matches grayscale STED images.

If future input representations include additional channels, such as a denoised
image, a local contrast map, or a confidence mask, this argument could be
increased.

#### `base_filters`

This controls the width of the network. The default is 32.

The channel widths are computed as:

```python
widths = [
    base_filters,
    base_filters * 2,
    base_filters * 4,
    base_filters * 8,
    base_filters * 12,
]
```

With `base_filters=32`, this becomes:

```text
32, 64, 128, 256, 384
```

The first four widths are used in the encoder and decoder. The fifth width is
used at the bottleneck.

The bottleneck uses `base_filters * 12`, not `base_filters * 16`. A classic
U-Net often doubles channels at every downsampling step, which would give 512
channels after 256. This implementation uses 384 instead. That keeps the model
smaller while still giving the bottleneck more capacity than the final encoder
stage.

#### `out_channels`

As explained above, this argument is stored but not used to build the heads. The
model always returns six channels.

#### `groups`

This controls the number of GroupNorm groups used inside `ResidualBlock2D` and
`ASPPBottleneck2D`.

The default is 8. This is a conventional choice for medium-width convolutional
networks. It gives each group enough channels to estimate stable statistics
while avoiding the batch-size dependency of BatchNorm.

### 6.2 Encoder Stage 1: `e1`

The first encoder stage is:

```python
self.e1 = ResidualBlock2D(in_channels, widths[0], groups=groups)
```

With default settings:

```text
input:  (B, 1, H, W)
output: (B, 32, H, W)
```

This stage extracts the first learned feature representation from the raw STED
image. Because it operates at full resolution, it is responsible for preserving
fine edge, ridge, and intensity information.

For skeletonization, this full-resolution stage is especially important. The
centerline target is thin, so the decoder later needs high-resolution details
from `e1` to place the output ridge accurately.

### 6.3 Shared Max Pooling Layer

The pooling layer is:

```python
self.pool = nn.MaxPool2d(2)
```

It is reused between encoder stages:

```python
e2 = self.e2(self.pool(e1))
e3 = self.e3(self.pool(e2))
e4 = self.e4(self.pool(e3))
b = self.bottleneck(self.pool(e4))
```

`MaxPool2d(2)` uses a 2x2 window with stride 2 by default. It halves height and
width.

For example:

```text
64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
```

Max pooling keeps the maximum activation in each 2x2 region. It is used to
reduce spatial resolution while increasing the effective receptive field.

Why it is there:

- It gives deeper layers access to wider context.
- It reduces computation at deeper stages.
- It lets the model learn increasingly abstract representations.

Why it is risky for this task:

- Pooling loses exact spatial detail.
- Thin centerlines can be only one pixel wide after target generation.
- Four pooling operations make the bottleneck very coarse.

The U-Net skip connections are the mechanism that compensates for this loss.
Without skips, four downsampling stages would be poorly suited to one-pixel
skeleton localization.

### 6.4 Encoder Stage 2: `e2`

The second encoder stage is:

```python
self.e2 = ResidualBlock2D(widths[0], widths[1], groups=groups)
```

With default settings:

```text
input:  (B, 32, H/2, W/2)
output: (B, 64, H/2, W/2)
```

This stage works at half resolution. It can combine local features from `e1`
into slightly larger patterns. At this scale, the model may begin learning
short fiber segment features rather than only pixel-level intensity changes.

### 6.5 Encoder Stage 3: `e3`

The third encoder stage is:

```python
self.e3 = ResidualBlock2D(widths[1], widths[2], groups=groups)
```

With default settings:

```text
input:  (B, 64, H/4, W/4)
output: (B, 128, H/4, W/4)
```

This stage sees a larger region of the original image. It can represent curved
fiber fragments, local crossings, parallel bundles, or broader blur patterns.

### 6.6 Encoder Stage 4: `e4`

The fourth encoder stage is:

```python
self.e4 = ResidualBlock2D(widths[2], widths[3], groups=groups)
```

With default settings:

```text
input:  (B, 128, H/8, W/8)
output: (B, 256, H/8, W/8)
```

This is the deepest encoder feature before the bottleneck. It contains
relatively high-level spatial context but still has more spatial resolution than
the bottleneck. The decoder later uses `e4` as the first skip connection after
upsampling from the bottleneck.

### 6.7 Bottleneck

The bottleneck is:

```python
self.bottleneck = nn.Sequential(
    ResidualBlock2D(widths[3], widths[4], groups=groups),
    ASPPBottleneck2D(widths[4], groups=groups),
)
```

With default settings:

```text
input:  (B, 256, H/16, W/16)
output: (B, 384, H/16, W/16)
```

The bottleneck has two jobs:

1. Convert the deepest encoder representation from 256 channels to 384
   channels.
2. Apply multi-scale context through ASPP.

The bottleneck is the most parameter-heavy part of the current model. With the
default configuration, it contains 6,151,296 parameters, more than half of the
entire model.

This makes sense from a general U-Net perspective because the bottleneck has
many channels. But for thin skeletonization, it is worth asking whether this is
the best place to spend model capacity. The final output depends heavily on
fine-resolution ridge placement, so capacity in high-resolution decoder
refinement may be more valuable than extra global context in a very coarse
bottleneck.

### 6.8 Decoder Overview

The decoder mirrors the encoder. Each decoder stage:

1. Upsamples the previous lower-resolution feature map with a transposed
   convolution.
2. Concatenates the matching encoder skip feature.
3. Applies a residual block to fuse coarse semantic information with
   high-resolution detail.

This is the defining U-Net pattern.

The purpose is to combine:

- deep context from the bottleneck,
- mid-level structure from encoder stages,
- full-resolution detail from early encoder stages.

For skeletonization, this is essential. The bottleneck can know that a fiber
probably continues through a region, but the early skip features help determine
exactly where the centerline lies.

### 6.9 Transposed Convolutions

The upsampling layers are:

```python
self.up4 = nn.ConvTranspose2d(widths[4], widths[3], kernel_size=2, stride=2)
self.up3 = nn.ConvTranspose2d(widths[3], widths[2], kernel_size=2, stride=2)
self.up2 = nn.ConvTranspose2d(widths[2], widths[1], kernel_size=2, stride=2)
self.up1 = nn.ConvTranspose2d(widths[1], widths[0], kernel_size=2, stride=2)
```

A transposed convolution with `kernel_size=2` and `stride=2` doubles the spatial
resolution.

Default channel changes:

```text
up4: 384 -> 256
up3: 256 -> 128
up2: 128 -> 64
up1: 64  -> 32
```

Why it is there:

- It restores spatial resolution after pooling.
- It learns how to map coarse features into a higher-resolution feature grid.
- It reduces channel count as the decoder moves toward the final output.

The chosen kernel and stride are aligned:

```text
kernel_size = stride = 2
```

This is less prone to checkerboard artifacts than transposed convolutions where
the kernel size is not divisible by the stride. Still, learned upsampling can
introduce artifacts, and this is one reason some architectures prefer bilinear
upsampling followed by a normal convolution.

### 6.10 Skip Concatenation: `_concat_skip`

The helper method is:

```python
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
```

This method does two things.

First, it checks whether the upsampled decoder tensor has the same spatial
height and width as the encoder skip tensor. If not, it resizes the decoder
tensor with bilinear interpolation to exactly match the skip shape.

This protects the model from odd input sizes. Pooling and transposed
convolution can produce off-by-one shape differences when the input height or
width is not divisible by 16.

Second, it concatenates the skip and upsampled tensors along the channel
dimension.

For example, at decoder stage 4:

```text
skip e4:       (B, 256, H/8, W/8)
upsampled b:  (B, 256, H/8, W/8)
concat:       (B, 512, H/8, W/8)
```

Concatenation is different from addition. Addition would require both tensors to
have the same channel count and would immediately mix them. Concatenation keeps
the encoder and decoder features separate at first, then lets the following
residual block learn how to combine them.

### 6.11 Decoder Stage 4: `up4` and `d4`

The first decoder stage is:

```python
self.up4 = nn.ConvTranspose2d(widths[4], widths[3], kernel_size=2, stride=2)
self.d4 = ResidualBlock2D(widths[3] + widths[3], widths[3], groups=groups)
```

With default widths:

```text
bottleneck:   (B, 384, H/16, W/16)
up4 output:   (B, 256, H/8,  W/8)
e4 skip:      (B, 256, H/8,  W/8)
concat:       (B, 512, H/8,  W/8)
d4 output:    (B, 256, H/8,  W/8)
```

This stage fuses the broad bottleneck context with the deepest encoder skip.
The result is still low resolution, but it has begun recovering spatial detail.

### 6.12 Decoder Stage 3: `up3` and `d3`

The second decoder stage is:

```python
self.up3 = nn.ConvTranspose2d(widths[3], widths[2], kernel_size=2, stride=2)
self.d3 = ResidualBlock2D(widths[2] + widths[2], widths[2], groups=groups)
```

With default widths:

```text
d4 output:    (B, 256, H/8, W/8)
up3 output:   (B, 128, H/4, W/4)
e3 skip:      (B, 128, H/4, W/4)
concat:       (B, 256, H/4, W/4)
d3 output:    (B, 128, H/4, W/4)
```

At this stage, the decoder starts to recover medium-resolution fiber geometry.
It can combine context with details about local curvature, crossings, and
neighboring fibers.

### 6.13 Decoder Stage 2: `up2` and `d2`

The third decoder stage is:

```python
self.up2 = nn.ConvTranspose2d(widths[2], widths[1], kernel_size=2, stride=2)
self.d2 = ResidualBlock2D(widths[1] + widths[1], widths[1], groups=groups)
```

With default widths:

```text
d3 output:    (B, 128, H/4, W/4)
up2 output:   (B, 64,  H/2, W/2)
e2 skip:      (B, 64,  H/2, W/2)
concat:       (B, 128, H/2, W/2)
d2 output:    (B, 64,  H/2, W/2)
```

This stage restores half-resolution detail. It is important for aligning
predictions with fine image structures.

### 6.14 Decoder Stage 1: `up1` and `d1`

The final decoder stage is:

```python
self.up1 = nn.ConvTranspose2d(widths[1], widths[0], kernel_size=2, stride=2)
self.d1 = ResidualBlock2D(widths[0] + widths[0], widths[0], groups=groups)
```

With default widths:

```text
d2 output:    (B, 64, H/2, W/2)
up1 output:   (B, 32, H,   W)
e1 skip:      (B, 32, H,   W)
concat:       (B, 64, H,   W)
d1 output:    (B, 32, H,   W)
```

This final full-resolution feature map `d1` is the shared representation used by
all five output heads.

For skeletonization, `d1` is critical. The centerline target is narrow, so the
quality of the final full-resolution features strongly affects whether the
model can place ridges accurately.

## 7. Output Heads

All output heads are 1x1 convolutions applied to `d1`.

A 1x1 convolution processes each pixel independently across channels. It does
not look at neighboring pixels. At this point, all spatial reasoning has already
been done by the encoder, bottleneck, decoder, and skip fusion. The heads simply
convert the final 32-channel feature vector at each pixel into task-specific
outputs.

### 7.1 Centerline Head

The centerline head is:

```python
self.centerline_head = nn.Conv2d(widths[0], 1, kernel_size=1)
```

With default widths:

```text
32 channels -> 1 channel
```

It outputs raw logits. During training, the loss applies sigmoid internally when
computing centerline probability. During inference, the logits are clipped and
then passed through sigmoid.

The centerline target comes from `StructuralTargetGenerator2D`, which creates a
soft centerline field using a Gaussian profile around each segment. The default
centerline sigma is 0.65 pixels. This means the target is very narrow.

The centerline head is the most important head for skeletonization. It tells the
decoder where candidate fiber centerlines are.

Why it is a separate head:

- Centerline confidence has different behavior from orientation or radius.
- It is trained with specialized losses including focal loss, soft Dice, and
  clDice.
- It needs to distinguish sparse line-like targets from a large background.

### 7.2 Orientation Head

The orientation head is:

```python
self.orientation_head = nn.Conv2d(widths[0], 2, kernel_size=1)
```

It outputs two channels:

```text
cos(2theta), sin(2theta)
```

This is a double-angle representation of orientation.

The reason for using `2theta` is that fiber direction is unoriented. A tangent
pointing left-to-right and a tangent pointing right-to-left represent the same
fiber orientation. In angle terms:

```text
theta and theta + pi are equivalent
```

If the model directly predicted `cos(theta), sin(theta)`, those two equivalent
directions would have opposite vectors. That would create an artificial
discontinuity in the target. The double-angle representation removes this
ambiguity:

```text
cos(2(theta + pi)) = cos(2theta + 2pi) = cos(2theta)
sin(2(theta + pi)) = sin(2theta + 2pi) = sin(2theta)
```

During inference, the project converts the double-angle representation back to
an arbitrary tangent vector with `orientation_to_vector_map_np`.

Why this head is present:

- The decoder uses orientation for ridge non-maximum suppression.
- Endpoint bridging uses orientation to decide whether two broken skeleton
  components plausibly belong to the same fiber.
- Orientation helps distinguish crossings, junctions, and nearby parallel
  structures.

The orientation loss is masked. It is only meaningful near centerlines and away
from junctions. This is important because background pixels do not have a true
fiber direction.

### 7.3 Traceability Head

The traceability head is:

```python
self.traceability_head = nn.Conv2d(widths[0], 1, kernel_size=1)
```

It outputs logits that are converted to a traceability probability.

Traceability is a support or confidence field. It represents whether a region
should be trusted as part of a traceable fiber structure. In the loss code, this
target is called `target_traceability`. In older naming it is related to
visibility.

Why this head is present:

- Some fibers may be visible but faint.
- Some predicted centerline responses may be low-confidence noise.
- The decoder multiplies centerline evidence by support.
- Hysteresis thresholding uses support floors for strong and weak candidates.
- Endpoint bridging uses support along the corridor between endpoints.

This head lets the model separate "there is a centerline-like response here"
from "this response is reliable enough to trace."

### 7.4 Radius Head

The radius head is:

```python
self.radius_head = nn.Conv2d(widths[0], 1, kernel_size=1)
```

It outputs logits that are converted with sigmoid to a normalized radius value.

The target radius comes from the synthetic segment thickness. In
`StructuralTargetGenerator2D`, radius is computed from:

```text
base_sigma * thickness_mult
```

and normalized by `radius_normalizer`.

Why this head is present:

- Fiber apparent thickness can vary.
- Radius may help interpret bundle structure and confidence.
- Radius is useful for visualization and downstream measurement.
- Predicting radius gives the shared representation pressure to understand more
  than just the centerline location.

The radius loss is only applied near centerline support. Background pixels do
not have a meaningful radius.

### 7.5 Bundle-Count Head

The bundle-count head is:

```python
self.bundle_count_head = nn.Conv2d(widths[0], 1, kernel_size=1)
```

It outputs logits that are converted with sigmoid to a normalized bundle-count
value.

Bundle count represents how many underlying fibers or segments contribute to a
local structure, normalized by a configured maximum. This is useful when several
fibers are close together and appear as a bundle.

Why this head is present:

- STED fiber images can contain overlapping or bundled structures.
- A single centerline probability does not fully describe local fiber density.
- Bundle count can help characterize dense regions and support downstream
  analysis.
- Like radius, it provides auxiliary supervision that encourages the backbone to
  learn structural properties of the fibers.

The bundle-count loss is also applied near centerlines, because background
bundle count is not structurally meaningful.

### 7.6 Why All Heads Share `d1`

All heads use the same final decoder feature map:

```python
d1 = self.d1(...)
```

This is a multi-task design. The model learns a shared representation of fiber
structure, then projects it into several related outputs.

This is beneficial because the tasks are correlated:

- Centerline position and orientation are directly related.
- Traceability depends on centerline confidence and local image quality.
- Radius depends on the local profile around the centerline.
- Bundle count depends on local structural density.

The risk is that the tasks may compete. A single shared `d1` may not be optimal
for every output. For example, centerline localization may benefit from very
sharp high-resolution features, while bundle count may benefit from broader
context. This motivates one of the proposed changes later: add shallow
task-specific refinement heads.

## 8. Forward Pass in Detail

The forward method is:

```python
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
```

For a 64x64 input and default widths, the shape flow is:

| Tensor | Operation | Shape |
| --- | --- | --- |
| `x` | input | `(B, 1, 64, 64)` |
| `e1` | residual block | `(B, 32, 64, 64)` |
| `pool(e1)` | max pool | `(B, 32, 32, 32)` |
| `e2` | residual block | `(B, 64, 32, 32)` |
| `pool(e2)` | max pool | `(B, 64, 16, 16)` |
| `e3` | residual block | `(B, 128, 16, 16)` |
| `pool(e3)` | max pool | `(B, 128, 8, 8)` |
| `e4` | residual block | `(B, 256, 8, 8)` |
| `pool(e4)` | max pool | `(B, 256, 4, 4)` |
| `b` | residual bottleneck plus ASPP | `(B, 384, 4, 4)` |
| `up4(b)` | transposed conv | `(B, 256, 8, 8)` |
| concat with `e4` | skip concat | `(B, 512, 8, 8)` |
| `d4` | residual block | `(B, 256, 8, 8)` |
| `up3(d4)` | transposed conv | `(B, 128, 16, 16)` |
| concat with `e3` | skip concat | `(B, 256, 16, 16)` |
| `d3` | residual block | `(B, 128, 16, 16)` |
| `up2(d3)` | transposed conv | `(B, 64, 32, 32)` |
| concat with `e2` | skip concat | `(B, 128, 32, 32)` |
| `d2` | residual block | `(B, 64, 32, 32)` |
| `up1(d2)` | transposed conv | `(B, 32, 64, 64)` |
| concat with `e1` | skip concat | `(B, 64, 64, 64)` |
| `d1` | residual block | `(B, 32, 64, 64)` |
| output | five heads plus concat | `(B, 6, 64, 64)` |

## 9. Parameter Distribution

With default settings, the model has 10,461,862 trainable parameters.

Approximate parameter distribution by top-level component:

| Component | Parameters |
| --- | ---: |
| `e1` | 9,760 |
| `e2` | 57,792 |
| `e3` | 230,272 |
| `e4` | 919,296 |
| `pool` | 0 |
| `bottleneck` | 6,151,296 |
| `up4` | 393,472 |
| `d4` | 1,902,336 |
| `up3` | 131,200 |
| `d3` | 476,032 |
| `up2` | 32,832 |
| `d2` | 119,232 |
| `up1` | 8,224 |
| `d1` | 29,920 |
| `centerline_head` | 33 |
| `orientation_head` | 66 |
| `traceability_head` | 33 |
| `radius_head` | 33 |
| `bundle_count_head` | 33 |

The bottleneck and deepest decoder block dominate the parameter count. The
output heads are tiny. This means almost all modeling capacity is in the shared
feature extractor, not in task-specific heads.

## 10. Receptive Field Considerations

The receptive field is the region of the input image that can influence one
output pixel.

A standard 3x3 convolution with stride 1 expands receptive field by one pixel in
each direction. Pooling increases the spacing between feature map positions,
which causes later convolutions to cover more input pixels.

For the current model, the approximate maximum receptive field is very large
relative to small crops. The deepest feature map is downsampled by a factor of
16. A dilated ASPP branch with dilation 8 has an effective 17x17 footprint at
that bottleneck resolution. In input-pixel terms, this can cover most or all of
a 64x64 crop.

This has both advantages and disadvantages.

Advantages:

- The model can use long-range context to bridge faint fiber regions.
- It can disambiguate noisy local patterns using broader image structure.
- It can recognize global bundle layout.

Disadvantages:

- Very large context may be unnecessary for exact centerline placement.
- Broad context can encourage smoother, less sharply localized responses.
- High dilation on very small bottleneck maps can interact heavily with zero
  padding.
- Capacity spent at coarse resolution may be less useful than capacity spent at
  high resolution for one-pixel skeleton targets.

The target centerline sigma is about 0.65 pixels. That is much narrower than
the model's deepest receptive field. Therefore, the architecture relies heavily
on skip connections and final decoder features to recover precise localization.

## 11. How the Architecture Connects to the Loss

The model architecture and loss are tightly coupled.

### 11.1 Centerline Loss

The centerline channel is trained with a combination of:

- focal loss,
- soft Dice loss,
- clDice loss,
- stability margin loss.

Focal loss helps with sparse positives. Most pixels are background, so ordinary
binary cross entropy can be dominated by easy background pixels. Focal loss
downweights easy examples and focuses training on difficult pixels.

Soft Dice helps optimize overlap between predicted centerline probability and
the target centerline field.

clDice encourages topological agreement. It is intended to reward centerline
connectivity and skeleton-like structure more directly than ordinary overlap.

The architecture supports these losses by producing a dense full-resolution
centerline logit map.

### 11.2 Orientation Loss

The orientation channels are normalized before loss computation:

```python
pred_orientation = normalize_orientation_torch(pred_orientation)
target_orientation = normalize_orientation_torch(target[:, 1:3])
```

Then the loss uses a dot-product style error:

```text
orientation_err = 1 - dot(pred_orientation, target_orientation)
```

The loss is masked by centerline confidence, traceability confidence, and a
junction mask. This means the architecture does not need to produce meaningful
orientation everywhere. It only needs to produce useful orientation near
traceable centerline regions.

### 11.3 Traceability Loss

Traceability is trained with binary cross entropy with logits. It is a dense
support field.

This supports the architecture's multi-head design: centerline confidence and
traceability are separate outputs. A pixel can have a local centerline-like
response but low traceability, or it can be part of a traceable region with
different centerline confidence.

### 11.4 Radius and Bundle-Count Losses

Radius and bundle count are passed through sigmoid and trained with Smooth L1
loss near centerline regions.

This is why they are output as logits but interpreted as normalized values. The
sigmoid constrains them to `[0, 1]`.

These auxiliary outputs encourage the shared backbone to learn structural
properties of fibers, not only binary centerline presence.

## 12. How the Architecture Connects to the Decoder

The model output is not the final skeleton. In inference, the pipeline does the
following:

1. Converts centerline logits to centerline probability.
2. Converts orientation channels to a tangent vector map.
3. Computes orientation confidence from the magnitude of the orientation output.
4. Converts traceability logits to traceability probability.
5. Converts radius and bundle-count logits to normalized scalar maps.
6. Runs `CenterlineGraphDecoder`.

The decoder uses:

- centerline probability for candidate ridge strength,
- orientation for non-maximum suppression perpendicular to the fiber direction,
- traceability for support weighting,
- orientation confidence as an additional support factor,
- endpoint bridging based on orientation and support.

This means the architecture is designed as a field predictor rather than a pure
segmentation model. The CNN produces dense geometric fields. The decoder turns
those fields into a topological skeleton.

This is a reasonable design. CNNs are good at dense local prediction, while
explicit decoders can impose geometric rules that are awkward to learn purely
from convolutional layers.

## 13. Evaluation of the Current Architecture for STED Skeletonization

The current architecture is a sound baseline. It has several appropriate
choices:

- U-Net structure is well suited to dense prediction.
- Skip connections preserve high-resolution information.
- Residual blocks improve trainability.
- GroupNorm is appropriate for small-batch microscopy training.
- Multi-head output matches the downstream skeletonization pipeline.
- Double-angle orientation is the correct representation for unoriented fibers.
- ASPP provides broad context for ambiguous or faint fiber regions.
- The output remains full resolution.

However, the architecture is somewhat generic. It is not strongly specialized
for thin curvilinear skeletons. The main concerns are:

- Four downsampling stages make the bottleneck coarse.
- ASPP dilation 8 at the bottleneck may be too broad for small crops.
- Most capacity is spent at low resolution.
- The final heads are only 1x1 convolutions, with no task-specific spatial
  refinement.
- Zero padding may create boundary artifacts.
- The model has no explicit line-filter, steerable-filter, or topology-aware
  architectural bias.
- The `out_channels` argument is misleading because it does not control the
  output.

The current architecture can work, but there is room to better align it with the
specific task.

## 14. Proposed Architecture Changes

This section describes architecture changes that may improve performance for
STED fiber skeletonization. These are not changes currently present in
`src/model.py`; they are recommended changes to consider.

### 14.1 Reduce the Aggressiveness of Bottleneck ASPP Dilation

Current ASPP dilation rates:

```text
2, 4, 8
```

Recommended first experiment:

```text
1, 2, 4
```

or:

```text
1, 2, 3
```

Why this may be better:

- The target centerline is very narrow.
- The bottleneck is already downsampled by 16.
- Dilation 8 at bottleneck resolution can be effectively global on small crops.
- Large dilation can emphasize broad context over exact location.
- Smaller dilation still provides multi-scale context but keeps it more local.

The goal is not to remove context. The goal is to make the context scale match
the problem scale. For thin fibers, local and medium-range continuity are often
more useful than extremely broad context.

### 14.2 Make Network Depth Configurable or Use Three Downsampling Stages

Current depth:

```text
H -> H/2 -> H/4 -> H/8 -> H/16
```

For a 64x64 crop:

```text
64 -> 32 -> 16 -> 8 -> 4
```

Recommended experiment:

```text
H -> H/2 -> H/4 -> H/8
```

This would use three pooling operations instead of four.

Why this may be better:

- The bottleneck would retain more spatial detail.
- A 64x64 crop would have an 8x8 bottleneck instead of 4x4.
- Thin centerline localization would rely less on reconstructing detail from a
  very coarse representation.
- The model would likely be lighter and faster.

The tradeoff is reduced global context. But if fibers are mostly determined by
local ridge evidence and medium-range continuity, this may be a good trade.

A practical design would make depth configurable so larger crops can still use a
deeper model.

### 14.3 Move Some Multi-Scale Context Into the Decoder

Current architecture applies multi-scale dilated context only at the bottleneck.

Recommended change:

- Keep a lighter ASPP bottleneck.
- Add modest dilation or multi-scale refinement in higher-resolution decoder
  stages, especially near `d2` or `d1`.

For example:

```text
full-resolution refinement:
  3x3 dilation 1 branch
  3x3 dilation 2 branch
  1x1 projection
```

Why this may be better:

- Multi-scale context near full resolution can help distinguish ridges without
  losing precise location.
- Dilation 2 at full resolution has a much more interpretable footprint than
  dilation 8 at 1/16 resolution.
- Fine-scale centerline placement benefits from local context around the final
  ridge.

This is especially relevant because the decoder performs ridge-like prediction,
not just coarse object segmentation.

### 14.4 Add a High-Resolution Refinement Block Before the Heads

Current final representation:

```text
d1 -> 1x1 heads
```

Recommended change:

```text
d1 -> high-resolution refinement block -> heads
```

The refinement block could be another `ResidualBlock2D(widths[0], widths[0])`
or a small multi-scale refinement module.

Why this may be better:

- The final decoder block has to fuse skip features and produce a shared
  representation for all tasks.
- A dedicated final refinement block gives the model one more opportunity to
  sharpen full-resolution ridge features.
- The cost is small because the channel width is only 32.

This is likely one of the highest-value low-risk changes.

### 14.5 Use Task-Specific Shallow Heads

Current heads:

```text
d1 -> 1x1 centerline
d1 -> 1x1 orientation
d1 -> 1x1 traceability
d1 -> 1x1 radius
d1 -> 1x1 bundle count
```

Recommended change:

```text
d1 -> shared refinement
   -> centerline head: 3x3 + norm + activation + 1x1
   -> orientation head: 3x3 + norm + activation + 1x1
   -> traceability head: 1x1 or 3x3 + 1x1
   -> radius head: 3x3 + 1x1
   -> bundle-count head: 3x3 + 1x1
```

Why this may be better:

- Centerline localization and orientation estimation are related but not
  identical tasks.
- Radius and bundle count may need slightly broader local context than
  centerline classification.
- A 1x1 head cannot perform any final spatial reasoning.
- Shallow task-specific heads add capacity exactly where dense prediction is
  made.

This would increase parameter count only modestly if the head width remains
small.

### 14.6 Replace Transposed Convolution With Bilinear Upsampling Plus Convolution

Current upsampling:

```python
ConvTranspose2d(kernel_size=2, stride=2)
```

Recommended experiment:

```text
bilinear upsample by 2
3x3 convolution to adjust channels
GroupNorm
GELU
```

Why this may be better:

- Bilinear upsampling is deterministic and smooth.
- The following convolution learns feature refinement without relying on
  transposed-convolution sampling.
- It can reduce upsampling artifacts.

The current transposed convolutions use `kernel_size=2` and `stride=2`, which is
already a relatively safe configuration. This proposed change is therefore not
as urgent as the ASPP and high-resolution refinement changes. It is still worth
testing if predictions show checkerboard-like artifacts or grid bias.

### 14.7 Use Reflection Padding Instead of Zero Padding

Current convolution padding is implicit zero padding.

Recommended change:

- Add explicit reflection padding before 3x3 convolutions.
- Set convolution padding to 0 after the reflection pad.

Why this may be better:

- STED crops may contain fibers touching crop boundaries.
- Zero padding creates artificial dark borders.
- Reflection padding gives boundary convolutions more realistic local context.

This can improve edge behavior, especially during tiled inference.

The tradeoff is a slightly more complex convolution block. It also changes
behavior compared with existing checkpoints, so it should be introduced only
when retraining.

### 14.8 Add Explicit Ridge-Oriented Inductive Bias

The current model uses generic isotropic 3x3 convolution. It can learn oriented
filters, but it is not forced to do so.

Possible changes:

- Add separable asymmetric convolutions such as 1x3 and 3x1 in early or final
  refinement blocks.
- Add a small bank of oriented filters or steerable-like convolutions.
- Add parallel line-sensitive branches in the high-resolution refinement module.

Why this may be better:

- Fibers are elongated structures.
- Ridge detection depends on comparing intensity along and across local
  directions.
- Explicit line-sensitive filters may improve sample efficiency and reduce
  false positives from blob-like noise.

This should be treated carefully. A generic U-Net may already learn adequate
oriented filters from data. Explicit ridge bias is useful if the model struggles
with faint fibers, crossings, or noisy punctate artifacts.

### 14.9 Add Orientation-Aware Coupling Between Centerline and Orientation

Current design:

- The centerline head and orientation head are independent 1x1 projections from
  `d1`.
- The loss couples them indirectly by applying orientation loss mainly near
  centerlines.
- The decoder couples them later during ridge non-maximum suppression.

Potential architecture change:

- Use a shared centerline-orientation refinement branch.
- Predict orientation confidence explicitly.
- Let centerline refinement use orientation features before the final
  centerline logit.

Why this may be better:

- A true centerline ridge should have a coherent tangent direction.
- False positives often have weak or inconsistent orientation.
- Coupling these predictions earlier may improve decoder-ready outputs.

This is more invasive than simply adding refinement blocks. It should be tested
after simpler changes.

### 14.10 Make Output Channel Configuration Explicit

Current issue:

```python
out_channels=6
```

is accepted but not used.

Recommended change:

- Remove `out_channels` from the constructor, or
- Validate that `out_channels == 6`, or
- Build heads based on a structured output configuration.

Why this may be better:

- It prevents silent user confusion.
- It makes checkpoint compatibility clearer.
- It documents that this is specifically a structural-v2 six-channel model.

This does not change predictive quality, but it improves maintainability.

## 15. Suggested Revised Architecture Direction

A conservative revised model could keep the same overall U-Net design while
changing only the parts most relevant to skeleton quality:

```text
Input
  |
Encoder with residual blocks and 3 or 4 scales
  |
Lighter ASPP, dilation rates 1/2/4 instead of 2/4/8
  |
Decoder with skip connections
  |
Full-resolution refinement residual block
  |
Small task-specific heads
  |
Six structural output channels
```

This preserves the strengths of the current model:

- dense U-Net prediction,
- residual trainability,
- GroupNorm for small batches,
- multi-scale context,
- multi-task structural outputs.

It adjusts the architecture toward the main difficulty of the task:

- thin centerline localization,
- stable ridge prediction,
- high-resolution orientation consistency,
- reduced over-reliance on very coarse context.

## 16. Priority of Recommended Changes

The most practical order to test changes is:

1. Add a full-resolution refinement block before the heads.
2. Reduce ASPP dilation rates to `1, 2, 4` or `1, 2, 3`.
3. Add shallow task-specific heads for centerline and orientation.
4. Test three downsampling stages for small crops.
5. Consider bilinear upsampling plus convolution if artifacts are visible.
6. Consider reflection padding if boundary artifacts are visible.
7. Explore explicit ridge-oriented filters if the model still confuses fibers
   with punctate or blob-like noise.

The first three changes are likely to give the best balance of implementation
simplicity and task alignment.

## 17. Final Assessment

The current architecture is appropriate as a strong baseline. It is a sensible
residual U-Net adapted for structural STED prediction, and its six-channel
output design matches the downstream decoder well.

The main architectural mismatch is scale. The model predicts a very narrow
centerline target, but it spends most of its capacity in a deep, highly
contextual bottleneck with large dilation rates. For this problem, more
high-resolution refinement and more modest multi-scale context may be better.

The most important principle for improving this architecture is:

```text
Keep enough context to follow fibers, but spend more modeling capacity at the
resolution where centerline placement is decided.
```

