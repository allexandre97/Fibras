## 1. Proposed Architecture Changes

This section describes architecture changes that may improve performance for
STED fiber skeletonization. These are not changes currently present in
`src/model.py`; they are recommended changes to consider.

### 1.1 Reduce the Aggressiveness of Bottleneck ASPP Dilation

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

### 1.2 Make Network Depth Configurable or Use Three Downsampling Stages

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

### 1.3 Move Some Multi-Scale Context Into the Decoder

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

### 1.4 Add a High-Resolution Refinement Block Before the Heads

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

### 1.5 Use Task-Specific Shallow Heads

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

### 1.6 Replace Transposed Convolution With Bilinear Upsampling Plus Convolution

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

### 1.7 Use Reflection Padding Instead of Zero Padding

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

### 1.8 Add Explicit Ridge-Oriented Inductive Bias

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

### 1.9 Add Orientation-Aware Coupling Between Centerline and Orientation

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

### 1.10 Make Output Channel Configuration Explicit

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

## 2. Suggested Revised Architecture Direction

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

## 3. Priority of Recommended Changes

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