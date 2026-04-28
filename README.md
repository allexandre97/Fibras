# Fibras

## Environment

This project is configured for NVIDIA GPU training. The Conda environment uses
the official PyTorch CUDA 12.8 wheels.

Rebuild the environment from scratch:

```bash
conda env remove -n fibras
conda env create -f environment.yml
conda activate fibras
python verify_cuda.py
```

`python verify_cuda.py` must report `cuda_available=True` before training.

If it reports `False`, fix the machine first:

```bash
nvidia-smi
```

That command must succeed before PyTorch can use the GPUs.

## Dataset

Build a calibrated synthetic 2D STED dataset:

```bash
python analyze_real_sted.py profile-real --real_dir /ssd/STED_dataset/data --output_dir reports/sted_real --patch_size 512
python generate_dataset.py \
  --output_dir /ssd/Fibras_Dataset/sted2d_calibrated_v1 \
  --bounds 1024 1024 \
  --synth_depth 16 \
  --calibration_profile reports/sted_real/sted_real_profile.json
```

## Training

Run 2D STED training:

```bash
python train.py fit \
  --gpus 0 \
  --data_dir /ssd/Fibras_Dataset/sted2d_calibrated_v1 \
  --dim 2 \
  --crop_size 512
```
