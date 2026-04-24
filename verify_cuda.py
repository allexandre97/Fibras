import sys


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        print(f"Failed to import torch: {exc}", file=sys.stderr)
        return 1

    print(f"python_executable={sys.executable}")
    print(f"torch_version={torch.__version__}")
    print(f"torch_cuda_build={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print(
            "CUDA is not available. Check that the NVIDIA driver is working "
            "first (nvidia-smi must succeed), then verify you installed the "
            "CUDA wheel build of PyTorch rather than a CPU-only build.",
            file=sys.stderr,
        )
        return 1

    for index in range(torch.cuda.device_count()):
        print(f"cuda_device_{index}={torch.cuda.get_device_name(index)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
