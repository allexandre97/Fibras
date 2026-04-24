import argparse

from inference import main as run_single_image_inference
from src.real_inference import add_inference_arguments
from visualize_dataset import show_dataset_sample, show_sted_debug, show_synthetic_data


def main():
    parser = argparse.ArgumentParser(description="Unified Fibras visualization entrypoint.")
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    inf_parser = subparsers.add_parser("inference", help="Run model inference on one image")
    add_inference_arguments(inf_parser)

    ds_parser = subparsers.add_parser("dataset", help="Visualize synthetic dataset samples")
    ds_parser.add_argument("--file", type=str, default="", help="Specific .pt file to open")
    ds_parser.add_argument("--data_dir", type=str, default="", help="Dataset root directory")
    ds_parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    ds_parser.add_argument("--index", type=int, default=0)
    ds_parser.add_argument("--random_sample", action="store_true")
    ds_parser.add_argument("--visibility_floor", type=float, default=0.25)

    debug_parser = subparsers.add_parser("sted-debug", help="Debug 3D-to-2D STED synthesis")
    debug_parser.add_argument("--bounds", type=int, nargs=2, default=[64, 64])
    debug_parser.add_argument("--synth_depth", type=int, default=16)
    debug_parser.add_argument("--label_slab_thickness", type=float, default=None)
    debug_parser.add_argument("--label_slab_scale", type=float, default=1.3)
    debug_parser.add_argument("--annotation_weight_floor", type=float, default=0.25)
    debug_parser.add_argument("--soft_skeleton_alpha", type=float, default=0.35)
    debug_parser.add_argument("--visibility_weight_floor", type=float, default=0.03)
    debug_parser.add_argument("--seed", type=int, default=None)
    debug_parser.add_argument("--save", type=str, default=None)
    debug_parser.add_argument("--no_show", action="store_true")

    args = parser.parse_args()

    if args.mode == "inference":
        run_single_image_inference(args)
    elif args.mode == "dataset":
        if args.file:
            show_synthetic_data(args.file, visibility_floor=args.visibility_floor)
        elif args.data_dir:
            show_dataset_sample(
                data_dir=args.data_dir,
                split=args.split,
                index=args.index,
                random_sample=args.random_sample,
                visibility_floor=args.visibility_floor,
            )
        else:
            raise ValueError("dataset mode requires either --file or --data_dir")
    elif args.mode == "sted-debug":
        show_sted_debug(
            bounds=args.bounds,
            synth_depth=args.synth_depth,
            label_slab_thickness=args.label_slab_thickness,
            label_slab_scale=args.label_slab_scale,
            annotation_weight_floor=args.annotation_weight_floor,
            soft_skeleton_alpha=args.soft_skeleton_alpha,
            visibility_weight_floor=args.visibility_weight_floor,
            seed=args.seed,
            save_path=args.save,
            show=not args.no_show,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
