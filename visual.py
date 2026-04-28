import argparse
import sys

from src.dataset_visualization import (
    export_dataset_pngs,
    export_sample_png,
    show_dataset_sample,
    show_sted_debug,
    show_synthetic_data,
)


def main():
    parser = argparse.ArgumentParser(description="Unified Fibras visualization entrypoint.")
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    inf_parser = subparsers.add_parser("inference", help="Run model inference on one image")
    if len(sys.argv) > 1 and sys.argv[1] == "inference":
        from src.real_inference import add_inference_arguments

        add_inference_arguments(inf_parser)

    ds_parser = subparsers.add_parser("dataset", help="Visualize synthetic dataset samples")
    ds_parser.add_argument("--file", type=str, default="", help="Specific .pt file to open")
    ds_parser.add_argument("--data_dir", type=str, default="", help="Dataset root directory")
    ds_parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    ds_parser.add_argument("--index", type=int, default=0)
    ds_parser.add_argument("--random_sample", action="store_true")
    ds_parser.add_argument("--score_floor", "--visibility_floor", dest="score_floor", type=float, default=0.25)

    export_parser = subparsers.add_parser("export-dataset", help="Export synthetic dataset samples to PNG files")
    export_parser.add_argument("--file", type=str, default="", help="Specific .pt file to export")
    export_parser.add_argument("--data_dir", type=str, default="", help="Dataset root directory")
    export_parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    export_parser.add_argument("--export_dir", type=str, required=True)
    export_parser.add_argument("--num_samples", type=int, default=0)
    export_parser.add_argument("--score_floor", "--visibility_floor", dest="score_floor", type=float, default=0.25)

    debug_parser = subparsers.add_parser("sted-debug", help="Debug 3D-to-2D STED synthesis")
    debug_parser.add_argument("--bounds", type=int, nargs=2, default=[64, 64])
    debug_parser.add_argument("--synth_depth", type=int, default=16)
    debug_parser.add_argument("--label_slab_thickness", type=float, default=None)
    debug_parser.add_argument("--label_slab_scale", type=float, default=1.3)
    debug_parser.add_argument("--annotation_weight_floor", type=float, default=0.25)
    debug_parser.add_argument("--soft_skeleton_alpha", type=float, default=0.35)
    debug_parser.add_argument("--visibility_weight_floor", type=float, default=0.03)
    debug_parser.add_argument(
        "--bundle_phenotype",
        action="store_true",
        help="Force coherent longitudinal bundles in the STED debug sample.",
    )
    debug_parser.add_argument("--bundle_probability", type=float, default=None)
    debug_parser.add_argument("--bundle_size_range", type=int, nargs=2, default=[2, 6])
    debug_parser.add_argument("--bundle_separation_range", type=float, nargs=2, default=[0.7, 2.4])
    debug_parser.add_argument("--bundle_direction_jitter_degrees", type=float, default=5.0)
    debug_parser.add_argument("--bundle_lateral_jitter_fraction", type=float, default=0.20)
    debug_parser.add_argument("--bundle_axial_jitter_fraction", type=float, default=0.25)
    debug_parser.add_argument("--seed", type=int, default=None)
    debug_parser.add_argument("--save", type=str, default=None)
    debug_parser.add_argument("--no_show", action="store_true")

    args = parser.parse_args()

    if args.mode == "inference":
        from inference import run_single as run_single_image_inference
        run_single_image_inference(args)
    elif args.mode == "dataset":
        if args.file:
            show_synthetic_data(args.file, score_floor=args.score_floor)
        elif args.data_dir:
            show_dataset_sample(
                data_dir=args.data_dir,
                split=args.split,
                index=args.index,
                random_sample=args.random_sample,
                score_floor=args.score_floor,
            )
        else:
            raise ValueError("dataset mode requires either --file or --data_dir")
    elif args.mode == "export-dataset":
        if args.file:
            export_sample_png(args.file, args.export_dir, score_floor=args.score_floor)
        elif args.data_dir:
            export_dataset_pngs(
                data_dir=args.data_dir,
                split=args.split,
                export_dir=args.export_dir,
                num_samples=args.num_samples,
                score_floor=args.score_floor,
            )
        else:
            raise ValueError("export-dataset mode requires either --file or --data_dir")
    elif args.mode == "sted-debug":
        show_sted_debug(
            bounds=args.bounds,
            synth_depth=args.synth_depth,
            label_slab_thickness=args.label_slab_thickness,
            label_slab_scale=args.label_slab_scale,
            annotation_weight_floor=args.annotation_weight_floor,
            soft_skeleton_alpha=args.soft_skeleton_alpha,
            visibility_weight_floor=args.visibility_weight_floor,
            bundle_phenotype=args.bundle_phenotype,
            bundle_probability=args.bundle_probability,
            bundle_size_range=args.bundle_size_range,
            bundle_separation_range=args.bundle_separation_range,
            bundle_direction_jitter_degrees=args.bundle_direction_jitter_degrees,
            bundle_lateral_jitter_fraction=args.bundle_lateral_jitter_fraction,
            bundle_axial_jitter_fraction=args.bundle_axial_jitter_fraction,
            seed=args.seed,
            save_path=args.save,
            show=not args.no_show,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
