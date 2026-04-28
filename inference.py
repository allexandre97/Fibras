import argparse
import sys

from src.batch_inference import add_batch_arguments, main as batch_main
from src.inference_calibration import add_calibration_arguments, main as calibrate_main
from src.real_inference import (
    add_inference_arguments,
    build_output_paths,
    load_image_for_inference,
    load_sted_model,
    resolve_output_base,
    run_real_image_inference,
    save_inference_outputs,
    save_preview_panel,
    show_interactive_result,
    summarize_inference_result,
)


def run_single(args):
    if args.dim != 2:
        raise ValueError("The upgraded STED inference path is 2D only. Use --dim 2.")

    if args.downsample > 1.0:
        print("Warning: --downsample is ignored by the 2D STED tiled inference path.")

    model, device = load_sted_model(
        model_path=args.model_path,
        base_filters=args.base_filters,
        device_spec=args.device,
    )
    print(f"Executing inference on: {device}")

    image = load_image_for_inference(args.image_path, dim=args.dim)
    print(
        f"Processing {image.shape} image with tile_size={args.tile_size}, "
        f"tile_overlap={args.tile_overlap}"
    )
    result = run_real_image_inference(
        model,
        image=image,
        image_path=args.image_path,
        device=device,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        centerline_threshold=args.centerline_threshold,
        use_amp=not args.no_amp,
    )
    print(f"Successfully reconstructed {len(result.component_paths)} fiber components.")

    base_path = resolve_output_base(args.image_path, output_dir=args.output_dir)
    output_paths = save_inference_outputs(result, base_path)
    metrics = summarize_inference_result(result, output_paths=output_paths)
    print(f"Skeleton graph exported to: {output_paths['skeleton']}")

    if args.save_preview:
        preview_path = args.preview_path or build_output_paths(base_path)["preview"]
        save_preview_panel(result, preview_path, metrics=metrics)
        print(f"Preview exported to: {preview_path}")

    if args.visualize:
        print("Launching Napari AdvancedVisualizer...")
        show_interactive_result(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2D STED inference tools.")
    subparsers = parser.add_subparsers(dest="mode")

    single_parser = subparsers.add_parser("single", help="Run inference on one image")
    add_inference_arguments(single_parser)

    batch_parser = subparsers.add_parser("batch", help="Run inference on a directory of TIFF images")
    add_batch_arguments(batch_parser)

    calibrate_parser = subparsers.add_parser("calibrate-decoder", help="Calibrate decoder parameters")
    add_calibration_arguments(calibrate_parser)

    return parser


def _normalize_programmatic_args(args):
    """Normalize already-parsed arguments passed by another entrypoint."""
    if getattr(args, "mode", None) in {None, "inference"}:
        args.mode = "single"
    return args


def main(argv=None):
    parser = build_parser()

    if isinstance(argv, argparse.Namespace):
        args = _normalize_programmatic_args(argv)
    elif isinstance(argv, dict):
        args = _normalize_programmatic_args(argparse.Namespace(**argv))
    else:
        argv = sys.argv[1:] if argv is None else list(argv)

        # Preserve the old single-image invocation style: python inference.py --model_path ...
        if argv and argv[0] not in {"single", "batch", "calibrate-decoder", "-h", "--help"}:
            argv = ["single", *argv]

        args = parser.parse_args(argv)

    if args.mode == "single":
        run_single(args)
    elif args.mode == "batch":
        batch_main(args)
    elif args.mode == "calibrate-decoder":
        calibrate_main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
