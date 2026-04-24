import argparse

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


def main(args):
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
        min_edt=args.min_edt,
        visibility_floor=args.visibility_floor,
        use_amp=not args.no_amp,
    )
    print(f"Successfully traced {len(result.streamlines)} fiber segments.")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_inference_arguments(parser)
    args = parser.parse_args()
    main(args)
