# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the reusable pipeline code: geometry and boundaries in `core.py`, synthetic generation in `synthesis.py`, rasterization and target construction in `rasterization.py` and `targets.py`, and model, tracking, inference, and visualization logic. Top-level scripts drive the workflow: `analyze_real_sted.py`, `generate_dataset.py`, `train.py`, `inference.py`, `visual.py`, `sweep.py`, and `verify_cuda.py`. `tests/` holds regression coverage for STED synthesis, target generation, calibration, decoding, and loss behavior. Treat `wandb/`, `weights/`, `reports/`, `__pycache__/`, generated images/TIFFs, and `*.npz` as generated artifacts, not source.

## Build, Test, and Development Commands
Use the `fibras` Conda environment defined in `environment.yml`.

`conda env create -f environment.yml && conda activate fibras`
Installs the runtime dependencies used by training, inference, and visualization.

`python generate_dataset.py --output_dir data/sted2d --bounds 64 64`
Builds synthetic train/val/test splits for 2D STED-style data.

`python train.py fit --data_dir data/sted2d --dim 2`
Trains `STEDResUNet2D` and saves checkpoints under `weights/`.

`python train.py evaluate --model_path weights/sted_resunet2d_final.pth --data_dir data/sted2d --dim 2`
Evaluates a saved model on the `test/` split.

`python -m unittest discover -s tests`
Runs the regression suite. For faster iteration, target a file: `python -m unittest tests.test_sted_targets -v`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, snake_case for functions, variables, and CLI flags, and CamelCase for classes such as `STEDResUNet2D`. Keep shared logic inside `src/` rather than duplicating it in top-level scripts. Group imports as standard library, third-party, then local modules. No formatter or linter config is checked in, so match the surrounding file style and keep argument names explicit, for example `--visibility_weight_floor`.

## Testing Guidelines
Write tests with `unittest` under `tests/`, using `test_*.py` filenames and `test_*` methods. Prefer deterministic regression tests for math-heavy code in `src/synthesis.py`, `src/targets.py`, and training losses over slow end-to-end jobs. There is no enforced coverage gate, but behavior changes in generation, supervision, or loss weighting should include focused tests.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects with narrow scope, for example `Add napari visualizer` and `Tune sweep ranges for visibility weighting hyperparameters`. Keep commits focused on one change. PRs should state the purpose, list affected scripts or modules, include the test commands you ran, note any dataset or checkpoint impact, and attach screenshots when changing Napari views or saved image outputs.
