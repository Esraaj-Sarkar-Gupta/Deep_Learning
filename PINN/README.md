# Cylinder flow PINN

Run training commands from this directory. `configs/`, `scripts/`, `tests/`, `prepared/`, and `runs/` contain the training configuration, code support, prepared data, and saved experiments. `outputs_try1/` contains the returned HPC results. See [HPC_HANDOFF.md](HPC_HANDOFF.md) for setup and training instructions.

The original flow and reaction datasets are in `../Data/`. The `../cylinder_eda/analyze.py` script uses the original flow dataset there by default; pass `--reaction-path ../Data/Re100_fixe00.reactions` from this directory to include reaction-force analysis.

