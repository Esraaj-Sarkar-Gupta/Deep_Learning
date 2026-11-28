# Airfoil-only PINN

This directory provides an airfoil-only command for the shared velocity PINN implementation in `generalized_PINN/train.py`. It runs the same `plain`, `attention`, or `segmented` experiment against the plunging-airfoil data and writes results under `airfoil_PINN/runs/airfoil/`.

From the repository root:

```bash
python airfoil_PINN/train.py train --device auto
```

The entry point defaults to `proprietary_data/300_1pt0/Spatial_files_with_body`, the airfoil-specific config in `configs/default.json`, and the `airfoil` dataset. It accepts the shared trainer's options, including `--experiment`, `--config`, `--airfoil-data`, `--airfoil-crop`, `--device`, and `--resume`. It also supports:

```bash
python airfoil_PINN/train.py verify
python airfoil_PINN/train.py train --experiment segmented --device cuda
```

It predicts only `u` and `v`, uses the moving ellipse geometry and no-slip motion from the airfoil dataset, and reports held-out velocity metrics. Training requires NumPy and PyTorch; Matplotlib is optional for plots.
