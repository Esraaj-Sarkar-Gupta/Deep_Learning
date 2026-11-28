# Generalized velocity PINNs

`train.py` uses one conditioned model implementation for the fixed cylinder data in `Data/` and the plunging elliptic-airfoil data in `proprietary_data/`. The default run trains each of the three existing experiment types on balanced batches from both cases, then reports held-out velocity scores separately for each case.

The attached paper, `docs/rahul_etal.pdf`, is used as a reference for the moving-boundary formulation. Its fixed Eulerian frame, body kinematics, fluid-only physics loss and moving no-slip condition inform the implementation. The paper is not treated as a separate instruction to reproduce its pressure-recovery experiment. This trainer predicts only `u` and `v`.

## Run

From the repository root:

```bash
python generalized_PINN/train.py verify
python generalized_PINN/train.py train --device auto
```

`verify` reads the configured data and checks their layouts. Training defaults to `generalized_PINN/configs/default.json` and writes checkpoints, metrics, sensor layouts and snapshot plots under `generalized_PINN/runs/both/`.

Training requires NumPy and PyTorch. Matplotlib is optional and is used only for the snapshot plots. `verify` can run without PyTorch.

Run one case, or choose one architecture:

```bash
python generalized_PINN/train.py train --dataset cylinder --experiment all
python generalized_PINN/train.py train --dataset airfoil --experiment attention
```

The model choices are `plain`, `attention`, and `segmented`. `segmented` transfers weights forward through equal time intervals; its configured `steps` count applies to each interval. Checkpoints can be resumed with `--resume` when the data and config are unchanged. Use `--config`, `--cylinder-data`, or `--airfoil-data` to point to other files. CUDA is selected automatically when available; pass `--device cuda` to require it.

## Data and physics

- Cylinder input defaults to `Data/prepared/cylinder_crop.npz` with `xy`, `times`, `fields`, and `re` arrays. The trainer reads only the first two field components. The cylinder is represented as a stationary circle centered at `(0, 0)` with radius `0.5`; those values can be changed with `--cylinder-center` and `--cylinder-radius`.
- Airfoil input defaults to `proprietary_data/300_1pt0/Spatial_files_with_body`. The reader uses `x,y,u,v,xb,yb`, ignores other scalar columns, and derives snapshot time from the filename step and `--airfoil-dt`. It fits the supplied body-center positions to the harmonic motion used by `convert_add_body_position.py`. For raw `Spatial_files`, pass that directory and the motion parameters (`--airfoil-amplitude`, `--airfoil-omega`, `--airfoil-phase`, `--airfoil-body-x`). The supplied defaults are `dt=0.0002`, `a=0.5`, `b=0.0625`, amplitude `0.25`, and angular frequency `4`.
- The default airfoil crop is `x=[-1.5,6]`, `y=[-1.5,1.5]`; override it with four `--airfoil-crop` values or use `--airfoil-crop` values covering the desired region.
- Both cases use fixed-frame inputs. Each input carries normalized coordinates/time, Reynolds number, ellipse axes, current body position and velocity, and an approximate signed distance to the body. Batches are balanced across cases.
- Training uses a deterministic set of 64 fixed spatial sensor nodes per case. Sensor labels inside the moving solid are skipped. Separate analytic no-slip samples use zero wall velocity for the cylinder and the prescribed vertical airfoil velocity.
- To avoid a pressure output, the physics loss combines incompressibility with the 2-D vorticity-transport equation, which is the curl of the momentum equations and eliminates pressure. Physics points are sampled in the fluid region. This is a velocity-only formulation; it does not recover pressure.

Scores use held-out spatial nodes at evenly spaced snapshots, with solid-interior points excluded. `metrics.json` and the combined `results.csv` report `u,v` RMSE and relative L2 error. Each case also gets a mid-window reference/prediction/error plot. The dense fields provide evaluation targets only at those held-out nodes; supervised fitting uses the configured sensors.

## What the two cases can establish

Joint conditioning makes it possible for the same architecture and weights to represent the available cylinder and moving-airfoil cases. The current pair changes object shape, motion, Reynolds number, grid and time window together. A score comparison therefore describes these two cases; it cannot isolate which change caused a performance difference or establish generalization to unseen shapes or Reynolds numbers. That needs more cases where shape and Reynolds number vary independently, ideally with held-out cases for evaluation.

The supplied geometry inputs cover circles and ellipses, including the present cylinder and elliptic airfoil. Other body shapes need a corresponding signed-distance and boundary-sampling implementation before they can be included.
