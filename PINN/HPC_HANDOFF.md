# HPC handoff

The Slurm job has **not** been submitted. The current real-data result is a short local CPU pilot, not a converged result. The supplied paper describes sequential learning; it does not specify a temporal-attention architecture. The attention model here uses three fixed temporal tokens per query. The segmented model transfers weights between equal time intervals; it does not enforce interface continuity explicitly. Velocity sensors supervise the network, and incompressible Navier–Stokes residuals provide physics loss. Pressure is inferred, not supervised.

## Transfer

Transfer `train.py`, `configs/`, `scripts/train.slurm`, `tests/smoke.py`, `prepared/cylinder_crop.npz`, and `runs/real/sensors.npz`. Optionally transfer `runs/real/results.csv`, `runs/real/plain/metrics.json`, `runs/real/plain/snapshot.png`, and `runs/real/interpolation.json` to retain local results. The synthetic outputs in `runs/synthetic/` are test artifacts and need not be transferred. Do not reuse the local pilot checkpoint with the overnight configuration: checkpoint configuration must match.

Transfer the **prepared crop**, not the 1.16 GB raw export, for overnight training. It contains all 201 times and nodes in x=[-1.5,8], y=[-2,2]; it was created by:

```bash
python train.py prepare --data-path /path/to/fixed_cylinder_atRe100 --output prepared/cylinder_crop.npz --crop -1.5 8 -2 2
```

To repeat full-domain interpolation or the original EDA, also transfer the raw field export. The separate reaction file is only needed for the EDA's optional force evaluation. The crop is derived from the raw export; keep the original safely available.

From a shell with SSH access, for example (replace the local and remote placeholders):

```bash
ssh USER@LOGIN 'mkdir -p /REMOTE/PROJECT/configs /REMOTE/PROJECT/scripts /REMOTE/PROJECT/tests /REMOTE/PROJECT/prepared /REMOTE/PROJECT/runs/real'
rsync -av train.py configs scripts tests prepared/cylinder_crop.npz USER@LOGIN:/REMOTE/PROJECT/
# The previous rsync puts cylinder_crop.npz in the project root; place it in prepared/:
ssh USER@LOGIN 'mv /REMOTE/PROJECT/cylinder_crop.npz /REMOTE/PROJECT/prepared/'
rsync -av runs/real/sensors.npz USER@LOGIN:/REMOTE/PROJECT/runs/real/
```

Alternatively copy each listed path with `scp` and preserve the directory layout. From Windows PowerShell, run these from the project root if OpenSSH and rsync are installed, or use your preferred file-transfer tool.

## Environment and verification

Use Python 3.10 or newer and a cluster-supported CUDA-enabled PyTorch installation matched to the cluster's driver. Install NumPy and Matplotlib in the same environment. For example, after loading the cluster's Python/CUDA modules as appropriate:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
# Install torch using the cluster's recommended CUDA build or module.
python -c 'import torch,numpy,matplotlib; print(torch.__version__, torch.cuda.is_available())'
python train.py verify --data-path prepared/cylinder_crop.npz --run-dir runs/real
python tests/smoke.py
```

Verification should show `(201, 43855, 3)`, Re 100, time 400–420 and 64 sensors. `verify` checks array shape and finite field values, and validates the saved sensor-coordinate signature. For byte-for-byte transfer verification, compare `Get-FileHash prepared/cylinder_crop.npz -Algorithm SHA256` in local PowerShell with `sha256sum prepared/cylinder_crop.npz` on the cluster. `torch.cuda.is_available()` may be false on a login node; check it inside a GPU allocation.

## Submit and monitor

Fill in your cluster's **account**, **partition**, **GPU request syntax**, and any required **module setup**. No example cluster script was present locally. `scripts/train.slurm` sets 12 hours, 4 CPUs and 16 GB; adjust them to your allocation and policy. It forces `--device cuda`, so a GPU and CUDA-capable PyTorch are required. Set the absolute Python executable; modules or activation can be added to the script if your cluster needs them. The budget is 10,000 optimizer steps for plain, 4 x 10,000 for segmented, then 10,000 for attention; overnight completion time is hardware-dependent.

```bash
cd /REMOTE/PROJECT
mkdir -p runs/overnight
cp runs/real/sensors.npz runs/overnight/sensors.npz
export DATA_PATH="$PWD/prepared/cylinder_crop.npz"
export RUN_DIR="$PWD/runs/overnight"
export CONFIG="$PWD/configs/overnight.json"
export PYTHON="$PWD/.venv/bin/python"
export EXPERIMENT=all
sbatch --account=ACCOUNT --partition=GPU_PARTITION --gres=gpu:1 --export=ALL scripts/train.slurm
squeue -u "$USER"
```

Replace `--gres=gpu:1` if your site uses a different GPU option. Standard output and errors go to `slurm-JOBID.out` in the submit directory. The script saves each model's checkpoint, metrics and figure before proceeding to the next. Checkpoints are atomic and written every 250 steps and at segment boundaries. To resume after a timeout, run the **same sbatch command** with the same environment variables, run directory, data and config; `--resume` is already in the script. Completed models are evaluated again, and remaining models continue. For separate allocations set `EXPERIMENT=plain`, `segmented`, or `attention` before submission; the shared run directory stores separate checkpoints.

Open `runs/overnight/results.csv` first, then `runs/overnight/plain/snapshot.png`. Per-model `metrics.json` reports the optimization budget and snapshot time. All scores evaluate non-sensor nodes at one held-out spatial snapshot, while interpolation reports all frames and nodes outside the sensor set. Pressure is aligned using the mean error on the evaluated snapshot, so pressure results are gauge-aligned and use evaluation truth for that offset. Linear interpolation uses nearest-sensor extrapolation outside the sensor convex hull and does not account for the cylinder obstruction; inspect its coverage before comparison. The local `runs/real/interpolation.json` is the full-domain interpolation with the same crop sensor locations; the local pilot snapshot is on the crop.

No cluster GPU, CUDA software, Slurm directives, walltime, queue policy, or transfer has been tested here.
