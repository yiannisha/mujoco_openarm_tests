# OpenArm MuJoCo Workspace

This repository contains Gymnasium environments for OpenArm MuJoCo scenes, scripted IK policies, ACT-style dataset collection, and ACT training utilities.

## Prerequisites

- Python 3.10 or newer.
- A working MuJoCo/OpenGL stack. Headless machines usually need EGL or OSMesa configured for offscreen rendering.
- Git submodules initialized if this repo is cloned with submodules:

```bash
git submodule update --init --recursive
```

For GPU training, install the PyTorch build that matches the target CUDA runtime before or after running the setup script. The default `requirements.txt` works for CPU and Apple Silicon setups through the normal PyTorch wheels.

## Setup

From the repository root:

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

The setup script creates `.venv`, installs Python dependencies, installs this repository in editable mode, installs the vendored ACT DETR module, and runs a headless environment reset as a smoke test.

To use a specific Python binary or virtualenv path:

```bash
PYTHON=python3.11 VENV_DIR=.venv311 bash scripts/setup.sh
```

## Smoke Tests

Headless policy rollout:

```bash
python run_ik_policy.py OpenArmBimanualReach-v0 none 1
```

Interactive rollout, if the machine has display support:

```bash
python run_ik_policy.py OpenArmBimanualReach-v0 human 1
```

Camera rendering check:

```bash
python view_wrist_cameras.py --env-id OpenArmBimanualReach-v0 --output-dir outputs/cameras
```

## Common Workflows

Collect ACT-style OpenArm rollouts:

```bash
python collect_act_dataset.py \
  --env-id OpenArmBimanualPickPlace-v0 \
  --episodes 10 \
  --output-dir datasets/act_openarm
```

Inspect a collected dataset:

```bash
python inspect_act_dataset.py datasets/act_openarm
```

Train the OpenArm ACT model:

```bash
python -m ml.act.train \
  --dataset-dir datasets/act_openarm \
  --ckpt-dir checkpoints/openarm_act \
  --num-epochs 2000
```

## Legacy ACT Environment

The upstream ACT code under `act/` also includes its original Conda environment at `act/conda_env.yaml`. Use that only if you specifically need the original ALOHA simulation scripts. The root setup path above is intended for the OpenArm environments and training scripts in this workspace.
