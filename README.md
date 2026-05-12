# OpenArm MuJoCo Workspace

This repository contains Gymnasium environments for OpenArm MuJoCo scenes, scripted IK policies, ACT-style dataset collection, and ACT policy training/evaluation utilities.

The root workflow is for the OpenArm pick-place and reach environments in `openarm_env/`. The upstream ACT code under `act/` is kept as a vendored reference and model implementation.

## Setup

Prerequisites:

- Python 3.10 or newer.
- A working MuJoCo/OpenGL stack. Headless machines usually need EGL or OSMesa configured for offscreen rendering.
- Git submodules initialized if this repo is cloned with submodules:

```bash
git submodule update --init --recursive
```

Create and verify the Python environment from the repository root:

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

The setup script creates `.venv`, installs `requirements.txt`, installs this repository in editable mode, installs the vendored DETR package at `act/detr`, and runs a headless MuJoCo reset smoke test.

To use a specific interpreter or virtualenv directory:

```bash
PYTHON=python3.11 VENV_DIR=.venv311 bash scripts/setup.sh
```

For CUDA training, install the PyTorch wheel matching the target CUDA runtime before or after running the setup script. The default requirements are suitable for CPU and Apple Silicon setups through the normal PyTorch wheels.

## Environment IDs

The import `import openarm_env` registers these Gymnasium IDs:

- `OpenArmReach-v0`: single-arm reach environment.
- `OpenArmBimanualReach-v0`: bimanual reach environment.
- `OpenArmBimanualPickPlace-v0`: bimanual cube pick-place environment and the main ACT dataset/training target.

Basic headless rollout:

```bash
python run_ik_policy.py OpenArmBimanualReach-v0 none 1
```

Interactive rollout, if the machine has display support:

```bash
python run_ik_policy.py OpenArmBimanualReach-v0 human 1
```

## Collecting And Inspecting Data

The ACT data pipeline is currently oriented around `OpenArmBimanualPickPlace-v0`. The collector records RGB from the head and wrist cameras, robot proprioception, scripted joint targets, low-level actuator commands, cube state, rewards, success flags, and episode metadata.

### Collect A Dataset

Collect a small dataset:

```bash
python collect_act_dataset.py \
  --env-id OpenArmBimanualPickPlace-v0 \
  --policy grasp \
  --episodes 20 \
  --seed 69 \
  --width 320 \
  --height 240 \
  --output-dir datasets/act_pick_place
```

Useful collection options:

- `--policy auto`: defaults to the grasp policy for pick-place.
- `--policy grasp`: uses the scripted grasp-and-place policy.
- `--policy push`: uses the scripted push-style pick-place policy.
- `--episodes`: number of rollouts to save.
- `--seed`: base seed. Episode `N` uses `seed + N`.
- `--width`, `--height`: camera render resolution saved into the dataset.
- `--sticky-grasp` / `--no-sticky-grasp`: toggles the pick-place sticky grasp helper.
- `--output-dir`: destination directory for `manifest.json` and `episode_XXXX.npz` files.

The collector prints one line per episode:

```text
episode=0000 steps=... success=... final_distance=... saved=episode_0000.npz
```

Training uses only successful episodes, so collect enough rollouts to produce a useful number of `success=1` files.

### Dataset Layout

A collection directory looks like:

```text
datasets/act_pick_place/
  manifest.json
  episode_0000.npz
  episode_0001.npz
  ...
```

`manifest.json` records the environment, policy, controller, camera resolution, base seed, sticky-grasp setting, and controller/policy config.

Each episode file contains arrays such as:

- `head_rgb`, `left_wrist_rgb`, `right_wrist_rgb`: RGB image sequences, shape `(T, H, W, 3)`.
- `robot_qpos`, `robot_qvel`: robot joint position and velocity sequences.
- `desired_left_arm_qpos`, `desired_right_arm_qpos`: scripted arm targets used as ACT labels.
- `desired_left_finger_target`, `desired_right_finger_target`: scripted gripper targets used as ACT labels.
- `action`: low-level actuator command sent to MuJoCo.
- `goal`, `cube_pos`, `cube_quat`, `cube_linear_velocity`, `cube_angular_velocity`: task state.
- `reward`, `is_success`, `terminated`, `truncated`: per-step rollout results.
- `sticky_attached`, `sticky_side`: sticky grasp state for pick-place episodes.
- `episode_length`, `episode_success`, `seed`, `initial_distance_to_goal`, `final_distance_to_goal`: episode-level metadata.

For ACT training, the target action vector is built from the desired joint targets:

```text
[left_arm_qpos(7), left_finger(1), right_arm_qpos(7), right_finger(1)]
```

### Inspect Dataset Contents

Open a dataset directory or a single episode:

```bash
python inspect_act_dataset.py datasets/act_pick_place --episode-index 0
python inspect_act_dataset.py datasets/act_pick_place/episode_0003.npz
```

The inspector prints episode metadata and every stored key with shape and dtype, then opens a Matplotlib viewer with camera frames and joint/action traces.

Viewer controls:

- Slider: scrub frame by frame.
- Left / Right: step by 1 frame.
- Up / Down: step by 10 frames.
- Home / End: jump to first or last frame.

Camera display modes:

```bash
python inspect_act_dataset.py datasets/act_pick_place --camera all
python inspect_act_dataset.py datasets/act_pick_place --camera head
python inspect_act_dataset.py datasets/act_pick_place --camera wrist
```

### Replay A Collected Episode

Replay saved scripted targets in MuJoCo:

```bash
python run_replay_policy.py datasets/act_pick_place OpenArmBimanualPickPlace-v0 none
```

Use `human` instead of `none` to render interactively:

```bash
python run_replay_policy.py datasets/act_pick_place/episode_0000.npz OpenArmBimanualPickPlace-v0 human
```

The replay runner uses the saved episode seed and desired joint targets. It is useful for checking whether a saved trajectory is coherent in the simulator.

### Collection Notes

- Larger image resolutions increase `.npz` size and training memory use.
- If offscreen renderer creation fails, verify MuJoCo/OpenGL support before collecting data.
- If training later reports no successful episodes, inspect the collection logs or `episode_success` fields and collect more data.
- Dataset and checkpoint directories are ignored by git by default.

## Training Policies

The OpenArm ACT training entry point is:

```bash
python -m ml.act.train
```

The trainer loads successful `episode_*.npz` files from `--dataset-dir`, computes normalization stats from the training split, samples image/proprioception/action chunks, and trains an ACT policy. It writes checkpoints, metrics, normalization stats, and optional evaluation rollouts to `--ckpt-dir`.

### Minimal Training Command

```bash
python -m ml.act.train \
  --dataset-dir datasets/act_pick_place \
  --ckpt-dir checkpoints/openarm_act \
  --num-epochs 2000 \
  --batch-size 8 \
  --chunk-size 100 \
  --device auto
```

For a quick plumbing test:

```bash
python -m ml.act.train \
  --dataset-dir datasets/act_pick_place \
  --ckpt-dir checkpoints/debug_act \
  --num-epochs 2 \
  --batch-size 2 \
  --eval-rollouts 1 \
  --checkpoint-every 1
```

### Important Training Options

- `--dataset-dir`: directory containing collected `episode_*.npz` files.
- `--ckpt-dir`: output directory for checkpoints and metrics.
- `--num-epochs`: number of training epochs.
- `--batch-size`: number of episodes sampled per batch.
- `--chunk-size`: number of future target actions predicted by ACT.
- `--camera-names`: image streams used by the policy. Defaults to `head_camera left_wrist_camera right_wrist_camera`.
- `--lr`, `--lr-backbone`, `--weight-decay`: optimizer settings.
- `--hidden-dim`, `--dim-feedforward`, `--enc-layers`, `--dec-layers`, `--nheads`: transformer/model size.
- `--pretrained-backbone` / `--no-pretrained-backbone`: controls torchvision backbone initialization.
- `--device auto|cpu|cuda|mps`: execution device.
- `--num-workers`: DataLoader worker count.
- `--temporal-agg`: enables ACT temporal action aggregation during evaluation.
- `--checkpoint-every`: save `policy_epoch_XXXX.ckpt` every N epochs. Set `0` to disable periodic checkpoints.

### Training Outputs

`--ckpt-dir` contains:

- `policy_best.ckpt`: best validation-loss checkpoint.
- `policy_last.ckpt`: final epoch checkpoint.
- `policy_epoch_XXXX.ckpt`: periodic checkpoints, if enabled.
- `policy_epoch_XXXX_best.ckpt`: copy of the best epoch at the end of training.
- `metrics.json`: per-epoch train/validation loss history.
- `train_summary.json`: best epoch, best validation loss, history, and metadata.
- `run_metadata.json`: dataset split, camera names, dimensions, seed, and device.
- `norm_stats.json`: qpos/action normalization statistics.
- `tensorboard/`: TensorBoard logs when `torch.utils.tensorboard` is available.
- `eval_final_best.json` and `eval_rollouts/final_best/`: final evaluation metrics and rollouts, if `--eval-rollouts > 0`.

Inspect TensorBoard:

```bash
tensorboard --logdir checkpoints/openarm_act/tensorboard
```

### Periodic Simulator Evaluation During Training

By default, final evaluation runs after training. To also evaluate during training:

```bash
python -m ml.act.train \
  --dataset-dir datasets/act_pick_place \
  --ckpt-dir checkpoints/openarm_act \
  --num-epochs 2000 \
  --eval-every 100 \
  --eval-rollouts 10
```

Each periodic evaluation loads the current `policy_best.ckpt`, runs simulator rollouts, writes `eval_epoch_....json`, and optionally saves rollout `.npz` files under `eval_rollouts/`.

To disable simulator evaluation during a training run:

```bash
python -m ml.act.train \
  --dataset-dir datasets/act_pick_place \
  --ckpt-dir checkpoints/openarm_act \
  --eval-rollouts 0
```

## Evaluating And Visualizing Policies

Evaluation loads a trained ACT checkpoint, runs it in `OpenArmBimanualPickPlace-v0`, reports success rate and final distance, and can save the evaluation rollouts in the same `.npz` format used by the dataset inspector.

### Evaluate A Checkpoint

Evaluate the best checkpoint in a checkpoint directory:

```bash
python -m ml.act.train \
  --eval-only \
  --ckpt-dir checkpoints/openarm_act \
  --eval-rollouts 20 \
  --device auto
```

Evaluate a specific checkpoint:

```bash
python -m ml.act.train \
  --eval-only \
  --ckpt-dir checkpoints/openarm_act \
  --ckpt-path checkpoints/openarm_act/policy_epoch_0500.ckpt \
  --eval-rollouts 20 \
  --eval-output-dir outputs/eval_rollouts
```

Useful evaluation options:

- `--eval-rollouts`: number of simulator rollouts.
- `--eval-seed-offset`: offset added to `--seed` for eval seeds.
- `--eval-width`, `--eval-height`: camera render resolution. Defaults are inferred from the dataset when training, or fall back to `320x240`.
- `--eval-max-steps`: cap rollout length.
- `--save-eval-rollouts` / `--no-save-eval-rollouts`: save or skip `.npz` rollout files.
- `--sticky-grasp` / `--no-sticky-grasp`: match the evaluation environment to the collection setting.
- `--temporal-agg`: evaluate with temporal action aggregation.

Evaluation writes:

- `eval_<tag>.json` in `--ckpt-dir`, containing `success_rate`, `mean_final_distance`, seeds, checkpoint path, and per-rollout results.
- `eval_rollouts/<tag>/manifest.json`, when rollout saving is enabled.
- `eval_rollouts/<tag>/episode_XXXX.npz`, when rollout saving is enabled.

### Visualize Evaluation Rollouts

Saved evaluation rollouts can be inspected with the same dataset viewer:

```bash
python inspect_act_dataset.py checkpoints/openarm_act/eval_rollouts/final_best --episode-index 0
```

or, if `--eval-output-dir` was used:

```bash
python inspect_act_dataset.py outputs/eval_rollouts/policy_epoch_0500 --episode-index 0
```

The inspector will show policy-generated target traces in the same slots as the demonstration target traces. Some evaluation files also include `predicted_joint_target` for the raw ACT target vector.

### Visualize Scripted Policies

Run an IK/scripted policy in the simulator:

```bash
python run_ik_policy.py OpenArmBimanualPickPlace-v0 none 1 grasp
python run_ik_policy.py OpenArmBimanualPickPlace-v0 human 1 grasp
python run_ik_policy.py OpenArmBimanualPickPlace-v0 human 1 push
```

`run_ik_policy.py` positional arguments are:

```text
python run_ik_policy.py [env_id] [render_mode] [num_episodes] [policy]
```

Use `render_mode=none` for headless logs and `render_mode=human` for an interactive MuJoCo viewer.

### Preview Wrist Cameras

Use the camera preview when adjusting camera poses in the XML scenes or validating rendering:

```bash
python view_wrist_cameras.py \
  --env-id OpenArmBimanualReach-v0 \
  --steps 250 \
  --episodes 1 \
  --width 480 \
  --height 360 \
  --output-dir outputs/camera_previews
```

If Matplotlib is available, this opens a live preview. If Matplotlib is unavailable, it saves preview images to `--output-dir`.

## Troubleshooting

- `No successful .npz episodes found`: training filters to `episode_success == 1`; collect more successful pick-place demonstrations or inspect failed episodes.
- MuJoCo renderer errors: reduce `--width/--height`, check OpenGL/EGL/OSMesa availability, and ensure the scene XML offscreen framebuffer is large enough.
- CUDA out of memory: lower `--batch-size`, lower camera resolution, lower `--chunk-size`, or use fewer cameras with `--camera-names`.
- Viewer does not open over SSH: use headless commands and inspect saved `.npz` files on a machine with display support, or configure X forwarding/EGL as appropriate.
- Dataset and checkpoint paths can grow quickly. Keep generated data under `datasets/`, `checkpoints/`, or `outputs/`, which are ignored by git.

## Legacy ACT Environment

The upstream ACT code under `act/` includes its original Conda environment at `act/conda_env.yaml`. Use that only if you specifically need the original ALOHA simulation scripts. The root setup path above is intended for the OpenArm environments and training scripts in this workspace.
