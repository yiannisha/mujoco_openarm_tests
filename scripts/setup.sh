#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
python -m pip install --no-build-isolation -e act/detr

python - <<'PY'
import gymnasium as gym

import openarm_env

env = gym.make(openarm_env.ENV_ID, render_mode=None)
observation, info = env.reset(seed=0)
print(f"verified {openarm_env.ENV_ID}: obs_shape={observation.shape} action_shape={env.action_space.shape}")
print(f"initial distance_to_goal={info.get('distance_to_goal', 'n/a')}")
env.close()
PY

cat <<EOF

Setup complete.

Activate the environment with:
  source $VENV_DIR/bin/activate

Run a headless smoke test with:
  python run_ik_policy.py OpenArmBimanualReach-v0 none 1
EOF
