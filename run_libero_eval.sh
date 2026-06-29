#!/usr/bin/env bash
# Run OpenPi π0.5-LIBERO in the LIBERO sim (two terminals, or use tmux).
#
# Requirements: NVIDIA GPU (>8GB), Ubuntu 22.04, EGL or GLX for MuJoCo.
# First run downloads ~several GB checkpoint to ~/.cache/openpi (or OPENPI_DATA_HOME).
set -euo pipefail

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$OPENPI_ROOT"

LIBERO_VENV="$OPENPI_ROOT/examples/libero/.venv"
LIBERO_PYTHON="$LIBERO_VENV/bin/python3.8"

deactivate_venv() {
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    deactivate 2>/dev/null || true
  fi
  unset VIRTUAL_ENV
}

if [[ ! -f "$HOME/.libero/config.yaml" ]]; then
  echo "Creating ~/.libero/config.yaml ..."
  mkdir -p "$HOME/.libero"
  python3 - <<PY
import os, yaml
root = "$OPENPI_ROOT/third_party/libero/libero/libero"
config = {
    "benchmark_root": root,
    "bddl_files": os.path.join(root, "bddl_files"),
    "init_states": os.path.join(root, "init_files"),
    "datasets": os.path.join(root, "../datasets"),
    "assets": os.path.join(root, "assets"),
}
with open(os.path.expanduser("~/.libero/config.yaml"), "w") as f:
    yaml.dump(config, f)
PY
fi

setup_libero_venv() {
  if [[ -x "$LIBERO_PYTHON" ]] && "$LIBERO_PYTHON" -c "import openpi_client" 2>/dev/null; then
    return
  fi
  if [[ ! -x "$LIBERO_PYTHON" ]]; then
    echo "Setting up LIBERO sim venv (one-time) ..."
    uv venv --python 3.8 "$LIBERO_VENV"
    UV_PROJECT_ENVIRONMENT="$LIBERO_VENV" uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cu113 \
      --index-strategy=unsafe-best-match \
      --python "$LIBERO_PYTHON"
  else
    echo "Repairing LIBERO sim venv (openpi_client missing) ..."
  fi
  UV_PROJECT_ENVIRONMENT="$LIBERO_VENV" uv pip install -e packages/openpi-client --python "$LIBERO_PYTHON"
  UV_PROJECT_ENVIRONMENT="$LIBERO_VENV" uv pip install -e third_party/libero --python "$LIBERO_PYTHON"
}

TASK_SUITE="${TASK_SUITE:-libero_spatial}"
CLIENT_ARGS="${CLIENT_ARGS:---args.task-suite-name $TASK_SUITE}"

case "${1:-}" in
  server)
    shift
    deactivate_venv
    exec uv run scripts/serve_policy.py --env LIBERO "$@"
    ;;
  reverie-sim)
    # LIBERO eval with Reverie in the perception loop (examples/libero/main_reverie.py).
    # Pass flags straight through, e.g.:
    #   ./run_libero_eval.sh reverie-sim --mode rerender --reverie-host localhost \
    #       --task-suite-name libero_spatial --num-trials-per-task 2
    shift
    deactivate_venv
    setup_libero_venv
    export PYTHONPATH="${PYTHONPATH:-}:$OPENPI_ROOT/third_party/libero"
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    if [[ "$MUJOCO_GL" == "egl" ]]; then
      nvidia_egl="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
      [[ -f "$nvidia_egl" ]] && export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-$nvidia_egl}"
      export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
      export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    fi
    exec "$LIBERO_PYTHON" examples/libero/main_reverie.py "$@"
    ;;
  sim)
    shift
    deactivate_venv
    setup_libero_venv
    export PYTHONPATH="${PYTHONPATH:-}:$OPENPI_ROOT/third_party/libero"
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    if [[ "$MUJOCO_GL" == "egl" ]]; then
      # Prefer NVIDIA's surfaceless EGL vendor so rendering does not depend on
      # /dev/dri render-node permissions (the usual headless failure mode).
      nvidia_egl="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
      if [[ -f "$nvidia_egl" ]]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-$nvidia_egl}"
      fi
      export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
      export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    elif [[ "$MUJOCO_GL" == "osmesa" ]]; then
      export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
    fi
    # shellcheck disable=SC2086
    exec "$LIBERO_PYTHON" examples/libero/main.py $CLIENT_ARGS "$@"
    ;;
  docker)
    sudo xhost +local:docker
    export SERVER_ARGS="${SERVER_ARGS:---env LIBERO}"
    exec docker compose -f examples/libero/compose.yml up --build
    ;;
  *)
    cat <<EOF
Usage:
  ./run_libero_eval.sh server          # Terminal 1: policy server (GPU, downloads pi05_libero)
  ./run_libero_eval.sh sim             # Terminal 2: LIBERO rollouts + videos in data/libero/videos

Env vars:
  TASK_SUITE=libero_spatial|libero_object|libero_goal|libero_10|libero_90
  MUJOCO_GL=egl|osmesa|glx
    egl    -> GPU headless (auto-uses NVIDIA EGL vendor; needs render/video group)
    osmesa -> CPU software render, slow but always works headless
              (requires: sudo apt-get install -y libosmesa6-dev)
    glx    -> needs an X display
  CLIENT_ARGS='--args.task-suite-name libero_10 --args.num-trials-per-task 5'
  OPENPI_DATA_HOME=~/.cache/openpi

Docker (recommended if deps fight you):
  ./run_libero_eval.sh docker
EOF
    exit 1
    ;;
esac
