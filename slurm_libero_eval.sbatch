#!/usr/bin/env bash
# Self-contained OpenPi pi05_libero eval on a single Slurm GPU node.
#
# Submit (set your GPU partition):
#   sbatch --partition=<gpu_partition> slurm_libero_eval.sbatch
#
# Runs the policy server + LIBERO sim on the SAME GPU node, so the server
# uses the GPU for inference and MuJoCo renders via NVIDIA EGL (no /dev/dri
# permissions, no X display needed). Rollout videos land in data/libero/videos.
#
#SBATCH --job-name=openpi-libero
#SBATCH --output=openpi-libero-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail

OPENPI_ROOT="/home/shivin/repos/spatio_monorepo/reverie/openpi"
cd "$OPENPI_ROOT"

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "ERROR: no GPU visible on this node"; exit 1; }

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TASK_SUITE="${TASK_SUITE:-libero_spatial}"
# Quick functional test by default; raise for a full benchmark run.
export CLIENT_ARGS="${CLIENT_ARGS:---args.task-suite-name $TASK_SUITE --args.num-trials-per-task 2}"

echo "=== Starting policy server (pi05_libero) ==="
./run_libero_eval.sh server &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "=== Waiting for server on 127.0.0.1:8000 ==="
for _ in $(seq 1 180); do
  if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',8000))==0 else 1)"; then
    echo "server up"; break
  fi
  sleep 5
done

echo "=== Running LIBERO sim ==="
./run_libero_eval.sh sim

echo "=== Done. Videos in $OPENPI_ROOT/data/libero/videos ==="
