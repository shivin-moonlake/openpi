#!/usr/bin/env bash
# Export one LIBERO bowl episode as a Reverie-standard gbuffer recording (zip).
# No Reverie in the loop -- just the pi0.5 policy driving the arm while we
# capture native-res agentview RGB.
#
# Submit:
#   sbatch --partition=all slurm_export_gbuffers.sbatch
#
#SBATCH --job-name=libero-gbuf-export
#SBATCH --output=libero-gbuf-export-%j.log
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
set -euo pipefail

export UV_CACHE_DIR="${HOME}/.cache/uv"
OPENPI_ROOT="/home/shivin/repos/spatio_monorepo/reverie/openpi"

TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TASK_ID="${TASK_ID:-0}"
EPISODE_IDX="${EPISODE_IDX:-0}"
RESOLUTION="${RESOLUTION:-512}"
CAPTURE_STEPS="${CAPTURE_STEPS:-221}"
RECORDING_ID="${RECORDING_ID:-libero_bowl_${TASK_SUITE}_t${TASK_ID}_e${EPISODE_IDX}}"
OUT_ROOT="${OUT_ROOT:-data/libero/gbuffer_exports}"

POLICY_PORT=$(( 8000 + (SLURM_JOB_ID % 1000) ))
IFS=',' read -ra _G <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
G_POLICY="${_G[0]:-0}"
G_SIM="${_G[1]:-${G_POLICY}}"

echo "=== Node: $(hostname) | suite=$TASK_SUITE task=$TASK_ID ep=$EPISODE_IDX res=$RESOLUTION steps=$CAPTURE_STEPS port=$POLICY_PORT gpus=[$G_POLICY,$G_SIM] ==="

wait_for_port() {
  local host="$1" port="$2" deadline=$(( $(date +%s) + ${3:-600} ))
  while [[ $(date +%s) -lt $deadline ]]; do
    if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('$host',$port))==0 else 1)"; then
      return 0
    fi
    sleep 5
  done
  echo "ERROR: $host:$port did not come up"; return 1
}

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

cd "$OPENPI_ROOT"
( CUDA_VISIBLE_DEVICES=$G_POLICY ./run_libero_eval.sh server --port "$POLICY_PORT" ) \
  > "$OPENPI_ROOT/gbuf_policy_server-$SLURM_JOB_ID.log" 2>&1 &
PIDS+=($!)
wait_for_port 127.0.0.1 "$POLICY_PORT" 600

echo "=== Running export (GPU$G_SIM) ==="
MUJOCO_EGL_DEVICE_ID=$G_SIM ./run_libero_eval.sh export-gbuffers \
  --host 127.0.0.1 --port "$POLICY_PORT" \
  --task-suite-name "$TASK_SUITE" \
  --task-id "$TASK_ID" \
  --episode-idx "$EPISODE_IDX" \
  --resolution "$RESOLUTION" \
  --capture-steps "$CAPTURE_STEPS" \
  --recording-id "$RECORDING_ID" \
  --out-root "$OUT_ROOT"

echo "=== Done. Recording + zip in $OPENPI_ROOT/$OUT_ROOT ==="
ls -la "$OUT_ROOT"/recording_"$RECORDING_ID".zip
