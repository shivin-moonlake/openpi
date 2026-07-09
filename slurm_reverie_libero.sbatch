#!/usr/bin/env bash
# Reverie-in-the-loop LIBERO eval on a single Slurm GPU node.
#
# Submit (GPU partition has 8x H100):
#   sbatch --partition=all slurm_reverie_libero.sbatch
#
# Launches THREE services on the same node and runs the eval:
#   GPU0: Reverie rerender server   (reverie venv,  port 8418)
#   GPU1: openpi pi0.5 policy server (openpi venv,   port 8000)
#   GPU2: LIBERO sim + main_reverie  (libero py3.8 venv, EGL render)
# All talk over localhost. Videos -> openpi/data/libero/videos_reverie.
#
#SBATCH --job-name=reverie-libero
#SBATCH --output=reverie-libero-%j.log
# All three services (reverie server, policy server, LIBERO sim) run on the
# batch host and talk over localhost; the sim's EGL/CUDA device ids come from
# this host's CUDA_VISIBLE_DEVICES. --nodes=1 pins all 3 GPUs to one node so
# those indices are valid locally (without it Slurm may spread GPUs across
# nodes, and MuJoCo EGL falls through to an inaccessible Mesa DRI device).
#SBATCH --nodes=1
#SBATCH --gpus=3
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
set -euo pipefail

# Use a writable uv cache; the submitting shell may export a sandbox UV_CACHE_DIR
# (e.g. /tmp/cursor-sandbox-cache/...) that compute nodes cannot write to.
export UV_CACHE_DIR="${HOME}/.cache/uv"

REVERIE_ROOT="/home/shivin/repos/spatio_monorepo/reverie"
OPENPI_ROOT="$REVERIE_ROOT/openpi"

# Tunables (override via: sbatch --export=ALL,MODE=raw_chunked,... )
MODE="${MODE:-rerender}"                       # rerender | raw_chunked | vanilla
REVERIE_N="${REVERIE_N:-3}"                     # 3 (chunks 9->12) | 1 (chunks 1->4, tighter loop)
REVERIE_MODEL="${REVERIE_MODEL:-sf}"           # sf (SF_independent N=3) | n18_distill_run1 (two-step DMD distill) | dualdmd
REPLAN_STEPS="${REPLAN_STEPS:-}"               # blank -> sim default (5)
ACTION_HORIZON="${ACTION_HORIZON:-}"           # blank -> model default (10 for pi05_libero)
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-2}"                   # per task; raise for full runs
MAX_TASKS="${MAX_TASKS:-1}"                     # cap tasks for a quick test
REVERIE_PROMPT="${REVERIE_PROMPT:-A robotic arm with segmented joints and a gripper moves above a tabletop workspace on a bright sunny day, with warm sunlight casting soft highlights and long shadows.}"
# Optional JSON prompt bank (env/robot/table/light fragments) -> a fresh
# composed prompt is sampled per task. Blank uses the single REVERIE_PROMPT.
PROMPT_TEMPLATES="${PROMPT_TEMPLATES:-}"

# Per-job ports so concurrent jobs on the same node don't collide.
POLICY_PORT=$(( 8000 + (SLURM_JOB_ID % 1000) ))
REVERIE_PORT=$(( 8418 + (SLURM_JOB_ID % 1000) ))

# Map the three services onto the GPUs Slurm actually allocated to THIS job.
# Slurm sets CUDA_VISIBLE_DEVICES to absolute device indices (e.g. "3,4,5");
# hardcoding 0/1/2 makes co-located jobs stomp the same physical GPUs.
IFS=',' read -ra _ALLOC_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
G_REVERIE="${_ALLOC_GPUS[0]:-0}"
G_POLICY="${_ALLOC_GPUS[1]:-${G_REVERIE}}"
G_SIM="${_ALLOC_GPUS[2]:-${G_REVERIE}}"

echo "=== Node: $(hostname) | mode=$MODE suite=$TASK_SUITE trials=$NUM_TRIALS tasks=$MAX_TASKS policy_port=$POLICY_PORT reverie_port=$REVERIE_PORT gpus=[$G_REVERIE,$G_POLICY,$G_SIM] ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

wait_for_port() {  # host port timeout_s
  local host="$1" port="$2" deadline=$(( $(date +%s) + ${3:-600} ))
  while [[ $(date +%s) -lt $deadline ]]; do
    if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('$host',$port))==0 else 1)"; then
      return 0
    fi
    sleep 5
  done
  echo "ERROR: $host:$port did not come up in time"; return 1
}

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

# --- GPU1: openpi policy server (always needed) ---
SERVER_EXTRA=""
[[ -n "$ACTION_HORIZON" ]] && SERVER_EXTRA="--action-horizon $ACTION_HORIZON"
echo "=== Starting openpi policy server (GPU$G_POLICY, :$POLICY_PORT) $SERVER_EXTRA ==="
( cd "$OPENPI_ROOT" && CUDA_VISIBLE_DEVICES=$G_POLICY ./run_libero_eval.sh server --port "$POLICY_PORT" $SERVER_EXTRA ) \
  > "$OPENPI_ROOT/policy_server-$SLURM_JOB_ID.log" 2>&1 &
PIDS+=($!)

# N selects chunk size (perception cadence) + stream mode + server config.
# N=3 is the trained ARI recipe; other N run in plain 'independent' mode.
# Pixel-frame counts (4x temporal VAE): first = 1+4*(N-1), later = 4*N.
if [[ "$REVERIE_N" == "3" ]]; then
  REVERIE_STREAM_MODE="autoregressive_independent"
else
  REVERIE_STREAM_MODE="independent"
fi
if [[ "$REVERIE_MODEL" != "sf" ]]; then
  # n18_distill_run1 | dualdmd | ... -> config.shivin.<model>.N<N>.yaml
  REVERIE_CONFIG="$REVERIE_ROOT/integration/simpler/config.shivin.${REVERIE_MODEL}.N${REVERIE_N}.yaml"
elif [[ "$REVERIE_N" == "3" ]]; then
  REVERIE_CONFIG="$REVERIE_ROOT/integration/simpler/config.shivin.yaml"
else
  REVERIE_CONFIG="$REVERIE_ROOT/integration/simpler/config.shivin.N${REVERIE_N}.yaml"
fi
FIRST_CHUNK=$(( 1 + 4 * (REVERIE_N - 1) ))
LATER_CHUNK=$(( 4 * REVERIE_N ))

# --- GPU0: Reverie rerender server (only for rerender mode) ---
if [[ "$MODE" == "rerender" ]]; then
  echo "=== Starting Reverie server (GPU$G_REVERIE, :$REVERIE_PORT, N=$REVERIE_N) ==="
  ( cd "$REVERIE_ROOT" \
      && CUDA_VISIBLE_DEVICES=$G_REVERIE \
         REVERIE_SIMPLER_CONFIG="$REVERIE_CONFIG" \
         REVERIE_SIMPLER_PORT="$REVERIE_PORT" \
         uv run --no-sync python integration/simpler/server.py ) \
    > "$OPENPI_ROOT/reverie_server-$SLURM_JOB_ID.log" 2>&1 &
  PIDS+=($!)
  wait_for_port 127.0.0.1 "$REVERIE_PORT" 1800  # 40GB load + float8/flash4 compile + warmup
  REVERIE_ARGS="--mode rerender --reverie-host 127.0.0.1 --reverie-port $REVERIE_PORT --reverie-mode $REVERIE_STREAM_MODE --reverie-prompt \"$REVERIE_PROMPT\""
  [[ -n "$PROMPT_TEMPLATES" ]] && REVERIE_ARGS="$REVERIE_ARGS --prompt-templates \"$PROMPT_TEMPLATES\""
elif [[ "$MODE" == "raw_chunked" ]]; then
  REVERIE_ARGS="--mode raw_chunked --first-chunk $FIRST_CHUNK --later-chunk $LATER_CHUNK"
else
  REVERIE_ARGS="--mode $MODE"
fi

wait_for_port 127.0.0.1 "$POLICY_PORT" 600

# --- GPU2: LIBERO sim + eval ---
echo "=== Running LIBERO sim (GPU2) mode=$MODE ==="
cd "$OPENPI_ROOT"
MAX_TASKS_ARG=""
[[ -n "$MAX_TASKS" ]] && MAX_TASKS_ARG="--max-tasks $MAX_TASKS"
REPLAN_ARG=""
[[ -n "$REPLAN_STEPS" ]] && REPLAN_ARG="--replan-steps $REPLAN_STEPS"
VIDEO_OUT="data/libero/videos_reverie/${MODE}_${REVERIE_MODEL}_N${REVERIE_N}_r${REPLAN_STEPS:-d}_h${ACTION_HORIZON:-d}_job${SLURM_JOB_ID}"
eval MUJOCO_EGL_DEVICE_ID=$G_SIM ./run_libero_eval.sh reverie-sim \
  --host 127.0.0.1 --port "$POLICY_PORT" \
  --task-suite-name "$TASK_SUITE" \
  --num-trials-per-task "$NUM_TRIALS" \
  --video-out-path "$VIDEO_OUT" \
  $REPLAN_ARG \
  $MAX_TASKS_ARG \
  $REVERIE_ARGS

echo "=== Done. Videos in $OPENPI_ROOT/$VIDEO_OUT ==="
