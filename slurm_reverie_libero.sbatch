#!/usr/bin/env bash
# Reverie-in-the-loop LIBERO eval on a single Slurm GPU node.
#
# Submit (GPU partition has 8x H100):
#   sbatch --partition=all slurm_reverie_libero.sbatch
#
# Launches TWO services on the same node and runs the eval:
#   GPU0: openpi pi0.5 policy server (openpi venv,   port 8000)
#   GPU1: LIBERO sim + main_reverie  (libero py3.8 venv, EGL render)
# Rerendering goes to the shared Reverie stream server (REVERIE_URL, /api
# endpoint of inference/stream/local_stream_server.py); the eval leases one
# of its GPUs for the whole run. Videos -> openpi/data/libero/videos_reverie.
#
#SBATCH --job-name=reverie-libero
#SBATCH --output=reverie-libero-%j.log
# Both local services (policy server, LIBERO sim) run on the batch host and
# talk over localhost; the sim's EGL/CUDA device ids come from this host's
# CUDA_VISIBLE_DEVICES. --nodes=1 pins both GPUs to one node so those indices
# are valid locally (without it Slurm may spread GPUs across nodes, and
# MuJoCo EGL falls through to an inaccessible Mesa DRI device).
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
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
REVERIE_URL="${REVERIE_URL:-ws://gmicloud-loki-g1-gpu-001:8422/api}"
# Optional config preset/YAML for the leased stream-server GPU (blank =
# whatever the server has resident, default dual_dmd_cf_optimized).
REVERIE_CONFIG="${REVERIE_CONFIG:-}"
REVERIE_N="${REVERIE_N:-3}"                     # 3 (chunks 9->12) | 1 (chunks 1->4); must match the server config's N
REPLAN_STEPS="${REPLAN_STEPS:-}"               # blank -> sim default (5)
ACTION_HORIZON="${ACTION_HORIZON:-}"           # blank -> model default (10 for pi05_libero)
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-2}"                   # per task; raise for full runs
MAX_TASKS="${MAX_TASKS:-1}"                     # cap tasks for a quick test
REVERIE_PROMPT="${REVERIE_PROMPT:-A robotic arm with segmented joints and a gripper moves above a tabletop workspace on a bright sunny day, with warm sunlight casting soft highlights and long shadows.}"
# Optional JSON prompt bank (env/robot/table/light fragments) -> 10 prompts
# are composed once and cycled across every task's episodes (50 trials =
# 10 prompts x5). Blank uses the single REVERIE_PROMPT.
PROMPT_TEMPLATES="${PROMPT_TEMPLATES:-}"

# Per-job port so concurrent jobs on the same node don't collide.
POLICY_PORT=$(( 8000 + (SLURM_JOB_ID % 1000) ))

# Map the two local services onto the GPUs Slurm actually allocated to THIS
# job. Slurm sets CUDA_VISIBLE_DEVICES to absolute device indices (e.g.
# "3,4"); hardcoding 0/1 makes co-located jobs stomp the same physical GPUs.
IFS=',' read -ra _ALLOC_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
G_POLICY="${_ALLOC_GPUS[0]:-0}"
G_SIM="${_ALLOC_GPUS[1]:-${G_POLICY}}"

echo "=== Node: $(hostname) | mode=$MODE suite=$TASK_SUITE trials=$NUM_TRIALS tasks=$MAX_TASKS policy_port=$POLICY_PORT reverie_url=$REVERIE_URL gpus=[$G_POLICY,$G_SIM] ==="
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

# --- GPU0: openpi policy server (always needed) ---
SERVER_EXTRA=""
[[ -n "$ACTION_HORIZON" ]] && SERVER_EXTRA="--action-horizon $ACTION_HORIZON"
echo "=== Starting openpi policy server (GPU$G_POLICY, :$POLICY_PORT) $SERVER_EXTRA ==="
( cd "$OPENPI_ROOT" && CUDA_VISIBLE_DEVICES=$G_POLICY ./run_libero_eval.sh server --port "$POLICY_PORT" $SERVER_EXTRA ) \
  > "$OPENPI_ROOT/policy_server-$SLURM_JOB_ID.log" 2>&1 &
PIDS+=($!)

# N selects chunk size (perception cadence) + stream mode; it must match the
# N of the config loaded on the stream server. N=3 is the trained ARI recipe;
# other N run in plain 'independent' mode. Pixel-frame counts (4x temporal
# VAE): first = 1+4*(N-1), later = 4*N.
if [[ "$REVERIE_N" == "3" ]]; then
  REVERIE_STREAM_MODE="autoregressive_independent"
else
  REVERIE_STREAM_MODE="independent"
fi
FIRST_CHUNK=$(( 1 + 4 * (REVERIE_N - 1) ))
LATER_CHUNK=$(( 4 * REVERIE_N ))

if [[ "$MODE" == "rerender" ]]; then
  REVERIE_ARGS="--mode rerender --reverie-url $REVERIE_URL --reverie-mode $REVERIE_STREAM_MODE --reverie-prompt \"$REVERIE_PROMPT\""
  [[ -n "$REVERIE_CONFIG" ]] && REVERIE_ARGS="$REVERIE_ARGS --reverie-config \"$REVERIE_CONFIG\""
  [[ -n "$PROMPT_TEMPLATES" ]] && REVERIE_ARGS="$REVERIE_ARGS --prompt-templates \"$PROMPT_TEMPLATES\""
elif [[ "$MODE" == "raw_chunked" ]]; then
  REVERIE_ARGS="--mode raw_chunked --first-chunk $FIRST_CHUNK --later-chunk $LATER_CHUNK"
else
  REVERIE_ARGS="--mode $MODE"
fi

wait_for_port 127.0.0.1 "$POLICY_PORT" 600

# --- GPU1: LIBERO sim + eval ---
echo "=== Running LIBERO sim (GPU$G_SIM) mode=$MODE ==="
cd "$OPENPI_ROOT"
MAX_TASKS_ARG=""
[[ -n "$MAX_TASKS" ]] && MAX_TASKS_ARG="--max-tasks $MAX_TASKS"
REPLAN_ARG=""
[[ -n "$REPLAN_STEPS" ]] && REPLAN_ARG="--replan-steps $REPLAN_STEPS"
VIDEO_OUT="data/libero/videos_reverie/${MODE}_N${REVERIE_N}_r${REPLAN_STEPS:-d}_h${ACTION_HORIZON:-d}_job${SLURM_JOB_ID}"
eval MUJOCO_EGL_DEVICE_ID=$G_SIM ./run_libero_eval.sh reverie-sim \
  --host 127.0.0.1 --port "$POLICY_PORT" \
  --task-suite-name "$TASK_SUITE" \
  --num-trials-per-task "$NUM_TRIALS" \
  --video-out-path "$VIDEO_OUT" \
  $REPLAN_ARG \
  $MAX_TASKS_ARG \
  $REVERIE_ARGS

echo "=== Done. Videos in $OPENPI_ROOT/$VIDEO_OUT ==="
