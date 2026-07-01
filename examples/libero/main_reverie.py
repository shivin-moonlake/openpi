"""LIBERO eval with a Reverie world-model in the perception loop.

Same task as examples/libero/main.py, but the policy's camera observations are
optionally re-rendered by the Reverie streaming server before being fed to
pi0.5. Because Reverie is an autoregressive *temporal* model (4x temporal VAE:
first chunk 1+4*(N-1) frames, later chunks 4*N), it cannot rerender a single
frame on demand. We therefore decouple perception from control:

  * raw sim frames are buffered into Reverie-sized chunks and flushed through
    the server; the latest rerendered frame becomes the policy's visual input,
  * the policy keeps replanning every ``replan_steps`` on that (possibly stale,
    <= chunk_frames old) image, with always-fresh proprioceptive state.

Modes (``--mode``):
  rerender    - both cameras streamed through Reverie (needs --reverie-host).
  raw_chunked - identical buffering/staleness, Reverie bypassed (A/B baseline).
  vanilla     - original per-step loop (no chunking), for reference.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import Literal, Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from reverie_client import ReverieClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    # Policy server (openpi serve_policy.py --env LIBERO)
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # Reverie rerender server (integration/simpler/server.py)
    mode: Literal["rerender", "raw_chunked", "vanilla"] = "rerender"
    reverie_host: Optional[str] = None
    reverie_port: int = 8418
    reverie_prompt: str = (
        "A robotic arm with segmented joints and a gripper moves above a tabletop "
        "workspace on a bright sunny day, with warm sunlight pouring in and casting "
        "soft highlights and long shadows across the surface and the objects it "
        "manipulates."
    )
    reverie_mode: str = "autoregressive_independent"
    reverie_seed: int = 123
    # Optional JSON prompt bank (isaaclab_templates.json style: keys env/robot/
    # table/light, each a list of fragments). When set, a fresh prompt is
    # composed per task by sampling one fragment from each of these categories.
    prompt_templates: Optional[str] = None
    prompt_template_keys: str = "env,robot,table,light"
    # Letterbox square sim frames to the server's inference aspect ratio
    # (edge-pad before sending, crop center back) so the 256->1280 stretch
    # doesn't distort the scene.
    pad_aspect: bool = True
    # Chunk sizes (only used by raw_chunked; rerender reads them from /info).
    first_chunk: int = 9
    later_chunk: int = 12

    # LIBERO
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    max_tasks: Optional[int] = None  # cap tasks for quick tests

    video_out_path: str = "data/libero/videos_reverie"
    save_reverie_video: bool = True
    seed: int = 7


def _resize224(frame_u8: np.ndarray, size: int) -> np.ndarray:
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(frame_u8, size, size))


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    if args.mode == "rerender" and not args.reverie_host:
        raise ValueError("--reverie-host is required for mode=rerender")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    if args.max_tasks is not None:
        num_tasks_in_suite = min(num_tasks_in_suite, args.max_tasks)
    logging.info("Task suite: %s (%d tasks)", args.task_suite_name, num_tasks_in_suite)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps = {
        "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
        "libero_10": 520, "libero_90": 400,
    }[args.task_suite_name]

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    reverie = None
    first_chunk, later_chunk = args.first_chunk, args.later_chunk
    aspect = None  # inference width/height; enables letterboxing when set
    if args.mode == "rerender":
        reverie = ReverieClient(args.reverie_host, args.reverie_port)
        info = reverie.info()
        first_chunk, later_chunk = info["first_chunk_frames"], info["chunk_frames"]
        if args.pad_aspect:
            aspect = info["inference_width"] / info["inference_height"]
        logging.info("Reverie server: N=%s first_chunk=%d later_chunk=%d inf=%dx%d pad_aspect=%s",
                     info["N"], first_chunk, later_chunk,
                     info["inference_height"], info["inference_width"], aspect)

    chunked = args.mode in ("rerender", "raw_chunked")

    prompt_bank = None
    if args.prompt_templates:
        prompt_bank = json.loads(pathlib.Path(args.prompt_templates).read_text())
    prompt_rng = np.random.default_rng(args.seed)
    template_keys = [k.strip() for k in args.prompt_template_keys.split(",") if k.strip()]

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Compose a fresh reverie style prompt per task (same across its episodes).
        if prompt_bank is not None:
            reverie_prompt = " ".join(
                str(prompt_rng.choice(prompt_bank[k])) for k in template_keys)
            logging.info("Reverie prompt [task %d]: %s", task_id, reverie_prompt)
        else:
            reverie_prompt = args.reverie_prompt

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info("\nTask: %s", task_description)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            action_plan = collections.deque()

            if reverie is not None:
                reverie.open_session(reverie_prompt, mode=args.reverie_mode,
                                     seed=args.reverie_seed, force=True)

            # Perception buffers (chunked modes).
            buf_agent: list = []
            buf_wrist: list = []
            next_chunk = first_chunk
            last_img: Optional[np.ndarray] = None
            last_wrist: Optional[np.ndarray] = None

            policy_view = []   # 224 images actually fed to the policy
            reverie_view = []  # full rerendered agentview stream (rerender mode)

            t, done = 0, False
            logging.info("Starting episode %d...", task_episodes + 1)

            # Warmup: let physics settle and, in chunked modes, prime the
            # perception buffer with dummy-action steps so the policy's first
            # real observation is already a (re)rendered frame -- the policy
            # never acts on a raw sim frame during the first chunk.
            while t < args.num_steps_wait or (chunked and last_img is None):
                agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                if chunked:
                    buf_agent.append(agent)
                    buf_wrist.append(wrist)
                    if len(buf_agent) == next_chunk:
                        out_agent, out_wrist = _rerender_chunk(
                            reverie, buf_agent, buf_wrist, aspect)
                        if args.save_reverie_video:
                            reverie_view.extend(out_agent)
                        last_img = _resize224(out_agent[-1], args.resize_size)
                        last_wrist = _resize224(out_wrist[-1], args.resize_size)
                        buf_agent, buf_wrist = [], []
                        next_chunk = later_chunk
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1

            while t < max_steps + args.num_steps_wait:
                # Upright native-res frames (180deg rotate matches train preproc).
                agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                if chunked:
                    buf_agent.append(agent)
                    buf_wrist.append(wrist)
                    if len(buf_agent) == next_chunk:
                        out_agent, out_wrist = _rerender_chunk(
                            reverie, buf_agent, buf_wrist, aspect)
                        if args.save_reverie_video:
                            reverie_view.extend(out_agent)
                        last_img = _resize224(out_agent[-1], args.resize_size)
                        last_wrist = _resize224(out_wrist[-1], args.resize_size)
                        buf_agent, buf_wrist = [], []
                        next_chunk = later_chunk
                        # Force a replan on the freshly rerendered frame so the
                        # policy never executes a stale plan across a flush.
                        action_plan.clear()

                if last_img is not None:
                    img224, wrist224 = last_img, last_wrist
                else:
                    # Only reached in vanilla (non-chunked) mode.
                    img224 = _resize224(agent, args.resize_size)
                    wrist224 = _resize224(wrist, args.resize_size)

                policy_view.append(img224)

                if not action_plan:
                    element = {
                        "observation/image": img224,
                        "observation/wrist_image": wrist224,
                        "observation/state": np.concatenate((
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )),
                        "prompt": str(task_description),
                    }
                    action_chunk = client.infer(element)["actions"]
                    assert len(action_chunk) >= args.replan_steps, (
                        f"replan every {args.replan_steps} but policy returns "
                        f"{len(action_chunk)}")
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            if reverie is not None:
                reverie.close()

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            seg = task_description.replace(" ", "_")[:80]
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"policy_{seg}_{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in policy_view], fps=10)
            if args.save_reverie_video and reverie_view:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"reverie_{seg}_{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in reverie_view], fps=10)

            logging.info("Success: %s | episodes=%d successes=%d (%.1f%%)",
                         done, total_episodes, total_successes,
                         total_successes / total_episodes * 100)

        logging.info("Task success rate: %.3f", task_successes / max(task_episodes, 1))

    logging.info("Total success rate: %.3f (%d episodes)",
                 total_successes / max(total_episodes, 1), total_episodes)


def _rerender_chunk(reverie, buf_agent, buf_wrist, aspect=None):
    """Return (agent_frames, wrist_frames) lists of native-res uint8 frames."""
    if reverie is None:  # raw_chunked baseline: identity
        return list(buf_agent), list(buf_wrist)
    # [2, n, H, W, 3]: batch dim = [agentview, wrist].
    batch = np.stack([np.stack(buf_agent, 0), np.stack(buf_wrist, 0)], axis=0)
    crop = None
    if aspect is not None:
        batch, crop = _pad_to_aspect(batch, aspect)
    out = reverie.process(batch)  # [2, n, H, W(padded), 3]
    if crop is not None:
        (h0, h1), (w0, w1) = crop
        out = out[:, :, h0:h1, w0:w1, :]
    return list(out[0]), list(out[1])


def _pad_to_aspect(batch, aspect):
    """Edge-pad [B,n,H,W,3] to width:height == aspect; return (batch, crop box)."""
    _, _, H, W, _ = batch.shape
    target_w = int(round(H * aspect))
    if target_w > W:
        pad = target_w - W
        left = pad // 2
        batch = np.pad(batch, ((0, 0), (0, 0), (0, 0), (left, pad - left), (0, 0)),
                       mode="edge")
        return batch, ((0, H), (left, left + W))
    target_h = int(round(W / aspect))
    if target_h > H:
        pad = target_h - H
        top = pad // 2
        batch = np.pad(batch, ((0, 0), (0, 0), (top, pad - top), (0, 0), (0, 0)),
                       mode="edge")
        return batch, ((top, top + H), (0, W))
    return batch, ((0, H), (0, W))


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
