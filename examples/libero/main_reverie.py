"""LIBERO eval with a Reverie world-model in the perception loop.

Same task as examples/libero/main.py, but the policy's camera observations are
optionally re-rendered by the Reverie streaming server before being fed to
pi0.5. The chunked perception loop (buffering, staleness, aspect padding)
lives in inference/stream/rerender_loop.py, shared with the YAM sims.

Modes (``--mode``):
  rerender    - both cameras streamed through the shared Reverie stream
                server (--reverie-url, /api WebSocket endpoint).
  raw_chunked - identical buffering/staleness, Reverie bypassed (A/B baseline).
  vanilla     - original per-step loop (no chunking), for reference.
"""
from __future__ import annotations

import collections
import dataclasses
import logging
import math
import pathlib
import sys
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

# reverie root (this file lives at reverie/openpi/examples/libero/).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
import inference.stream.rerender_loop as rerender_loop  # noqa: E402

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    # Policy server (openpi serve_policy.py --env LIBERO)
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # Shared Reverie stream server (inference/stream/local_stream_server.py,
    # /api endpoint). Connecting leases one GPU for the whole eval.
    mode: Literal["rerender", "raw_chunked", "vanilla"] = "rerender"
    reverie_url: str = "ws://gmicloud-loki-g1-gpu-001:8422/api"
    # Optional config preset/YAML to load on the leased GPU (blank = whatever
    # the server has resident, default dual_dmd_cf_optimized).
    reverie_config: str = ""
    reverie_prompt: str = (
        "A robotic arm with segmented joints and a gripper moves above a tabletop "
        "workspace on a bright sunny day, with warm sunlight pouring in and casting "
        "soft highlights and long shadows across the surface and the objects it "
        "manipulates."
    )
    reverie_mode: str = "autoregressive_independent"
    reverie_seed: int = 123
    # Optional JSON prompt bank (isaaclab_templates.json style). When set,
    # num_reverie_prompts prompts are composed once and cycled across every
    # task's episodes: episode e uses prompt e % num, so 50 trials cover 10
    # prompts 5x each.
    prompt_templates: Optional[str] = None
    prompt_template_keys: str = "env,robot,table,light"
    num_reverie_prompts: int = 10
    # Letterbox square sim frames to the server's inference aspect ratio
    # (edge-pad before sending, crop center back) so the 256->1280 stretch
    # doesn't distort the scene.
    pad_aspect: bool = True
    # Chunk sizes (only used by raw_chunked; rerender reads them from /info).
    first_chunk: int = 9
    later_chunk: int = 12
    # Render the cameras at 20*mult fps: (mult - 1) extra TRUE frames are
    # rendered inside each env.step's physics substep loop and fed to Reverie,
    # so chunks fill mult-x faster and the policy gets a fresh rerendered
    # frame every later_chunk/mult steps instead of every later_chunk. Sim
    # dynamics and the 20Hz policy cadence are untouched.
    reverie_fps_mult: int = 1

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


def _step_with_midstep_renders(env, action, mult):
    """LIBERO env.step with (mult - 1) evenly spaced TRUE camera renders inside
    the physics substep loop -> 20*mult fps perception, unchanged 20Hz
    dynamics/policy cadence.

    Reimplements the two step() layers this replaces (pinned versions):
    robosuite base.Environment.step (substep loop) and LIBERO
    bddl_base_domain.step (OSC_POSITION conversion + done=_check_success).
    Returns (obs, reward, done, info, mids); mids are (agentview, wrist)
    frames in the same convention as obs["agentview_image"].
    """
    if mult <= 1:
        obs, reward, done, info = env.step(action)
        return obs, reward, done, info, []

    import robosuite.utils.macros as macros
    import robosuite.utils.mjcf_utils as mjcf_utils

    raw = env.env  # OffScreenRenderEnv/ControlEnv -> robosuite env
    assert not raw.done, "executing action in terminated episode"
    if raw.action_dim == 4 and len(action) > 4:
        action = np.concatenate((np.asarray(action)[:3], np.asarray(action)[-1:]), axis=-1)

    conv = mjcf_utils.IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

    def render_cams():
        agent = raw.sim.render(camera_name="agentview",
                               width=LIBERO_ENV_RESOLUTION,
                               height=LIBERO_ENV_RESOLUTION)[::conv]
        wrist = raw.sim.render(camera_name="robot0_eye_in_hand",
                               width=LIBERO_ENV_RESOLUTION,
                               height=LIBERO_ENV_RESOLUTION)[::conv]
        return agent, wrist

    raw.timestep += 1
    n_substeps = int(raw.control_timestep / raw.model_timestep)
    # Substeps after which to grab an intermediate frame (the step's final
    # state is rendered by the caller's next observe()).
    render_at = {int(round(n_substeps * k / mult)) for k in range(1, mult)}
    mids = []
    policy_step = True
    for i in range(n_substeps):
        raw.sim.forward()
        raw._pre_action(action, policy_step)
        raw.sim.step()
        raw._update_observables()
        policy_step = False
        if (i + 1) in render_at:
            mids.append(render_cams())
    raw.cur_time += raw.control_timestep
    reward, done, info = raw._post_action(action)
    done = raw._check_success()
    obs = raw.viewer._get_observations() if raw.viewer_get_obs else raw._get_observations()
    return obs, reward, done, info, mids


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    if args.mode == "rerender" and not args.reverie_url:
        raise ValueError("--reverie-url is required for mode=rerender")

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

    streamers = None
    if args.mode == "rerender":
        streamers = rerender_loop.connect(
            args.reverie_url, num_views=2, config=args.reverie_config)
        m = streamers[0].meta
        logging.info("Reverie server: N=%s first_chunk=%s later_chunk=%s inf=%sx%s",
                     m["n"], m["first_chunk_frames"], m["later_chunk_frames"],
                     m["inference_h"], m["inference_w"])

    chunked = args.mode in ("rerender", "raw_chunked")
    rerenderer = None
    if chunked:
        # Views: [agentview, wrist].
        rerenderer = rerender_loop.ChunkedRerenderer(
            streamers, num_views=2, first_chunk=args.first_chunk,
            later_chunk=args.later_chunk, pad_aspect=args.pad_aspect)

    # One fixed set of composed reverie prompts, shared by all tasks and
    # cycled across each task's episodes (episode e -> prompt e % num).
    reverie_prompts = [args.reverie_prompt]
    if args.prompt_templates:
        reverie_prompts = rerender_loop.compose_prompts(
            args.prompt_templates, args.num_reverie_prompts, args.seed,
            keys=args.prompt_template_keys)
        for i, p in enumerate(reverie_prompts):
            logging.info("Reverie prompt %d: %s", i, p)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            reverie_prompt = reverie_prompts[episode_idx % len(reverie_prompts)]
            logging.info("\nTask: %s", task_description)
            logging.info("Reverie prompt [task %d ep %d]: %s",
                         task_id, episode_idx, reverie_prompt)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            action_plan = collections.deque()

            if rerenderer is not None:
                rerenderer.start_episode(reverie_prompt, mode=args.reverie_mode,
                                         seed=args.reverie_seed)

            last_img: Optional[np.ndarray] = None
            last_wrist: Optional[np.ndarray] = None

            policy_view = []   # 224 images actually fed to the policy
            reverie_view = []  # full rerendered agentview stream (rerender mode)

            t, done = 0, False
            logging.info("Starting episode %d...", task_episodes + 1)

            def observe():
                """Upright native-res frames (180deg rotate = train preproc)."""
                agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                return agent, wrist

            def feed(agent, wrist):
                """Buffer one frame; on a chunk flush, refresh the policy's
                rerendered views. Returns True if a flush happened."""
                nonlocal last_img, last_wrist
                out = rerenderer.add([agent, wrist])
                if out is None:
                    return False
                if args.save_reverie_video:
                    # Full inference-res agentview generation (the
                    # native `out` is downscaled back to 256px).
                    reverie_view.extend(rerenderer.hires_chunk[0])
                last_img = _resize224(out[0][-1], args.resize_size)
                last_wrist = _resize224(out[1][-1], args.resize_size)
                return True

            def feed_mids(mids):
                """Feed the true mid-step frames rendered inside env.step
                (same 180deg rotate as observe()). Returns True on a flush."""
                flushed = False
                for m_agent, m_wrist in mids:
                    flushed |= feed(np.ascontiguousarray(m_agent[::-1, ::-1]),
                                    np.ascontiguousarray(m_wrist[::-1, ::-1]))
                return flushed

            # Warmup: let physics settle and, in chunked modes, prime the
            # perception buffer with dummy-action steps so the policy's first
            # real observation is already a (re)rendered frame -- the policy
            # never acts on a raw sim frame during the first chunk.
            while t < args.num_steps_wait or (chunked and last_img is None):
                agent, wrist = observe()
                if chunked:
                    feed(agent, wrist)
                obs, reward, done, info, mids = _step_with_midstep_renders(
                    env, LIBERO_DUMMY_ACTION, args.reverie_fps_mult if chunked else 1)
                if chunked:
                    feed_mids(mids)
                t += 1

            while t < max_steps + args.num_steps_wait:
                agent, wrist = observe()

                if chunked and feed(agent, wrist):
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
                obs, reward, done, info, mids = _step_with_midstep_renders(
                    env, action.tolist(), args.reverie_fps_mult if chunked else 1)
                if chunked and feed_mids(mids):
                    action_plan.clear()
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

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
                    [np.asarray(x) for x in reverie_view], fps=10, quality=8)

            logging.info("Success: %s | episodes=%d successes=%d (%.1f%%)",
                         done, total_episodes, total_successes,
                         total_successes / total_episodes * 100)

        logging.info("Task success rate: %.3f", task_successes / max(task_episodes, 1))

    if rerenderer is not None:
        rerenderer.close()

    logging.info("Total success rate: %.3f (%d episodes)",
                 total_successes / max(total_episodes, 1), total_episodes)


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
