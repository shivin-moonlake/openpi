"""Export one LIBERO bowl episode as a Reverie-standard gbuffer recording.

Runs a single episode of a LIBERO task against the openpi policy server (no
Reverie in the loop), captures the native-resolution agentview RGB per control
step, and writes a recording in the standard game-recording layout:

    recording_<id>/
      albedo/000000.png ...      # RGB frames (color-only models read this)
      color/000000.png  ...      # identical RGB copy
      albedo.mp4  color.mp4
      albedo_frametimes.txt  color_frametimes.txt   # ffconcat
      scene_descriptions.json

Only the RGB buffers are produced (MuJoCo has no native albedo/normal pass);
`albedo` doubles as the color/RGB input, which is exactly what the color-only
checkpoints (dual_dmd_cf / self_forcing, gbuffer_keys=[color]) consume.

The frame count is trimmed to the largest N <= captured with (N-1) % 4 == 0 so
the chunk-causal one-shot VAE encode contract holds. The recording dir is also
zipped (recording_<id>/ as the top-level entry) for upload.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import pathlib
import shutil

import imageio.v2 as imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass
class Args:
    # Policy server (openpi serve_policy.py --env LIBERO)
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO episode selection
    task_suite_name: str = "libero_spatial"  # black-bowl suite
    task_id: int = 0
    episode_idx: int = 0
    num_steps_wait: int = 10
    # Native render resolution for the exported frames (square).
    resolution: int = 512
    # Control steps to run (frames captured). Trimmed to largest (N-1)%4==0.
    capture_steps: int = 221

    # Output
    out_root: str = "data/libero/gbuffer_exports"
    recording_id: str = "libero_bowl_spatial_t0_e0"
    make_zip: bool = True
    seed: int = 7


def _resize224(frame_u8: np.ndarray, size: int) -> np.ndarray:
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(frame_u8, size, size))


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _largest_valid_len(n: int) -> int:
    """Largest m <= n with (m - 1) % 4 == 0 (chunk-causal encode contract)."""
    m = n - ((n - 1) % 4)
    return max(m, 1)


def _write_frametimes(path: pathlib.Path, subdir: str, n: int, fps: int) -> None:
    dur = 1.0 / fps
    lines = ["ffconcat version 1.0"]
    for i in range(n):
        lines.append(f"file '{subdir}/{i:06d}.png'")
        lines.append(f"duration {dur:.6f}")
    path.write_text("\n".join(lines) + "\n")


def export(args: Args) -> None:
    np.random.seed(args.seed)
    fps = 24

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    env, task_description = _get_libero_env(task, args.resolution, args.seed)
    logging.info("Task[%d]: %s", args.task_id, task_description)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    env.reset()
    obs = env.set_init_state(initial_states[args.episode_idx])

    import collections
    action_plan = collections.deque()
    frames: list[np.ndarray] = []

    # Let physics settle before capturing.
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    done = False
    for _ in tqdm.tqdm(range(args.capture_steps), desc="capture"):
        # 180deg rotate matches the training preprocessing (upright frame).
        agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        frames.append(agent)

        if not action_plan:
            element = {
                "observation/image": _resize224(agent, args.resize_size),
                "observation/wrist_image": _resize224(wrist, args.resize_size),
                "observation/state": np.concatenate((
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )),
                "prompt": str(task_description),
            }
            action_chunk = client.infer(element)["actions"]
            action_plan.extend(action_chunk[: args.replan_steps])

        action = action_plan.popleft()
        obs, _, done, _ = env.step(action.tolist())

    captured = len(frames)
    n = _largest_valid_len(captured)
    frames = frames[:n]
    logging.info("Captured %d frames -> keeping %d (task_success_seen=%s)",
                 captured, n, done)

    rec_dir = pathlib.Path(args.out_root) / f"recording_{args.recording_id}"
    if rec_dir.exists():
        shutil.rmtree(rec_dir)
    for buf in ("albedo", "color"):
        (rec_dir / buf).mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames):
            imageio.imwrite(rec_dir / buf / f"{i:06d}.png", fr)
        imageio.mimwrite(rec_dir / f"{buf}.mp4", frames, fps=fps,
                         codec="libx264", quality=8)
        _write_frametimes(rec_dir / f"{buf}_frametimes.txt", buf, n, fps)

    h, w = frames[0].shape[:2]
    scene = {
        "scene_description": (
            f"A Franka Emika Panda robot arm performing a tabletop manipulation "
            f"task: {task_description}. The scene is the LIBERO {args.task_suite_name} "
            f"simulation environment, with the arm, a table surface, and several "
            f"small objects including bowls, a plate, and a ramekin."
        ),
        "entities": {"robot_0": "A Franka Emika Panda 7-DoF robot arm with a parallel-jaw gripper"},
        "source": "libero",
        "task_suite": args.task_suite_name,
        "task_id": args.task_id,
        "episode_idx": args.episode_idx,
        "task_language": str(task_description),
        "num_frames": n,
        "resolution": [h, w],
        "fps": fps,
    }
    (rec_dir / "scene_descriptions.json").write_text(json.dumps(scene, indent=2))

    logging.info("Wrote recording: %s (%d frames, %dx%d)", rec_dir, n, h, w)

    if args.make_zip:
        zip_base = pathlib.Path(args.out_root) / f"recording_{args.recording_id}"
        # root_dir=out_root, base_dir=recording_<id> -> zip has recording_<id>/ at top.
        shutil.make_archive(str(zip_base), "zip", root_dir=args.out_root,
                            base_dir=rec_dir.name)
        logging.info("Wrote zip: %s.zip", zip_base)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    export(tyro.cli(Args))
