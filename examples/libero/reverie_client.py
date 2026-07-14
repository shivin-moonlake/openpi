"""Client for the shared Reverie stream server (/api WebSocket endpoint,
inference/stream/local_stream_server.py).

Replaces the per-job SIMPLER HTTP server: the eval now leases one GPU of the
long-lived multi-GPU stream server for the lifetime of this client (models
stay resident in VRAM there). py3.8-compatible for the LIBERO sim venv; deps
are numpy + websockets (already required by openpi_client).

Wire format: JSON text commands + .npy-encoded uint8 frames, resized to the
model's inference resolution and back server-side (same native-res contract
as the old integration/simpler/server.py).
"""
from __future__ import annotations

import io
import json
from typing import Optional

import numpy as np
import websockets.sync.client as ws_sync_client

WS_MAX_SIZE = 256 * 1024 * 1024


class ReverieClient:
    def __init__(self, url: str, timeout: float = 600.0, config: str = ""):
        self.ws = ws_sync_client.connect(
            url, max_size=WS_MAX_SIZE, open_timeout=timeout)
        hello = json.loads(self.ws.recv())
        if hello.get("type") == "error":
            self.ws.close()
            raise RuntimeError(f"stream server refused connection: {hello}")
        assert hello["type"] == "hello", hello
        self.meta = hello
        if config:
            self.meta = self._command({"cmd": "ensure_config", "config": config})

    def _command(self, request: dict) -> dict:
        self.ws.send(json.dumps(request))
        reply = json.loads(self.ws.recv())
        assert reply.get("type") == "ok", f"server error: {reply}"
        return reply

    def info(self) -> dict:
        m = self.meta
        return {
            "N": m["n"],
            "first_chunk_frames": m["first_chunk_frames"],
            "chunk_frames": m["later_chunk_frames"],
            "inference_height": m["inference_h"],
            "inference_width": m["inference_w"],
            "model_config": m["config_path"],
        }

    def open_session(self, prompt: str, mode: Optional[str] = None,
                     seed: Optional[int] = None) -> None:
        """set_prompt + reset_stream: starts a fresh stream on the leased GPU."""
        self._command({"cmd": "set_prompt", "prompt": prompt})
        request = {"cmd": "reset_stream"}
        if mode is not None:
            request["mode"] = mode
        if seed is not None:
            request["seed"] = seed
        self.meta = self._command(request)

    def process(self, frames: np.ndarray) -> np.ndarray:
        """frames: uint8 [B,n,H,W,3] (or [n,H,W,3]) -> same shape, rerendered."""
        if frames.dtype != np.uint8:
            raise ValueError(f"frames must be uint8, got {frames.dtype}")
        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(frames))
        self.ws.send(buf.getvalue())
        reply = self.ws.recv()
        assert isinstance(reply, bytes), f"server error: {reply}"
        return np.load(io.BytesIO(reply), allow_pickle=False)

    def close(self) -> None:
        """Release the leased GPU."""
        self.ws.close()
