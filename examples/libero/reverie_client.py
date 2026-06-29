"""Minimal HTTP client for the Reverie rerender server (integration/simpler/server.py).

Stdlib-only (urllib) + numpy so it runs inside the LIBERO py3.8 venv without
extra deps. Wire format matches the server: npz-encoded uint8 frames.
"""
from __future__ import annotations

import io
import json
import urllib.request
from typing import Optional

import numpy as np


class ReverieClient:
    def __init__(self, host: str, port: int, timeout: float = 600.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self.session_id: Optional[str] = None

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(self.base + path, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def _post_json(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def _post_npz(self, path: str, **arrays) -> dict:
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        req = urllib.request.Request(
            self.base + path, data=buf.getvalue(), method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return dict(np.load(io.BytesIO(resp.read())))

    def info(self) -> dict:
        return self._get("/info")

    def open_session(self, prompt: str, mode: Optional[str] = None,
                     seed: Optional[int] = None, force: bool = True) -> str:
        body = {"prompt": prompt}
        if mode is not None:
            body["mode"] = mode
        if seed is not None:
            body["seed"] = seed
        path = "/session?force=true" if force else "/session"
        out = self._post_json(path, body)
        self.session_id = out["session_id"]
        return self.session_id

    def process(self, frames: np.ndarray) -> np.ndarray:
        """frames: uint8 [B,n,H,W,3] (or [n,H,W,3]) -> same shape, rerendered."""
        assert self.session_id is not None, "call open_session() first"
        if frames.dtype != np.uint8:
            raise ValueError(f"frames must be uint8, got {frames.dtype}")
        out = self._post_npz(f"/session/{self.session_id}/process", frames=frames)
        return out["frames"]

    def close(self) -> None:
        if self.session_id is None:
            return
        req = urllib.request.Request(
            f"{self.base}/session/{self.session_id}", method="DELETE")
        try:
            urllib.request.urlopen(req, timeout=self.timeout).read()
        finally:
            self.session_id = None
