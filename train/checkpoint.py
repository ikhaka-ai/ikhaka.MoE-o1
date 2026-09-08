"""
Checkpoint/resume, adapted from the design discussed earlier to match what
this repo actually has: MixedDomainLoader's per-domain state_dict (see
data/loader.py) rather than a single loader, and MoETransformer's
aux_state()/load_aux_state() for the expert token counters.

Scope decision worth being explicit about: this class guarantees a
consistent, atomically-committed checkpoint on LOCAL disk. Getting that
checkpoint onto R2 is treated as a separate, pluggable concern -- pair this
with a periodic `rclone sync checkpoints/ r2:bucket/prefix` (or a cron'd
`aws s3 sync`) rather than baking boto3 credentials into this class. That
split means this file is fully testable with no cloud account, and the
sync command is one line to add later without touching training code.

Design recap (see the original checkpointing discussion):
  - A checkpoint is committed only when latest.json lands, written LAST,
    after every other file for that step exists. A crash mid-write leaves
    an orphaned step directory that nothing will ever read.
  - Periodic saving is the real safety net; a SIGTERM/SIGINT handler is
    best-effort on top of it, not a replacement for it.
  - RNG state is captured because MixedDomainLoader.__iter__ shuffles batch
    row order with the global `random` module (see data/loader.py) -- skip
    this and a resumed run's batches are still correct, just not
    byte-identical to what an uninterrupted run would have produced.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)

MANIFEST_NAME = "latest.json"


def capture_rng() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@dataclass
class CheckpointConfig:
    local_dir: Path
    keep_last: int = 3
    save_every_seconds: float = 1500.0   # ~25 min, see design doc Section 7
    save_every_steps: int | None = None  # alternative/additional trigger


class CheckpointManager:
    def __init__(self, cfg: CheckpointConfig):
        self.cfg = cfg
        self.cfg.local_dir = Path(cfg.local_dir)
        self.cfg.local_dir.mkdir(parents=True, exist_ok=True)
        self._last_save_time = time.monotonic()
        self._last_save_step = 0
        self._preempted = threading.Event()
        self._prev_sigterm = signal.signal(signal.SIGTERM, self._on_preempt)
        self._prev_sigint = signal.signal(signal.SIGINT, self._on_preempt)

    def _on_preempt(self, signum, frame):  # noqa: ARG002
        log.warning("preemption signal %s received", signum)
        self._preempted.set()

    @property
    def preempted(self) -> bool:
        return self._preempted.is_set()

    def should_save(self, step: int) -> bool:
        if self._preempted.is_set():
            return True
        if (time.monotonic() - self._last_save_time) >= self.cfg.save_every_seconds:
            return True
        if self.cfg.save_every_steps and step - self._last_save_step >= self.cfg.save_every_steps:
            return True
        return False

    def save(self, step: int, payload: dict[str, Any]) -> Path:
        """
        `payload` should carry model/optimizer/scheduler state dicts, the
        loader's state_dict(), RNG state, and the model's aux_state(). All
        of it lands in one state.pt -- simpler to reason about than several
        files at this checkpoint size (tens of MB at DEBUG_CONFIG, single-
        digit GB at BASE_CONFIG per the design doc's 4.53GB estimate).
        """
        step_dir = self.cfg.local_dir / f"step_{step:09d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        tmp = step_dir / "state.pt.tmp"
        torch.save(payload, tmp)
        final = step_dir / "state.pt"
        os.replace(tmp, final)   # atomic within this filesystem

        digest = _sha256(final)
        (step_dir / "meta.json").write_text(
            json.dumps({"step": step, "sha256": digest,
                       "bytes": final.stat().st_size}, indent=2)
        )

        # COMMIT POINT. latest.json is the only file load_latest() reads --
        # written last, so a crash before this line leaves step_dir
        # orphaned but invisible, never mistaken for a valid checkpoint.
        manifest_tmp = self.cfg.local_dir / (MANIFEST_NAME + ".tmp")
        manifest_tmp.write_text(json.dumps({
            "step": step, "dir": step_dir.name, "sha256": digest,
            "wall_time": time.time(),
        }, indent=2))
        os.replace(manifest_tmp, self.cfg.local_dir / MANIFEST_NAME)

        self._last_save_time = time.monotonic()
        self._last_save_step = step
        self._prune()
        log.info("checkpoint committed at step %d (%s)", step, final)
        return final

    def _prune(self) -> None:
        dirs = sorted(self.cfg.local_dir.glob("step_*"))
        for d in dirs[:-self.cfg.keep_last]:
            shutil.rmtree(d, ignore_errors=True)

    def load_latest(self, map_location: str = "cpu") -> dict[str, Any] | None:
        manifest_path = self.cfg.local_dir / MANIFEST_NAME
        if not manifest_path.exists():
            log.info("no checkpoint manifest at %s -- starting from scratch",
                     manifest_path)
            return None

        manifest = json.loads(manifest_path.read_text())
        state_path = self.cfg.local_dir / manifest["dir"] / "state.pt"
        if not state_path.exists():
            raise RuntimeError(
                f"manifest points at {state_path} but it doesn't exist -- "
                f"the checkpoint directory may have been partially deleted"
            )
        if _sha256(state_path) != manifest["sha256"]:
            raise RuntimeError(
                f"checksum mismatch at step {manifest['step']} -- "
                f"{state_path} is corrupt; restore an earlier step_* "
                f"directory by hand and point latest.json at it, or "
                f"restart from scratch"
            )
        log.info("resuming from step %d (%s)", manifest["step"], state_path)
        return torch.load(state_path, map_location=map_location, weights_only=False)

    def close(self) -> None:
        """Restore the previous signal handlers. Call this when done with
        the manager in a process that outlives training (tests, notebooks)
        -- otherwise the next thing to catch SIGINT in this interpreter is
        silently this object instead of the default handler."""
        signal.signal(signal.SIGTERM, self._prev_sigterm)
        signal.signal(signal.SIGINT, self._prev_sigint)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
