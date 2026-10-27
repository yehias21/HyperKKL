"""Disk cache with a creation lock, so concurrent jobs never build the same artefact twice."""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

import torch


def cached(path: Path, build, timeout_s: float = 4 * 3600, poll_s: float = 1.0):
    """Load ``path`` or build it under an exclusive flock on ``path.lock``. The kernel
    releases a flock when its holder dies, so a killed builder never leaves a stale lock
    and exactly one waiter takes over. Whoever holds the lock re-checks ``path`` first, so
    the lock file may be removed once ``path`` exists (never before: a newcomer could then
    lock a fresh inode while a waiter holds the old one, and both would build)."""
    path = Path(path)
    if path.exists():
        return torch.load(path, weights_only=False, map_location="cpu")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with open(lock, "a") as fh:
        t0 = time.time()
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:                   # someone else is building it
                if time.time() - t0 > timeout_s:      # builder hung: build it ourselves
                    break
                time.sleep(poll_s)
        if path.exists():
            return torch.load(path, weights_only=False, map_location="cpu")
        obj = build()
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        torch.save(obj, tmp)
        tmp.rename(path)
        lock.unlink(missing_ok=True)
        return obj


DATA_MODULES = ("systems", "systems_extra", "signals", "design", "data")   # systems_extra: battery, network
MODEL_MODULES = DATA_MODULES + ("nets", "train", "conditioning", "observers", "monitor")


def module_hash(names=MODEL_MODULES) -> str:
    """Hash of the source of the modules an artefact depends on. Part of every cache key,
    so a code fix can never be masked by an artefact built with the old code."""
    import hashlib
    pkg = Path(__file__).resolve().parent
    h = hashlib.sha1()
    for n in names:
        h.update((pkg / f"{n}.py").read_bytes())
    return h.hexdigest()[:10]
