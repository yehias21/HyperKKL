"""Fetch the raw files of a real-data track from the ORIGINAL source, never from us.

    hyperkkl-bench fetch battery_pf                      # download from Mendeley Data, check SHA-256
    hyperkkl-bench fetch battery_pf --from-local DIR     # copy from a local copy (checked the same way)
    hyperkkl-bench fetch battery_pf --dry-run            # check URLs are reachable; download nothing

Every file has a SHA-256 recorded in ``resources/real_<dataset>.json`` (for Mendeley: the hash
the repository itself publishes). A file whose hash does not match is deleted and reported; it
is never used. Archives (``extract``) are unpacked with a path check (no absolute paths, no
``..``); nothing inside a download is ever executed. This package never uploads anything.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .registry import raw_dir, spec

UA = {"User-Agent": "hyperkkl-bench-fetch/1.1 (+research benchmark; read-only)"}


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_file(path, sha: str, size: int | None = None) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    if size is not None and p.stat().st_size != int(size):
        return False
    return sha256_file(p) == sha


def reachable(url: str, timeout: float = 30.0) -> tuple:
    """(ok, status or error). GET of the first byte (some hosts refuse HEAD)."""
    req = urllib.request.Request(url, headers=dict(UA, Range="bytes=0-0"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            st = getattr(r, "status", None)
            st = 200 if st is None else st             # file:// URLs (tests) have no HTTP status
            return (200 <= st < 300), st
    except Exception as e:  # noqa: BLE001  (network errors are reported, not raised)
        return False, f"{type(e).__name__}: {e}"


def _download(url: str, dest: Path, timeout: float) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dest)


def _safe_extract(zpath: Path, out: Path, members=None) -> list:
    out = out.resolve()
    done = []
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = info.filename
            if members is not None and name not in members:
                continue
            tgt = (out / name).resolve()
            if not str(tgt).startswith(str(out) + "/") and tgt != out:
                raise ValueError(f"{zpath.name}: unsafe member path {name!r}")
            if info.is_dir():
                continue
            tgt.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(tgt, "wb") as dst:
                shutil.copyfileobj(src, dst)
            done.append(name)
    return done


def _find_local(root: Path, rel: str) -> Path | None:
    for c in (root / rel, root / Path(rel).name):
        if c.is_file():
            return c
    hits = list(root.rglob(Path(rel).name))
    return hits[0] if len(hits) == 1 else None


def fetch(dataset: str, from_local=None, dry_run: bool = False, root=None, timeout: float = 120.0,
          log=print) -> dict:
    """Make the raw files of ``dataset`` available under raw_dir(dataset). Returns a summary dict
    {file: status}. Raises RuntimeError if any file is missing or fails its checksum (unless dry_run)."""
    s = spec(dataset)
    out = raw_dir(dataset, root)
    status, bad = {}, []
    for f in s["files"]:
        rel, url, sha, size = f["path"], f["url"], f["sha256"], f.get("size")
        dest = out / rel
        if check_file(dest, sha, size):
            status[rel] = "present (sha256 ok)"
            continue
        if dry_run:
            ok, st = reachable(url)
            status[rel] = f"{'reachable' if ok else 'UNREACHABLE'} ({st}) {url}"
            if not ok:
                bad.append(rel)
            continue
        if from_local:
            src = _find_local(Path(from_local), rel)
            if src is None:
                status[rel] = "not found in --from-local folder"
                bad.append(rel)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            how = f"copied from {src}"
        else:
            log(f"[fetch] {dataset}: {rel} <- {url}")
            try:
                _download(url, dest, timeout)
            except Exception as e:  # noqa: BLE001
                status[rel] = f"download failed: {type(e).__name__}: {e}"
                bad.append(rel)
                continue
            how = "downloaded"
        if not check_file(dest, sha, size):
            got = sha256_file(dest) if dest.exists() else None
            dest.unlink(missing_ok=True)
            status[rel] = f"CHECKSUM MISMATCH (got {got}, expected {sha}); file deleted"
            bad.append(rel)
            continue
        status[rel] = f"{how} (sha256 ok)"
        if f.get("extract"):
            names = _safe_extract(dest, out / f.get("extract_to", Path(rel).stem), f.get("members"))
            status[rel] += f", extracted {len(names)} files"
    for f in s.get("derived_checks", []):                # files inside archives with their own hashes
        p = out / f["path"]
        if not dry_run and not check_file(p, f["sha256"]):
            status[f["path"]] = "CHECKSUM MISMATCH after extraction"
            bad.append(f["path"])
    for k, v in status.items():
        log(f"  {k}: {v}")
    if bad and not dry_run:
        raise RuntimeError(f"{dataset}: {len(bad)} file(s) not available or failed the checksum: {bad}")
    return {"dataset": dataset, "raw_dir": str(out), "status": status, "failed": bad, "dry_run": dry_run}
