"""Fetch the shipped checkpoints from a release instead of keeping them in git."""

import hashlib
import json
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = (
    "https://github.com/divyanshg03/Causal-Trajectory-Analysis/releases/download/checkpoints-v1"
)
MANIFEST = "MANIFEST.json"  # {filename: sha256}, committed next to the checkpoints


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(directory):
    """Hash every ``*.pt`` in ``directory`` into ``MANIFEST.json``."""
    d = Path(directory)
    manifest = {p.name: sha256_file(p) for p in sorted(d.glob("*.pt"))}
    (d / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def download_checkpoints(directory="checkpoints", base_url=DEFAULT_BASE_URL, force=False):
    """Download every file listed in ``<directory>/MANIFEST.json`` that is missing or has the
    wrong hash, verifying each against its SHA-256. Returns the list of files fetched."""
    d = Path(directory)
    manifest_path = d / MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found; cannot tell what to download.")
    manifest = json.loads(manifest_path.read_text())
    fetched = []
    for name, digest in manifest.items():
        target = d / name
        if not force and target.is_file() and sha256_file(target) == digest:
            continue
        url = f"{base_url.rstrip('/')}/{name}"
        tmp = target.with_suffix(".part")
        try:
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https base URL
        except OSError as err:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download {url}: {err}") from err
        if sha256_file(tmp) != digest:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {name}; refusing to keep it.")
        tmp.replace(target)
        fetched.append(name)
    return fetched
