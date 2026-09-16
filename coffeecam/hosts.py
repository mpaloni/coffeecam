"""Loads local, non-secret-but-not-public host/IP config from ``hosts.env``.

``hosts.env`` (gitignored, lives at the repo root) holds the real homelab
addresses -- e.g. ``COFFEECAM_SOURCE_URL=http://192.168.50.10:8888`` -- so
they never end up hardcoded in source that's mirrored to the public GitHub
repo. ``hosts.env.example`` is the committed template with placeholders.

Values already set in the environment win (so systemd ``Environment=`` lines
and ad-hoc ``FOO=bar python -m ...`` invocations still take precedence).
"""

from __future__ import annotations

import os
from pathlib import Path

_HOSTS_FILE = Path(__file__).resolve().parent.parent / "hosts.env"
_loaded = False


def load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    if not _HOSTS_FILE.exists():
        return
    for line in _HOSTS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


load()
