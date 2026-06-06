from __future__ import annotations

import uuid
from pathlib import Path


def get_node_id() -> str:
    """Return a stable unique ID for this Node, persisted across restarts."""
    path = Path.home() / ".coremind" / "node-id"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        return path.read_text().strip()
    nid = str(uuid.uuid4())
    path.write_text(nid)
    return nid
