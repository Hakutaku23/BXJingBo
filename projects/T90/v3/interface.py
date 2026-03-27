from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def recommend_t90_v3(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder V3 interface.

    This function intentionally returns a structured scaffold response so V3
    can define its real task cleanly before implementation begins.
    """

    base_dir = Path(__file__).resolve().parent
    return {
        "status": "scaffold_only",
        "version": "v3",
        "workspace_root": str(base_dir),
        "message": (
            "V3 has been scaffolded but not implemented yet. "
            "Use projects/T90/prior/ to choose the next objective before "
            "adding core logic."
        ),
        "received_keys": sorted(list(input_data.keys())),
    }
