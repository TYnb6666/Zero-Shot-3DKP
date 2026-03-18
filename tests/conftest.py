"""pytest configuration: add patchalign3d internals to sys.path once."""
from __future__ import annotations

import sys
from pathlib import Path

_PA3D = Path(__file__).resolve().parent.parent / "patchalign3d"
for _p in [str(_PA3D), str(_PA3D / "models")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
