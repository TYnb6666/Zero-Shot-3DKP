"""Feature-cache helpers for ZeroKey experiments."""

from typing import Any

__all__ = ["MolmoVitFeatureSaver"]


def __getattr__(name: str) -> Any:
    """Lazily import heavyweight feature helpers only when requested."""
    if name == "MolmoVitFeatureSaver":
        from zerokey.features.molmo_vit import MolmoVitFeatureSaver

        return MolmoVitFeatureSaver
    raise AttributeError(name)
