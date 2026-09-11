"""`Prior` value object — validated at construction, converts to `bambi.Prior`.

`create_model(priors=...)` accepts both `Prior` instances and the legacy
dict format `{"dist": "Normal", "mu": 0, "sigma": 1}`. Either way validation
happens immediately, before `bambi` is imported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hbsaemp._exceptions import PriorSpecError

__all__: list[str] = ["Prior", "validate_priors"]


@dataclass(frozen=True)
class Prior:
    """Immutable prior specification — validated at construction.

    Args:
        dist: PyMC distribution name (e.g. `"Normal"`, `"HalfNormal"`).
            Must be a non-empty string.
        **params: Distribution parameters forwarded to `bambi.Prior`.
            At least one must be supplied.

    Raises:
        PriorSpecError: `dist` empty / non-string, or no params given.
    """

    dist: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.dist, str) or not self.dist.strip():
            raise PriorSpecError(
                f"Prior 'dist' must be a non-empty string, got {self.dist!r}."
            )
        if not self.params:
            raise PriorSpecError(
                f"Prior for dist={self.dist!r} has no parameters. "
                "Provide at least one keyword argument, e.g. Prior('Normal', mu=0, sigma=1)."
            )

    def __init__(self, dist: str, **params: Any) -> None:
        # Custom __init__ so callers write Prior("Normal", mu=0, sigma=1)
        # rather than Prior(dist="Normal", params={"mu": 0, "sigma": 1}).
        # Uses object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "dist", dist)
        object.__setattr__(self, "params", dict(params))
        self.__post_init__()

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, param: str | None = None) -> Prior:
        """Build a `Prior` from the legacy `{"dist": ..., ...}` dict format.

        Validates immediately. `param` is the model parameter name the prior
        belongs to (used in error messages only).
        """
        if not isinstance(d, dict) or "dist" not in d:
            raise PriorSpecError(
                f"Prior specification must be a dict with a 'dist' key, got {d!r}.",
                param=param,
            )
        dist = d["dist"]
        params = {k: v for k, v in d.items() if k != "dist"}
        return cls(dist, **params)

    def to_bambi(self, bmb_module: Any) -> Any:
        """Return a `bambi.Prior(dist, **params)`."""
        return bmb_module.Prior(self.dist, **self.params)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"Prior({self.dist!r}, {params_str})"


def validate_priors(priors: dict[str, Any] | None) -> None:
    """Validate a prior mapping eagerly, before any `bambi` import.

    `Prior` instances are validated at construction already; legacy dict
    entries are checked through `Prior.from_dict`. `None` is accepted.

    Raises:
        PriorSpecError: If a dict entry is malformed.
    """
    if priors is None:
        return
    for name, spec in priors.items():
        if not isinstance(spec, Prior):
            Prior.from_dict(spec, param=name)
