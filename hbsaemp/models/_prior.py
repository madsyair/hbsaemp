"""`Prior` value object — structurally validated, converts to `bambi.Prior`.

`create_model(priors=...)` accepts both `Prior` instances and the legacy
dict format `{"dist": "Normal", "mu": 0, "sigma": 1}`. Either way the
*structure* is checked immediately, before `bambi` is imported: `dist` must be
a non-empty string and at least one parameter must be supplied. Whether the
distribution exists and whether its parameter values make sense is resolved by
the backend at `fit()` time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hbsaemp._exceptions import PriorSpecError

__all__: list[str] = ["Prior", "validate_priors"]


@dataclass(frozen=True)
class Prior:
    """Prior specification — structurally validated at construction.

    Attribute rebinding is blocked, but the `params` dict itself is not
    deep-frozen; treat it as read-only by convention.

    Args:
        dist: PyMC distribution name (e.g. `"Normal"`, `"HalfNormal"`).
            Must be a non-empty string. Whether the distribution exists is
            resolved by the backend at `fit()` time, not here.
        **params: Distribution parameters forwarded to `bambi.Prior`.
            At least one must be supplied; values are not type-checked.

    Raises:
        PriorSpecError: `dist` empty / non-string, no params given, or
            `params` passed as a parameter name.
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
        if "params" in params:
            # `dataclasses.replace()` re-calls __init__ with the *field* names,
            # so its `params={...}` gets swallowed by **params and nested one
            # level deep — passing __post_init__ and yielding a Prior the
            # backend rejects much later. No PyMC distribution takes a
            # parameter called "params", so refusing it here turns that silent
            # corruption into an immediate error. Use copy.replace() instead.
            raise PriorSpecError(
                "'params' is not a valid distribution parameter name. "
                "Build a new Prior directly, e.g. Prior('Normal', mu=0, sigma=1)."
            )
        object.__setattr__(self, "dist", dist)
        object.__setattr__(self, "params", dict(params))
        self.__post_init__()

    def __replace__(self, **changes: Any) -> Prior:
        """Return a copy with *changes* applied (`copy.replace()`, Python 3.13+).

        Defined because the generic `dataclasses.replace()` path cannot work
        here: it would pass `params=` as a keyword, which `__init__` refuses.
        """
        dist = changes.pop("dist", self.dist)
        params = changes.pop("params", self.params)
        return type(self)(dist, **{**params, **changes})

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
