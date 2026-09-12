"""`ModelConfig` — one reusable bundle of sampler settings.

Holds draws, chains, tune, etc.; passed via `create_model(config=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__: list[str] = ["ModelConfig", "DEFAULT_CONFIG"]


@dataclass
class ModelConfig:
    """NUTS sampler configuration shared across all model families.

    Pass the same `ModelConfig` to multiple `create_model()` calls to keep
    sampling settings consistent across a comparison study.

    Args:
        draws: Post-warmup draws per chain (default 1000). Total posterior
            samples = `draws * chains`.
        tune: Warmup steps per chain (default 1000).
        chains: Independent Markov chains (default 4). Use >= 4 for reliable
            r-hat.
        cores: CPU cores for parallel sampling (default 1).
        target_accept: NUTS acceptance rate target (default 0.8). Raise to
            0.9-0.95 for complex posterior geometry.
        random_seed: Integer seed for reproducibility (default None).
        progressbar: Show sampling progress bar (default True).
        max_treedepth: Maximum NUTS tree depth (default None → PyMC's own
            default of 10). Raise it when PyMC warns "reached the maximum tree
            depth".
        sampler_kwargs: Extra keyword arguments forwarded verbatim to
            `pm.sample` via Bambi (e.g. `init`, `nuts_sampler`). Must not
            override any managed key produced by `to_sampler_kwargs()`.
    """

    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    cores: int = 1
    target_accept: float = 0.8
    random_seed: int | None = None
    progressbar: bool = True
    max_treedepth: int | None = None
    sampler_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.draws < 1:
            raise ValueError(f"`draws` must be >= 1, got {self.draws}")
        if self.tune < 0:
            raise ValueError(f"`tune` must be >= 0, got {self.tune}")
        if self.chains < 1:
            raise ValueError(f"`chains` must be >= 1, got {self.chains}")
        if self.cores < 1:
            raise ValueError(f"`cores` must be >= 1, got {self.cores}")
        if not (0 < self.target_accept < 1):
            raise ValueError(
                f"`target_accept` must be in (0, 1), got {self.target_accept}"
            )
        if self.max_treedepth is not None and self.max_treedepth < 1:
            raise ValueError(
                f"`max_treedepth` must be >= 1, got {self.max_treedepth}"
            )
        if not isinstance(self.sampler_kwargs, dict):
            raise TypeError(
                f"`sampler_kwargs` must be a dict, got {type(self.sampler_kwargs).__name__}"
            )

    @property
    def total_draws(self) -> int:
        """Total posterior draws across all chains: `draws * chains`."""
        return self.draws * self.chains

    def to_sampler_kwargs(self) -> dict:
        """Dict for `**` unpacking into `bambi.Model.fit()`.

        `max_treedepth` is included only when set. `sampler_kwargs` is merged
        last as a passthrough to `pm.sample`, but may not override any
        hbsaemp-managed key (raises `ValueError`) so the fit contract stays intact.
        """
        base = {
            "inference_method": "pymc",  # canonical name in Bambi 0.18+ ("mcmc" was deprecated)
            "draws": self.draws,
            "tune": self.tune,
            "chains": self.chains,
            "cores": self.cores,
            "target_accept": self.target_accept,
            "random_seed": self.random_seed,
            "progressbar": self.progressbar,
            "include_response_params": False,  # pin lazy-μ contract (predict computes mu/p on demand)
        }
        if self.max_treedepth is not None:
            base["max_treedepth"] = self.max_treedepth
        if overlap := (set(base) & set(self.sampler_kwargs)):
            raise ValueError(
                f"sampler_kwargs may not override managed keys: {sorted(overlap)}"
            )
        base.update(self.sampler_kwargs)
        return base


# Default used when `config=None` is passed to `create_model()`.
DEFAULT_CONFIG: ModelConfig = ModelConfig()
