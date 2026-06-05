"""`ModelConfig` — one reusable bundle of sampler settings.

Holds draws, chains, tune, etc.; passed via `create_model(config=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass

from hbsaemp._types import SamplePriorLiteral

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
        sample_prior: `"no"` (posterior) or `"only"` (prior predictive, used
            by `check_prior`). Default `"no"`.
        progressbar: Show sampling progress bar (default True).
    """

    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    cores: int = 1
    target_accept: float = 0.8
    random_seed: int | None = None
    sample_prior: SamplePriorLiteral = "no"
    progressbar: bool = True

    def __post_init__(self) -> None:
        if self.draws < 1:
            raise ValueError(f"`draws` must be ≥ 1, got {self.draws}")
        if self.tune < 0:
            raise ValueError(f"`tune` must be ≥ 0, got {self.tune}")
        if self.chains < 1:
            raise ValueError(f"`chains` must be ≥ 1, got {self.chains}")
        if self.cores < 1:
            raise ValueError(f"`cores` must be ≥ 1, got {self.cores}")
        if not (0 < self.target_accept < 1):
            raise ValueError(
                f"`target_accept` must be in (0, 1), got {self.target_accept}"
            )
        if self.sample_prior not in ("no", "only"):
            raise ValueError(
                f"`sample_prior` must be 'no' or 'only', got {self.sample_prior!r}"
            )

    @property
    def total_draws(self) -> int:
        """Total posterior draws across all chains: `draws * chains`."""
        return self.draws * self.chains

    def to_sampler_kwargs(self) -> dict:
        """Dict for `**` unpacking into `bambi.Model.fit()`."""
        return {
            "inference_method": "pymc",  # canonical name in Bambi 0.18+ ("mcmc" was deprecated)
            "draws": self.draws,
            "tune": self.tune,
            "chains": self.chains,
            "cores": self.cores,
            "target_accept": self.target_accept,
            "random_seed": self.random_seed,
            "progressbar": self.progressbar,
        }

    def __repr__(self) -> str:
        return (
            f"ModelConfig(draws={self.draws}, tune={self.tune}, "
            f"chains={self.chains}, cores={self.cores}, "
            f"target_accept={self.target_accept}, "
            f"sample_prior={self.sample_prior!r})"
        )


# Default used when `config=None` is passed to `create_model()`.
DEFAULT_CONFIG: ModelConfig = ModelConfig()
