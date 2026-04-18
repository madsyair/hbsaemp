"""MCMC sampler configuration for hbsaemp.

:class:`ModelConfig` bundles all sampler settings into one reusable object,
analogous to ``trainControl()`` in R's caret package.

.. code-block:: python

    # caret (R)                         # hbsaemp (Python)
    ctrl <- trainControl(               cfg = ModelConfig(
        verboseIter = TRUE,                 draws  = 2000,
    )                                       chains = 4,
                                            cores  = 2,
                                        )
    m1 <- train(..., trControl=ctrl)    m1 = hbm(..., config=cfg)
    m2 <- train(..., trControl=ctrl)    m2 = hbm(..., config=cfg)
"""

from __future__ import annotations

from dataclasses import dataclass

from hbsaemp._types import SamplePriorLiteral

__all__: list[str] = ["ModelConfig", "DEFAULT_CONFIG"]


@dataclass
class ModelConfig:
    """MCMC sampler configuration — shared across all model families.

    Pass a single ``ModelConfig`` to multiple :func:`hbm` calls to ensure
    consistent sampling settings across a model comparison study.

    Mapping to Bambi parameters (v1):
    ----------------------------------
    .. code-block:: text

        ModelConfig.draws          →  model.fit(draws=...)
        ModelConfig.tune           →  model.fit(tune=...)      # warmup
        ModelConfig.chains         →  model.fit(chains=...)
        ModelConfig.cores          →  model.fit(cores=...)
        ModelConfig.target_accept  →  model.fit(target_accept=...)
        ModelConfig.random_seed    →  model.fit(random_seed=...)

    Mapping to R hbsaems parameters:
    ----------------------------------
    .. code-block:: text

        ModelConfig.draws   ←→  iter - warmup
        ModelConfig.tune    ←→  warmup
        ModelConfig.chains  ←→  chains
        ModelConfig.cores   ←→  cores

    Args:
        draws: Post-warmup draws per chain. Default 1000.
            Total posterior samples = ``draws × chains``.
        tune: Warmup / adaptation steps per chain. Default 1000.
        chains: Independent Markov chains. Default 4.
            Use ≥ 4 for reliable :math:`\\hat{R}`.
        cores: CPU cores for parallel sampling. Default 1.
        target_accept: NUTS acceptance rate target. Default 0.8.
            Increase to 0.9–0.95 for complex posterior geometry.
        random_seed: Optional integer for reproducibility. Default ``None``.
        sample_prior: ``"no"`` (posterior) or ``"only"`` (prior predictive,
            used by :func:`check_prior`). Default ``"no"``.
        progressbar: Show sampling progress bar. Default ``True``.

    Example:
        >>> cfg = ModelConfig(draws=2000, tune=1000, chains=4, cores=2)
        >>> m1 = hbm("y ~ x1", family="gaussian", data=df, config=cfg)
        >>> m2 = hbm("y ~ x1", family="beta",     data=df, config=cfg)
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
        """Validate all fields on construction."""
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
        """Total posterior draws across all chains: ``draws × chains``."""
        return self.draws * self.chains

    def to_sampler_kwargs(self) -> dict:
        """Keyword arguments for ``bambi.Model.fit()`` (v1).

        Returns:
            Dict suitable for ``**`` unpacking into ``model.fit()``.
        """
        return {
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


#: Default config — used when ``config=None`` is passed to :func:`hbm`.
DEFAULT_CONFIG: ModelConfig = ModelConfig()
