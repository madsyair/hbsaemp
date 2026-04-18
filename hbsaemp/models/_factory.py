"""Factory function and model registry.

:func:`create_model` is the single entry point for all HBSAE models —
analogous to ``caret::train()``.  The ``family=`` parameter selects the
model class from :data:`MODEL_REGISTRY`, just as ``method=`` does in caret.

The R-style alias ``hbm = create_model`` is provided for users migrating
from the R ``hbsaems`` package.

.. code-block:: python

    # caret (R)                        # hbsaemp (Python)
    cfg = trainControl(...)            cfg = ModelConfig(draws=2000, chains=4)

    m1 = train(f, data,                m1 = create_model("y ~ x1 + (1|area)",
               method="glm",                              family="gaussian",
               trControl=cfg)                             data=df, config=cfg)
    m2 = train(f, data,                m2 = create_model("y ~ x1",
               method="rf",                               family="beta",
               trControl=cfg)                             data=df, n="n",
                                                          deff="deff",
                                                          config=cfg)
    # Compare                          m1.fit(); m2.fit()
    resamples(list(m1,m2))             compare_models([m1, m2])

Extending the registry
-----------------------
To add a new family (e.g. for v2+ spatial models)::

    from hbsaemp.models._factory import MODEL_REGISTRY
    from my_module import GaussianSpatialModel
    MODEL_REGISTRY["gaussian_spatial"] = GaussianSpatialModel
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from hbsaemp._exceptions import ModelRegistryError
from hbsaemp._logging import get_logger
from hbsaemp._types import FamilyLiteral, FormulaStr, MissingStrategyLiteral, PriorDict
from hbsaemp.models._base import BaseModel
from hbsaemp.models._config import DEFAULT_CONFIG, ModelConfig
from hbsaemp.models._gaussian import GaussianModel
from hbsaemp.models._beta import BetaModel
from hbsaemp.models._binomial import BinomialModel
from hbsaemp.models._lognormal import LognormalModel

logger = get_logger(__name__)

__all__: list[str] = ["create_model", "hbm", "MODEL_REGISTRY"]


# ---------------------------------------------------------------------------
# MODEL_REGISTRY — maps family strings to model classes
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "gaussian":  GaussianModel,
    "beta":      BetaModel,
    "binomial":  BinomialModel,
    "lognormal": LognormalModel,
}


# ---------------------------------------------------------------------------
# create_model() — unified entry point (= caret's train())
# ---------------------------------------------------------------------------

def create_model(
    formula: FormulaStr,
    family: FamilyLiteral,
    data: pd.DataFrame,
    *,
    config: ModelConfig | None = None,
    priors: PriorDict | None = None,
    group: str | None = None,
    handle_missing: MissingStrategyLiteral = "deleted",
    # ── Family-specific optional parameters ──────────────────────────────
    trials: str | None = None,
    # ^ binomial: column name for number of trials
    n: str | None = None,
    # ^ beta: column name for survey sample size  (to compute phi)
    deff: str | None = None,
    # ^ beta: column name for design effect       (to compute phi)
    link: str | None = None,
    # ^ override default link for the family
    **kwargs: Any,
) -> BaseModel:
    """Create an unfitted HBSAE model — the single entry point for all families.

    Analogous to ``caret::train()``.  Call ``.fit()`` on the returned object
    to start MCMC sampling.

    The R-style alias ``hbm`` points to this function for users migrating
    from the R ``hbsaems`` package.

    Args:
        formula: R/lme4-style formula string.

            - Fixed effects only: ``"y ~ x1 + x2 + x3"``
            - With random intercept: ``"y ~ x1 + x2 + (1|group)"``
            - The ``group`` parameter adds ``(1|group)`` automatically
              if not already present in *formula*.

        family: Distribution family.  Must be a key in :data:`MODEL_REGISTRY`.
            Choices: ``"gaussian"``, ``"beta"``, ``"binomial"``,
            ``"lognormal"``.
        data: :class:`pandas.DataFrame` containing all variables in *formula*.
        config: Sampler settings (:class:`ModelConfig`).  ``None`` uses
            :data:`DEFAULT_CONFIG`.  Reuse one config object across multiple
            ``create_model()`` calls for consistent settings.
        priors: Prior specification dict or ``None`` (Bambi auto-priors in v1).
            Format: ``{"x1": {"dist": "Normal", "mu": 0, "sigma": 1}}``.
        group: Grouping column for random effects.  If provided and
            ``(1|group)`` is not already in *formula*, it is appended
            automatically.
        handle_missing: Missing data strategy. ``"deleted"`` (default) removes
            rows with ``NaN`` via ``pandas.dropna``.
        trials: *Binomial only.* Column with trial counts :math:`n_i`.
            Required when ``family="binomial"``.
        n: *Beta only.* Column with survey sample size.
            Used with *deff* to compute :math:`\\phi_i = n_i / \\text{deff}_i - 1`.
        deff: *Beta only.* Column with design effect.  Must accompany *n*.
        link: Override the family default link function.
        **kwargs: Additional keyword arguments forwarded to the model class.

    Returns:
        An *unfitted* :class:`BaseModel` subclass.  Call ``.fit()`` to run
        MCMC sampling.

    Raises:
        ModelRegistryError: If *family* is not in :data:`MODEL_REGISTRY`.
        TypeError: If *data* is not a :class:`pandas.DataFrame`.
        ValueError: For missing required family-specific parameters.

    Examples:
        Gaussian model with random effects::

            cfg = ModelConfig(draws=2000, chains=4)
            model = create_model("y ~ x1 + x2", family="gaussian",
                                 data=df, group="area", config=cfg)
            result = model.fit()

        Beta model with survey weights::

            model = create_model("y ~ x1 + x2", family="beta",
                                 data=df, n="n_sample", deff="design_eff",
                                 config=cfg)

        Binomial model::

            model = create_model("y ~ x1", family="binomial",
                                 data=df, trials="n_trials", config=cfg)

        Model comparison (caret's ``resamples`` equivalent)::

            m1 = create_model("y ~ x1", family="gaussian",  data=df, config=cfg)
            m2 = create_model("y ~ x1", family="lognormal", data=df, config=cfg)
            m1.fit(); m2.fit()
            result = compare_models([m1, m2])
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"`data` must be a pandas DataFrame, got {type(data).__name__!r}"
        )

    resolved_config: ModelConfig = config if config is not None else DEFAULT_CONFIG

    if family not in MODEL_REGISTRY:
        raise ModelRegistryError(
            f"Unknown family {family!r}. "
            f"Registered families: {sorted(MODEL_REGISTRY)}.",
            family=family,
            registered=sorted(MODEL_REGISTRY),
        )

    # Family-specific parameter validation
    if family == "binomial" and trials is None:
        raise ValueError(
            "family='binomial' requires the `trials` parameter: "
            "the column name containing the number of trials per area."
        )

    if family == "beta" and (n is None) != (deff is None):
        raise ValueError(
            "family='beta': provide both `n` and `deff`, or neither. "
            "Both are needed to compute the precision parameter phi."
        )

    # Auto-inject random intercept if group is given and not in formula
    resolved_formula = formula
    if group is not None and f"(1|{group})" not in formula:
        resolved_formula = f"{formula} + (1|{group})"
        logger.debug("Auto-injected random effect: %r", resolved_formula)

    logger.info(
        "create_model: family=%r, n=%d, formula=%r",
        family, len(data), resolved_formula,
    )

    model_class = MODEL_REGISTRY[family]
    return model_class(
        resolved_formula,
        family,
        data,
        resolved_config,
        priors=priors,
        group=group,
        handle_missing=handle_missing,
        # Beta-specific
        n_col=n,
        deff_col=deff,
        # Binomial-specific
        trials_col=trials,
        # Common optional
        link=link,
        **kwargs,
    )


#: R-style alias — ``hbm(...)`` is identical to ``create_model(...)``.
#: Provided for users migrating from R ``hbsaems``.
hbm = create_model
