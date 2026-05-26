"""hbsaemp v1.0.0 — Hierarchical Bayesian Small Area Estimation.

A Python port of R package ``hbsaems`` (Choir et al., 2025).
Design: **one function for all models**, like ``caret::train()``.

Quick start::

    from hbsaemp import create_model, ModelConfig

    cfg = ModelConfig(draws=2000, tune=1000, chains=4, cores=2)

    m1 = create_model("y ~ x1 + x2", family="gaussian",  data=df, group="area",  config=cfg)
    m2 = create_model("y ~ x1 + x2", family="beta",       data=df, n="n", deff="deff", config=cfg)
    m3 = create_model("y ~ x1 + x2", family="binomial",   data=df, trials="n",   config=cfg)
    m4 = create_model("y ~ x1 + x2", family="lognormal",  data=df, group="area",  config=cfg)

    m1.fit()                             # v1+: Bambi MCMC
    check_prior(m1)                      # prior predictive check  (alias: hbpc)
    check_convergence(m1)                # Rhat, ESS, trace plots  (alias: hbcc)
    compare_models([m1, m2])             # LOO, WAIC, pp_check     (alias: hbmc)
    estimate_areas(m1)                   # RSE, MSE, RMSE, CI      (alias: hbsae)
    launch_app()                         # GUI dashboard           (≡ run_sae_app)

Status: v1.0.0=Bambi | v2.0.0=spatial (PyMC, planned)
"""
from __future__ import annotations
import sys
__version__: str = "1.0.0"
if sys.version_info < (3, 11):
    raise RuntimeError(f"hbsaemp requires Python >= 3.11 (got {sys.version}).")

# ── Core infrastructure (concrete, all versions) ─────────────────────────────
from hbsaemp._logging import configure_logging as configure_logging
from hbsaemp._exceptions import (
    HBSAEError as HBSAEError,
    ValidationError as ValidationError,
    DataValidationError as DataValidationError,
    FormulaError as FormulaError,
    SpatialMatrixError as SpatialMatrixError,
    ModelRegistryError as ModelRegistryError,
    ModelNotFittedError as ModelNotFittedError,
    EstimationError as EstimationError,
    ConvergenceWarning as ConvergenceWarning,
)

# ── Model layer (stubs v0, Bambi v1) ─────────────────────────────────────────
from hbsaemp.models._config import ModelConfig as ModelConfig, DEFAULT_CONFIG as DEFAULT_CONFIG
from hbsaemp.models._base import BaseModel as BaseModel, ModelResult as ModelResult
from hbsaemp.models._factory import (
    create_model as create_model,
    hbm as hbm,               # R-style alias for create_model
    MODEL_REGISTRY as MODEL_REGISTRY,
)

# ── Data layer (stubs v0, concrete v1) ──────────────────────────────────────
from hbsaemp.data._validator import DataValidator as DataValidator
from hbsaemp.data._preprocessor import DataPreprocessor as DataPreprocessor
from hbsaemp.data.datasets import load_dataset as load_dataset, AVAILABLE_DATASETS as AVAILABLE_DATASETS

# ── Diagnostics (stubs v0, arviz v1) ────────────────────────────────────────
from hbsaemp.diagnostics.convergence import (
    ConvergenceResult as ConvergenceResult,
    check_convergence as check_convergence,
    hbcc as hbcc,
)
from hbsaemp.diagnostics.prior_check import (
    PriorCheckResult as PriorCheckResult,
    check_prior as check_prior,
    hbpc as hbpc,
)
from hbsaemp.diagnostics.comparison import (
    ComparisonResult as ComparisonResult,
    compare_models as compare_models,
    hbmc as hbmc,
)

# ── Estimation (stubs v0, concrete v1) ──────────────────────────────────────
from hbsaemp.estimation.areas import (
    AreaEstimatesResult as AreaEstimatesResult,
    estimate_areas as estimate_areas,
    hbsae as hbsae,
)
from hbsaemp.estimation.update import update_model as update_model, update_hbm as update_hbm

# ── GUI (stub v0, Panel v1) ──────────────────────────────────────────────────
from hbsaemp.app import (
    launch_app as launch_app,
    App as App,
    AppConfig as AppConfig,
    DEFAULT_APP_CONFIG as DEFAULT_APP_CONFIG,
)

__all__: list[str] = [
    "__version__", "configure_logging",
    # Exceptions
    "HBSAEError", "ValidationError", "DataValidationError", "FormulaError",
    "SpatialMatrixError", "ModelRegistryError", "ModelNotFittedError",
    "EstimationError", "ConvergenceWarning",
    # Model layer
    "ModelConfig", "DEFAULT_CONFIG", "BaseModel", "ModelResult",
    "create_model",          # primary Python name
    "hbm",                   # R-style alias for create_model
    "MODEL_REGISTRY",
    # Data layer
    "DataValidator", "DataPreprocessor", "load_dataset", "AVAILABLE_DATASETS",
    # Diagnostics (Python names + R aliases)
    "ConvergenceResult", "check_convergence", "hbcc",
    "PriorCheckResult", "check_prior", "hbpc",
    "ComparisonResult", "compare_models", "hbmc",
    # Estimation (Python names + R aliases)
    "AreaEstimatesResult", "estimate_areas", "hbsae",
    "update_model", "update_hbm",
    # GUI
    "launch_app", "App", "AppConfig", "DEFAULT_APP_CONFIG",
]
