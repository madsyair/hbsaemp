"""hbsaemp — Hierarchical Bayesian Small Area Estimation.

Python port of R `hbsaems` (Choir et al., 2025). Single caret-style entry
point:

    from hbsaemp import create_model, ModelConfig
    cfg = ModelConfig(draws=2000, chains=4)
    m = create_model("y ~ x1 + x2", family="gaussian", data=df, group="area",
                     config=cfg)
    m.fit()
    estimate_areas(m)             # alias: hbsae
    check_convergence(m)          # alias: hbcc
    compare_models([m1, m2])      # alias: hbmc
"""
from __future__ import annotations

import sys

__version__: str = "1.0.0"
if sys.version_info < (3, 12):
    raise RuntimeError(f"hbsaemp requires Python >= 3.12 (got {sys.version}).")

# Core
from hbsaemp._exceptions import (
    ConvergenceWarning as ConvergenceWarning,
    DataValidationError as DataValidationError,
    EstimationError as EstimationError,
    FormulaError as FormulaError,
    HBSAEError as HBSAEError,
    ModelNotFittedError as ModelNotFittedError,
    ModelRegistryError as ModelRegistryError,
    SpatialMatrixError as SpatialMatrixError,
    ValidationError as ValidationError,
)
from hbsaemp._logging import configure_logging as configure_logging

# GUI (frontend)
from hbsaemp.app import (
    DEFAULT_APP_CONFIG as DEFAULT_APP_CONFIG,
    App as App,
    AppConfig as AppConfig,
    launch_app as launch_app,
)
from hbsaemp.data._preprocessor import DataPreprocessor as DataPreprocessor

# Data layer
from hbsaemp.data._validator import DataValidator as DataValidator
from hbsaemp.data.datasets import (
    AVAILABLE_DATASETS as AVAILABLE_DATASETS,
    load_dataset as load_dataset,
)
from hbsaemp.diagnostics.comparison import (
    ComparisonResult as ComparisonResult,
    compare_models as compare_models,
    hbmc as hbmc,
)

# Diagnostics
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

# Estimation
from hbsaemp.estimation.areas import (
    AreaEstimatesResult as AreaEstimatesResult,
    estimate_areas as estimate_areas,
    hbsae as hbsae,
)
from hbsaemp.estimation.update import update_hbm as update_hbm, update_model as update_model
from hbsaemp.models._base import BaseModel as BaseModel, ModelResult as ModelResult

# Model layer
from hbsaemp.models._config import DEFAULT_CONFIG as DEFAULT_CONFIG, ModelConfig as ModelConfig
from hbsaemp.models._factory import (
    MODEL_REGISTRY as MODEL_REGISTRY,
    create_model as create_model,
    hbm as hbm,
)
from hbsaemp.models._family_spec import (
    FamilySpec as FamilySpec,
    get_family_spec as get_family_spec,
    list_families as list_families,
)
from hbsaemp.models._flex import hbm_flex as hbm_flex
from hbsaemp.models._prior import Prior as Prior
from hbsaemp.models._shortcuts import (
    hbm_beta as hbm_beta,
    hbm_binomial as hbm_binomial,
    hbm_gaussian as hbm_gaussian,
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
    "hbm_flex",              # tier 2: response + auxiliary list
    "hbm_beta", "hbm_gaussian", "hbm_binomial",  # tier 3
    "MODEL_REGISTRY",
    "Prior",                 # validated prior value object
    "FamilySpec",            # frozen family-metadata type
    "list_families", "get_family_spec",  # read-only family registry accessors
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
