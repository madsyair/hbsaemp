"""hbsaemp.models — model classes, config, factory, and family specs."""
from __future__ import annotations

from hbsaemp.models._base import BaseModel, ModelResult
from hbsaemp.models._config import DEFAULT_CONFIG, ModelConfig
from hbsaemp.models._factory import MODEL_REGISTRY, create_model, hbm
from hbsaemp.models._family_spec import FAMILY_SPECS, FamilySpec
from hbsaemp.models._flex import hbm_flex
from hbsaemp.models._prior import Prior
from hbsaemp.models._shortcuts import (
    hbm_beta,
    hbm_binomial,
    hbm_gaussian,
)

__all__: list[str] = [
    "BaseModel",
    "ModelResult",
    "ModelConfig",
    "DEFAULT_CONFIG",
    "MODEL_REGISTRY",
    "create_model",
    "hbm",
    "hbm_flex",
    "hbm_beta",
    "hbm_gaussian",
    "hbm_binomial",
    "FamilySpec",
    "FAMILY_SPECS",
    "Prior",
]
