"""Shared type aliases and literals for hbsaemp.

All types are stable across v0 -> v1 -> v2+.
"""
from __future__ import annotations
from typing import Any, Literal, TypeAlias
import numpy as np

__all__: list[str] = [
    "FormulaStr", "ColumnName", "GroupName",
    "FamilyLiteral", "LinkLiteral",
    "MissingStrategyLiteral", "SamplePriorLiteral",
    "SpatialTypeLiteral", "CARTypeLiteral", "SARTypeLiteral",
    "DiagTestLiteral", "PlotTypeLiteral", "ComparisonMetricLiteral",
    "DrawsArray", "AdjacencyMatrix", "WeightMatrix", "PriorDict",
]

FormulaStr: TypeAlias = str
ColumnName: TypeAlias = str
GroupName: TypeAlias = str

FamilyLiteral: TypeAlias = Literal["gaussian", "beta", "binomial"]
LinkLiteral: TypeAlias = Literal["identity", "log", "logit", "probit", "cloglog"]
MissingStrategyLiteral: TypeAlias = Literal["deleted"]
SamplePriorLiteral: TypeAlias = Literal["no", "only"]
SpatialTypeLiteral: TypeAlias = Literal["car", "sar"]
CARTypeLiteral: TypeAlias = Literal["icar", "escar", "esicar", "bym2"]
SARTypeLiteral: TypeAlias = Literal["lag", "error"]
DiagTestLiteral: TypeAlias = Literal["rhat", "ess", "geweke", "heidel", "raftery"]
PlotTypeLiteral: TypeAlias = Literal["trace", "dens", "acf", "pair", "rhat", "neff", "energy"]
ComparisonMetricLiteral: TypeAlias = Literal["loo", "bf"]

DrawsArray: TypeAlias = np.ndarray
AdjacencyMatrix: TypeAlias = np.ndarray
WeightMatrix: TypeAlias = np.ndarray
PriorDict: TypeAlias = dict[str, Any]
