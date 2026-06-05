"""Shared type aliases and literals for hbsaemp.

All types are stable across v0 -> v1 -> v2+.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

__all__: list[str] = [
    "FormulaStr", "ColumnName", "GroupName",
    "FamilyLiteral", "LinkLiteral",
    "MissingStrategyLiteral", "SamplePriorLiteral",
    "SpatialTypeLiteral", "CARTypeLiteral", "SARTypeLiteral",
    "DiagTestLiteral", "PlotTypeLiteral", "ComparisonMetricLiteral",
    "DrawsArray", "AdjacencyMatrix", "WeightMatrix", "PriorDict",
]

type FormulaStr = str
type ColumnName = str
type GroupName = str

type FamilyLiteral = Literal["gaussian", "beta", "binomial"]
type LinkLiteral = Literal["identity", "log", "logit", "probit", "cloglog"]
type MissingStrategyLiteral = Literal["deleted"]
type SamplePriorLiteral = Literal["no", "only"]
type SpatialTypeLiteral = Literal["car", "sar"]
type CARTypeLiteral = Literal["icar", "escar", "esicar", "bym2"]
type SARTypeLiteral = Literal["lag", "error"]
type DiagTestLiteral = Literal["rhat", "ess", "geweke", "heidel", "raftery"]
type PlotTypeLiteral = Literal["trace", "dens", "acf", "pair", "rhat", "neff", "energy"]
type ComparisonMetricLiteral = Literal["loo", "bf"]

type DrawsArray = np.ndarray
type AdjacencyMatrix = np.ndarray
type WeightMatrix = np.ndarray
type PriorDict = dict[str, Any]
