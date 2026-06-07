"""Shared type aliases and literals for hbsaemp.

All types are stable across v0 -> v1 -> v2+.
"""
from __future__ import annotations

from typing import Any, Literal

__all__: list[str] = [
    "FormulaStr",
    "FamilyLiteral",
    "MissingStrategyLiteral",
    "PriorDict",
]

type FormulaStr = str

type FamilyLiteral = Literal["gaussian", "beta", "binomial"]
type MissingStrategyLiteral = Literal["deleted"]

type PriorDict = dict[str, Any]
