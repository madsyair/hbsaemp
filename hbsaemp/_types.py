"""Shared type aliases and literals for hbsaemp (PEP 695 `type` statements).

Only aliases with a real consumer live here — speculative ones were removed.
Two will widen in v2: `FamilyLiteral` gains `"lognormal"`
(`docs/lognormal-v2.md`), and `MissingStrategyLiteral` widens only when a
second strategy lands in **both** `DataValidator` and `DataPreprocessor`.
Spatial aliases are parked in `docs/spatial-v2.md`, not defined here.
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
