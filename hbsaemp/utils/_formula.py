"""Formula parsing utilities — v0 stub.

v1: Parse R/lme4-style formula strings into fixed-effect terms,
    random-effect groups, and response variable.
"""
from __future__ import annotations
__all__: list[str] = ["parse_formula"]


def parse_formula(formula: str) -> dict:
    """Parse a formula string into components.

    Args:
        formula: R/lme4-style formula, e.g. ``"y ~ x1 + x2 + (1|group)"``.

    Returns:
        Dict with keys ``response``, ``fixed``, ``random_groups``.

    Raises:
        NotImplementedError: In v0.
    """
    raise NotImplementedError("parse_formula() requires v1 (formulae package).")
