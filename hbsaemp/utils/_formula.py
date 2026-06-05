"""Parse R/lme4-style formula strings into response, fixed effects, and random-effect groups."""
from __future__ import annotations

import re

from hbsaemp._exceptions import FormulaError

__all__: list[str] = ["parse_formula"]


def parse_formula(formula: str) -> dict:
    """Parse an R/lme4-style formula string into its components.

    Handles standard fixed-effect formulas and random-intercept terms
    ``(1|group)``. The response is the first identifier on the left-hand
    side; a leading ``lhs | ...`` is tolerated, with the part after ``|``
    ignored.

    Args:
        formula: R/lme4-style formula string, e.g.
            ``"y ~ x1 + x2 + (1|group)"``.

    Returns:
        Dict with keys:

        - ``"response"`` (:class:`str`): left-hand-side response name.
        - ``"fixed"`` (:class:`list[str]`): fixed-effect predictor names.
        - ``"random_groups"`` (:class:`list[str]`): grouping factor names
          from ``(.*|group)`` terms.

    Raises:
        FormulaError: If ``"~"`` is absent or the response cannot be parsed.

    Examples:
        >>> parse_formula("y ~ x1 + x2 + (1|area)")
        {'response': 'y', 'fixed': ['x1', 'x2'], 'random_groups': ['area']}

        >>> parse_formula("y ~ x1")
        {'response': 'y', 'fixed': ['x1'], 'random_groups': []}
    """
    if "~" not in formula:
        raise FormulaError(
            "Formula must contain '~' separating response from predictors.",
            formula=formula,
        )

    lhs, rhs = formula.split("~", 1)

    # Response: first identifier on lhs (before any '|' or whitespace)
    response_match = re.match(r"\s*(\w+)", lhs)
    if not response_match:
        raise FormulaError(
            "Cannot parse response variable from formula LHS.",
            formula=formula,
        )
    response = response_match.group(1)

    # Extract all random-effect grouping factors: (anything | groupname)
    # \s* around the group name supports R-style spacing: (1 | group)
    random_groups: list[str] = re.findall(r"\([^|)]*\|\s*([A-Za-z_]\w*)\s*\)", rhs)

    # Remove all parenthesised groups (random effects) from RHS, then parse fixed
    rhs_fixed = re.sub(r"\([^)]*\)", "", rhs)
    fixed: list[str] = []
    for token in rhs_fixed.split("+"):
        term = token.strip()
        if not term or term in ("0", "1", "-1"):
            continue
        if not re.match(r"^[A-Za-z_]\w*$", term):
            raise FormulaError(
                f"Formula term {term!r} is not a valid column identifier. "
                f"Pre-compute any transformations (e.g. log, interactions) "
                f"in the DataFrame before calling create_model().",
                formula=formula,
            )
        fixed.append(term)

    return {
        "response": response,
        "fixed": fixed,
        "random_groups": random_groups,
    }
