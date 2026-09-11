"""Parse R/lme4-style formula strings into response, fixed effects, and random-effect groups."""
from __future__ import annotations

import re

from hbsaemp._exceptions import FormulaError

__all__: list[str] = ["parse_formula", "update_formula"]


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
        FormulaError: If ``"~"`` is absent, the response cannot be parsed, or a
            fixed-effect term is not a bare column identifier (e.g. an in-formula
            transform such as ``log(x1)``/``C(x1)``/``I(x1**2)`` or an interaction
            ``x1:x2``). Pre-compute such terms as DataFrame columns first.

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

    # Strip only random-effect groups (those containing '|'); any remaining '(' marks a
    # function term and will fail the bare-identifier check below — never silent-drop.
    rhs_fixed = re.sub(r"\([^)]*\|[^)]*\)", "", rhs)
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


def _split_terms(rhs: str) -> list[tuple[str, str]]:
    """Split *rhs* on top-level ``+``/``-`` into ``(sign, term)`` pairs.

    Signs inside parentheses belong to the term, so ``(0 + x1 | g)`` stays whole.
    """
    terms: list[tuple[str, str]] = []
    sign, depth, start = "+", 0, 0
    for i, char in enumerate(rhs):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char in "+-" and depth == 0:
            term = rhs[start:i].strip()
            if term:
                terms.append((sign, term))
            sign, start = char, i + 1
    term = rhs[start:].strip()
    if term:
        terms.append((sign, term))
    return terms


def _normalize_term(term: str) -> str:
    """Whitespace-free form of *term*, so ``(1 | g)`` matches ``(1|g)``."""
    return re.sub(r"\s+", "", term)


def update_formula(old: str, template: str) -> str:
    """Apply an R ``update.formula``-style *template* to *old*.

    ``.`` on the left stands for the old response and ``.`` on the right for
    the old right-hand side. ``+ term`` adds a term (once) and ``- term``
    removes it; ``- 1`` or ``+ 0`` drops the intercept. A template without
    ``.`` replaces *old* outright.

    Args:
        old: Current formula, e.g. ``"y ~ x1 + x2 + (1|group)"``.
        template: Update template, e.g. ``". ~ . + x3 - x1"``.

    Returns:
        The updated formula, validated by ``parse_formula()``.

    Raises:
        FormulaError: If *template* has no ``"~"``, or the result has no
            response or contains a term that is not a bare column identifier.
    """
    if "~" not in template:
        raise FormulaError(
            "Formula update template must contain '~', e.g. '. ~ . + x3'.",
            formula=template,
        )
    old_lhs, old_rhs = old.split("~", 1)
    new_lhs, new_rhs = template.split("~", 1)
    lhs = old_lhs.strip() if new_lhs.strip() == "." else new_lhs.strip()

    operations: list[tuple[str, str]] = []
    for sign, term in _split_terms(new_rhs):
        operations.extend(_split_terms(old_rhs) if term == "." else [(sign, term)])

    kept: list[str] = []
    intercept = True
    for sign, term in operations:
        if term in ("0", "1"):
            # "+ 0" and "- 1" drop the intercept; "+ 1" restores it.
            intercept = (term == "1") == (sign == "+")
            continue
        key = _normalize_term(term)
        if sign == "+":
            if all(_normalize_term(k) != key for k in kept):
                kept.append(term)
        else:
            kept = [k for k in kept if _normalize_term(k) != key]

    rhs_terms = kept if intercept else ["0", *kept]
    formula = f"{lhs} ~ {' + '.join(rhs_terms) or '1'}"
    parse_formula(formula)  # raises FormulaError on an invalid result
    return formula
