"""Convergence diagnostics for fitted HBSAE models.

`check_convergence()` (alias `hbcc`) computes r-hat / ESS via ArviZ, emits
`ConvergenceWarning` when thresholds are breached, and renders trace / dens /
acf / rhat / neff / energy plots.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from hbsaemp._exceptions import ConvergenceWarning
from hbsaemp._logging import get_logger
from hbsaemp.diagnostics._plot_utils import (
    _CONVERGENCE_PLOT_CANDIDATES,
    _NO_VAR_NAMES_PLOTS,
    _diagnostic_var_names,
    _ensure_headless_matplotlib,
    _fig_from_axes,
)
from hbsaemp.models._base import BaseModel

logger = get_logger(__name__)
__all__: list[str] = ["ConvergenceResult", "check_convergence", "hbcc"]

# Default diagnostic settings
_DEFAULT_DIAG_TESTS: list[str] = ["rhat", "ess"]
_DEFAULT_PLOT_TYPES: list[str] = ["trace", "dens", "acf", "rhat", "neff", "energy"]

# Convergence thresholds
_RHAT_THRESHOLD: float = 1.01
_ESS_THRESHOLD: int = 400

# Spec-driven warning emission: each tuple is
#   (diag_test_key, summary_column, threshold, op, label, suffix)
# ``op`` is "gt" (col > threshold) for Rhat and "lt" (col < threshold) for ESS.
# ``label`` is rendered in the warning text; ``suffix`` is appended verbatim
# (used only by Tail ESS to mention credible-interval reliability).
_WARN_SPECS: tuple[tuple[str, str, float, str, str, str], ...] = (
    ("rhat", "r_hat",    _RHAT_THRESHOLD, "gt", "Rhat",     ""),
    ("ess",  "ess_bulk", _ESS_THRESHOLD,  "lt", "Bulk ESS", ""),
    ("ess",  "ess_tail", _ESS_THRESHOLD,  "lt", "Tail ESS",
     " — credible intervals may be unreliable."),
)


def _emit_convergence_warning(
    label: str,
    threshold: float,
    bad: pd.DataFrame,
    *,
    op: str,
    suffix: str = "",
) -> None:
    """Emit one ``ConvergenceWarning`` and a logger entry for a diagnostic.

    ``stacklevel=3`` attributes the warning to the caller of
    :func:`check_convergence`: frames are warn (1) ← this helper (2) ←
    ``check_convergence`` (3) ← user.
    """
    params = ", ".join(bad.index.tolist()[:5])
    symbol = ">" if op == "gt" else "<"
    warnings.warn(
        f"{label} {symbol} {threshold} for {len(bad)} parameter(s): "
        f"{params}{'...' if len(bad) > 5 else ''}{suffix}",
        ConvergenceWarning,
        stacklevel=3,
    )
    logger.warning(
        "check_convergence: %s %s %s for %d param(s).",
        label, symbol, threshold, len(bad),
    )


@dataclass
class ConvergenceResult:
    """Container for convergence diagnostic results.

    Attributes:
        rhat_ess: DataFrame with ``mean``, ``sd``, ``r_hat``, ``ess_bulk``,
            ``ess_tail`` per parameter (ArviZ summary).
        plots: Dict of ``matplotlib.Figure`` keyed by plot-type name.
            Keys from ``["trace", "dens", "acf", "rhat", "neff", "energy"]``.
        plot_errors: Dict keyed by plot-type name → error message, for plots
            that failed to render (surfaced rather than silently swallowed).
        geweke: Geweke Z-scores (v2+, ``None`` in v1).
        heidel: Heidelberger-Welch results (v2+, ``None`` in v1).
        raftery: Raftery-Lewis results (v2+, ``None`` in v1).
    """

    rhat_ess: Any = None
    plots: dict[str, Any] = field(default_factory=dict)
    plot_errors: dict[str, str] = field(default_factory=dict)
    geweke: Any = None
    heidel: Any = None
    raftery: Any = None

    def summary(self) -> str:
        """Human-readable convergence summary."""
        if self.rhat_ess is None:
            return "ConvergenceResult [not computed — call check_convergence()]"

        def _extreme(col: str, fn: str) -> float | None:
            # Returns ``None`` for missing columns AND all-NaN aggregates
            # (the latter happens after ArviZ writes "—" / "NA" strings that
            # ``pd.to_numeric(errors="coerce")`` turned into NaN).
            if col not in self.rhat_ess.columns:
                return None
            val = getattr(self.rhat_ess[col], fn)()
            return float(val) if pd.notna(val) else None

        max_rhat     = _extreme("r_hat",    "max")
        min_ess      = _extreme("ess_bulk", "min")
        min_ess_tail = _extreme("ess_tail", "min")

        status = "OK"
        if max_rhat is not None and max_rhat > _RHAT_THRESHOLD:
            status = f"WARNING: max Rhat={max_rhat:.4f} > {_RHAT_THRESHOLD}"
        elif min_ess is not None and min_ess < _ESS_THRESHOLD:
            status = f"WARNING: min Bulk ESS={min_ess:.0f} < {_ESS_THRESHOLD}"
        elif min_ess_tail is not None and min_ess_tail < _ESS_THRESHOLD:
            status = f"WARNING: min Tail ESS={min_ess_tail:.0f} < {_ESS_THRESHOLD}"

        max_rhat_str     = f"{max_rhat:.4f}"     if max_rhat     is not None else "N/A"
        min_ess_str      = f"{min_ess:.0f}"      if min_ess      is not None else "N/A"
        min_ess_tail_str = f"{min_ess_tail:.0f}" if min_ess_tail is not None else "N/A"
        return (
            f"ConvergenceResult\n"
            f"  Parameters   : {len(self.rhat_ess)}\n"
            f"  Max Rhat     : {max_rhat_str}\n"
            f"  Min ESS bulk : {min_ess_str}\n"
            f"  Min ESS tail : {min_ess_tail_str}\n"
            f"  Plots        : {list(self.plots.keys())}\n"
            + (f"  Plot errors  : {len(self.plot_errors)}\n" if self.plot_errors else "")
            + f"  Status       : {status}"
        )

    def __repr__(self) -> str:
        n = len(self.rhat_ess) if self.rhat_ess is not None else 0
        return f"ConvergenceResult(n_params={n}, plots={list(self.plots.keys())})"


def check_convergence(
    model: BaseModel,
    *,
    diag_tests: list[str] | None = None,
    plot_types: list[str] | None = None,
) -> ConvergenceResult:
    """Assess MCMC convergence of a fitted model.

    Python equivalent of ``hbcc()`` in R hbsaems.

    Computes:

    * **Rhat** (:math:`\\hat{R}`) — potential scale reduction factor per
      parameter.  Values > 1.01 trigger a :class:`~hbsaemp._exceptions.ConvergenceWarning`.
    * **ESS** (bulk / tail) — effective sample size.
      Values < 400 trigger a warning.

    Generates diagnostic plots (each stored as a ``matplotlib.Figure``):

    * ``"trace"``  — trace plots
    * ``"dens"``   — marginal posterior densities (``plot_dist``)
    * ``"acf"``    — autocorrelation plots
    * ``"rhat"``   — R-hat forest plot
    * ``"neff"``   — effective sample size plot
    * ``"energy"`` — NUTS energy / BFMI diagnostic

    Args:
        model: A fitted :class:`~hbsaemp.models._base.BaseModel`.
        diag_tests: Diagnostic tests to run.  Default ``["rhat", "ess"]``.
        plot_types: Plots to generate.  Default: all five listed above.
            Pass ``[]`` to skip plotting.

    Returns:
        :class:`ConvergenceResult` with ``rhat_ess`` DataFrame and ``plots``
        dict.

    Raises:
        ModelNotFittedError: If *model* has not been fitted.
        ImportError: If ``arviz`` is not installed.
    """
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "check_convergence() requires arviz>=1.1. "
            "Install with: pip install 'hbsaemp[bambi]'"
        ) from exc

    _ensure_headless_matplotlib()

    # guard
    result = model.result  # raises ModelNotFittedError if not fitted
    idata = result.idata

    resolved_tests = diag_tests if diag_tests is not None else _DEFAULT_DIAG_TESTS
    resolved_plots = plot_types if plot_types is not None else _DEFAULT_PLOT_TYPES

    logger.info(
        "check_convergence(): family=%r, tests=%r, plots=%r",
        result.family, resolved_tests, resolved_plots,
    )

    # 1. Rhat / ESS via ArviZ summary
    # Drop per-observation response params (mu/p/kappa) written by
    # include_response_params=True — their n_obs rows inflate the parameter
    # count and can skew the min-ESS / max-Rhat aggregates. Keep group-level
    # random effects (1|group): they ARE sampled and can fail to converge
    # independently. Explicit guard, NOT ``or None``: var_names=[] raises
    # TypeError, var_names=None processes ALL rows — both reintroduce the bug.
    summary_vars = _diagnostic_var_names(idata.posterior, policy="scalar_and_group")
    if summary_vars:
        summary_df = az.summary(idata, var_names=summary_vars)
    else:
        summary_df = pd.DataFrame()  # no scalar params (defensive; never in practice)
    keep_cols = ["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]
    available = [c for c in keep_cols if c in summary_df.columns]
    rhat_ess = summary_df[available].copy()
    # ArviZ ≥0.20 returns ``r_hat`` / ``ess_bulk`` / ``ess_tail`` as ``object``
    # dtype when any value renders as a string (e.g. "—" for diverged params).
    # Coerce upfront so threshold comparisons below and ``.max()`` / ``.min()``
    # in ``ConvergenceResult.summary()`` see plain floats (NaN where unparseable).
    for col in ("r_hat", "ess_bulk", "ess_tail"):
        if col in rhat_ess.columns:
            rhat_ess[col] = pd.to_numeric(rhat_ess[col], errors="coerce")

    # 2. Issue warnings
    # Spec-driven loop over ``_WARN_SPECS`` keeps the three Rhat/Bulk/Tail
    # branches in sync — same message shape, same logger call, single
    # ``stacklevel=3`` attribution to the caller of ``check_convergence``.
    for test, col, thr, op, label, suffix in _WARN_SPECS:
        if test not in resolved_tests or col not in rhat_ess.columns:
            continue
        bad = (
            rhat_ess[rhat_ess[col] > thr] if op == "gt"
            else rhat_ess[rhat_ess[col] < thr]
        )
        if not bad.empty:
            _emit_convergence_warning(label, thr, bad, op=op, suffix=suffix)

    # 3. Generate plots
    # ``_CONVERGENCE_PLOT_CANDIDATES`` (in ``_plot_utils``) lists fallback
    # ArviZ function names per plot type; the first callable on the installed
    # ArviZ wins.  ``_fig_from_axes`` then turns whatever the plot fn returns
    # (Axes / ndarray / Figure / tuple / PlotCollection) into a Figure or None.
    # Plots use ``scalar_only`` — one subplot per variable, so group effects
    # (and per-obs params) would blow past max_subplots=40 at large n_area.
    # Failures land in ``plot_errors`` instead of being swallowed to the log.
    plot_vars = _diagnostic_var_names(idata.posterior, policy="scalar_only")
    plots: dict[str, Any] = {}
    plot_errors: dict[str, str] = {}
    for ptype in resolved_plots:
        if ptype not in _CONVERGENCE_PLOT_CANDIDATES:
            logger.debug("check_convergence: unknown plot_type %r — skipped.", ptype)
            continue
        fn = next(
            (
                (getattr(az, name), kwargs)
                for name, kwargs in _CONVERGENCE_PLOT_CANDIDATES[ptype]
                if callable(getattr(az, name, None))
            ),
            None,
        )
        if fn is None:
            logger.warning(
                "check_convergence: no compatible ArviZ function for plot %r.", ptype
            )
            continue
        plot_fn, kwargs = fn
        # plot_energy reads sample_stats, not posterior vars, so it rejects
        # var_names; inject var_names only for the posterior-based plots.
        call_kwargs = dict(kwargs)
        if ptype not in _NO_VAR_NAMES_PLOTS:
            call_kwargs["var_names"] = plot_vars
        try:
            fig = _fig_from_axes(plot_fn(idata, **call_kwargs))
        except Exception as exc:  # noqa: BLE001
            plot_errors[ptype] = str(exc)
            logger.warning(
                "check_convergence: could not generate %r plot: %s", ptype, exc
            )
            continue
        # A None here means the plot ran but the Figure could not be extracted
        # from the ArviZ return value — record it instead of storing None silently.
        if fig is None:
            plot_errors[ptype] = (
                f"could not extract a Figure from {plot_fn.__name__}() return value."
            )
            logger.warning(
                "check_convergence: %r plot produced no extractable Figure.", ptype
            )
            continue
        plots[ptype] = fig
        logger.debug("check_convergence: generated %r plot.", ptype)

    logger.info(
        "check_convergence(): complete — %d params, %d plots generated.",
        len(rhat_ess), len(plots),
    )

    return ConvergenceResult(rhat_ess=rhat_ess, plots=plots, plot_errors=plot_errors)


#: R-style alias — ``hbcc(model)`` is equivalent to ``check_convergence(model)``.
hbcc = check_convergence
