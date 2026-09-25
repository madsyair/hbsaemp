"""Convergence diagnostics for fitted HBSAE models.

`check_convergence()` (alias `hbcc`) applies the ArviZ convergence checks —
rank-normalised split R-hat, bulk / tail ESS, divergences, maximum tree depth
and E-BFMI — emits `ConvergenceWarning` when a check fails, and renders
trace / dens / acf / rhat / neff / energy plots.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from hbsaemp._exceptions import ConvergenceWarning
from hbsaemp._logging import get_logger
from hbsaemp.diagnostics._plot_utils import (
    _CONVERGENCE_PLOTS,
    _diagnostic_var_names,
    _ensure_headless_matplotlib,
    _fig_from_axes,
)
from hbsaemp.models._base import BaseModel

logger = get_logger(__name__)
__all__: list[str] = ["ConvergenceResult", "check_convergence", "hbcc"]

# Default diagnostic settings
_DEFAULT_DIAG_TESTS: list[str] = ["rhat", "ess", "divergences", "treedepth", "bfmi"]
_DEFAULT_PLOT_TYPES: list[str] = ["trace", "dens", "acf", "rhat", "neff", "energy"]

# Convergence thresholds — the ``arviz.diagnose`` defaults (Vehtari et al.
# 2021; Betancourt 2016). The ESS floor scales with the chain count.
_RHAT_THRESHOLD: float = 1.01
_ESS_PER_CHAIN: int = 100
_BFMI_THRESHOLD: float = 0.3

# Per-parameter checks on the summary table: each tuple is
#   (diag_test_key, summary_column, op, label, suffix)
# ``op`` is "gt" (col > threshold) for Rhat and "lt" (col < threshold) for ESS.
# The threshold is looked up by ``diag_test_key`` at call time because the ESS
# floor depends on the chain count. ``suffix`` is appended verbatim (used only
# by Tail ESS to mention credible-interval reliability).
_WARN_SPECS: tuple[tuple[str, str, str, str, str], ...] = (
    ("rhat", "r_hat",    "gt", "Rhat",     ""),
    ("ess",  "ess_bulk", "lt", "Bulk ESS", ""),
    ("ess",  "ess_tail", "lt", "Tail ESS",
     " — credible intervals may be unreliable."),
)


def _emit_convergence_warning(message: str) -> None:
    """Emit one ``ConvergenceWarning`` and a matching logger entry.

    ``stacklevel=3`` attributes the warning to the caller of
    :func:`check_convergence`: frames are warn (1) ← this helper (2) ←
    ``check_convergence`` (3) ← user.
    """
    warnings.warn(message, ConvergenceWarning, stacklevel=3)
    logger.warning("check_convergence: %s", message)


def _table_issues(
    rhat_ess: pd.DataFrame,
    tests: list[str],
    thresholds: dict[str, float],
) -> list[str]:
    """Describe per-parameter R-hat / ESS breaches in the summary table."""
    issues: list[str] = []
    for test, col, op, label, suffix in _WARN_SPECS:
        if test not in tests or col not in rhat_ess.columns:
            continue
        thr = thresholds[test]
        bad = (
            rhat_ess[rhat_ess[col] > thr] if op == "gt"
            else rhat_ess[rhat_ess[col] < thr]
        )
        if bad.empty:
            continue
        params = ", ".join(bad.index.tolist()[:5])
        symbol = ">" if op == "gt" else "<"
        issues.append(
            f"{label} {symbol} {thr} for {len(bad)} parameter(s): "
            f"{params}{'...' if len(bad) > 5 else ''}{suffix}"
        )
    return issues


def _sampler_issues(diag: dict[str, Any], tests: list[str]) -> list[str]:
    """Describe NUTS sampler problems reported by ``arviz.diagnose``.

    The R-hat and ESS entries of *diag* are not re-reported: the summary table
    already carries the same R-hat (ArviZ's ``rank`` method is the maximum of
    the bulk and folded variants) and per-parameter bulk / tail ESS, and the
    ``ESS / N < 0.001`` ratio check is covered by the ``100 × n_chains`` floor
    for any run under 100 000 draws per chain.
    """
    issues: list[str] = []
    div = diag.get("divergent")
    if "divergences" in tests and div and div["n_divergent"] > 0:
        issues.append(
            f"{div['n_divergent']} of {div['total_samples']} ({div['pct']:.2f}%) "
            "transitions diverged — raise target_accept or reparameterise the model."
        )
    depth = diag.get("treedepth")
    if "treedepth" in tests and depth and depth["n_max"] > 0:
        issues.append(
            f"{depth['n_max']} of {depth['total_samples']} ({depth['pct']:.2f}%) "
            "transitions hit the maximum tree depth — exploration may be inefficient."
        )
    bfmi = diag.get("bfmi")
    if "bfmi" in tests and bfmi and bfmi["failed_chains"]:
        issues.append(
            f"E-BFMI < {bfmi['threshold']} in chain(s) {bfmi['failed_chains']} "
            f"(min {float(bfmi['bfmi_values'].min()):.3f}) — the sampler may "
            "struggle to explore the posterior."
        )
    return issues


@dataclass
class ConvergenceResult:
    """Container for convergence diagnostic results.

    Attributes:
        rhat_ess: DataFrame with ``mean``, ``sd``, ``r_hat``, ``ess_bulk``,
            ``ess_tail`` per parameter (ArviZ summary, unrounded floats).
        plots: Dict of ``matplotlib.Figure`` keyed by plot-type name.
            Keys from ``["trace", "dens", "acf", "rhat", "neff", "energy"]``
            plus ``"pair"`` when requested.
        plot_errors: Error message per plot type that failed to render.
        diagnose: Detailed results of ``arviz.diagnose`` keyed
            ``"divergent"``, ``"treedepth"``, ``"bfmi"``, ``"ess"`` and
            ``"rhat"``. The first three are absent when the sampler did not
            record the underlying statistic.
        ess_threshold: Bulk and tail ESS floor applied, ``100 × n_chains``.
    """

    rhat_ess: Any = None
    plots: dict[str, Any] = field(default_factory=dict)
    plot_errors: dict[str, str] = field(default_factory=dict)
    diagnose: dict[str, Any] | None = None
    ess_threshold: int | None = None

    def summary(self) -> str:
        """Human-readable convergence summary."""
        if self.rhat_ess is None:
            return "ConvergenceResult [not computed — call check_convergence()]"

        def _extreme(col: str, fn: str) -> float | None:
            # ``None`` for a missing column or an all-NaN aggregate.
            if col not in self.rhat_ess.columns:
                return None
            val = getattr(self.rhat_ess[col], fn)()
            return float(val) if pd.notna(val) else None

        def _fmt(val: float | None, spec: str) -> str:
            return format(val, spec) if val is not None else "N/A"

        max_rhat     = _extreme("r_hat",    "max")
        min_ess      = _extreme("ess_bulk", "min")
        min_ess_tail = _extreme("ess_tail", "min")

        diag  = self.diagnose or {}
        div   = diag.get("divergent")
        depth = diag.get("treedepth")
        bfmi  = diag.get("bfmi")
        min_bfmi = float(bfmi["bfmi_values"].min()) if bfmi else None

        issues: list[str] = []
        if max_rhat is not None and max_rhat > _RHAT_THRESHOLD:
            issues.append(f"max Rhat={max_rhat:.4f} > {_RHAT_THRESHOLD}")
        if self.ess_threshold is not None:
            if min_ess is not None and min_ess < self.ess_threshold:
                issues.append(f"min Bulk ESS={min_ess:.0f} < {self.ess_threshold}")
            if min_ess_tail is not None and min_ess_tail < self.ess_threshold:
                issues.append(f"min Tail ESS={min_ess_tail:.0f} < {self.ess_threshold}")
        if div and div["n_divergent"] > 0:
            issues.append(f"{div['n_divergent']} divergent transition(s)")
        if depth and depth["n_max"] > 0:
            issues.append(f"{depth['n_max']} max tree depth hit(s)")
        if bfmi and bfmi["failed_chains"]:
            issues.append(f"min E-BFMI={min_bfmi:.3f} < {bfmi['threshold']}")
        status = ("WARNING: " + "; ".join(issues)) if issues else "OK"

        div_str   = f"{div['n_divergent']} ({div['pct']:.2f}%)"  if div   else "N/A"
        depth_str = f"{depth['n_max']} ({depth['pct']:.2f}%)"    if depth else "N/A"
        return (
            f"ConvergenceResult\n"
            f"  Parameters   : {len(self.rhat_ess)}\n"
            f"  Max Rhat     : {_fmt(max_rhat, '.4f')}\n"
            f"  Min ESS bulk : {_fmt(min_ess, '.0f')}\n"
            f"  Min ESS tail : {_fmt(min_ess_tail, '.0f')}\n"
            f"  ESS threshold: {self.ess_threshold if self.ess_threshold is not None else 'N/A'}\n"
            f"  Divergences  : {div_str}\n"
            f"  Tree depth   : {depth_str}\n"
            f"  Min E-BFMI   : {_fmt(min_bfmi, '.3f')}\n"
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
    r"""Assess MCMC convergence of a fitted model.

    Python equivalent of ``hbcc()`` in R hbsaems.

    Applies the ArviZ checks (``arviz.summary`` and ``arviz.diagnose``).
    Each failing check emits a :class:`~hbsaemp.ConvergenceWarning`:

    * ``"rhat"``: rank-normalised split :math:`\hat{R}` (the larger of the
      bulk and folded variants) above 1.01 for any parameter.
    * ``"ess"``: bulk or tail effective sample size below
      :math:`100 \times` the number of chains.
    * ``"divergences"``: any divergent NUTS transition.
    * ``"treedepth"``: any transition that hit the maximum tree depth.
    * ``"bfmi"``: E-BFMI below 0.3 in any chain.

    Plots, each a ``matplotlib.Figure``:

    * ``"trace"``: trace plots.
    * ``"dens"``: marginal posterior densities (``plot_dist``).
    * ``"acf"``: autocorrelation plots.
    * ``"rhat"``: distribution of R-hat over the scalar and group-level
      parameters (``plot_convergence_dist``).
    * ``"neff"``: effective sample size.
    * ``"energy"``: NUTS energy and BFMI.
    * ``"pair"``: pairwise posterior scatter with divergent draws
      highlighted. Slow, so not drawn by default.

    Args:
        model: A fitted :class:`~hbsaemp.BaseModel`.
        diag_tests: Checks allowed to emit warnings. Default: all five above.
            Every check is computed and stored either way.
        plot_types: Plots to draw. Default: all except ``"pair"``. Pass
            ``[]`` to draw none.

    Returns:
        :class:`ConvergenceResult` with the ``rhat_ess`` table, the
        ``diagnose`` sampler checks and the ``plots`` dict.

    Raises:
        ModelNotFittedError: If *model* has not been fitted.
        ImportError: If ``arviz`` is not installed.
    """
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "check_convergence() requires arviz>=1.2. "
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
    # Drop per-observation response params (mu/p/kappa, materialised by
    # predict(kind="response_params")) — their n_obs rows inflate the parameter
    # count and can skew the min-ESS / max-Rhat aggregates. Keep group-level
    # random effects (1|group): they ARE sampled and can fail to converge
    # independently. Explicit guard, NOT ``or None``: var_names=[] raises
    # TypeError, var_names=None processes ALL rows — both reintroduce the bug.
    summary_vars = _diagnostic_var_names(idata.posterior, policy="scalar_and_group")
    if summary_vars:
        # round_to="none": the default "auto" rounds R-hat to two decimals and
        # returns strings, so an R-hat of 1.0147 reads as "1.01" and slips past
        # the > 1.01 check.
        summary_df = az.summary(idata, var_names=summary_vars, round_to="none")
    else:
        summary_df = pd.DataFrame()  # no scalar params (defensive; never in practice)
    keep_cols = ["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]
    available = [c for c in keep_cols if c in summary_df.columns]
    rhat_ess = summary_df[available].copy()

    # 2. Sampler-level checks via ArviZ diagnose
    # Divergences, tree depth and E-BFMI live in ``sample_stats`` and have no
    # per-parameter row in the summary table.
    ess_threshold = _ESS_PER_CHAIN * int(idata.posterior.sizes["chain"])
    diag: dict[str, Any] | None = None
    if summary_vars:
        _, diag = az.diagnose(
            idata,
            var_names=summary_vars,
            rhat_max=_RHAT_THRESHOLD,
            ess_threshold=ess_threshold,
            bfmi_threshold=_BFMI_THRESHOLD,
            show_diagnostics=False,
            return_diagnostics=True,
        )

    # 3. Issue warnings
    # Pure helpers collect the messages; emitting them from here keeps every
    # warning's ``stacklevel=3`` attribution pointing at the caller.
    thresholds = {"rhat": _RHAT_THRESHOLD, "ess": ess_threshold}
    issues = _table_issues(rhat_ess, resolved_tests, thresholds)
    if diag is not None:
        issues += _sampler_issues(diag, resolved_tests)
    for message in issues:
        _emit_convergence_warning(message)

    # 4. Generate plots
    # ``_CONVERGENCE_PLOTS`` (in ``_plot_utils``) maps each plot type to its
    # ArviZ function, kwargs and variable policy; ``_fig_from_axes`` extracts
    # the Figure from the returned PlotCollection. Per-variable plots use
    # ``scalar_only`` — one subplot per variable, so group effects (and per-obs
    # params) would blow past max_subplots=40 at large n_area.
    # Failures land in ``plot_errors`` instead of being swallowed to the log.
    var_names_by_policy = {
        "scalar_only": _diagnostic_var_names(idata.posterior, policy="scalar_only"),
        "scalar_and_group": summary_vars,
    }
    plots: dict[str, Any] = {}
    plot_errors: dict[str, str] = {}
    for ptype in resolved_plots:
        if ptype not in _CONVERGENCE_PLOTS:
            logger.debug("check_convergence: unknown plot_type %r — skipped.", ptype)
            continue
        name, kwargs, policy = _CONVERGENCE_PLOTS[ptype]
        plot_fn = getattr(az, name)
        call_kwargs = dict(kwargs)
        if policy is not None:
            call_kwargs["var_names"] = var_names_by_policy[policy]
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
            plot_errors[ptype] = f"could not extract a Figure from {name}() return value."
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

    return ConvergenceResult(
        rhat_ess=rhat_ess,
        plots=plots,
        plot_errors=plot_errors,
        diagnose=diag,
        ess_threshold=ess_threshold,
    )


#: R-style alias — ``hbcc(model)`` is equivalent to ``check_convergence(model)``.
hbcc = check_convergence
