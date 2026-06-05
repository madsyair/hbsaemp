"""Model comparison (alias `hbmc`): PSIS-LOO-CV via ArviZ.

Also builds a pp_check and marginal posterior plots for the first model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hbsaemp._logging import get_logger
from hbsaemp.diagnostics._plot_utils import (
    _diagnostic_var_names,
    _ensure_headless_matplotlib,
    _fig_from_axes,
    _idata_groups,
)
from hbsaemp.models._base import BaseModel

logger = get_logger(__name__)
__all__: list[str] = ["ComparisonResult", "compare_models", "hbmc"]


@dataclass
class ComparisonResult:
    """Container for model comparison results.

    Attributes:
        loo: PSIS-LOO-CV result (``arviz.ELPDData`` for single model;
            ``dict[str, ELPDData]`` for multiple models).
        bayes_factor: Bayes Factor (v2+, ``None`` in v1).
        pp_check_plot: ``matplotlib.Figure`` — posterior predictive check
            for the first (or only) model.
        params_plot: ``matplotlib.Figure`` — marginal posterior distributions
            for the first (or only) model.
        compare_plot: ``matplotlib.Figure`` — model-comparison summary
            (``arviz.plot_compare``); only when *models* is a list with ≥ 2
            elements, else ``None``.
        prior_sensitivity: Prior sensitivity results (v2+, ``None`` in v1).
        comparison_table: ``pandas.DataFrame`` from ``arviz.compare()``
            (only when *models* is a list with ≥ 2 elements; else ``None``).
    """

    loo: Any = None
    bayes_factor: Any = None
    pp_check_plot: Any = None
    params_plot: Any = None
    compare_plot: Any = None
    prior_sensitivity: Any = None
    comparison_table: Any = None

    def summary(self) -> str:
        """Human-readable comparison summary."""
        if self.loo is None:
            return "ComparisonResult [not computed — call compare_models()]"

        def _elpd_val(v: Any) -> str:
            # ArviZ 1.1 renamed ELPDData's elpd accessor to the metric-agnostic
            # `elpd` (was `elpd_loo`). Read the canonical name first, falling
            # back to the legacy one for older ArviZ. `?` only when none are
            # present (genuinely missing), never just because the name moved.
            for name in ("elpd", "elpd_loo"):
                val = getattr(v, name, None)
                if val is not None:
                    return f"{float(val):.2f}"
            return "?"

        def _fmt_elpd(elpd: Any) -> str:
            if elpd is None:
                return "N/A"
            if isinstance(elpd, dict):
                return "{" + ", ".join(f"{k}: {_elpd_val(v)}" for k, v in elpd.items()) + "}"
            return _elpd_val(elpd)

        lines = ["ComparisonResult"]
        lines.append(f"  LOO  : {_fmt_elpd(self.loo)}")
        if self.comparison_table is not None:
            lines.append(f"  Comparison table: {len(self.comparison_table)} models")
        lines.append(
            f"  Plots: pp_check={'yes' if self.pp_check_plot else 'no'}, "
            f"params={'yes' if self.params_plot else 'no'}, "
            f"compare={'yes' if self.compare_plot else 'no'}"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        has_table = self.comparison_table is not None
        return (
            f"ComparisonResult("
            f"loo={'yes' if self.loo is not None else 'no'}, "
            f"comparison_table={has_table})"
        )


# Only PSIS-LOO-CV is supported. compare_models() validates against this set and
# raises on anything else rather than accepting an argument it would ignore.
_SUPPORTED_METRICS: frozenset[str] = frozenset({"loo"})


def _validate_compare_args(
    *,
    metrics: list[str] | str | None,
    n_draws_ppc: int,
    run_prior_sensitivity: bool,
    sensitivity_vars: list[str] | None,
) -> tuple[str, ...]:
    """Normalise and validate ``compare_models`` arguments (fail-fast).

    Returns the lower-cased metric tuple. Arguments that are advertised but not
    yet implemented (non-LOO metrics, prior sensitivity) raise instead of being
    silently ignored, and ``n_draws_ppc`` must be a positive integer.
    """
    if metrics is None:
        metrics_out: tuple[str, ...] = ("loo",)
    elif isinstance(metrics, str):
        metrics_out = (metrics.lower(),)
    else:
        metrics_out = tuple(str(m).lower() for m in metrics)

    if not metrics_out:
        raise ValueError(
            "compare_models(metrics=...) must contain at least one metric. "
            "Currently supported: ['loo']."
        )

    unsupported = sorted(set(metrics_out) - _SUPPORTED_METRICS)
    if unsupported:
        raise NotImplementedError(
            "compare_models() currently supports only metrics=['loo']. "
            f"Unsupported metric(s): {unsupported}. WAIC was removed from ArviZ "
            "(>=0.20) in favour of PSIS-LOO-CV; other metrics must be added "
            "explicitly in a future version, not silently ignored."
        )

    if isinstance(n_draws_ppc, bool) or not isinstance(n_draws_ppc, int) or n_draws_ppc <= 0:
        raise ValueError(
            f"compare_models(n_draws_ppc=...) must be a positive integer; got {n_draws_ppc!r}."
        )

    if run_prior_sensitivity:
        raise NotImplementedError(
            "compare_models(run_prior_sensitivity=True) is not implemented yet; "
            "do not request it until prior sensitivity analysis lands."
        )

    if sensitivity_vars is not None:
        raise NotImplementedError(
            "compare_models(sensitivity_vars=...) is not implemented yet because "
            "prior sensitivity analysis is not implemented."
        )

    return metrics_out


def _require_log_likelihood(idata: Any, *, model_name: str) -> None:
    """Raise a clear error when *idata* lacks the ``log_likelihood`` group.

    ``az.loo()`` needs it; ``BaseModel.fit()`` always writes it via
    ``bmodel.compute_log_likelihood(idata)``. Guarding here turns a missing
    group into an actionable message instead of an opaque ArviZ failure.
    """
    if "log_likelihood" not in _idata_groups(idata):
        raise ValueError(
            f"{model_name} is missing the 'log_likelihood' group required by "
            "az.loo(). Fit the model via BaseModel.fit(), which calls "
            "bambi.Model.compute_log_likelihood(idata) after sampling."
        )


def compare_models(
    models: BaseModel | list[BaseModel],
    *,
    metrics: list[str] | str | None = None,
    n_draws_ppc: int = 100,
    run_prior_sensitivity: bool = False,
    sensitivity_vars: list[str] | None = None,
) -> ComparisonResult:
    """Compute goodness-of-fit metrics and compare models.

    Python equivalent of ``hbmc()`` in R hbsaems.

    For each model (or the single model):

    * Computes **PSIS-LOO-CV** (Pareto-smoothed importance-sampling
      leave-one-out cross-validation) via ``arviz.loo()``.

    WAIC is **not** computed: ArviZ removed ``arviz.waic()`` in ≥0.20 in
    favour of PSIS-LOO-CV, which is more robust, has better theoretical
    properties, and ships reliability diagnostics (Pareto-k). Use LOO for
    model selection.

    For the **first** (or only) model:

    * Generates a **posterior predictive check** plot via
      ``arviz_plots.plot_ppc_dist()`` (``az.plot_ppc`` was removed in ArviZ
      1.1).  When ``posterior_predictive`` is missing it is built on a
      temporary idata via the public ``model.predictive_idata()``
      (``inplace=False``) — the stored ``result.idata`` is never mutated.
    * Generates a **marginal posterior** plot via ``az.plot_dist`` — the
      ArviZ 1.1 replacement for the removed ``plot_posterior`` / ``plot_density``
      — falling back to ``plot_trace`` (best-effort).

    When ≥ 2 models are passed, also computes a **comparison table** via
    ``arviz.compare()`` (LOO-based stacking weights) and a **comparison plot**
    via ``arviz.plot_compare()`` (ELPD ± SE per model).

    The output container shape mirrors the input container:

    * ``compare_models(model)`` → *loo* is an ``arviz.ELPDData``.
    * ``compare_models([model])`` → *loo* is a ``dict[str, ELPDData]``
      keyed by ``"model_0"`` (the list form always returns a dict, even
      with a single element).
    * ``compare_models([m0, m1, ...])`` → *loo* is a ``dict[str, ELPDData]``
      keyed by ``"model_0"``, ``"model_1"``, etc.

    Args:
        models: A single fitted model or a list of fitted models.
        metrics: Metrics to compute.  Only ``["loo"]`` is supported (default).
            Any other metric — including ``"waic"``, removed from ArviZ ≥0.20 in
            favour of PSIS-LOO-CV — raises ``NotImplementedError`` rather than
            being silently ignored.  An empty list raises ``ValueError``.
        n_draws_ppc: Number of posterior predictive draws for the pp-check
            plot.  Must be a positive integer (default 100).
        run_prior_sensitivity: Not implemented — ``True`` raises
            ``NotImplementedError``.
        sensitivity_vars: Not implemented — a non-``None`` value raises
            ``NotImplementedError``.

    Returns:
        :class:`ComparisonResult` with LOO, plots, and comparison table.

    Raises:
        ValueError: If *models* is an empty list, ``metrics`` is empty,
            ``n_draws_ppc`` is not a positive integer, or a model's idata lacks
            the ``log_likelihood`` group required by ``az.loo()``.
        NotImplementedError: If an unsupported metric, ``run_prior_sensitivity``,
            or ``sensitivity_vars`` is requested.
        ModelNotFittedError: If any model in *models* has not been fitted.
        ImportError: If ``arviz`` is not installed.
    """
    # Fail fast on advertised-but-unsupported arguments before importing arviz —
    # a typo like metrics=["wiac"] should error clearly even in a minimal install.
    metrics = _validate_compare_args(
        metrics=metrics,
        n_draws_ppc=n_draws_ppc,
        run_prior_sensitivity=run_prior_sensitivity,
        sensitivity_vars=sensitivity_vars,
    )

    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "compare_models() requires arviz>=1.1. "
            "Install with: pip install 'hbsaemp[bambi]'"
        ) from exc

    _ensure_headless_matplotlib()

    # 1. Normalise to list
    # ``input_was_single`` distinguishes the bare-model and single-element-list
    # forms — only the bare form unwraps to a flat ``ELPDData`` so the output
    # container shape stays predictable from the input shape.
    input_was_single = isinstance(models, BaseModel)
    if input_was_single:
        model_list: list[BaseModel] = [models]
    else:
        model_list = list(models)

    if not model_list:
        raise ValueError(
            "compare_models() requires at least one fitted model; got an empty list."
        )

    n_models = len(model_list)
    logger.info("compare_models(): %d model(s), metrics=%r", n_models, metrics)

    # 2. ModelNotFittedError guard for all models
    idatas: list[Any] = []
    for i, m in enumerate(model_list):
        r = m.result  # raises ModelNotFittedError if not fitted
        idatas.append(r.idata)

    # 3. PSIS-LOO-CV per model. WAIC is intentionally not computed: ArviZ
    # removed az.waic in ≥0.20 (lower-bound is now ≥1.1) in favour of LOO.
    loo_results: dict[str, Any] = {}

    for i, idata in enumerate(idatas):
        key = f"model_{i}"
        _require_log_likelihood(idata, model_name=key)
        loo_results[key] = az.loo(idata)
        logger.debug("compare_models: LOO for %s computed.", key)

    # For bare single-model input: unwrap to bare ELPDData. List input keeps
    # the dict shape even with a single element (see docstring contract).
    if input_was_single:
        loo_out: Any = loo_results.get("model_0")
    else:
        loo_out = loo_results

    # 4. Posterior predictive check (first model)
    # az.plot_ppc was removed in ArviZ 1.1; PPC plotting moved to the separate
    # ``arviz_plots`` package. ``plot_ppc_dist`` is the closest analogue (KDE/
    # dist overlay of observed vs Y_rep). Lazy + guarded import: arviz_plots is a
    # declared dep (pyproject [bambi]) and also transitive via bambi+arviz>=1.1,
    # but the callable() gate still protects minimal/partial installs. PPC is an
    # optional enhancement — its failure must not fail the whole comparison,
    # hence the try/except.
    pp_check_plot: Any = None
    first_model = model_list[0]
    first_idata = idatas[0]

    try:
        import arviz_plots as azp
    except ImportError:
        azp = None
    plot_ppc_fn = getattr(azp, "plot_ppc_dist", None) if azp is not None else None
    if callable(plot_ppc_fn):
        try:
            # Ensure posterior_predictive is populated (lazy pattern) WITHOUT
            # mutating the stored idata: predictive_idata() (inplace=False)
            # returns a fresh idata carrying the Y_rep group; predict() no
            # longer mutates result.idata, so we plot from that.
            # _idata_groups() is used instead of `.groups()` because in ArviZ
            # 1.1 idata is a DataTree where `.groups` is a property (tuple),
            # not a method — `.groups()` would raise TypeError.
            if "posterior_predictive" not in _idata_groups(first_idata):
                ppc_idata = first_model.predictive_idata()
            else:
                ppc_idata = first_idata
            pp_check_plot = _fig_from_axes(
                plot_ppc_fn(ppc_idata, num_samples=n_draws_ppc)  # was num_pp_samples
            )
            logger.debug("compare_models: pp_check plot generated.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("compare_models: pp_check plot failed: %s", exc)

    # 5. Marginal posterior plot (first model)
    # ArviZ 1.1 removed ``plot_posterior`` / ``plot_density`` from the namespace;
    # ``plot_dist`` is the canonical marginal-density plot, with ``plot_trace``
    # as a last-resort fallback (best-effort — failure is logged, not raised).
    params_plot: Any = None
    plot_fn = next(
        (
            getattr(az, name)
            for name in ("plot_dist", "plot_trace")
            if callable(getattr(az, name, None))
        ),
        None,
    )
    if plot_fn is not None:
        try:
            # Keep only scalar params. Per-area random effects (`1|group`) and
            # per-obs params (mu/kappa) carry extra dims and would explode the
            # subplot count past matplotlib's `rcParams["plot.max_subplots"]=40`
            # cap. `scalar_only` is the shared helper used by convergence plots.
            common_vars = _diagnostic_var_names(first_idata.posterior, policy="scalar_only")
            kwargs = {"var_names": common_vars} if common_vars else {}
            params_plot = _fig_from_axes(plot_fn(first_idata, **kwargs))
            if params_plot is None:
                logger.warning(
                    "compare_models: params_plot produced no extractable Figure from %s().",
                    plot_fn.__name__,
                )
            else:
                logger.debug(
                    "compare_models: params_plot generated via %s (%d vars).",
                    plot_fn.__name__, len(common_vars) or len(first_idata.posterior.data_vars),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("compare_models: params_plot failed: %s", exc)

    # 6. Comparison table + plot (multi-model only). az.compare reuses the
    # already-computed ELPDData objects (no second LOO pass) and defaults to LOO
    # since ArviZ 0.21 (no ic kwarg). The table is the core output of a
    # multi-model comparison — a failure here is fatal, not swallowed to a
    # warning (only the plot below stays best-effort).
    comparison_table: Any = None
    compare_plot: Any = None
    if n_models > 1:
        comparison_table = az.compare(loo_results)
        logger.debug("compare_models: comparison table computed.")
        # Visualise the LOO ranking (ELPD ± SE per model) from the table.
        # az.plot_compare is the arviz-plots summary plot; best-effort like the
        # pp-check — a plotting failure must not fail the comparison.
        plot_compare_fn = getattr(az, "plot_compare", None)
        if callable(plot_compare_fn):
            try:
                compare_plot = _fig_from_axes(plot_compare_fn(comparison_table))
                logger.debug("compare_models: compare plot generated.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("compare_models: compare plot failed: %s", exc)

    logger.info(
        "compare_models(): complete — LOO computed, "
        "pp_check=%s, table=%s",
        pp_check_plot is not None,
        comparison_table is not None,
    )

    return ComparisonResult(
        loo=loo_out,
        pp_check_plot=pp_check_plot,
        params_plot=params_plot,
        compare_plot=compare_plot,
        comparison_table=comparison_table,
    )


#: R-style alias.
hbmc = compare_models
