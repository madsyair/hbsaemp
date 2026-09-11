"""Model comparison (alias `hbmc`): PSIS-LOO-CV, Bayes factors and prior sensitivity via ArviZ.

Also builds a pp_check and marginal posterior plots for the first model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from hbsaemp._logging import get_logger
from hbsaemp.diagnostics._plot_utils import (
    _diagnostic_var_names,
    _ensure_headless_matplotlib,
    _fig_from_axes,
    _idata_groups,
)
from hbsaemp.models._base import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)
__all__: list[str] = ["ComparisonResult", "compare_models", "hbmc"]

# Diagnoses ``arviz.psense_summary`` assigns to sensitive parameters, matched as
# substrings (ArviZ prefixes them with "potential "); any other value in its
# ``diagnosis`` column means "not sensitive".
_PSENSE_DIAGNOSES: tuple[str, ...] = ("prior-data conflict", "strong prior / weak likelihood")


def _per_model(value: Any, fmt: Callable[[Any], str]) -> str:
    """Format a per-model result that is either bare or keyed ``model_<i>``."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {fmt(v)}" for k, v in value.items()) + "}"
    return fmt(value)


def _elpd_val(elpd: Any) -> str:
    # ArviZ 1.1 renamed ELPDData's elpd accessor to the metric-agnostic
    # `elpd` (was `elpd_loo`). Read the canonical name first, falling
    # back to the legacy one for older ArviZ. `?` only when none are
    # present (genuinely missing), never just because the name moved.
    for name in ("elpd", "elpd_loo"):
        val = getattr(elpd, name, None)
        if val is not None:
            return f"{float(val):.2f}"
    return "?"


def _pareto_k_status(elpd: Any) -> str:
    """Count observations whose Pareto k exceeds ArviZ's ``good_k``."""
    pareto_k = getattr(elpd, "pareto_k", None)
    good_k = getattr(elpd, "good_k", None)
    if pareto_k is None or good_k is None:
        return "N/A"
    k = np.asarray(pareto_k)
    n_bad = int((k > good_k).sum())
    return f"{n_bad}/{k.size} obs > {good_k:.2f}" + (" (LOO unreliable)" if n_bad else "")


def _bf10_str(table: pd.DataFrame) -> str:
    if table.empty:
        return "no coefficients"
    return ", ".join(f"{name}={bf:.3g}" for name, bf in table["BF10"].items())


def _psense_str(table: pd.DataFrame) -> str:
    # Counts only: ArviZ marks unflagged rows with a check-mark glyph that a
    # cp1252 console cannot encode when the output is piped.
    diagnosis = table["diagnosis"].astype(str)
    counts = {d: int(diagnosis.str.contains(d, regex=False).sum()) for d in _PSENSE_DIAGNOSES}
    parts = [f"{n} {d}" for d, n in counts.items() if n]
    return ", ".join(parts) if parts else f"no sensitivity issues ({len(table)} params)"


@dataclass
class ComparisonResult:
    """Container for model comparison results.

    ``loo``, ``bayes_factor`` and ``prior_sensitivity`` mirror the input
    container: a bare model gives a bare value, a list gives a ``dict`` keyed
    ``"model_0"``, ``"model_1"``, ...

    Attributes:
        loo: PSIS-LOO-CV result (``arviz.ELPDData``); ``None`` unless
            ``"loo"`` is in *metrics*.
        bayes_factor: ``pandas.DataFrame`` indexed by fixed-effect coefficient
            with ``BF10`` / ``BF01`` columns (Savage-Dickey, H0: coefficient
            = 0); ``None`` unless ``"bf"`` is in *metrics*.
        pp_check_plot: ``matplotlib.Figure`` — posterior predictive check
            for the first (or only) model.
        params_plot: ``matplotlib.Figure`` — marginal posterior distributions
            for the first (or only) model.
        compare_plot: ``matplotlib.Figure`` — model-comparison summary
            (``arviz.plot_compare``); only when *models* is a list with ≥ 2
            elements, else ``None``.
        prior_sensitivity: ``pandas.DataFrame`` from ``arviz.psense_summary``
            with ``prior`` / ``likelihood`` sensitivity and a ``diagnosis``
            per parameter; ``None`` unless *run_prior_sensitivity*.
        comparison_table: ``pandas.DataFrame`` from ``arviz.compare()``
            (only when *models* is a list with ≥ 2 elements; else ``None``).
        plot_errors: Dict keyed by plot name (``"pp_check"``, ``"params"``,
            ``"compare"``) → error message, for plots that failed to render.
    """

    loo: Any = None
    bayes_factor: Any = None
    pp_check_plot: Any = None
    params_plot: Any = None
    compare_plot: Any = None
    prior_sensitivity: Any = None
    comparison_table: Any = None
    plot_errors: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable comparison summary."""
        if self.loo is None and self.bayes_factor is None and self.prior_sensitivity is None:
            return "ComparisonResult [not computed — call compare_models()]"

        lines = ["ComparisonResult"]
        if self.loo is not None:
            lines.append(f"  LOO  : {_per_model(self.loo, _elpd_val)}")
            lines.append(f"  Pareto k: {_per_model(self.loo, _pareto_k_status)}")
        if self.comparison_table is not None:
            lines.append(f"  Comparison table: {len(self.comparison_table)} models")
        if self.bayes_factor is not None:
            lines.append(f"  BF10 : {_per_model(self.bayes_factor, _bf10_str)}")
        if self.prior_sensitivity is not None:
            lines.append(
                f"  Prior sensitivity: {_per_model(self.prior_sensitivity, _psense_str)}"
            )
        lines.append(
            f"  Plots: pp_check={'yes' if self.pp_check_plot else 'no'}, "
            f"params={'yes' if self.params_plot else 'no'}, "
            f"compare={'yes' if self.compare_plot else 'no'}"
        )
        if self.plot_errors:
            lines.append(f"  Plot errors: {len(self.plot_errors)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        has_table = self.comparison_table is not None
        return (
            f"ComparisonResult("
            f"loo={'yes' if self.loo is not None else 'no'}, "
            f"comparison_table={has_table})"
        )


# compare_models() validates against this set and raises on anything else
# rather than accepting an argument it would ignore.
_SUPPORTED_METRICS: frozenset[str] = frozenset({"loo", "bf"})

# Prior draws for the Savage-Dickey density estimate at the reference value.
_BF_PRIOR_DRAWS: int = 4000


def _validate_compare_args(
    *,
    metrics: list[str] | str | None,
    n_draws_ppc: int,
    run_prior_sensitivity: bool,
    sensitivity_vars: list[str] | None,
) -> tuple[str, ...]:
    """Normalise and validate ``compare_models`` arguments (fail-fast).

    Returns the lower-cased metric tuple. Unknown metrics raise instead of
    being silently ignored, ``n_draws_ppc`` must be a positive integer, and
    ``sensitivity_vars`` is only meaningful with ``run_prior_sensitivity=True``.
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
            f"Supported: {sorted(_SUPPORTED_METRICS)}."
        )

    unsupported = sorted(set(metrics_out) - _SUPPORTED_METRICS)
    if unsupported:
        raise NotImplementedError(
            f"compare_models() supports metrics {sorted(_SUPPORTED_METRICS)}. "
            f"Unsupported metric(s): {unsupported}. WAIC was removed from ArviZ "
            "(>=0.20) in favour of PSIS-LOO-CV; other metrics must be added "
            "explicitly in a future version, not silently ignored."
        )

    if isinstance(n_draws_ppc, bool) or not isinstance(n_draws_ppc, int) or n_draws_ppc <= 0:
        raise ValueError(
            f"compare_models(n_draws_ppc=...) must be a positive integer; got {n_draws_ppc!r}."
        )

    if sensitivity_vars is not None and not run_prior_sensitivity:
        raise ValueError(
            "compare_models(sensitivity_vars=...) requires run_prior_sensitivity=True."
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


def _check_same_observations(idatas: list[Any]) -> None:
    """Raise when the models were not fitted to the same observations.

    ELPD sums over observations, so models fitted to different responses are
    not comparable. ``az.compare`` only checks the observation count, which
    two different datasets of equal length pass.
    """
    observed: list[np.ndarray] = []
    for i, idata in enumerate(idatas):
        if "observed_data" not in _idata_groups(idata):
            raise ValueError(
                f"model_{i} has no 'observed_data' group, so compare_models() "
                "cannot confirm the models share the same observations."
            )
        obs = idata.observed_data
        observed.append(np.asarray(obs[next(iter(obs.data_vars))].values))

    ref = observed[0]
    for i, y in enumerate(observed[1:], start=1):
        if y.shape != ref.shape or not np.allclose(y, ref, equal_nan=True):
            raise ValueError(
                f"model_{i} was fitted to different observations than model_0. "
                "compare_models() only compares models fitted to the same "
                "response and rows."
            )


def _common_terms(model: BaseModel) -> list[str]:
    """Fixed-effect coefficient names of the mean component (intercept excluded)."""
    bmodel = model.result.backend_model
    # Bambi keeps the mean-parameter terms on the component of the likelihood's
    # parent parameter (``mu`` / ``p``); ``common_terms`` excludes the intercept.
    component = bmodel.components[bmodel.family.likelihood.parent]
    return list(component.common_terms)


def _bayes_factor_table(model: BaseModel, az: Any) -> pd.DataFrame:
    """Savage-Dickey Bayes factors (H0: coefficient = 0) per fixed effect."""
    terms = _common_terms(model)
    if not terms:
        return pd.DataFrame(columns=["BF10", "BF01"], dtype=float)
    prior = model.prior_predictive_idata(draws=_BF_PRIOR_DRAWS)
    # The prior group is attached to a copy only: result.idata stays as fitted.
    dt = model.result.idata.copy()
    dt["prior"] = prior["prior"]
    bf = az.bayes_factor(dt, var_names=terms, ref_vals=0)
    return pd.DataFrame.from_dict(bf, orient="index")[["BF10", "BF01"]].astype(float)


def _log_prior_idata(model: BaseModel) -> Any:
    """Copy of ``result.idata`` carrying a ``log_prior`` group.

    Bambi drops the non-centred ``<term>_offset`` variables from the posterior
    (``omit_offsets=True``), yet PyMC's ``compute_log_prior`` needs a value for
    every free variable, so they are rebuilt exactly as ``term / term_sigma``
    on the copy. Only the non-offset priors enter ``log_prior``: the offsets'
    standard-normal priors belong to the hierarchy rather than to the prior
    being assessed — the same split as the brms ``lprior`` used by R's
    priorsense.
    """
    import pymc as pm

    pm_model = model.result.backend_model.backend.model
    free = [rv.name for rv in pm_model.free_RVs]
    offsets = [name for name in free if name.endswith("_offset")]

    dt = model.result.idata.copy()
    posterior = dt["posterior"].to_dataset()
    for offset in offsets:
        term = offset.removesuffix("_offset")
        posterior[offset] = posterior[term] / posterior[f"{term}_sigma"]
    dt["posterior"] = posterior
    pm.stats.compute_log_prior(
        dt,
        var_names=[name for name in free if name not in offsets],
        model=pm_model,
        extend_inferencedata=True,
        progressbar=False,
    )
    return dt


def _prior_sensitivity_table(
    model: BaseModel, var_names: list[str] | None, az: Any
) -> pd.DataFrame:
    """Power-scaling prior / likelihood sensitivity per parameter."""
    # Variables are selected on the stored posterior, which excludes the
    # offsets rebuilt on the copy.
    if var_names is None:
        var_names = _diagnostic_var_names(
            model.result.idata.posterior, policy="scalar_and_group"
        )
    return az.psense_summary(_log_prior_idata(model), var_names=var_names)


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

    For each model, depending on *metrics* and *run_prior_sensitivity*:

    * ``"loo"`` (default) — **PSIS-LOO-CV** via ``arviz.loo()``. Observations
      whose Pareto *k* exceeds ArviZ's ``good_k`` make the estimate
      unreliable; :meth:`ComparisonResult.summary` reports their count.
    * ``"bf"`` — **Savage-Dickey Bayes factors** via ``arviz.bayes_factor()``
      for every fixed-effect coefficient (intercept excluded), testing
      H0: coefficient = 0 from the prior and posterior densities at zero.
      Prior draws come from ``model.prior_predictive_idata()``. This is a
      per-coefficient test within one model, not the model-versus-model
      Bayes factor of R's ``hbmc()`` (bridge sampling), and it depends on the
      coefficient's prior: a vague prior favours H0.
    * ``run_prior_sensitivity=True`` — **power-scaling prior / likelihood
      sensitivity** via ``arviz.psense_summary()`` (the method of R's
      priorsense). Sensitivity above 0.05 is flagged in its ``diagnosis``
      column.

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
      — falling back to ``plot_trace``.

    Plots are best-effort: a failure is recorded in ``plot_errors``.

    When ≥ 2 models are passed they must be fitted to the same observations
    (checked against ``observed_data``); ``"loo"`` then also yields a
    **comparison table** via ``arviz.compare()`` (LOO-based stacking weights)
    and a **comparison plot** via ``arviz.plot_compare()``.

    ``loo``, ``bayes_factor`` and ``prior_sensitivity`` mirror the input
    container:

    * ``compare_models(model)`` → a bare value (e.g. ``arviz.ELPDData``).
    * ``compare_models([model])`` → a ``dict`` keyed by ``"model_0"`` (the
      list form always returns a dict, even with a single element).
    * ``compare_models([m0, m1, ...])`` → a ``dict`` keyed by ``"model_0"``,
      ``"model_1"``, etc.

    Args:
        models: A single fitted model or a list of fitted models.
        metrics: Metrics to compute: ``"loo"`` (default) and/or ``"bf"``.
            Any other metric — including ``"waic"``, removed from ArviZ ≥0.20 in
            favour of PSIS-LOO-CV — raises ``NotImplementedError`` rather than
            being silently ignored.  An empty list raises ``ValueError``.
        n_draws_ppc: Number of posterior predictive draws for the pp-check
            plot.  Must be a positive integer (default 100).
        run_prior_sensitivity: Compute power-scaling prior sensitivity.
        sensitivity_vars: Posterior variables for the sensitivity table.
            Default: scalar and group-level parameters.  Requires
            ``run_prior_sensitivity=True``.

    Returns:
        :class:`ComparisonResult` with the requested metrics, plots, and
        comparison table.

    Raises:
        ValueError: If *models* is an empty list, ``metrics`` is empty,
            ``n_draws_ppc`` is not a positive integer, ``sensitivity_vars`` is
            given without ``run_prior_sensitivity``, the models were fitted to
            different observations, or a model's idata lacks the
            ``log_likelihood`` group required by ``az.loo()``.
        NotImplementedError: If an unsupported metric is requested.
        ModelNotFittedError: If any model in *models* has not been fitted.
        ImportError: If ``arviz`` is not installed.
    """
    # Fail fast on unsupported arguments before importing arviz — a typo like
    # metrics=["wiac"] should error clearly even in a minimal install.
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
    # forms — only the bare form unwraps to a flat value so the output
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
    keys = [f"model_{i}" for i in range(n_models)]
    logger.info(
        "compare_models(): %d model(s), metrics=%r, prior_sensitivity=%s",
        n_models, metrics, run_prior_sensitivity,
    )

    def _shape(per_model: dict[str, Any]) -> Any:
        # Bare-model input unwraps to a bare value; list input keeps the dict.
        if not per_model:
            return None
        return per_model["model_0"] if input_was_single else per_model

    # 2. ModelNotFittedError guard for all models
    idatas: list[Any] = [m.result.idata for m in model_list]

    # 3. ELPD sums over observations: models fitted to different data are
    # not comparable, whatever their row count.
    if n_models > 1:
        _check_same_observations(idatas)

    # 4. PSIS-LOO-CV per model. WAIC is intentionally not computed: ArviZ
    # removed az.waic in ≥0.20 (lower-bound is now ≥1.1) in favour of LOO.
    loo_results: dict[str, Any] = {}
    if "loo" in metrics:
        for key, idata in zip(keys, idatas, strict=True):
            _require_log_likelihood(idata, model_name=key)
            loo_results[key] = az.loo(idata)
            logger.debug("compare_models: LOO for %s computed.", key)

    # 5. Savage-Dickey Bayes factors per model
    bf_results: dict[str, pd.DataFrame] = {}
    if "bf" in metrics:
        for key, m in zip(keys, model_list, strict=True):
            bf_results[key] = _bayes_factor_table(m, az)
            logger.debug("compare_models: Bayes factors for %s computed.", key)

    # 6. Power-scaling prior sensitivity per model
    psense_results: dict[str, pd.DataFrame] = {}
    if run_prior_sensitivity:
        for key, m in zip(keys, model_list, strict=True):
            psense_results[key] = _prior_sensitivity_table(m, sensitivity_vars, az)
            logger.debug("compare_models: prior sensitivity for %s computed.", key)

    plot_errors: dict[str, str] = {}

    def _plot_failed(name: str, message: str) -> None:
        plot_errors[name] = message
        logger.warning("compare_models: %s plot failed: %s", name, message)

    # 7. Posterior predictive check (first model)
    # az.plot_ppc was removed in ArviZ 1.1; PPC plotting moved to the separate
    # ``arviz_plots`` package. ``plot_ppc_dist`` is the closest analogue (KDE/
    # dist overlay of observed vs Y_rep). Lazy + guarded import: arviz_plots is a
    # declared dep (pyproject [bambi]) and also transitive via bambi+arviz>=1.1,
    # but the callable() gate still protects minimal/partial installs.
    pp_check_plot: Any = None
    first_model = model_list[0]
    first_idata = idatas[0]

    try:
        import arviz_plots as azp
    except ImportError:
        azp = None
    plot_ppc_fn = getattr(azp, "plot_ppc_dist", None) if azp is not None else None
    if not callable(plot_ppc_fn):
        _plot_failed("pp_check", "arviz_plots.plot_ppc_dist is not available.")
    else:
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
            if pp_check_plot is None:
                _plot_failed("pp_check", "could not extract a Figure from plot_ppc_dist().")
            else:
                logger.debug("compare_models: pp_check plot generated.")
        except Exception as exc:  # noqa: BLE001
            _plot_failed("pp_check", str(exc))

    # 8. Marginal posterior plot (first model)
    # ArviZ 1.1 removed ``plot_posterior`` / ``plot_density`` from the namespace;
    # ``plot_dist`` is the canonical marginal-density plot, with ``plot_trace``
    # as a last-resort fallback.
    params_plot: Any = None
    plot_fn = next(
        (
            getattr(az, name)
            for name in ("plot_dist", "plot_trace")
            if callable(getattr(az, name, None))
        ),
        None,
    )
    if plot_fn is None:
        _plot_failed("params", "no compatible ArviZ marginal-posterior plot function.")
    else:
        try:
            # Keep only scalar params. Per-area random effects (`1|group`) and
            # per-obs params (mu/kappa) carry extra dims and would explode the
            # subplot count past matplotlib's `rcParams["plot.max_subplots"]=40`
            # cap. `scalar_only` is the shared helper used by convergence plots.
            common_vars = _diagnostic_var_names(first_idata.posterior, policy="scalar_only")
            kwargs = {"var_names": common_vars} if common_vars else {}
            params_plot = _fig_from_axes(plot_fn(first_idata, **kwargs))
            if params_plot is None:
                _plot_failed(
                    "params", f"could not extract a Figure from {plot_fn.__name__}()."
                )
            else:
                logger.debug(
                    "compare_models: params_plot generated via %s (%d vars).",
                    plot_fn.__name__, len(common_vars) or len(first_idata.posterior.data_vars),
                )
        except Exception as exc:  # noqa: BLE001
            _plot_failed("params", str(exc))

    # 9. Comparison table + plot (multi-model LOO only). az.compare reuses the
    # already-computed ELPDData objects (no second LOO pass) and defaults to LOO
    # since ArviZ 0.21 (no ic kwarg). The table is the core output of a
    # multi-model comparison — a failure here is fatal, not swallowed to a
    # warning (only the plot below stays best-effort).
    comparison_table: Any = None
    compare_plot: Any = None
    if n_models > 1 and loo_results:
        comparison_table = az.compare(loo_results)
        logger.debug("compare_models: comparison table computed.")
        # Visualise the LOO ranking (ELPD ± SE per model) from the table.
        plot_compare_fn = getattr(az, "plot_compare", None)
        if not callable(plot_compare_fn):
            _plot_failed("compare", "arviz.plot_compare is not available.")
        else:
            try:
                compare_plot = _fig_from_axes(plot_compare_fn(comparison_table))
                if compare_plot is None:
                    _plot_failed("compare", "could not extract a Figure from plot_compare().")
                else:
                    logger.debug("compare_models: compare plot generated.")
            except Exception as exc:  # noqa: BLE001
                _plot_failed("compare", str(exc))

    logger.info(
        "compare_models(): complete — loo=%s, bf=%s, prior_sensitivity=%s, "
        "table=%s, plot_errors=%d",
        bool(loo_results), bool(bf_results), bool(psense_results),
        comparison_table is not None, len(plot_errors),
    )

    return ComparisonResult(
        loo=_shape(loo_results),
        bayes_factor=_shape(bf_results),
        pp_check_plot=pp_check_plot,
        params_plot=params_plot,
        compare_plot=compare_plot,
        prior_sensitivity=_shape(psense_results),
        comparison_table=comparison_table,
        plot_errors=plot_errors,
    )


#: R-style alias.
hbmc = compare_models
