"""Model comparison and goodness-of-fit checks.

Python equivalent of R hbsaems::hbmc().

v0: compare_models() stub + ComparisonResult dataclass.
v1: LOO/WAIC via arviz; pp_check + marginal posterior plots.
v2+: Bayes Factor via bridge sampling; prior sensitivity loop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from inspect import signature as _signature
from typing import Any

from hbsaemp._logging import get_logger
from hbsaemp.diagnostics._plot_utils import _ensure_headless_matplotlib, _fig_from_axes
from hbsaemp.models._base import BaseModel

logger = get_logger(__name__)
__all__: list[str] = ["ComparisonResult", "compare_models", "hbmc"]


@dataclass
class ComparisonResult:
    """Container for model comparison results.

    Attributes:
        loo: LOO-CV result (``arviz.ELPDData`` for single model;
            ``dict[str, ELPDData]`` for multiple models).
        waic: WAIC result (same structure as *loo*).
        bayes_factor: Bayes Factor (v2+, ``None`` in v1).
        pp_check_plot: ``matplotlib.Figure`` — posterior predictive check
            for the first (or only) model.
        params_plot: ``matplotlib.Figure`` — marginal posterior distributions
            for the first (or only) model.
        prior_sensitivity: Prior sensitivity results (v2+, ``None`` in v1).
        comparison_table: ``pandas.DataFrame`` from ``arviz.compare()``
            (only when *models* is a list with ≥ 2 elements; else ``None``).
    """

    loo: Any = None
    waic: Any = None
    bayes_factor: Any = None
    pp_check_plot: Any = None
    params_plot: Any = None
    prior_sensitivity: Any = None
    comparison_table: Any = None

    def summary(self) -> str:
        """Human-readable comparison summary."""
        if self.loo is None and self.waic is None:
            return "ComparisonResult [not computed — call compare_models()]"

        def _elpd_val(v: Any) -> str:
            val = getattr(v, "elpd_loo", None)
            if val is None:
                val = getattr(v, "elpd_waic", None)
            return f"{val:.2f}" if val is not None else "?"

        def _fmt_elpd(elpd: Any) -> str:
            if elpd is None:
                return "N/A"
            if isinstance(elpd, dict):
                return "{" + ", ".join(f"{k}: {_elpd_val(v)}" for k, v in elpd.items()) + "}"
            return _elpd_val(elpd)

        lines = ["ComparisonResult"]
        lines.append(f"  LOO  : {_fmt_elpd(self.loo)}")
        lines.append(f"  WAIC : {_fmt_elpd(self.waic)}")
        if self.comparison_table is not None:
            lines.append(f"  Comparison table: {len(self.comparison_table)} models")
        lines.append(
            f"  Plots: pp_check={'yes' if self.pp_check_plot else 'no'}, "
            f"params={'yes' if self.params_plot else 'no'}"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        has_table = self.comparison_table is not None
        return (
            f"ComparisonResult("
            f"loo={'yes' if self.loo is not None else 'no'}, "
            f"waic={'yes' if self.waic is not None else 'no'}, "
            f"comparison_table={has_table})"
        )


def compare_models(
    models: BaseModel | list[BaseModel],
    *,
    metrics: list[str] | None = None,
    n_draws_ppc: int = 100,
    run_prior_sensitivity: bool = False,
    sensitivity_vars: list[str] | None = None,
) -> ComparisonResult:
    """Compute goodness-of-fit metrics and compare models.

    Python equivalent of ``hbmc()`` in R hbsaems.

    For each model (or the single model):

    * Computes **LOO** (leave-one-out cross-validation) via ``arviz.loo()``.
    * Computes **WAIC** (widely applicable information criterion) via
      ``arviz.waic()``.

    For the **first** (or only) model:

    * Generates a **posterior predictive check** plot via ``az.plot_ppc()``.
      Calls ``model.predict()`` to populate ``posterior_predictive`` in
      ``idata`` if not already present.
    * Generates a **marginal posterior** plot via ``az.plot_posterior()``.

    When ≥ 2 models are passed, also computes a **comparison table** via
    ``arviz.compare()`` (LOO-based stacking weights).

    The output container shape mirrors the input container:

    * ``compare_models(model)`` → *loo*/*waic* are ``arviz.ELPDData``.
    * ``compare_models([model])`` → *loo*/*waic* are ``dict[str, ELPDData]``
      keyed by ``"model_0"`` (the list form always returns a dict, even
      with a single element).
    * ``compare_models([m0, m1, ...])`` → *loo*/*waic* are
      ``dict[str, ELPDData]`` keyed by ``"model_0"``, ``"model_1"``, etc.

    Args:
        models: A single fitted model or a list of fitted models.
        metrics: Metrics to compute.  Default ``["loo", "waic"]``.
            (Ignored in v1 — both are always computed.)
        n_draws_ppc: Number of posterior predictive draws for the pp-check
            plot.  Default 100.
        run_prior_sensitivity: (v2+) Not implemented in v1.
        sensitivity_vars: (v2+) Not implemented in v1.

    Returns:
        :class:`ComparisonResult` with LOO/WAIC, plots, and comparison table.

    Raises:
        ModelNotFittedError: If any model in *models* has not been fitted.
        ImportError: If ``arviz`` is not installed.
    """
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "compare_models() requires arviz>=0.18. "
            "Install with: pip install 'hbsaemp[bambi]'"
        ) from exc

    _ensure_headless_matplotlib()

    # ── 1. Normalise to list ──────────────────────────────────────────────────
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
    logger.info("compare_models(): %d model(s)", n_models)

    # ── 2. ModelNotFittedError guard for all models ───────────────────────────
    idatas: list[Any] = []
    for i, m in enumerate(model_list):
        r = m.result  # raises ModelNotFittedError if not fitted
        idatas.append(r.idata)

    # ── 3. LOO and WAIC per model ─────────────────────────────────────────────
    # ArviZ ≥0.20 removed ``az.waic``; LOO stays mandatory and WAIC degrades
    # gracefully to ``None`` so downstream summaries (which already handle
    # missing ELPD) keep working.
    loo_results: dict[str, Any] = {}
    waic_results: dict[str, Any] = {}
    waic_fn = getattr(az, "waic", None)

    for i, idata in enumerate(idatas):
        key = f"model_{i}"
        loo_results[key] = az.loo(idata)
        logger.debug("compare_models: LOO for %s computed.", key)
        if waic_fn is None:
            waic_results[key] = None
            continue
        try:
            waic_results[key] = waic_fn(idata)
            logger.debug("compare_models: WAIC for %s computed.", key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("compare_models: WAIC for %s failed: %s", key, exc)
            waic_results[key] = None

    # For bare single-model input: unwrap to bare ELPDData. List input keeps
    # the dict shape even with a single element (see docstring contract).
    if input_was_single:
        loo_out: Any = loo_results.get("model_0")
        waic_out: Any = waic_results.get("model_0")
    else:
        loo_out = loo_results
        waic_out = waic_results

    # ── 4. Posterior predictive check (first model) ───────────────────────────
    # ArviZ ≥0.21 split plotting into ``arviz-plots``: top-level names may be
    # missing or aliased to non-callables. ``callable()`` is the only safe
    # gate; ``hasattr()`` reports True for those aliases.
    pp_check_plot: Any = None
    first_model = model_list[0]
    first_idata = idatas[0]

    plot_ppc_fn = getattr(az, "plot_ppc", None)
    if callable(plot_ppc_fn):
        try:
            # Ensure posterior_predictive is populated (lazy pattern).
            # Use idata.groups() — robust across ArviZ versions vs hasattr().
            if "posterior_predictive" not in first_idata.groups():
                first_model.predict()
            pp_check_plot = _fig_from_axes(
                plot_ppc_fn(first_idata, num_pp_samples=n_draws_ppc)
            )
            logger.debug("compare_models: pp_check plot generated.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("compare_models: pp_check plot failed: %s", exc)

    # ── 5. Marginal posterior plot (first model) ──────────────────────────────
    # ``plot_posterior`` moved to arviz-plots in ≥0.21; fall back to
    # ``plot_density`` then ``plot_trace`` so users still get a marginal
    # density per parameter (best-effort — failure is logged, not raised).
    params_plot: Any = None
    plot_fn = next(
        (
            getattr(az, name)
            for name in ("plot_posterior", "plot_density", "plot_trace")
            if callable(getattr(az, name, None))
        ),
        None,
    )
    if plot_fn is not None:
        try:
            # Drop per-level random effects (Bambi names them with a ``__N``
            # suffix). Otherwise an HBSAE model with N areas produces N+ subplots
            # and breaches matplotlib's ``rcParams["plot.max_subplots"]=40`` cap.
            common_vars = [
                v for v in first_idata.posterior.data_vars
                if not re.search(r"__\d+$", v)
            ]
            kwargs = {"var_names": common_vars} if common_vars else {}
            params_plot = _fig_from_axes(plot_fn(first_idata, **kwargs))
            logger.debug(
                "compare_models: params_plot generated via %s (%d vars).",
                plot_fn.__name__, len(common_vars) or len(first_idata.posterior.data_vars),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("compare_models: params_plot failed: %s", exc)

    # ── 6. Comparison table (multi-model only) ────────────────────────────────
    # ArviZ ≥0.21 dropped the ``ic`` kwarg on ``az.compare`` (LOO is the only
    # supported information criterion now); older versions accept ic="loo".
    # Detect via the live signature so both APIs keep working.
    comparison_table: Any = None
    if n_models > 1:
        valid_idatas = {
            f"model_{i}": idata
            for i, idata in enumerate(idatas)
            if idata is not None
        }
        if len(valid_idatas) >= 2:
            compare_kwargs: dict[str, Any] = {}
            if "ic" in _signature(az.compare).parameters:
                compare_kwargs["ic"] = "loo"
            try:
                comparison_table = az.compare(valid_idatas, **compare_kwargs)
                logger.debug("compare_models: comparison table computed.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("compare_models: comparison table failed: %s", exc)

    logger.info(
        "compare_models(): complete — LOO/WAIC computed, "
        "pp_check=%s, table=%s",
        pp_check_plot is not None,
        comparison_table is not None,
    )

    return ComparisonResult(
        loo=loo_out,
        waic=waic_out,
        pp_check_plot=pp_check_plot,
        params_plot=params_plot,
        comparison_table=comparison_table,
    )


#: R-style alias.
hbmc = compare_models
