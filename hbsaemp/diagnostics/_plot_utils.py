"""Shared private helpers for diagnostic plotting.

Centralises matplotlib backend hygiene, the ArviZ figure-extraction helper,
and the convergence-plot fallback chains used by
:mod:`hbsaemp.diagnostics.convergence` and
:mod:`hbsaemp.diagnostics.comparison`.

This module is **internal** — names are not part of the public ``hbsaemp``
API.  Keeping the helpers here avoids the previous verbatim duplication of
``_fig_from_axes`` across the two diagnostic modules and removes the
private cross-import ``comparison.py → convergence.py``.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

__all__: list[str] = [
    "_CONVERGENCE_PLOT_CANDIDATES",
    "_NO_VAR_NAMES_PLOTS",
    "_ensure_headless_matplotlib",
    "_fig_from_axes",
    "_diagnostic_var_names",
    "_idata_groups",
]


def _idata_groups(idata: Any) -> tuple[str, ...]:
    """Group names from an idata, robust to the ArviZ 1.1 DataTree change.

    In ArviZ 1.1 ``InferenceData`` maps to xarray's ``DataTree``, where
    ``groups`` is a **property** returning a tuple of paths
    (``("/posterior", "/log_likelihood", ...)``) — not a method. Older
    ``InferenceData`` exposed ``groups()`` as a method. Calling ``.groups()``
    on the new object raises ``TypeError: 'tuple' object is not callable``.

    Resolves either shape and normalises leading slashes
    (``"/posterior"`` → ``"posterior"``) so membership tests work uniformly.
    """
    g = idata.groups
    names = g() if callable(g) else g
    return tuple(str(n).strip("/") for n in names)

# Bambi's per-observation dim, written by include_response_params=True. Stable
# across families (Beta/Gaussian/Binomial). Variables carrying it (mu/p/kappa)
# are deterministic functions of the sampled coefficients and scale with n_obs.
_OBS_DIM = "__obs__"

DiagnosticVarPolicy = Literal["scalar_only", "scalar_and_group"]


def _diagnostic_var_names(
    posterior: Any,
    *,
    policy: DiagnosticVarPolicy,
    obs_dim: str = _OBS_DIM,
) -> list[str]:
    """Select posterior variable names for diagnostics and plotting.

    Always drops per-observation response params (mu/p/kappa — dims include
    ``obs_dim``): they are deterministic functions of the sampled coefficients
    and scale with n_obs, blowing past matplotlib's ``max_subplots=40`` cap and
    polluting ESS/Rhat aggregates with no diagnostic value.

    Args:
        posterior: ``idata.posterior`` (anything exposing ``data_vars.items()``
            yielding ``(name, da)`` where ``da.dims`` is a tuple).
        policy: ``"scalar_only"`` keeps only scalar/model-level params (used for
            plots — always subplot-safe). ``"scalar_and_group"`` also keeps
            group-level random effects (``1|group``, dims include
            ``group__factor_dim``) — used for the convergence summary, since
            those ARE sampled and can fail to converge independently.
        obs_dim: Per-observation dim name (default ``"__obs__"``).

    Returns:
        Variable names in posterior order.
    """
    names: list[str] = []
    for name, da in posterior.data_vars.items():
        extra_dims = set(da.dims) - {"chain", "draw"}
        if obs_dim in extra_dims:
            continue                       # per-obs (mu/p/kappa) — always drop
        if not extra_dims:
            names.append(name)             # scalar — always keep
        elif policy == "scalar_and_group":
            names.append(name)             # group-level — keep (summary only)
    return names


# Convergence-plot fallback chains.  Each plot type lists the candidate
# ArviZ function names in preference order; the first one resolving to a
# callable on the installed ArviZ is used.  Keeps the diagnostic alive
# across the ArviZ → arviz-plots split: ArviZ 1.1 removed ``plot_density`` /
# ``plot_posterior`` / ``plot_kde`` from the namespace — ``plot_dist`` is the
# canonical marginal-density replacement (``plot_forest`` also lost the legacy
# ``r_hat`` kwarg, hence ``plot_rank`` as fallback).
_CONVERGENCE_PLOT_CANDIDATES: dict[str, tuple[tuple[str, dict[str, Any]], ...]] = {
    "trace":  (("plot_trace", {}),),
    "dens":   (("plot_dist", {}), ("plot_trace", {})),
    "acf":    (("plot_autocorr", {}),),
    "rhat":   (("plot_forest", {}), ("plot_rank", {})),
    "neff":   (("plot_ess", {}),),
    "pair":   (("plot_pair", {"divergences": True}),),
    "energy": (("plot_energy", {}),),
}

# Plot types whose ArviZ function reads ``sample_stats`` (not posterior vars)
# and therefore rejects the ``var_names=`` kwarg the convergence loop injects
# for every other candidate.  ``plot_energy`` overlays BFMI itself
# (``show_bfmi=True, threshold=0.3``) — so energy + BFMI is a single call.
_NO_VAR_NAMES_PLOTS: frozenset[str] = frozenset({"energy"})


def _ensure_headless_matplotlib() -> None:
    """Force matplotlib to the non-interactive ``Agg`` backend when none has
    been chosen yet.

    Calling ArviZ/arviz-plots will lazy-import a GUI backend (Qt5/Tk) on
    Windows, which can crash with ``WinError 0xc0000139`` if the Qt
    runtime is broken or Tcl/Tk is misconfigured.  We avoid the GUI path
    entirely for diagnostics; ``force=False`` defers to any explicit
    user choice (e.g. set by the Panel app).
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
    except Exception:  # noqa: BLE001
        # Backend already initialised or matplotlib unavailable — proceed:
        # the worst case is the same crash we were trying to avoid, which
        # is already surfaced by the user's environment.
        pass


def _fig_from_axes(obj: Any) -> Any:
    """Extract the matplotlib Figure from an ArviZ plot return value.

    Handles every shape ArviZ has shipped: a single ``Figure``, an ``Axes``,
    an ndarray of ``Axes``, a ``(fig, axes)`` tuple, or an arviz-plots
    ``PlotCollection`` (ArviZ ≥0.21).  Returns ``None`` when no Figure can
    be located instead of raising — plot capture is best-effort and the
    caller already treats missing plots as a logged warning.
    """
    import matplotlib.figure as mfig
    if isinstance(obj, mfig.Figure):
        return obj
    if isinstance(obj, tuple) and obj:
        return _fig_from_axes(obj[0])
    # Axes-like with a direct ``.figure`` handle (classic ArviZ).
    fig = getattr(obj, "figure", None)
    if isinstance(fig, mfig.Figure):
        return fig
    # arviz-plots PlotCollection: figure lives in ``viz``. Older releases
    # exposed it as an attribute; arviz-plots ≥1.1 stores it as an xarray
    # data variable on a DataTree — a 0-d object array retrieved by item
    # access and unwrapped via ``.item()`` (cf. PlotCollection._display_).
    viz = getattr(obj, "viz", None)
    if viz is not None:
        for attr in ("figure", "fig", "_fig"):
            candidate = getattr(viz, attr, None)
            if isinstance(candidate, mfig.Figure):
                return candidate
        try:
            if "figure" in viz:
                candidate = viz["figure"]
                candidate = candidate.item() if hasattr(candidate, "item") else candidate
                if isinstance(candidate, mfig.Figure):
                    return candidate
        except (TypeError, KeyError):  # viz not a container in some versions
            pass
    # Last resort: ndarray of Axes.
    try:
        axes = np.atleast_1d(obj).ravel()
        first_fig = getattr(axes[0], "figure", None)
        if isinstance(first_fig, mfig.Figure):
            return first_fig
    except Exception:  # noqa: BLE001
        pass
    return None
