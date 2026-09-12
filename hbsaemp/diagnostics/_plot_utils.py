"""Shared private helpers for diagnostic plotting.

Centralises matplotlib backend hygiene, Figure extraction from ArviZ plot
return values, diagnostic variable selection, and the convergence-plot table
used by :mod:`hbsaemp.diagnostics.convergence` and
:mod:`hbsaemp.diagnostics.comparison`.

This module is **internal** — names are not part of the public ``hbsaemp``
API.
"""
from __future__ import annotations

from typing import Any, Literal

__all__: list[str] = [
    "_CONVERGENCE_PLOTS",
    "_ensure_headless_matplotlib",
    "_fig_from_axes",
    "_diagnostic_var_names",
    "_idata_groups",
]


def _idata_groups(idata: Any) -> tuple[str, ...]:
    """Group names of an ArviZ 1.1 idata (an xarray ``DataTree``).

    ``DataTree.groups`` is a property returning slash-prefixed paths
    (``("/", "/posterior", ...)``); the leading slash is stripped so
    membership tests read ``"posterior" in _idata_groups(idata)``.
    """
    return tuple(str(n).strip("/") for n in idata.groups)

# Bambi's per-observation dim, attached to the response params (mu/p/kappa)
# when they are materialised via predict(kind="response_params"). Stable across
# families (Beta/Gaussian/Binomial). Variables carrying it are deterministic
# functions of the sampled coefficients and scale with n_obs.
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


# Convergence plots: plot type -> (ArviZ function, kwargs, variable policy).
# The policy selects ``var_names`` through ``_diagnostic_var_names``; ``None``
# means the function takes no ``var_names`` (``plot_energy`` reads
# ``sample_stats`` and rejects it).
# * ``rhat`` plots the distribution of rank-normalised R-hat values against
#   the 1.01 reference line — one aggregated panel, so group effects fit.
#   ``plot_forest`` cannot show R-hat in ArviZ 1.1.
# * ``pair`` highlights divergent draws; ArviZ 1.1 enables them through
#   ``visuals`` and rejects the old ``divergences`` keyword.
# * ``energy`` takes no kwargs — it is the energy transition plot only. E-BFMI
#   is checked separately by ``az.diagnose(bfmi_threshold=...)`` in
#   ``convergence.py``, not read off this figure.
_CONVERGENCE_PLOTS: dict[str, tuple[str, dict[str, Any], DiagnosticVarPolicy | None]] = {
    "trace":  ("plot_trace", {}, "scalar_only"),
    "dens":   ("plot_dist", {}, "scalar_only"),
    "acf":    ("plot_autocorr", {}, "scalar_only"),
    "rhat":   ("plot_convergence_dist", {"diagnostics": ["rhat_rank"]}, "scalar_and_group"),
    "neff":   ("plot_ess", {}, "scalar_only"),
    "pair":   ("plot_pair", {"visuals": {"divergence": True}}, "scalar_only"),
    "energy": ("plot_energy", {}, None),
}


def _ensure_headless_matplotlib() -> None:
    """Select the non-interactive ``Agg`` backend when none has been chosen.

    Otherwise the first ArviZ plot lets matplotlib resolve its default
    backend, which lazy-imports a GUI toolkit (Qt5/Tk) and can crash on
    Windows with ``WinError 0xc0000139`` when that runtime is broken. A
    backend already selected — explicitly, via ``MPLBACKEND``, or by Jupyter's
    inline backend — is left untouched.
    """
    import matplotlib

    try:
        chosen = matplotlib.get_backend(auto_select=False)  # matplotlib >= 3.10
    except TypeError:
        # matplotlib < 3.10 has no auto_select; this private accessor performs
        # the same check without triggering backend resolution.
        chosen = matplotlib.rcParams._get_backend_or_none()
    if chosen is None:
        matplotlib.use("Agg")


def _fig_from_axes(obj: Any) -> Any:
    """Extract the matplotlib Figure from an ArviZ plot return value.

    ArviZ 1.1 plot functions return an arviz-plots ``PlotCollection`` (or
    ``PlotMatrix``) whose ``viz`` DataTree stores the Figure as a 0-d object
    variable. Returns ``None`` when no Figure can be located, so callers can
    record the failure instead of storing ``None`` silently.
    """
    import matplotlib.figure as mfig

    viz = getattr(obj, "viz", None)
    if viz is None or "figure" not in viz:
        return None
    candidate = viz["figure"]
    candidate = candidate.item() if hasattr(candidate, "item") else candidate
    return candidate if isinstance(candidate, mfig.Figure) else None
