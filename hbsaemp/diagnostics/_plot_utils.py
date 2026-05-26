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

from typing import Any

import numpy as np

__all__: list[str] = [
    "_CONVERGENCE_PLOT_CANDIDATES",
    "_ensure_headless_matplotlib",
    "_fig_from_axes",
]


# Convergence-plot fallback chains.  Each plot type lists the candidate
# ArviZ function names in preference order; the first one resolving to a
# callable on the installed ArviZ is used.  Keeps the diagnostic alive
# across the ArviZ → arviz-plots split (≥0.21 dropped ``plot_posterior``
# from the top-level namespace; ``plot_forest`` lost the ``r_hat`` kwarg).
_CONVERGENCE_PLOT_CANDIDATES: dict[str, tuple[tuple[str, dict[str, Any]], ...]] = {
    "trace": (("plot_trace",    {}),),
    "dens":  (("plot_density",  {}), ("plot_kde", {}), ("plot_trace", {})),
    "acf":   (("plot_autocorr", {}),),
    "rhat":  (("plot_forest",   {}), ("plot_rank", {})),
    "neff":  (("plot_ess",      {}),),
    "pair":  (("plot_pair",     {"divergences": True}),),
}


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
    # arviz-plots PlotCollection: figure lives one level deeper, but the
    # public attribute name is unstable across releases — probe a few.
    viz = getattr(obj, "viz", None)
    if viz is not None:
        for attr in ("figure", "fig", "_fig"):
            candidate = getattr(viz, attr, None)
            if isinstance(candidate, mfig.Figure):
                return candidate
    # Last resort: ndarray of Axes.
    try:
        axes = np.atleast_1d(obj).ravel()
        first_fig = getattr(axes[0], "figure", None)
        if isinstance(first_fig, mfig.Figure):
            return first_fig
    except Exception:  # noqa: BLE001
        pass
    return None
