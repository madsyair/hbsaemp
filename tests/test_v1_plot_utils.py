"""Unit tests for diagnostic plot helpers — no Bambi / MCMC.

Uses lightweight stubs so the structural logic in ``_diagnostic_var_names``
and the figure-extraction in ``_fig_from_axes`` can be exercised without a
fitted model.
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from hbsaemp.diagnostics._plot_utils import (
    _CONVERGENCE_PLOTS,
    _diagnostic_var_names,
    _fig_from_axes,
    _idata_groups,
)


def _fake_posterior() -> types.SimpleNamespace:
    """Mimic a Beta posterior with response params (mu/kappa) materialised."""
    def da(dims: tuple[str, ...]) -> types.SimpleNamespace:
        return types.SimpleNamespace(dims=dims)

    return types.SimpleNamespace(
        data_vars={
            "Intercept":       da(("chain", "draw")),
            "x1":              da(("chain", "draw")),
            "kappa_Intercept": da(("chain", "draw")),
            "1|group":         da(("chain", "draw", "group__factor_dim")),
            "mu":              da(("chain", "draw", "__obs__")),
            "kappa":           da(("chain", "draw", "__obs__")),
        }
    )


def test_diagnostic_var_names_scalar_only():
    names = _diagnostic_var_names(_fake_posterior(), policy="scalar_only")
    assert "Intercept" in names
    assert "x1" in names
    assert "kappa_Intercept" in names
    # per-obs and group-level dropped
    assert "mu" not in names
    assert "kappa" not in names
    assert "1|group" not in names


def test_diagnostic_var_names_scalar_and_group():
    names = _diagnostic_var_names(_fake_posterior(), policy="scalar_and_group")
    assert "Intercept" in names
    # per-obs still dropped
    assert "mu" not in names
    assert "kappa" not in names
    # group-level retained for the convergence summary
    assert "1|group" in names


# _fig_from_axes — Figure extraction from ArviZ 1.1 PlotCollection


class _FakeDataArray:
    """0-d object DataArray stand-in: arviz-plots wraps the Figure like this."""

    def __init__(self, obj: object) -> None:
        self._obj = obj

    def item(self) -> object:
        return self._obj


class _FakeViz:
    """DataTree/Dataset stand-in supporting ``"figure" in viz`` and ``viz["figure"]``."""

    def __init__(self, mapping: dict) -> None:
        self._m = mapping

    def __contains__(self, key: str) -> bool:
        return key in self._m

    def __getitem__(self, key: str):
        return self._m[key]


def test_fig_from_axes_extracts_plotcollection_figure():
    """ArviZ ≥1.1: figure is viz["figure"].item(), not an attribute."""
    mfig = pytest.importorskip("matplotlib.figure")
    fig = mfig.Figure()
    pc = types.SimpleNamespace(viz=_FakeViz({"figure": _FakeDataArray(fig)}))
    assert _fig_from_axes(pc) is fig


def test_fig_from_axes_returns_none_when_absent():
    pytest.importorskip("matplotlib.figure")
    pc = types.SimpleNamespace(viz=_FakeViz({"plot": _FakeDataArray(object())}))
    assert _fig_from_axes(pc) is None


def test_fig_from_axes_returns_none_without_viz():
    pytest.importorskip("matplotlib.figure")
    assert _fig_from_axes(object()) is None


# _idata_groups — ArviZ 1.1 DataTree


def test_idata_groups_datatree_attribute_tuple():
    """ArviZ 1.1 DataTree: `.groups` is a property/tuple of slash-prefixed paths.
    `.groups()` would raise TypeError — the helper must read it as an attribute
    and normalise the leading slash."""
    idata = types.SimpleNamespace(groups=("/posterior", "/log_likelihood"))
    names = _idata_groups(idata)
    assert names == ("posterior", "log_likelihood")
    assert isinstance(names, tuple)
    assert "posterior" in names


# _CONVERGENCE_PLOTS — ArviZ 1.1 function per plot type (no MCMC)


def test_every_convergence_plot_function_exists():
    az = pytest.importorskip("arviz")
    for ptype, (name, _, _) in _CONVERGENCE_PLOTS.items():
        assert callable(getattr(az, name, None)), f"{ptype}: arviz.{name} is missing"


def test_dens_uses_plot_dist():
    """ArviZ 1.1 removed plot_density/plot_posterior/plot_kde; plot_dist is the
    canonical marginal-density plot."""
    assert _CONVERGENCE_PLOTS["dens"][0] == "plot_dist"


def test_rhat_plots_rhat_values():
    """plot_forest cannot show R-hat in ArviZ 1.1 — the rhat plot must be the
    R-hat distribution, over scalar and group-level parameters."""
    name, kwargs, policy = _CONVERGENCE_PLOTS["rhat"]
    assert name == "plot_convergence_dist"
    assert kwargs == {"diagnostics": ["rhat_rank"]}
    assert policy == "scalar_and_group"


def test_pair_highlights_divergences_via_visuals():
    """ArviZ 1.1 rejects the old ``divergences=True`` keyword."""
    name, kwargs, _ = _CONVERGENCE_PLOTS["pair"]
    assert name == "plot_pair"
    assert kwargs == {"visuals": {"divergence": True}}


def test_energy_takes_no_var_names():
    """plot_energy reads sample_stats and rejects var_names."""
    assert _CONVERGENCE_PLOTS["energy"] == ("plot_energy", {}, None)


# _ensure_headless_matplotlib — the backend is process-global state, so each
# case runs in a fresh interpreter.


def _backend_after_helper(setup: str) -> str:
    pytest.importorskip("matplotlib")
    code = (
        "import matplotlib\n"
        f"{setup}\n"
        "from hbsaemp.diagnostics._plot_utils import _ensure_headless_matplotlib\n"
        "_ensure_headless_matplotlib()\n"
        "print(matplotlib.get_backend())\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True, env=env,
        cwd=Path(__file__).resolve().parents[1],
    )
    return out.stdout.strip().lower()


def test_headless_backend_selected_when_none_chosen():
    assert _backend_after_helper("") == "agg"


def test_user_backend_is_left_untouched():
    assert _backend_after_helper('matplotlib.use("svg")') == "svg"
