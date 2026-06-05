"""Unit tests for diagnostic plot helpers — no Bambi / MCMC.

Uses lightweight stubs so the structural logic in ``_diagnostic_var_names``
and the figure-extraction in ``_fig_from_axes`` can be exercised without a
fitted model.
"""
from __future__ import annotations

import types

import pytest

from hbsaemp.diagnostics._plot_utils import (
    _CONVERGENCE_PLOT_CANDIDATES,
    _NO_VAR_NAMES_PLOTS,
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


# _fig_from_axes — figure extraction across ArviZ return shapes


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
    """ArviZ ≥1.1: figure is viz["figure"].item(), not an attribute (the bug)."""
    mfig = pytest.importorskip("matplotlib.figure")
    fig = mfig.Figure()
    pc = types.SimpleNamespace(viz=_FakeViz({"figure": _FakeDataArray(fig)}))
    assert _fig_from_axes(pc) is fig


def test_fig_from_axes_legacy_attribute_still_works():
    """Older arviz-plots exposed the figure as viz.figure — must still resolve."""
    mfig = pytest.importorskip("matplotlib.figure")
    fig = mfig.Figure()
    pc = types.SimpleNamespace(viz=types.SimpleNamespace(figure=fig))
    assert _fig_from_axes(pc) is fig


def test_fig_from_axes_returns_none_when_absent():
    pytest.importorskip("matplotlib.figure")
    pc = types.SimpleNamespace(viz=_FakeViz({"plot": _FakeDataArray(object())}))
    assert _fig_from_axes(pc) is None


# _idata_groups — robust to ArviZ 1.1 DataTree (.groups attribute) vs legacy method


def test_idata_groups_datatree_attribute_tuple():
    """ArviZ 1.1 DataTree: `.groups` is a property/tuple of slash-prefixed paths.
    `.groups()` would raise TypeError — the helper must read it as an attribute
    and normalise the leading slash."""
    idata = types.SimpleNamespace(groups=("/posterior", "/log_likelihood"))
    names = _idata_groups(idata)
    assert names == ("posterior", "log_likelihood")
    assert isinstance(names, tuple)
    assert "posterior" in names


def test_idata_groups_legacy_method():
    """Older InferenceData exposed `.groups()` as a method — still resolved."""
    idata = types.SimpleNamespace(groups=lambda: ["posterior", "posterior_predictive"])
    names = _idata_groups(idata)
    assert names == ("posterior", "posterior_predictive")


# _CONVERGENCE_PLOT_CANDIDATES — ArviZ 1.1 function names (no MCMC)


def test_dens_candidate_uses_plot_dist():
    """ArviZ 1.1 removed plot_density/plot_posterior/plot_kde; plot_dist is the
    canonical marginal-density plot. A stale name here silently degrades 'dens'
    to a trace plot (the bug this test guards)."""
    assert _CONVERGENCE_PLOT_CANDIDATES["dens"][0][0] == "plot_dist"
    flat = [name for cands in _CONVERGENCE_PLOT_CANDIDATES.values() for name, _ in cands]
    assert "plot_density" not in flat
    assert "plot_kde" not in flat


def test_energy_candidate_present_and_var_names_exempt():
    """Energy/BFMI plot is wired and exempt from the var_names injection
    (plot_energy reads sample_stats, not posterior vars, and rejects var_names)."""
    assert _CONVERGENCE_PLOT_CANDIDATES["energy"] == (("plot_energy", {}),)
    assert "energy" in _NO_VAR_NAMES_PLOTS
