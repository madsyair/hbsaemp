"""Unit tests for check_convergence() thresholds and sampler checks — no MCMC.

Synthetic draws built with ``arviz.from_dict`` stand in for a fitted model, so
the R-hat / ESS / divergence / tree-depth / E-BFMI logic runs in milliseconds
and each failure mode can be triggered deterministically.
"""
from __future__ import annotations

import types
import warnings

import numpy as np
import pytest

import hbsaemp as hb

az = pytest.importorskip("arviz")


def _fitted(
    posterior: dict[str, np.ndarray],
    sample_stats: dict[str, np.ndarray] | None = None,
) -> types.SimpleNamespace:
    """Duck-typed stand-in for a fitted model exposing ``.result.idata``."""
    groups = {"posterior": posterior}
    if sample_stats is not None:
        groups["sample_stats"] = sample_stats
    return types.SimpleNamespace(
        result=types.SimpleNamespace(idata=az.from_dict(groups), family="gaussian")
    )


def _clean_stats(rng: np.random.Generator, chains: int, draws: int) -> dict[str, np.ndarray]:
    """Sampler stats of a healthy NUTS run: no divergences, iid energy."""
    return {
        "diverging": np.zeros((chains, draws), dtype=bool),
        "reached_max_treedepth": np.zeros((chains, draws), dtype=bool),
        "energy": rng.normal(size=(chains, draws)),
    }


def _run(model: types.SimpleNamespace, **kwargs) -> tuple[hb.ConvergenceResult, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = hb.check_convergence(model, plot_types=[], **kwargs)
    messages = [str(w.message) for w in caught if issubclass(w.category, hb.ConvergenceWarning)]
    return result, messages


def test_rhat_breach_not_hidden_by_rounding():
    """R-hat ≈ 1.0147 must warn — ArviZ's display rounding shows it as "1.01"."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 1000))
    a[3] += 0.3
    result, messages = _run(_fitted({"a": a}, _clean_stats(rng, 4, 1000)))
    assert result.rhat_ess["r_hat"].dtype.kind == "f"
    assert 1.01 < result.rhat_ess.loc["a", "r_hat"] < 1.02
    assert any(m.startswith("Rhat > 1.01") for m in messages)


def test_ess_threshold_scales_with_chains():
    rng = np.random.default_rng(1)
    result, _ = _run(_fitted({"a": rng.normal(size=(2, 500))}, _clean_stats(rng, 2, 500)))
    assert result.ess_threshold == 200


def test_low_ess_warns_against_chain_scaled_floor():
    """Strongly autocorrelated chains fall under the 100 × n_chains floor."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(4, 500)).cumsum(axis=1)  # random walk
    _, messages = _run(_fitted({"a": a}, _clean_stats(rng, 4, 500)))
    assert any(m.startswith("Bulk ESS < 400") for m in messages)


def test_sampler_problems_warn():
    rng = np.random.default_rng(3)
    stats = _clean_stats(rng, 4, 1000)
    stats["diverging"][0, :5] = True
    stats["reached_max_treedepth"][1, :3] = True
    # A random walk moves little between draws relative to its spread → low E-BFMI.
    stats["energy"] = rng.normal(size=(4, 1000)).cumsum(axis=1)
    result, messages = _run(_fitted({"a": rng.normal(size=(4, 1000))}, stats))

    assert any("diverged" in m for m in messages)
    assert any("maximum tree depth" in m for m in messages)
    assert any(m.startswith("E-BFMI < 0.3") for m in messages)
    assert result.diagnose["divergent"]["n_divergent"] == 5
    assert result.diagnose["treedepth"]["n_max"] == 3
    assert "WARNING" in result.summary()


def test_healthy_run_has_no_warnings():
    rng = np.random.default_rng(4)
    result, messages = _run(
        _fitted({"a": rng.normal(size=(4, 1000))}, _clean_stats(rng, 4, 1000))
    )
    assert messages == []
    assert result.summary().endswith("Status       : OK")


def test_diag_tests_gates_sampler_warnings():
    """Excluded checks stay silent but are still computed and stored."""
    rng = np.random.default_rng(5)
    stats = _clean_stats(rng, 4, 1000)
    stats["diverging"][0, 0] = True
    result, messages = _run(
        _fitted({"a": rng.normal(size=(4, 1000))}, stats), diag_tests=["rhat", "ess"]
    )
    assert not any("diverged" in m for m in messages)
    assert result.diagnose["divergent"]["n_divergent"] == 1


def test_missing_sample_stats_skips_sampler_checks():
    """Without NUTS stats R-hat / ESS still run; sampler entries are absent."""
    rng = np.random.default_rng(6)
    result, messages = _run(_fitted({"a": rng.normal(size=(4, 1000))}))
    assert messages == []
    assert "divergent" not in result.diagnose
    assert "Divergences  : N/A" in result.summary()
