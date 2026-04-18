"""Pytest shared fixtures for hbsaemp test suite.

Fixture tiers:
    v0 (no marker):  pure pandas/numpy, no MCMC — always run.
    v1 (@slow):      require Bambi + real MCMC.     pytest -m "not slow"
    v2 (@spatial):   require PyMC + libpysal.       pytest -m "not spatial"
    gui (@gui):      require Panel.                 pytest -m "not gui"
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

SEED: int = 42


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def data_gaussian(rng: np.random.Generator) -> pd.DataFrame:
    """100-row DataFrame mimicking data_fhnorm (Gaussian/Fay-Herriot)."""
    n, g = 100, 10
    group = np.repeat(np.arange(1, g + 1), n // g)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    re = rng.normal(0, 0.5, g)[group - 1]
    y = 1.0 + 0.3 * x1 - 0.2 * x2 + 0.1 * x3 + re + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3,
                         "group": group, "sre": group})


@pytest.fixture(scope="session")
def data_beta(rng: np.random.Generator) -> pd.DataFrame:
    """100-row DataFrame for Beta / BetaLogitNorm — y in (0,1)."""
    n, g = 100, 10
    group = np.repeat(np.arange(1, g + 1), n // g)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    mu = 1 / (1 + np.exp(-(- 0.5 + 0.3 * x1 - 0.2 * x2)))
    y = np.clip(rng.beta(mu * 5, (1 - mu) * 5, n), 1e-6, 1 - 1e-6)
    n_sample = rng.integers(50, 200, n)
    deff = rng.uniform(1.0, 2.5, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3,
                         "group": group, "n": n_sample, "deff": deff})


@pytest.fixture(scope="session")
def data_binomial(rng: np.random.Generator) -> pd.DataFrame:
    """100-row DataFrame for Binomial / BinLogitNorm."""
    n, g = 100, 10
    group = np.repeat(np.arange(1, g + 1), n // g)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    n_trials = rng.integers(20, 100, n)
    p = 1 / (1 + np.exp(-(- 0.5 + 0.3 * x1 - 0.2 * x2)))
    y = rng.binomial(n_trials, p)
    return pd.DataFrame({"y": y, "n": n_trials, "x1": x1, "x2": x2,
                         "x3": x3, "group": group})


@pytest.fixture(scope="session")
def data_lognormal(rng: np.random.Generator) -> pd.DataFrame:
    """100-row DataFrame for Lognormal / LNLN — y > 0."""
    n, g = 100, 10
    group = np.repeat(np.arange(1, g + 1), n // g)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = rng.lognormal(1.0 + 0.2 * x1 - 0.1 * x2, 0.5, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "group": group})


@pytest.fixture(scope="session")
def data_with_missing(data_gaussian: pd.DataFrame) -> pd.DataFrame:
    df = data_gaussian.copy()
    df.loc[[2, 5, 10, 20, 30], "y"] = np.nan
    df.loc[[7, 15, 25], "x1"] = np.nan
    return df


@pytest.fixture(scope="session")
def default_config():
    """A ModelConfig suitable for fast v0/stub tests (no MCMC)."""
    from hbsaemp import ModelConfig
    return ModelConfig(draws=100, tune=50, chains=2, cores=1)
