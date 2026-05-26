"""Built-in datasets matching R hbsaems package.

v0: load_dataset() stub — raises NotImplementedError.
v1: Returns real DataFrames generated from reproducible seeds.

Available datasets (same names as R hbsaems):
    data_fhnorm         — Gaussian/Fay-Herriot normal
    data_betalogitnorm  — Beta logit-normal
    data_binlogitnorm   — Binomial logit-normal
    data_lnln           — Lognormal-lognormal

All datasets: m=30 areas, seed=42, area-level (one row per area).
Column names match the R hbsaems package.
"""
from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd

__all__: list[str] = ["load_dataset", "AVAILABLE_DATASETS"]

DatasetName = Literal[
    "data_fhnorm",
    "data_betalogitnorm",
    "data_binlogitnorm",
    "data_lnln",
]

AVAILABLE_DATASETS: list[str] = [
    "data_fhnorm",
    "data_betalogitnorm",
    "data_binlogitnorm",
    "data_lnln",
]

_SEED: int = 42
_N_AREAS: int = 30


def load_dataset(name: DatasetName) -> pd.DataFrame:
    """Load a built-in hbsaemp dataset.

    Returns a :class:`pandas.DataFrame` with one row per small area, column
    names matching the R ``hbsaems`` package.  All datasets are generated
    from a fixed seed (42) and are reproducible across runs.

    Args:
        name: Dataset identifier. One of :data:`AVAILABLE_DATASETS`.

    Returns:
        A :class:`pandas.DataFrame` with :data:`_N_AREAS` rows.

    Raises:
        ValueError: If *name* is not in :data:`AVAILABLE_DATASETS`.

    Examples:
        >>> df = load_dataset("data_fhnorm")
        >>> df.shape
        (30, 9)
        >>> list(df.columns)
        ['y', 'D', 'x1', 'x2', 'x3', 'theta_true', 'u', 'group', 'sre']
    """
    if name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}. "
            f"Available: {AVAILABLE_DATASETS}"
        )
    _generators: dict[str, object] = {
        "data_fhnorm":        _make_fhnorm,
        "data_betalogitnorm": _make_betalogitnorm,
        "data_binlogitnorm":  _make_binlogitnorm,
        "data_lnln":          _make_lnln,
    }
    return _generators[name]()  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Private generators — one per dataset
# ---------------------------------------------------------------------------

def _rng() -> np.random.Generator:
    return np.random.default_rng(_SEED)


def _make_fhnorm() -> pd.DataFrame:
    """Gaussian Fay-Herriot normal.

    Model: y_i = theta_i + e_i,  e_i ~ N(0, D_i)
           theta_i = x_i^T beta + u_i,  u_i ~ N(0, sigma_u^2)

    Columns: y, D, x1, x2, x3, theta_true, u, group, sre
    """
    rng = _rng()
    m = _N_AREAS

    x1 = rng.normal(0, 1, m)
    x2 = rng.normal(0, 1, m)
    x3 = rng.normal(0, 1, m)

    # True area effects
    sigma_u = 0.5
    u = rng.normal(0, sigma_u, m)
    theta_true = 5.0 + 0.4 * x1 - 0.3 * x2 + 0.2 * x3 + u

    # Known sampling variances D_i (vary by area)
    D = rng.uniform(0.1, 0.5, m)

    # Direct estimates: y_i = theta_i + e_i
    e = rng.normal(0, np.sqrt(D), m)
    y = theta_true + e

    group = np.arange(1, m + 1)
    return pd.DataFrame({
        "y": y, "D": D,
        "x1": x1, "x2": x2, "x3": x3,
        "theta_true": theta_true, "u": u,
        "group": group, "sre": group,
    })


def _make_betalogitnorm() -> pd.DataFrame:
    """Beta logit-normal.

    Model: y_i | eta_i ~ Beta(mu_i * phi_i, (1 - mu_i) * phi_i)
           logit(mu_i) = x_i^T beta + u_i,  u_i ~ N(0, sigma_u^2)
           phi_i = n_i / deff_i - 1  (precision, derived from survey design)

    Columns: y, theta, x1, x2, x3, n, deff, group, sre
    """
    rng = _rng()
    m = _N_AREAS

    x1 = rng.normal(0, 1, m)
    x2 = rng.normal(0, 1, m)
    x3 = rng.normal(0, 1, m)

    sigma_u = 0.4
    u = rng.normal(0, sigma_u, m)
    eta = -0.5 + 0.3 * x1 - 0.2 * x2 + 0.15 * x3 + u
    mu = 1.0 / (1.0 + np.exp(-eta))  # logistic

    # Survey design parameters
    n_sample = rng.integers(50, 200, m).astype(float)
    deff = rng.uniform(1.2, 2.5, m)
    phi = n_sample / deff - 1  # precision parameter

    # Draw proportions from Beta
    a = mu * phi
    b = (1.0 - mu) * phi
    y = rng.beta(a, b)
    # Ensure strict (0, 1) — numerical safety
    y = np.clip(y, 1e-6, 1 - 1e-6)

    group = np.arange(1, m + 1)
    return pd.DataFrame({
        "y": y, "theta": mu,
        "x1": x1, "x2": x2, "x3": x3,
        "n": n_sample, "deff": deff,
        "group": group, "sre": group,
    })


def _make_binlogitnorm() -> pd.DataFrame:
    """Binomial logit-normal.

    Model: y_i | eta_i ~ Binomial(n_i, p_i)
           logit(p_i) = x_i^T beta + u_i,  u_i ~ N(0, sigma_u^2)

    Columns: n, y, p, x1, x2, x3, u_true, eta_true, p_true,
             psi_i, y_obs, p_obs, group, sre
    """
    rng = _rng()
    m = _N_AREAS

    x1 = rng.normal(0, 1, m)
    x2 = rng.normal(0, 1, m)
    x3 = rng.normal(0, 1, m)

    sigma_u = 0.5
    u_true = rng.normal(0, sigma_u, m)
    eta_true = -0.5 + 0.3 * x1 - 0.2 * x2 + 0.15 * x3 + u_true
    p_true = 1.0 / (1.0 + np.exp(-eta_true))

    n_trials = rng.integers(30, 120, m)
    y = rng.binomial(n_trials, p_true)

    p_obs = y / n_trials  # direct proportion estimate
    y_obs = y.astype(float)

    # Sampling variance of logit-scale direct estimate
    psi_i = 1.0 / (n_trials * p_true * (1.0 - p_true))

    group = np.arange(1, m + 1)
    return pd.DataFrame({
        "n": n_trials, "y": y,
        "p": p_obs,
        "x1": x1, "x2": x2, "x3": x3,
        "u_true": u_true, "eta_true": eta_true, "p_true": p_true,
        "psi_i": psi_i, "y_obs": y_obs, "p_obs": p_obs,
        "group": group, "sre": group,
    })


def _make_lnln() -> pd.DataFrame:
    """Lognormal-lognormal.

    Model: log(y_i) | eta_i ~ N(eta_i, psi_i)   [psi_i = known sampling var]
           eta_i = x_i^T beta + u_i,  u_i ~ N(0, sigma_u^2)

    The response for model fitting is ``y_log_obs`` (log-scale direct estimate).
    ``y_obs`` is the original scale; ``lambda_dir`` is the direct estimate on
    the log scale from the sample.

    Columns: group, x1, x2, x3, u_true, theta_true, mu_orig_true,
             n, y_obs, lambda_dir, y_log_obs, psi_i, sre
    """
    rng = _rng()
    m = _N_AREAS

    x1 = rng.normal(0, 1, m)
    x2 = rng.normal(0, 1, m)
    x3 = rng.normal(0, 1, m)

    sigma_u = 0.4
    u_true = rng.normal(0, sigma_u, m)
    theta_true = 2.0 + 0.3 * x1 - 0.2 * x2 + 0.1 * x3 + u_true  # log-scale true mean
    mu_orig_true = np.exp(theta_true + 0.5 * sigma_u**2)          # original-scale E[Y]

    # Sample sizes (area-level)
    n = rng.integers(30, 150, m)

    # Known sampling variance on log scale (approx from delta method)
    psi_i = (sigma_u**2 + 0.25) / n

    # Direct estimates: log-scale
    lambda_dir = rng.normal(theta_true, np.sqrt(psi_i))
    y_log_obs = lambda_dir                              # alias, used as model response
    y_obs = np.exp(lambda_dir)                         # back-transformed direct estimate

    group = np.arange(1, m + 1)
    return pd.DataFrame({
        "group": group,
        "x1": x1, "x2": x2, "x3": x3,
        "u_true": u_true, "theta_true": theta_true, "mu_orig_true": mu_orig_true,
        "n": n.astype(float), "y_obs": y_obs, "lambda_dir": lambda_dir,
        "y_log_obs": y_log_obs, "psi_i": psi_i,
        "sre": group,
    })
