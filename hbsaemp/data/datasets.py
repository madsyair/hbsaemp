"""Built-in datasets matching R hbsaems package.

v0: load_dataset() stub — raises NotImplementedError.
v1: Returns real DataFrames generated from reproducible seeds.

Available datasets (same names as R hbsaems):
    data_fhnorm         — Gaussian/Fay-Herriot normal
    data_betalogitnorm  — Beta logit-normal
    data_binlogitnorm   — Binomial logit-normal
    data_lnln           — Lognormal-lognormal
"""
from __future__ import annotations
from typing import Literal
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


def load_dataset(name: DatasetName) -> pd.DataFrame:
    """Load a built-in hbsaemp dataset.

    Args:
        name: Dataset identifier. One of :data:`AVAILABLE_DATASETS`.

    Returns:
        A :class:`pandas.DataFrame` with 100 rows.

    Raises:
        ValueError: If *name* is not in :data:`AVAILABLE_DATASETS`.
        NotImplementedError: In v0.

    Example:
        >>> df = load_dataset("data_fhnorm")
        >>> df.shape
        (100, 6)
    """
    if name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}. "
            f"Available: {AVAILABLE_DATASETS}"
        )
    raise NotImplementedError(
        f"load_dataset({name!r}) requires v1 (numpy + pandas datasets module)."
    )
