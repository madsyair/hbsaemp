"""Single source of truth for family metadata *and* behavior.

`FAMILY_SPECS` is keyed by the user-facing family label (`"gaussian"` etc.);
every component that needs to know anything about a family reads it here —
metadata (backend family, mean param, links, pipeline fields, kwarg mapping)
*and* behavior (response-domain check, offset preprocessing, and the formula
addition template). Behavior is stored as **module-level functions** (never
subclass method references), so this module imports nothing from `hbsaemp`
except the exception hierarchy (`_exceptions`) and the logger (`_logging`) —
neither has a back-dependency, so the module stays cycle-free.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hbsaemp._exceptions import DataValidationError, ModelRegistryError
from hbsaemp._logging import get_logger

logger = get_logger(__name__)

__all__: list[str] = ["FamilySpec", "FAMILY_SPECS", "list_families", "get_family_spec"]

#: Raise `DataValidationError` when the response (or its auxiliary columns)
#: violates the family's domain; return `None` on success. `ctx` carries the
#: family's pipeline kwargs (`n_col`, `deff_col`, `sampling_var_col`,
#: `trials_col`, `squeeze`).
ResponseCheck = Callable[[pd.DataFrame, str, dict], None]
#: Return a transformed copy of the frame with any offset columns added
#: (e.g. `log_phi`, `log_sqrt_D`). `ctx` carries the same pipeline kwargs.
Preprocess = Callable[[pd.DataFrame, str, dict], pd.DataFrame]


@dataclass(frozen=True)
class FamilySpec:
    """Immutable metadata and behavior for one HBSAE family.

    Attributes:
        bambi_family: Family string passed to `bambi.Model(family=...)`.
        preproc_family: Family string for `DataValidator` and
            `DataPreprocessor`. Same as `bambi_family` in practice; kept
            separate so a future family can request different preprocessing.
        mean_param_key: Key in `idata.posterior` for the latent mean
            parameter; `"mu"` for Gaussian/Beta, `"p"` for Binomial.
            Read by `predict(kind="response_params")`.
        default_link: Default link applied to mu (or p) when no `link=` is
            given. Read by each subclass constructor via `_default_link`.
        supported_links: Valid link strings, enforced by
            `BaseModel._validate_link()` (run from `_pre_fit_checks()`).
        pipeline_fields: Internal attr names (`self._<name>`) forwarded as
            kwargs to validator and preprocessor. Read by
            `BaseModel._extra_pipeline_kwargs()`.
        user_params: `{internal_attr -> user_kwarg}` mapping accepted by
            `create_model()`. Keys mirror `self._<key>` on the model.
        response_check: Domain validator run by `DataValidator.validate()`,
            or `None` to skip. Raises `DataValidationError` on failure.
        preprocess: Offset/transform step run by `DataPreprocessor.process()`,
            or `None` when the family needs no transform.
        addition_template: `str.format` template for the formula left-hand
            side when the family wraps the response (e.g. Binomial
            `"p({response}, {trials_col})"`), or `None` for a bare response.
    """

    bambi_family: str
    preproc_family: str
    mean_param_key: str
    default_link: str
    supported_links: frozenset[str]
    pipeline_fields: tuple[str, ...]
    user_params: dict[str, str] = field(default_factory=dict)
    # Behavior lives here too — read by the data layer and formula builders.
    response_check: ResponseCheck | None = None
    preprocess: Preprocess | None = None
    addition_template: str | None = None


# Family behavior — module-level functions (cycle-free: only numpy/pandas/
# _exceptions/_logging). Each reads the family's pipeline kwargs from `ctx`.

def _check_gaussian(df: pd.DataFrame, response: str, ctx: dict) -> None:
    """Reject non-positive sampling variances for the Gaussian Fay-Herriot offset."""
    sampling_var_col = ctx.get("sampling_var_col")
    if sampling_var_col is not None:
        D = df[sampling_var_col].dropna()
        n_bad = int((D <= 0).sum())
        if n_bad:
            raise DataValidationError(
                f"Gaussian FH: {sampling_var_col!r} must be positive. "
                f"Found {n_bad} non-positive value(s).",
                column=sampling_var_col,
                context={"n_nonpositive": n_bad},
            )


def _preprocess_gaussian(df: pd.DataFrame, response: str, ctx: dict) -> pd.DataFrame:
    """Add ``log_sqrt_D = 0.5 * log(D)`` for Gaussian Fay-Herriot."""
    sampling_var_col = ctx.get("sampling_var_col")
    if sampling_var_col is not None:
        D = df[sampling_var_col].to_numpy(dtype=float)
        df["log_sqrt_D"] = 0.5 * np.log(D)
        logger.debug(
            "Gaussian FH: log_sqrt_D in [%.3f, %.3f].",
            df["log_sqrt_D"].min(), df["log_sqrt_D"].max(),
        )
    return df


def _check_beta(df: pd.DataFrame, response: str, ctx: dict) -> None:
    """Validate the Beta response domain and the precision phi = n/deff - 1.

    With ``squeeze=True`` the response domain is relaxed to closed ``[0, 1]``
    (the Smithson-Verkuilen transform will map boundary values into the open
    interval before fitting); otherwise it must be strictly open ``(0, 1)``.
    """
    n_col = ctx.get("n_col")
    deff_col = ctx.get("deff_col")
    squeeze = ctx.get("squeeze", False)
    y = df[response].dropna()
    if squeeze:
        # Smithson-Verkuilen will map [0,1] → (0,1); allow closed interval.
        n_bad = int(((y < 0) | (y > 1)).sum())
        domain_msg = "[0, 1]"
    else:
        # No squeeze: Bambi Beta requires strictly open (0, 1).
        n_bad = int(((y <= 0) | (y >= 1)).sum())
        domain_msg = "(0, 1)"
    if n_bad:
        raise DataValidationError(
            f"Beta family requires response in {domain_msg}. "
            f"Found {n_bad} value(s) outside this range in {response!r}. "
            + ("" if squeeze else "Pass squeeze=True to handle boundary values 0 and 1."),
            column=response,
            context={"n_boundary_values": n_bad},
        )

    if n_col is not None and deff_col is not None:
        phi_df = df[[n_col, deff_col]].dropna()
        if (phi_df[n_col] <= 0).any():
            raise DataValidationError(
                f"{n_col!r}: sample sizes must be positive.", column=n_col
            )
        if (phi_df[deff_col] <= 0).any():
            raise DataValidationError(
                f"{deff_col!r}: design effects must be positive.", column=deff_col
            )
        phi = phi_df[n_col].values / phi_df[deff_col].values - 1
        n_bad_phi = int((phi <= 0).sum())
        if n_bad_phi:
            raise DataValidationError(
                f"Beta precision phi = n/deff - 1 must be > 0. "
                f"Found {n_bad_phi} row(s) where n/deff ≤ 1.",
                context={"n_invalid_phi": n_bad_phi},
            )


def _preprocess_beta(df: pd.DataFrame, response: str, ctx: dict) -> pd.DataFrame:
    """Add ``log_phi`` column; optionally apply the Smithson-Verkuilen squeeze.

    Only active when both ``n_col`` and ``deff_col`` are provided.
    ``log_phi = log(n/deff - 1)`` is always computed when the precision is
    pinned from data — it is required by the Bambi distributional formula
    ``"kappa ~ 1 + offset(log_phi)"``. The squeeze ``(y*(n-1)+0.5)/n`` is
    applied only when ``squeeze=True`` (for datasets with boundary ``y=0``/
    ``y=1``); otherwise the response is left untouched.
    """
    n_col = ctx.get("n_col")
    deff_col = ctx.get("deff_col")
    squeeze = ctx.get("squeeze", False)
    if n_col is not None and deff_col is not None:
        n_vals = df[n_col].to_numpy(dtype=float)
        deff_vals = df[deff_col].to_numpy(dtype=float)

        if squeeze:
            y = df[response].to_numpy(dtype=float)
            df[response] = (y * (n_vals - 1) + 0.5) / n_vals
            logger.debug(
                "Beta: Smithson-Verkuilen squeeze applied (squeeze=True)."
            )

        df["log_phi"] = np.log(n_vals / deff_vals - 1)
        logger.debug(
            "Beta: log_phi in [%.3f, %.3f].",
            df["log_phi"].min(), df["log_phi"].max(),
        )
    return df


def _check_binomial(df: pd.DataFrame, response: str, ctx: dict) -> None:
    """Require non-negative integer successes not exceeding positive-integer trials."""
    trials_col = ctx.get("trials_col")
    y = df[response].dropna()
    if (y < 0).any():
        raise DataValidationError(
            f"Binomial family: {response!r} must be non-negative.", column=response
        )
    if not (y % 1 == 0).all():
        raise DataValidationError(
            f"Binomial family: {response!r} must contain integers.", column=response
        )
    if trials_col is not None:
        check = df[[response, trials_col]].dropna()
        n = check[trials_col]
        if (n < 1).any() or not (n % 1 == 0).all():
            raise DataValidationError(
                f"Binomial family: {trials_col!r} must contain positive integers.",
                column=trials_col,
            )
        n_bad = int((check[response].values > check[trials_col].values).sum())
        if n_bad:
            raise DataValidationError(
                f"Binomial family: {response!r} must not exceed {trials_col!r}. "
                f"Found {n_bad} row(s) where y > n.",
                column=response,
                context={"n_exceeding_trials": n_bad},
            )


# Adding a new family is a near-one-file change: (1) a FamilySpec entry here
# carrying metadata AND behavior (response_check / preprocess /
# addition_template), plus (2) a MODEL_REGISTRY entry in _factory.py. A
# BaseModel subclass is needed ONLY for formula behavior a template cannot
# express, or prior workarounds — never to re-declare metadata or domain/offset
# logic, which the data layer reads straight from this spec.
FAMILY_SPECS: dict[str, FamilySpec] = {
    "gaussian": FamilySpec(
        bambi_family="gaussian",
        preproc_family="gaussian",
        mean_param_key="mu",
        default_link="identity",
        supported_links=frozenset({"identity", "log"}),
        pipeline_fields=("sampling_var_col",),
        user_params={"sampling_var_col": "sampling_var", "link": "link"},
        response_check=_check_gaussian,
        preprocess=_preprocess_gaussian,
    ),
    "beta": FamilySpec(
        bambi_family="beta",
        preproc_family="beta",
        mean_param_key="mu",
        default_link="logit",
        supported_links=frozenset({"logit", "probit"}),
        pipeline_fields=("n_col", "deff_col", "squeeze"),
        user_params={
            "n_col": "n",
            "deff_col": "deff",
            "squeeze": "squeeze",
            "link": "link",
        },
        response_check=_check_beta,
        preprocess=_preprocess_beta,
    ),
    "binomial": FamilySpec(
        bambi_family="binomial",
        preproc_family="binomial",
        mean_param_key="p",
        default_link="logit",
        supported_links=frozenset({"logit", "probit"}),
        pipeline_fields=("trials_col",),
        user_params={"trials_col": "trials", "link": "link"},
        response_check=_check_binomial,
        preprocess=None,  # binomial needs no offset transform
        addition_template="p({response}, {trials_col})",
    ),
}


def list_families() -> list[str]:
    """Return the registered family names, sorted (mirrors R `list_hbsae_models()`)."""
    return sorted(FAMILY_SPECS)


def get_family_spec(name: str) -> FamilySpec:
    """Return the `FamilySpec` for family *name* (mirrors R `get_hbsae_model()`).

    Raises:
        ModelRegistryError: When *name* is not a registered family.
    """
    try:
        return FAMILY_SPECS[name]
    except KeyError:
        raise ModelRegistryError(
            f"Unknown family {name!r}. Registered families: {list_families()}.",
            family=name,
            registered=list_families(),
        ) from None
