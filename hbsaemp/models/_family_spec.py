"""Single source of truth for family metadata.

`FAMILY_SPECS` is keyed by the user-facing family label (`"gaussian"` etc.);
every component that needs to know anything about a family reads it here.
This module imports nothing from the rest of `hbsaemp` to keep it
cycle-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__: list[str] = ["FamilySpec", "FAMILY_SPECS"]


@dataclass(frozen=True)
class FamilySpec:
    """Immutable metadata for one HBSAE family.

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
    """

    bambi_family: str
    preproc_family: str
    mean_param_key: str
    default_link: str
    supported_links: frozenset[str]
    pipeline_fields: tuple[str, ...]
    user_params: dict[str, str] = field(default_factory=dict)


# Adding a new implemented family typically needs: (1) a FamilySpec entry
# here, (2) a BaseModel subclass for family-specific *behavior* only —
# formula rewriting, prior workarounds (no metadata constants), (3) a
# MODEL_REGISTRY entry in _factory.py, (4) tests + public shortcuts if
# exposed. All family metadata (links, mean param, pipeline fields, backend
# family) lives here and is read by BaseModel — never re-declared on the
# subclass.
FAMILY_SPECS: dict[str, FamilySpec] = {
    "gaussian": FamilySpec(
        bambi_family="gaussian",
        preproc_family="gaussian",
        mean_param_key="mu",
        default_link="identity",
        supported_links=frozenset({"identity", "log"}),
        pipeline_fields=("sampling_var_col",),
        user_params={"sampling_var_col": "sampling_var", "link": "link"},
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
    ),
    "binomial": FamilySpec(
        bambi_family="binomial",
        preproc_family="binomial",
        mean_param_key="p",
        default_link="logit",
        supported_links=frozenset({"logit", "probit"}),
        pipeline_fields=("trials_col",),
        user_params={"trials_col": "trials", "link": "link"},
    ),
}
