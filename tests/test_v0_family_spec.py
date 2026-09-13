"""FAMILY_SPECS single-source contract — no Bambi, no MCMC.

Locks the P1 invariant: family metadata (mean param, links, pipeline
fields, backend/preproc family) lives only in `FAMILY_SPECS` and is read by
`BaseModel`. These tests fail loudly if a subclass re-introduces a local
metadata constant or drifts from the spec.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hbsaemp as hb
from hbsaemp._exceptions import DataValidationError
from hbsaemp.models._factory import MODEL_REGISTRY
from hbsaemp.models._family_spec import FAMILY_SPECS

FAMILIES = ["gaussian", "beta", "binomial"]

# Minimum required family kwargs so create_model() succeeds and a later
# _pre_fit_checks() reaches the link check instead of failing earlier on a
# missing required argument.
_REQUIRED_KW: dict[str, dict[str, str]] = {
    "gaussian": {},
    "beta": {"n": "n", "deff": "deff"},
    "binomial": {"trials": "n"},
}


@pytest.fixture
def data_for(data_gaussian, data_beta, data_binomial):
    """Map family -> a fixture DataFrame valid for that family."""
    return {
        "gaussian": data_gaussian,
        "beta": data_beta,
        "binomial": data_binomial,
    }


def _build(family, data_for, default_config, *, link=None, **extra):
    kw = dict(_REQUIRED_KW[family], **extra)
    return hb.create_model(
        "y ~ x1", family, data_for[family],
        link=link, config=default_config, **kw,
    )


# Registry <-> spec consistency

def test_model_registry_matches_family_specs():
    # The sets must be equal: a subset would silently allow an unregistered
    # family to slip through.
    assert set(MODEL_REGISTRY) == set(FAMILY_SPECS)


def test_mean_param_keys_in_spec():
    assert FAMILY_SPECS["gaussian"].mean_param_key == "mu"
    assert FAMILY_SPECS["beta"].mean_param_key == "mu"
    assert FAMILY_SPECS["binomial"].mean_param_key == "p"


# Anti-drift: subclasses must NOT re-declare metadata constants

@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "attr", ["_MEAN_PARAM_KEY", "_SUPPORTED_LINKS", "_EXTRA_FIELD_NAMES"]
)
def test_subclass_has_no_local_metadata_constant(family, attr):
    """Re-introducing a local constant defeats the single source — forbid it."""
    cls = MODEL_REGISTRY[family]
    assert not hasattr(cls, attr), (
        f"{cls.__name__}.{attr} re-introduced — read FAMILY_SPECS instead."
    )


# BaseModel reads metadata from the spec

@pytest.mark.parametrize("family", FAMILIES)
def test_instance_mean_param_key_reads_spec(family, data_for, default_config):
    m = _build(family, data_for, default_config)
    assert m._mean_param_key == FAMILY_SPECS[family].mean_param_key


@pytest.mark.parametrize("family", FAMILIES)
def test_instance_default_link_reads_spec(family, data_for, default_config):
    # No link= given -> the family's spec default is used.
    m = _build(family, data_for, default_config)
    assert m._link == FAMILY_SPECS[family].default_link


@pytest.mark.parametrize("family", FAMILIES)
def test_backend_family_reads_spec(family, data_for, default_config):
    m = _build(family, data_for, default_config)
    assert m._bambi_family() == FAMILY_SPECS[family].bambi_family


@pytest.mark.parametrize("family", FAMILIES)
def test_data_layer_is_keyed_by_the_registry_name(family, data_for, default_config):
    """The validator and preprocessor look the spec up by the family string.

    They both do `FAMILY_SPECS[family]`, so whatever the model hands them has
    to be the registry key itself. A separate `preproc_family` field used to
    sit here promising it could differ; it could not, and it is gone.
    """
    m = _build(family, data_for, default_config)
    assert m._family == family
    assert m._family in FAMILY_SPECS


# pipeline_fields are backed by real instance attributes

@pytest.mark.parametrize("family", FAMILIES)
def test_pipeline_fields_have_backing_attributes(family, data_for, default_config):
    m = _build(family, data_for, default_config)
    for name in FAMILY_SPECS[family].pipeline_fields:
        assert hasattr(m, f"_{name}"), f"{family}: missing _{name}"
    # And the derived kwargs dict keys match the spec's pipeline_fields.
    assert set(m._extra_pipeline_kwargs()) == set(FAMILY_SPECS[family].pipeline_fields)


# Link validation reads supported_links from the spec

@pytest.mark.parametrize("family", FAMILIES)
def test_unsupported_link_rejected_from_spec(family, data_for, default_config):
    m = _build(family, data_for, default_config, link="not_a_real_link")
    with pytest.raises(ValueError, match="not supported"):
        m._pre_fit_checks()


def test_cross_family_link_rejected(data_for, default_config):
    # "logit" is valid for beta/binomial but NOT gaussian — the spec, not a
    # global list, decides. Confirms per-family link sets are honoured.
    m = _build("gaussian", data_for, default_config, link="logit")
    with pytest.raises(ValueError, match="not supported"):
        m._pre_fit_checks()


@pytest.mark.parametrize("family", FAMILIES)
def test_default_link_passes_pre_fit_checks(family, data_for, default_config):
    # The spec default link must itself be in supported_links.
    m = _build(family, data_for, default_config)
    m._pre_fit_checks()  # must not raise


# Beta squeeze guard (added with P1 — stricter than before)

def test_beta_squeeze_without_n_deff_rejected(data_beta, default_config):
    m = hb.create_model("y ~ x1", "beta", data_beta,
                        squeeze=True, config=default_config)
    with pytest.raises(ValueError, match="squeeze=True"):
        m._pre_fit_checks()


def test_beta_squeeze_with_n_deff_ok(data_beta, default_config):
    m = hb.create_model("y ~ x1", "beta", data_beta,
                        n="n", deff="deff", squeeze=True, config=default_config)
    m._pre_fit_checks()  # squeeze with the precision offset active is allowed


# Behavior callables live in the spec too (P0-1) — no Bambi, direct on the spec

def test_addition_template_only_for_binomial():
    # Only families that wrap the response carry a template; others are None.
    assert FAMILY_SPECS["binomial"].addition_template == "p({response}, {trials_col})"
    assert FAMILY_SPECS["gaussian"].addition_template is None
    assert FAMILY_SPECS["beta"].addition_template is None


def test_addition_template_formats_lhs():
    lhs = FAMILY_SPECS["binomial"].addition_template.format(response="y", trials_col="n")
    assert lhs == "p(y, n)"


def test_response_check_rejects_out_of_domain_beta():
    bad = pd.DataFrame({"y": [0.2, 1.5]})  # 1.5 is outside (0, 1)
    with pytest.raises(DataValidationError):
        FAMILY_SPECS["beta"].response_check(bad, "y", {"squeeze": False})


def test_response_check_rejects_non_integer_binomial():
    bad = pd.DataFrame({"y": [1.5, 2.0]})  # non-integer successes
    with pytest.raises(DataValidationError):
        FAMILY_SPECS["binomial"].response_check(bad, "y", {"trials_col": None})


# Survey-design domain guards.
#
# Each of these is the last thing standing between bad design columns and a
# silently wrong offset: log(D) on a non-positive D, or log(n/deff - 1) on a
# ratio <= 1, both yield NaN/-inf that only surfaces deep inside the sampler.


@pytest.mark.parametrize("bad_d", [0.0, -0.5])
def test_response_check_rejects_non_positive_sampling_variance(bad_d):
    """Fay-Herriot needs D_i > 0 — `log_sqrt_D = 0.5*log(D)` is undefined otherwise."""
    bad = pd.DataFrame({"y": [5.0, 6.0], "D": [0.25, bad_d]})
    with pytest.raises(DataValidationError, match="must be positive"):
        FAMILY_SPECS["gaussian"].response_check(bad, "y", {"sampling_var_col": "D"})


@pytest.mark.parametrize("bad_n", [0.0, -10.0])
def test_response_check_rejects_non_positive_sample_size(bad_n):
    bad = pd.DataFrame({"y": [0.5, 0.5], "n": [100.0, bad_n], "deff": [1.5, 1.5]})
    with pytest.raises(DataValidationError, match="sample sizes must be positive"):
        FAMILY_SPECS["beta"].response_check(
            bad, "y", {"n_col": "n", "deff_col": "deff", "squeeze": False}
        )


@pytest.mark.parametrize("bad_deff", [0.0, -1.0])
def test_response_check_rejects_non_positive_design_effect(bad_deff):
    bad = pd.DataFrame({"y": [0.5, 0.5], "n": [100.0, 100.0], "deff": [1.5, bad_deff]})
    with pytest.raises(DataValidationError, match="design effects must be positive"):
        FAMILY_SPECS["beta"].response_check(
            bad, "y", {"n_col": "n", "deff_col": "deff", "squeeze": False}
        )


def test_response_check_rejects_non_positive_precision():
    """phi = n/deff - 1 must exceed 0, so n/deff <= 1 is refused."""
    bad = pd.DataFrame({"y": [0.5], "n": [2.0], "deff": [4.0]})  # n/deff = 0.5
    with pytest.raises(DataValidationError, match="must be > 0") as exc_info:
        FAMILY_SPECS["beta"].response_check(
            bad, "y", {"n_col": "n", "deff_col": "deff", "squeeze": False}
        )
    str(exc_info.value).encode("cp1252")  # message must survive a cp1252 console


def test_response_check_rejects_negative_successes():
    bad = pd.DataFrame({"y": [5.0, -1.0]})
    with pytest.raises(DataValidationError, match="must be non-negative"):
        FAMILY_SPECS["binomial"].response_check(bad, "y", {"trials_col": None})


@pytest.mark.parametrize("bad_trials", [0.0, -5.0, 10.5])
def test_response_check_rejects_invalid_trial_counts(bad_trials):
    """Trials must be positive integers — zero, negative, and fractional all fail."""
    bad = pd.DataFrame({"y": [5.0, 5.0], "n": [25.0, bad_trials]})
    with pytest.raises(DataValidationError, match="positive integers"):
        FAMILY_SPECS["binomial"].response_check(bad, "y", {"trials_col": "n"})


def test_response_check_rejects_successes_exceeding_trials():
    bad = pd.DataFrame({"y": [5.0, 30.0], "n": [25.0, 25.0]})
    with pytest.raises(DataValidationError, match="must not exceed"):
        FAMILY_SPECS["binomial"].response_check(bad, "y", {"trials_col": "n"})


def test_preprocess_beta_adds_log_phi():
    df = pd.DataFrame({"y": [0.2], "n": [100.0], "deff": [2.0]})
    out = FAMILY_SPECS["beta"].preprocess(
        df.copy(), "y", {"n_col": "n", "deff_col": "deff", "squeeze": False}
    )
    assert "log_phi" in out.columns


def test_preprocess_gaussian_adds_log_sqrt_d():
    df = pd.DataFrame({"y": [5.0], "D": [0.25]})
    out = FAMILY_SPECS["gaussian"].preprocess(df.copy(), "y", {"sampling_var_col": "D"})
    assert "log_sqrt_D" in out.columns


def test_binomial_has_no_preprocess():
    # Binomial needs no offset transform — the spec stores None and the
    # preprocessor skips it (mirrors the response_check=None branch).
    assert FAMILY_SPECS["binomial"].preprocess is None


# FixedParam.offset_col names the column preprocess adds

def test_offset_col_in_spec():
    assert FAMILY_SPECS["gaussian"].fixed_params[0].offset_col == "log_sqrt_D"
    assert FAMILY_SPECS["beta"].fixed_params[0].offset_col == "log_phi"
    assert FAMILY_SPECS["binomial"].fixed_params == ()


@pytest.mark.parametrize(
    ("family", "df", "ctx"),
    [
        ("gaussian", pd.DataFrame({"y": [5.0], "D": [0.25]}), {"sampling_var_col": "D"}),
        ("beta", pd.DataFrame({"y": [0.2], "n": [100.0], "deff": [2.0]}),
         {"n_col": "n", "deff_col": "deff", "squeeze": False}),
    ],
)
def test_preprocess_adds_exactly_offset_col(family, df, ctx):
    """The pin's offset_col and the column preprocess writes must not drift."""
    out = FAMILY_SPECS[family].preprocess(df.copy(), "y", ctx)
    assert set(out.columns) - set(df.columns) == {
        fp.offset_col for fp in FAMILY_SPECS[family].fixed_params
    }


# fixed_params — the fix-param contract
#
# A family declares which distributional parameters it computes from the survey
# design instead of sampling. BaseModel builds the entire offset-and-pin
# machinery from that declaration, so these tests guard the declaration itself:
# if it drifts from pipeline_fields or from what preprocess writes, the formula
# silently references a column that is not there.


@pytest.mark.parametrize("family", FAMILIES)
def test_fixed_param_sources_are_declared_pipeline_fields(family):
    """Every source column of a pin must be a pipeline field of that family.

    `_active_fixed_params()` reads `self._<name>` for each source field, so a
    name that is not in `pipeline_fields` has no backing attribute and the pin
    can never activate.
    """
    spec = FAMILY_SPECS[family]
    for fp in spec.fixed_params:
        missing = set(fp.source_fields) - set(spec.pipeline_fields)
        assert not missing, (
            f"{family}: {fp.param!r} names source field(s) {sorted(missing)} "
            f"that are not in pipeline_fields."
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_fixed_param_links_are_real(family):
    """A pinned parameter's link has to be one Bambi understands."""
    for fp in FAMILY_SPECS[family].fixed_params:
        assert fp.link in {"identity", "log", "logit", "probit", "inverse"}


@pytest.mark.parametrize("family", FAMILIES)
def test_offset_properties_agree_with_fixed_params(family):
    """`offset_source_fields` is derived, so it cannot drift from the pins."""
    spec = FAMILY_SPECS[family]
    assert spec.offset_source_fields == tuple(
        name for fp in spec.fixed_params for name in fp.source_fields
    )


def test_binomial_pins_nothing():
    """Binomial's only parameter IS the estimand, so nothing is fixed.

    This is the `fixed_params = ()` case — "no fixed params" has to be a
    first-class state, not an accident of leaving a field unset.
    """
    spec = FAMILY_SPECS["binomial"]
    assert spec.fixed_params == ()
    assert spec.offset_source_fields == ()
    assert spec.pinnable_params == {}


@pytest.mark.parametrize(
    ("family", "extra", "expected"),
    [
        ("gaussian", {"sampling_var": "sre"}, ("sigma",)),
        ("gaussian", {}, ()),
        ("beta", {"n": "n", "deff": "deff"}, ("kappa",)),
        ("beta", {}, ()),
        ("binomial", {"trials": "n"}, ()),
    ],
)
def test_active_fixed_params_follow_supplied_columns(
    family, extra, expected, data_for, default_config
):
    """A pin activates only when every source column was supplied.

    This reproduces exactly the conditions that used to be hardcoded twice per
    subclass (`sampling_var_col is not None`, `n_col is not None and deff_col
    is not None`) — now derived from the spec instead.
    """
    m = hb.create_model(
        "y ~ x1 + (1|group)", family=family, data=data_for[family],
        config=default_config, **extra,
    )
    assert tuple(fp.param for fp in m._active_fixed_params()) == expected


def test_pinning_is_all_or_nothing_per_fixed_param(data_beta, default_config):
    """Half a pin is refused: n without deff cannot compute phi."""
    with pytest.raises(ValueError, match="provide all of"):
        hb.create_model(
            "y ~ x1", family="beta", data=data_beta,
            config=default_config, n="n",
        )


# fixed_params= — the caller-facing door onto the same machinery
#
# A user pin and a family-declared pin produce the same FixedParam, so nothing
# downstream needs to know which it came from. What differs is where the
# values originate, and therefore what can go wrong with them.


@pytest.mark.parametrize("source", ["sd_col", 2.5])
def test_user_pin_becomes_a_fixed_param(source, data_gaussian, default_config):
    """A caller pin joins `_active_fixed_params()` with no source columns."""
    df = data_gaussian.assign(sd_col=2.0)
    m = hb.create_model(
        "y ~ x1", family="gaussian", data=df, config=default_config,
        fixed_params={"sigma": source},
    )
    (fp,) = m._active_fixed_params()
    assert fp.param == "sigma"
    assert fp.link == "log"
    assert fp.source_fields == (), "a caller pin needs no design column"
    assert "sigma" in fp.offset_col


def test_user_pin_materialises_its_column(data_gaussian, default_config):
    """The pinned value reaches the frame on the parameter's link scale."""
    df = data_gaussian.assign(sd_col=2.0)
    m = hb.create_model(
        "y ~ x1", family="gaussian", data=df, config=default_config,
        fixed_params={"sigma": "sd_col"},
    )
    clean = m.check_data()
    (fp,) = m._active_fixed_params()
    assert clean[fp.offset_col].iloc[0] == pytest.approx(np.log(2.0))


def test_pinnable_params_come_from_fixed_params():
    """Declaring a FixedParam is what makes a parameter pinnable by a caller."""
    assert FAMILY_SPECS["gaussian"].pinnable_params == {"sigma": "log"}
    assert FAMILY_SPECS["beta"].pinnable_params == {"kappa": "log"}
    assert FAMILY_SPECS["binomial"].pinnable_params == {}


@pytest.mark.parametrize(
    ("family", "extra", "pins", "match"),
    [
        ("gaussian", {}, {"nu": 4}, "cannot pin"),
        ("binomial", {"trials": "n"}, {"sigma": 1.0}, "cannot pin"),
        ("gaussian", {"sampling_var": "sre"}, {"sigma": 1.0}, "pinned twice"),
        ("gaussian", {}, {"sigma": [1, 2]}, "column name or a number"),
        ("gaussian", {}, {"sigma": True}, "column name or a number"),
    ],
)
def test_user_pin_rejections(family, extra, pins, match, data_for, default_config):
    """Every way a caller pin can be wrong is refused before Bambi is reached."""
    with pytest.raises(ValueError, match=match):
        hb.create_model(
            "y ~ x1", family=family, data=data_for[family],
            config=default_config, fixed_params=pins, **extra,
        )


def test_user_pin_outside_the_link_domain_is_refused(data_gaussian, default_config):
    """A log link cannot carry a non-positive value; say so, don't emit NaN."""
    m = hb.create_model(
        "y ~ x1", family="gaussian", data=data_gaussian,
        config=default_config, fixed_params={"sigma": -1.0},
    )
    with pytest.raises(DataValidationError, match="must be positive"):
        m.check_data()


def test_nan_in_a_pin_column_drops_the_row(data_gaussian, default_config):
    """A pin column follows the same NaN rule as a survey-design column.

    Regression guard: the pin column was not in the preprocessor's dropna
    subset, so the row survived and `apply_link` turned it into `log(NaN)` —
    a NaN offset that reaches the sampler, which is exactly what the
    link-domain check above exists to prevent.
    """
    df = data_gaussian.assign(sd_col=2.0)
    df.loc[df.index[0], "sd_col"] = np.nan
    m = hb.create_model(
        "y ~ x1", family="gaussian", data=df, config=default_config,
        fixed_params={"sigma": "sd_col"},
    )

    clean = m.check_data()
    (fp,) = m._active_fixed_params()

    assert len(clean) == len(df) - 1
    assert not clean[fp.offset_col].isna().any()


def test_user_pin_is_recorded_in_the_result_extra(data_gaussian, default_config):
    """`fixed_params` is the one model input that is not a pipeline field.

    Everything else `create_model()` accepts reaches `ModelResult.extra`
    through `_extra_pipeline_kwargs()`. A caller pin does not, so without an
    explicit entry anyone rebuilding a model from `result.extra` — the GUI's
    code export among them — would silently drop the pin and fit an
    unpinned model instead.
    """
    df = data_gaussian.assign(sd_col=2.0)
    m = hb.create_model(
        "y ~ x1", family="gaussian", data=df, config=default_config,
        fixed_params={"sigma": "sd_col"},
    )

    assert m._extra_result_dict()["fixed_params"] == {"sigma": "sd_col"}


def test_no_pin_leaves_the_result_extra_alone(data_gaussian, default_config):
    """An unpinned model gains no `fixed_params` key — absent, not empty."""
    m = hb.create_model(
        "y ~ x1", family="gaussian", data=data_gaussian, config=default_config,
    )

    assert "fixed_params" not in m._extra_result_dict()


def test_a_parameter_is_never_pinned_twice(data_gaussian, default_config):
    """Both pin doors open at once must still yield one pin, not two.

    `create_model()` refuses this combination outright, but a subclass built
    directly bypasses that screen exactly as it bypasses the family screen
    above. Two `FixedParam`s for one parameter would emit two sub-formulas
    for it and leave Bambi to resolve the conflict; the caller's pin wins.
    """
    from hbsaemp.models._gaussian import GaussianModel

    df = data_gaussian.assign(D=0.25, sd_col=2.0)
    m = GaussianModel(
        "y ~ x1", "gaussian", df, default_config,
        sampling_var_col="D",
        fixed_params={"sigma": "sd_col"},
    )

    (fp,) = m._active_fixed_params()
    assert fp.param == "sigma"
    assert fp.offset_col == "hbsaemp_sigma_fixed"


# The two-entry claim: a new family needs a FamilySpec and a MODEL_REGISTRY
# entry, and nothing else. Asserted here because it is the whole point of the
# fix-param refactor and nothing else in the suite would notice it regressing.


class _FakeBmb:
    """Stand-in for the `bambi` module — this file never imports the real one."""

    class Formula:
        def __init__(self, *parts):
            self.parts = parts

    class Prior:
        def __init__(self, dist, **params):
            self.dist = dist
            self.params = params


def _check_throwaway(df, response, ctx):
    """Domain rule the spec carries; proves `response_check` is reached."""
    if (df[response] <= 0).any():
        raise DataValidationError("Throwaway response must be positive.")


def _preprocess_throwaway(df, response, ctx):
    """Writes the pin column from a field neither data-layer class knows."""
    scale_col = ctx.get("scale_col")
    if scale_col is not None:
        df["throwaway_offset"] = np.log(df[scale_col].to_numpy(dtype=float))
    return df


@pytest.fixture
def throwaway_family(monkeypatch):
    """Register a family that overrides no hook at all, then tear it down."""
    from hbsaemp.models._base import BaseModel
    from hbsaemp.models._family_spec import FamilySpec, FixedParam

    class ThrowawayModel(BaseModel):
        # The entire subclass: store the fields the spec declares. No
        # `_build_formula_and_link`, no `_workaround_priors`, no
        # `_response_pp_key`, no `_pre_fit_checks` — if any of those became
        # necessary again, this class would fail rather than quietly grow.
        def __init__(self, *args, scale_col=None, size_col=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._scale_col = scale_col
            self._size_col = size_col

    spec = FamilySpec(
        bambi_family="gaussian",
        mean_param_key="mu",
        default_link="identity",
        supported_links=frozenset({"identity"}),
        # Neither user kwarg name exists anywhere in create_model's signature.
        pipeline_fields=("scale_col", "size_col"),
        user_params={"scale_col": "scale", "size_col": "size"},
        response_check=_check_throwaway,
        preprocess=_preprocess_throwaway,
        addition_template="w({response}, {size_col})",
        addition_field="size_col",
        fixed_params=(
            FixedParam("sigma", "log", "throwaway_offset", ("scale_col",)),
        ),
    )
    monkeypatch.setitem(FAMILY_SPECS, "throwaway", spec)
    monkeypatch.setitem(MODEL_REGISTRY, "throwaway", ThrowawayModel)
    return ThrowawayModel


@pytest.fixture
def throwaway_data(data_gaussian):
    """Frame valid for the throwaway family (positive response + its columns)."""
    return data_gaussian.assign(
        y=np.abs(data_gaussian["y"]) + 0.1, scale=2.0, size=10,
    )


def test_new_family_needs_no_hook_override(
    throwaway_family, throwaway_data, default_config
):
    """A new family writes no formula, prior, or PP-key code of its own.

    `scale_col` / `size_col` are pipeline fields neither `DataValidator` nor
    `DataPreprocessor` has ever heard of; both take the family's fields as
    `**kwargs` and read the string values as column names, so they flow
    through untouched.
    """
    m = throwaway_family(
        "y ~ x1", "throwaway", throwaway_data, default_config,
        scale_col="scale", size_col="size",
    )

    clean = m.check_data()
    assert clean["throwaway_offset"].iloc[0] == pytest.approx(np.log(2.0))

    formula, link = m._build_formula_and_link(_FakeBmb, "y")
    assert formula.parts == (
        "w(y, size) ~ x1", "sigma ~ 1 + offset(throwaway_offset)",
    )
    assert link == {"mu": "identity", "sigma": "log"}

    # The PP key must be the same literal LHS Bambi was handed, derived from
    # the template rather than re-spelled.
    assert m._response_pp_key("y") == "w(y, size)"

    priors = m._workaround_priors(_FakeBmb)
    assert priors["sigma"]["Intercept"].params == {"mu": 0.0, "sigma": 1e-3}


def test_addition_template_is_not_inert(
    throwaway_family, throwaway_data, default_config
):
    """A declared addition template must reach the formula.

    Regression guard: `BaseModel` used to go straight to the distributional
    builder, so a family that declared `addition_template` without also
    overriding `_build_formula_and_link` got a formula with the addition term
    silently missing — a wrong model, no error, no warning. Every other
    `FamilySpec` field was honoured by the base class; this one was not.
    """
    m = throwaway_family(
        "y ~ x1", "throwaway", throwaway_data, default_config,
        scale_col="scale", size_col="size",
    )

    formula, _ = m._build_formula_and_link(_FakeBmb, "y")

    assert formula.parts[0].startswith("w(y, size) ~")


def test_new_family_reaches_the_front_door(
    throwaway_family, throwaway_data, default_config
):
    """`create_model()` accepts a family whose kwargs its signature never names.

    Regression guard for the sharpest edge of all: family kwargs were looked
    up in a literal dict inside `create_model`, so a family declaring a new
    `user_params` value died on `KeyError: 'scale'` — with no message naming
    the cause, and no way to pass the argument even if you knew. The two tests
    above did not catch it because they build the subclass directly; only
    going through the front door does.
    """
    m = hb.create_model(
        "y ~ x1", family="throwaway", data=throwaway_data,
        config=default_config,
        aux_args={"scale": "scale", "size": "size"},
    )

    assert m._scale_col == "scale"
    assert m._size_col == "size"
    assert m.check_data()["throwaway_offset"].iloc[0] == pytest.approx(np.log(2.0))


def test_new_family_reaches_tier_two(
    throwaway_family, throwaway_data, default_config
):
    """`hbm_flex()` too — via the generic channels, with no signature edit."""
    m = hb.hbm_flex(
        "y", ["x1"], throwaway_data, family="throwaway",
        area_var="group", config=default_config,
        addition_var="size", aux_args={"scale": "scale"},
    )

    assert m.formula == "y ~ x1 + (1|group)"
    assert m._size_col == "size"          # addition_var landed on the right field
    assert m._scale_col == "scale"


def test_new_family_gets_typo_protection(
    throwaway_family, throwaway_data, default_config
):
    """`aux_args` is checked against the spec, so a new family is not defenceless.

    The built-in families get this from `create_model`'s explicit signature
    (a typo is a Python `TypeError`). A family reached through `aux_args` has
    no signature to lean on, so the check has to be explicit.
    """
    with pytest.raises(ValueError, match="not valid for family='throwaway'"):
        hb.create_model(
            "y ~ x1", family="throwaway", data=throwaway_data,
            config=default_config, aux_args={"scl": "scale"},
        )


def test_new_family_domain_check_is_reached(
    throwaway_family, data_gaussian, default_config
):
    """The spec's own `response_check` runs — no per-family branch needed."""
    df = data_gaussian.assign(scale=2.0, size=10)  # y has negative values
    m = throwaway_family(
        "y ~ x1", "throwaway", df, default_config,
        scale_col="scale", size_col="size",
    )

    with pytest.raises(DataValidationError, match="must be positive"):
        m.check_data()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"squeeze": "yes"},
        {"aux_args": {"n": "n", "deff": "deff", "squeeze": "yes"}},
    ],
    ids=["signature", "aux_args"],
)
def test_squeeze_must_be_bool_through_either_door(kwargs, data_beta, default_config):
    """The type guard cannot be side-stepped by the generic channel.

    `squeeze` reaches the preprocessor as a plain truth test, so a stray
    string would enable the Smithson-Verkuilen transform *and* relax the
    response-domain check with it. The guard therefore runs after the
    `aux_args` merge, not on the named parameter alone.
    """
    with pytest.raises(TypeError, match="must be a bool"):
        hb.create_model(
            "y ~ x1", family="beta", data=data_beta,
            config=default_config, **kwargs,
        )
