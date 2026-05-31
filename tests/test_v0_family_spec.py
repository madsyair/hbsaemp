"""FAMILY_SPECS single-source contract — no Bambi, no MCMC.

Locks the P1 invariant: family metadata (mean param, links, pipeline
fields, backend/preproc family) lives only in `FAMILY_SPECS` and is read by
`BaseModel`. These tests fail loudly if a subclass re-introduces a local
metadata constant or drifts from the spec.
"""
from __future__ import annotations

import pytest

import hbsaemp as hb
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
def test_backend_and_preproc_family_read_spec(family, data_for, default_config):
    m = _build(family, data_for, default_config)
    assert m._bambi_family() == FAMILY_SPECS[family].bambi_family
    assert m._preproc_family == FAMILY_SPECS[family].preproc_family


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
