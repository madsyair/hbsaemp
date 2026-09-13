"""Wiring & delegation tests for tier 2/3 API — no Bambi, no MCMC.

Covers `hbm_flex()` (tier 2) and `hbm_{beta,gaussian,binomial}()`
(tier 3). MCMC behaviour is already exercised by `tests/test_v1_*.py` —
this file only verifies formula construction, validation, and delegation
to `create_model()`.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import hbsaemp as hb
from hbsaemp.models._family_spec import FAMILY_SPECS

# Public API surface

def test_tier_exports_present():
    for name in ("hbm_flex", "hbm_beta", "hbm_gaussian", "hbm_binomial"):
        assert hasattr(hb, name), f"Missing top-level export: {name}"


def test_lognormal_shortcut_not_exported_in_v1():
    assert not hasattr(hb, "hbm_lognormal")


def test_tiers_not_aliased_to_create_model():
    """Tier 2/3 are wrappers, not the same callable as hbm()/create_model()."""
    assert hb.hbm_flex is not hb.create_model
    assert hb.hbm_beta is not hb.create_model


# hbm_flex — formula construction

class TestHbmFlexFormula:

    def test_builds_formula_from_response_and_auxiliary(self, data_gaussian, default_config):
        m = hb.hbm_flex("y", ["x1", "x2"], data_gaussian,
                        family="gaussian", config=default_config)
        assert m.formula == "y ~ x1 + x2"

    def test_intercept_false_emits_zero(self, data_gaussian, default_config):
        m = hb.hbm_flex("y", ["x1", "x2"], data_gaussian,
                        family="gaussian", intercept=False,
                        config=default_config)
        assert m.formula == "y ~ 0 + x1 + x2"

    def test_area_var_injects_random_intercept(self, data_gaussian, default_config):
        m = hb.hbm_flex("y", ["x1"], data_gaussian,
                        family="gaussian", area_var="group",
                        config=default_config)
        assert "(1|group)" in m.formula

    def test_single_auxiliary_works(self, data_gaussian, default_config):
        m = hb.hbm_flex("y", ["x1"], data_gaussian,
                        family="gaussian", config=default_config)
        assert m.formula == "y ~ x1"


# hbm_flex — validation

class TestHbmFlexValidation:

    def test_empty_auxiliary_raises(self, data_gaussian, default_config):
        with pytest.raises(ValueError, match="non-empty"):
            hb.hbm_flex("y", [], data_gaussian,
                        family="gaussian", config=default_config)

    def test_string_auxiliary_raises(self, data_gaussian, default_config):
        # A bare string would silently iterate per character — reject upfront.
        with pytest.raises(TypeError, match="sequence"):
            hb.hbm_flex("y", "x1", data_gaussian,
                        family="gaussian", config=default_config)

    def test_non_identifier_response_raises(self, data_gaussian, default_config):
        with pytest.raises(ValueError, match="response"):
            hb.hbm_flex("np.log(y)", ["x1"], data_gaussian,
                        family="gaussian", config=default_config)

    def test_non_identifier_auxiliary_raises(self, data_gaussian, default_config):
        with pytest.raises(ValueError, match=r"auxiliary\[1\]"):
            hb.hbm_flex("y", ["x1", "np.log(x2)"], data_gaussian,
                        family="gaussian", config=default_config)

    def test_non_identifier_area_var_raises(self, data_gaussian, default_config):
        with pytest.raises(ValueError, match="area_var"):
            hb.hbm_flex("y", ["x1"], data_gaussian,
                        family="gaussian", area_var="grp(1)",
                        config=default_config)


# hbm_flex — delegation to create_model

class TestHbmFlexDelegation:

    def test_addition_var_forwarded(self, data_binomial, default_config):
        """`addition_var` reaches the family's own field without naming it."""
        m = hb.hbm_flex("y", ["x1"], data_binomial,
                        family="binomial", addition_var="n",
                        config=default_config)
        assert type(m).__name__ == "BinomialModel"
        assert m._trials_col == "n"

    def test_aux_args_forwarded(self, data_beta, default_config):
        m = hb.hbm_flex("y", ["x1"], data_beta,
                        family="beta", aux_args={"n": "n", "deff": "deff"},
                        config=default_config)
        assert m._n_col == "n" and m._deff_col == "deff"

    def test_family_names_are_not_in_the_tier2_signature(self, data_beta,
                                                         default_config):
        """Tier 2 knows no family vocabulary — that is the point of it.

        Adding a family must not touch this signature, so the per-family
        names live one tier up (`_shortcuts.py`) and reach tier 2 only
        through `aux_args` / `addition_var`.
        """
        import inspect
        params = set(inspect.signature(hb.hbm_flex).parameters)
        assert not params & {"n", "deff", "squeeze", "trials"}
        with pytest.raises(TypeError, match="unexpected keyword argument 'n'"):
            hb.hbm_flex("y", ["x1"], data_beta, family="beta", n="n",
                        config=default_config)

    def test_cross_family_aux_args_rejected(self, data_gaussian, default_config):
        # Validation still happens centrally, in create_model.
        with pytest.raises(ValueError, match="not valid for family='gaussian'"):
            hb.hbm_flex("y", ["x1"], data_gaussian,
                        family="gaussian", aux_args={"trials": "n"},
                        config=default_config)

    def test_addition_var_rejected_when_family_takes_none(self, data_gaussian,
                                                          default_config):
        with pytest.raises(ValueError, match="takes no addition variable"):
            hb.hbm_flex("y", ["x1"], data_gaussian,
                        family="gaussian", addition_var="n",
                        config=default_config)

    def test_unknown_family_raises(self, data_gaussian, default_config):
        with pytest.raises(hb.ModelRegistryError):
            hb.hbm_flex("y", ["x1"], data_gaussian,
                        family="tweedie", config=default_config)

    def test_returns_unfitted_model(self, data_gaussian, default_config):
        m = hb.hbm_flex("y", ["x1"], data_gaussian,
                        family="gaussian", config=default_config)
        assert isinstance(m, hb.BaseModel)
        assert not m.is_fitted


# hbm_<specific> — family hardcoding

class TestShortcutHardcoding:

    def test_hbm_gaussian_sets_family(self, data_gaussian, default_config):
        m = hb.hbm_gaussian("y", ["x1"], data_gaussian, config=default_config)
        assert m.family == "gaussian"
        assert type(m).__name__ == "GaussianModel"

    def test_hbm_beta_sets_family(self, data_beta, default_config):
        m = hb.hbm_beta("y", ["x1"], data_beta,
                        n="n", deff="deff", config=default_config)
        assert m.family == "beta"

    def test_hbm_binomial_sets_family(self, data_binomial, default_config):
        m = hb.hbm_binomial("y", ["x1"], data_binomial,
                             trials="n", config=default_config)
        assert m.family == "binomial"


# hbm_<specific> — keyword surface

class TestShortcutSignatures:

    def test_hbm_binomial_requires_trials(self, data_binomial, default_config):
        # trials= is keyword-only with no default → Python TypeError before
        # _validate_family_args sees it.
        with pytest.raises(TypeError):
            hb.hbm_binomial("y", ["x1"], data_binomial, config=default_config)

    def test_hbm_gaussian_rejects_trials_kwarg(self, data_gaussian, default_config):
        with pytest.raises(TypeError):
            hb.hbm_gaussian("y", ["x1"], data_gaussian,
                            trials="n", config=default_config)  # type: ignore[call-arg]

    def test_hbm_gaussian_rejects_n_deff(self, data_gaussian, default_config):
        with pytest.raises(TypeError):
            hb.hbm_gaussian("y", ["x1"], data_gaussian,
                            n="n", deff="deff",  # type: ignore[call-arg]
                            config=default_config)

    def test_hbm_beta_rejects_sampling_var(self, data_beta, default_config):
        with pytest.raises(TypeError):
            hb.hbm_beta("y", ["x1"], data_beta,
                        n="n", deff="deff",
                        sampling_var="deff",  # type: ignore[call-arg]
                        config=default_config)

    @pytest.mark.parametrize("family", sorted(FAMILY_SPECS))
    def test_fixed_params_exposed_iff_family_pins_something(self, family):
        """Tier 3 takes `fixed_params=` exactly when the family can pin.

        Binomial's only parameter is the estimand, so the argument could only
        ever raise `ValueError` downstream — a `TypeError` at the call is the
        better failure. Derived from the spec so adding a pinnable family and
        forgetting its shortcut argument fails here, not in a user's script.
        """
        shortcut = getattr(hb, f"hbm_{family}")
        accepts = "fixed_params" in inspect.signature(shortcut).parameters

        assert accepts is bool(FAMILY_SPECS[family].pinnable_params)

    def test_hbm_gaussian_forwards_fixed_params(self, data_gaussian, default_config):
        m = hb.hbm_gaussian("y", ["x1"], data_gaussian, config=default_config,
                            fixed_params={"sigma": 2.0})
        assert tuple(fp.param for fp in m._active_fixed_params()) == ("sigma",)

    def test_hbm_beta_forwards_fixed_params(self, data_beta, default_config):
        m = hb.hbm_beta("y", ["x1"], data_beta, config=default_config,
                        fixed_params={"kappa": 49.0})
        assert tuple(fp.param for fp in m._active_fixed_params()) == ("kappa",)


# hbm_<specific> — area_var pass-through

class TestShortcutAreaVar:

    def test_hbm_gaussian_area_var(self, data_gaussian, default_config):
        m = hb.hbm_gaussian("y", ["x1"], data_gaussian,
                            area_var="group", config=default_config)
        assert "(1|group)" in m.formula

    def test_hbm_beta_area_var_and_squeeze(self, data_beta, default_config):
        m = hb.hbm_beta("y", ["x1"], data_beta,
                        n="n", deff="deff", squeeze=True,
                        area_var="group", config=default_config)
        assert "(1|group)" in m.formula
        assert m._squeeze is True


# Delegation chain — hbm_<specific> goes through hbm_flex

def test_delegation_chain_consistency(data_beta, default_config):
    """hbm_beta(...) must yield the same model spec as the equivalent
    hbm_flex(...) call — confirms tier 3 routes through tier 2."""
    m_specific = hb.hbm_beta("y", ["x1", "x2"], data_beta,
                              n="n", deff="deff", area_var="group",
                              config=default_config)
    m_flex = hb.hbm_flex("y", ["x1", "x2"], data_beta,
                          family="beta", area_var="group",
                          aux_args={"n": "n", "deff": "deff"},
                          config=default_config)
    assert type(m_specific) is type(m_flex)
    assert m_specific.formula == m_flex.formula
    assert m_specific.family == m_flex.family
    assert m_specific._n_col == m_flex._n_col
    assert m_specific._deff_col == m_flex._deff_col


# Bambi handoff containment — single contact point in BaseModel

def test_bambi_handoff_only_in_base():
    """Lock the handoff contract: only `_base.py` may import `bambi`.

    Family subclasses receive the `bmb` module as a hook argument from
    `BaseModel.fit()`; tier 2/3 and the factory stay Bambi-free. Any other
    `import bambi` under `models/` bypasses the single handoff point.
    """
    models_dir = Path(hb.models.__file__).parent
    bambi_import = re.compile(r"^\s*(?:import bambi|from bambi)\b", re.MULTILINE)
    offenders = sorted(
        path.name
        for path in models_dir.glob("*.py")
        if path.name != "_base.py"
        and bambi_import.search(path.read_text(encoding="utf-8"))
    )
    assert offenders == [], (
        f"`import bambi` found outside _base.py: {offenders} — "
        f"the Bambi handoff lives only in BaseModel."
    )
