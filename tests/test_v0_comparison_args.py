"""compare_models() argument validation — fast, no Bambi/ArviZ/MCMC.

Locks the P1-4 contract: compare_models() must fail fast on unsupported or
inconsistent arguments instead of silently ignoring them. These tests target
the pure-Python helpers ``_validate_compare_args``, ``_require_log_likelihood``
and ``_check_same_observations`` directly, so they run in the default
(non-slow) lane without a fitted model.

Run with:
    pytest tests/test_v0_comparison_args.py
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hbsaemp.diagnostics.comparison import (
    _SUPPORTED_METRICS,
    _check_same_observations,
    _require_log_likelihood,
    _validate_compare_args,
)


def _validate(**overrides):
    """Call _validate_compare_args with valid defaults, overriding as needed."""
    kwargs = dict(
        metrics=None,
        n_draws_ppc=100,
        run_prior_sensitivity=False,
        sensitivity_vars=None,
    )
    kwargs.update(overrides)
    return _validate_compare_args(**kwargs)


class TestValidateMetrics:

    def test_none_defaults_to_loo(self):
        assert _validate(metrics=None) == ("loo",)

    def test_loo_list_accepted(self):
        assert _validate(metrics=["loo"]) == ("loo",)

    def test_single_string_accepted(self):
        assert _validate(metrics="loo") == ("loo",)

    def test_case_insensitive(self):
        assert _validate(metrics=["LOO"]) == ("loo",)
        assert _validate(metrics="Loo") == ("loo",)

    def test_supported_metrics(self):
        assert _SUPPORTED_METRICS == frozenset({"loo", "bf"})

    def test_bf_accepted(self):
        assert _validate(metrics="bf") == ("bf",)
        assert _validate(metrics=["loo", "BF"]) == ("loo", "bf")

    def test_empty_list_rejected(self):
        with pytest.raises(ValueError, match="at least one metric"):
            _validate(metrics=[])

    def test_empty_tuple_rejected(self):
        with pytest.raises(ValueError, match="at least one metric"):
            _validate(metrics=())

    def test_waic_rejected(self):
        with pytest.raises(NotImplementedError, match="loo"):
            _validate(metrics=["waic"])

    def test_mixed_supported_and_unsupported_rejected(self):
        # A valid metric next to an invalid one must still raise — no partial run.
        with pytest.raises(NotImplementedError):
            _validate(metrics=["loo", "waic"])

    def test_unsupported_message_lists_the_offender(self):
        with pytest.raises(NotImplementedError, match="waic"):
            _validate(metrics=["WAIC"])


class TestValidateNDrawsPpc:

    def test_positive_int_accepted(self):
        _validate(n_draws_ppc=1)
        _validate(n_draws_ppc=500)

    def test_zero_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            _validate(n_draws_ppc=0)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            _validate(n_draws_ppc=-10)

    def test_float_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            _validate(n_draws_ppc=1.5)

    def test_string_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            _validate(n_draws_ppc="100")

    def test_bool_rejected(self):
        # bool is an int subclass; True would slip past a naive isinstance/>0 check.
        with pytest.raises(ValueError, match="positive integer"):
            _validate(n_draws_ppc=True)


class TestValidatePriorSensitivity:

    def test_run_prior_sensitivity_true_accepted(self):
        assert _validate(run_prior_sensitivity=True) == ("loo",)

    def test_run_prior_sensitivity_false_accepted(self):
        _validate(run_prior_sensitivity=False)

    def test_sensitivity_vars_with_run_accepted(self):
        _validate(run_prior_sensitivity=True, sensitivity_vars=["x1", "x2"])

    def test_sensitivity_vars_without_run_rejected(self):
        with pytest.raises(ValueError, match="run_prior_sensitivity"):
            _validate(sensitivity_vars=["x1", "x2"])

    def test_sensitivity_vars_empty_list_without_run_rejected(self):
        # The contract is "any non-None value", not "any non-empty value".
        with pytest.raises(ValueError):
            _validate(sensitivity_vars=[])

    def test_sensitivity_vars_none_accepted(self):
        assert _validate(sensitivity_vars=None) == ("loo",)


class FakeIdata:
    """Minimal stand-in exposing ``.groups`` like an ArviZ idata.

    ArviZ 1.1 maps idata to a DataTree where ``.groups`` is a tuple property;
    older InferenceData exposed ``.groups()`` as a method. ``callable_groups``
    switches between the two shapes so both code paths are exercised.
    """

    def __init__(self, groups, *, callable_groups=False):
        if callable_groups:
            self.groups = lambda: tuple(groups)
        else:
            self.groups = tuple(groups)


class TestRequireLogLikelihood:

    def test_present_passes(self):
        idata = FakeIdata(["posterior", "log_likelihood", "sample_stats"])
        _require_log_likelihood(idata, model_name="model_0")  # no raise

    def test_absent_raises(self):
        idata = FakeIdata(["posterior", "sample_stats"])
        with pytest.raises(ValueError, match="log_likelihood"):
            _require_log_likelihood(idata, model_name="model_0")

    def test_error_names_the_model(self):
        idata = FakeIdata(["posterior"])
        with pytest.raises(ValueError, match="model_3"):
            _require_log_likelihood(idata, model_name="model_3")

    def test_slash_prefixed_groups_handled(self):
        # ArviZ 1.1 DataTree paths are slash-prefixed; _idata_groups strips them.
        idata = FakeIdata(["/posterior", "/log_likelihood"])
        _require_log_likelihood(idata, model_name="model_0")  # no raise

    def test_legacy_callable_groups_handled(self):
        idata = FakeIdata(["posterior", "log_likelihood"], callable_groups=True)
        _require_log_likelihood(idata, model_name="model_0")  # no raise


def _observed(y) -> FakeIdata:
    """FakeIdata carrying an ``observed_data`` Dataset with response *y*."""
    idata = FakeIdata(["/posterior", "/observed_data"])
    idata.observed_data = xr.Dataset({"y": ("__obs__", np.asarray(y, dtype=float))})
    return idata


class TestCheckSameObservations:

    def test_identical_observations_pass(self):
        _check_same_observations([_observed([1, 2, 3]), _observed([1, 2, 3])])

    def test_same_length_different_values_rejected(self):
        # az.compare only checks the count; equal-length datasets must still fail.
        with pytest.raises(ValueError, match="model_1"):
            _check_same_observations([_observed([1, 2, 3]), _observed([1, 2, 4])])

    def test_different_length_rejected(self):
        with pytest.raises(ValueError, match="different observations"):
            _check_same_observations([_observed([1, 2, 3]), _observed([1, 2])])

    def test_missing_observed_data_rejected(self):
        with pytest.raises(ValueError, match="observed_data"):
            _check_same_observations([_observed([1, 2]), FakeIdata(["posterior"])])
