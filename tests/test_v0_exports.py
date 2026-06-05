"""Top-level export surface (P0-2) — no Bambi, no MCMC.

Locks the package front door: symbols recommended in docstrings and the README
quick-start must be importable straight from `hbsaemp`, and the family registry
is reachable only through the read-only accessors (not the raw dict).
"""
from __future__ import annotations

import pytest

import hbsaemp as hb

_FRONT_DOOR = ("Prior", "FamilySpec", "list_families", "get_family_spec")


@pytest.mark.parametrize("name", _FRONT_DOOR)
def test_symbol_exported_from_front_door(name):
    assert hasattr(hb, name), f"hbsaemp is missing top-level {name!r}"
    assert name in hb.__all__, f"{name!r} missing from hbsaemp.__all__"


def test_list_families_sorted_and_complete():
    families = hb.list_families()
    assert families == sorted(families)
    assert set(families) == {"gaussian", "beta", "binomial"}


def test_get_family_spec_returns_spec():
    spec = hb.get_family_spec("gaussian")
    assert isinstance(spec, hb.FamilySpec)
    assert spec.mean_param_key == "mu"


def test_get_family_spec_unknown_raises():
    with pytest.raises(hb.ModelRegistryError):
        hb.get_family_spec("poisson")


def test_raw_family_specs_dict_stays_a_subpackage_detail():
    # Decision: accessors are the front door; the mutable FAMILY_SPECS dict is
    # not promoted to the top-level namespace (still on hbsaemp.models).
    assert "FAMILY_SPECS" not in hb.__all__
    from hbsaemp.models import FAMILY_SPECS  # still reachable for advanced use

    assert hb.get_family_spec("beta") is FAMILY_SPECS["beta"]
