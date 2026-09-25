"""Tests for the bundled real datasets (Papua case study) — no Bambi, no MCMC.

Both frames are shipped as published, defects included: the tutorial's data
preparation step is where they get fixed, so the tests pin the defects too.
"""
from __future__ import annotations

import pandas as pd
import pytest

import hbsaemp as hb

SUSENAS = "susenas2023_papua"
PODES = "podes2021_papua"


def test_real_datasets_listed_apart_from_synthetic():
    assert hb.REAL_DATASETS == [SUSENAS, PODES]
    # The GUI dataset selector reads AVAILABLE_DATASETS; it must stay synthetic.
    assert not set(hb.REAL_DATASETS) & set(hb.AVAILABLE_DATASETS)


def test_susenas_shape_and_columns():
    df = hb.load_dataset(SUSENAS)
    assert df.shape == (42, 6)
    assert list(df.columns) == ["idkab", "nama_kab", "nama_prov", "morbidity_rate", "se", "rse"]
    assert df["idkab"].is_monotonic_increasing
    assert set(df["nama_prov"]) == {"Papua", "Papua Barat"}


def test_susenas_ships_its_published_defects():
    df = hb.load_dataset(SUSENAS).set_index("nama_kab")
    # Deiyai has no published rate or SE, only an RSE.
    assert pd.isna(df.loc["Deiyai", "morbidity_rate"]) and pd.isna(df.loc["Deiyai", "se"])
    assert df.loc["Deiyai", "rse"] == pytest.approx(60.63)
    # Kota Jayapura is published under 9437, not the official 9471.
    assert df.loc["Kota Jayapura", "idkab"] == 9437
    rate = df["morbidity_rate"].dropna()
    assert len(rate) == 41 and rate.between(0, 100, inclusive="neither").all()


def test_podes_shape_and_values():
    df = hb.load_dataset(PODES)
    assert df.shape == (42, 18)
    assert [f"X{i}" for i in range(1, 11)] == list(df.columns[-10:])
    fakfak = df.set_index("nama_kab").loc["Fakfak"]
    assert fakfak["n_desa"] == 149
    assert fakfak["X1"] == pytest.approx(98.657718)


def test_keys_match_once_jayapura_is_recoded():
    susenas = hb.load_dataset(SUSENAS)["idkab"].replace({9437: 9471})
    assert set(susenas) == set(hb.load_dataset(PODES)["idkab"])


def test_unknown_name_lists_both_catalogues():
    with pytest.raises(ValueError, match="susenas2023_papua") as err:
        hb.load_dataset("nonexistent_dataset")
    assert "data_fhnorm" in str(err.value)
