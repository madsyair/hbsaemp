from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import hbsaemp as hb

pn = pytest.importorskip("panel")
pytestmark = pytest.mark.gui

from hbsaemp.app._app import App, AppState  # noqa: E402
from hbsaemp.app.tabs import DataTab, ExploreTab, ModelTab, ResultsTab, UpdateModelTab  # noqa: E402
from hbsaemp.app.tabs.model_tab import _HBM_DISPATCH  # noqa: E402

APP_DIR = Path(__file__).resolve().parent.parent / "hbsaemp" / "app"


# ---------------------------------------------------------------------------
# AppConfig 
# ---------------------------------------------------------------------------

def test_app_config_defaults():
    cfg = hb.DEFAULT_APP_CONFIG
    assert cfg.port == 8080
    assert cfg.open_browser is True


def test_app_config_custom():
    cfg = hb.AppConfig(port=9000, title="Test", open_browser=False)
    assert cfg.port == 9000


@pytest.mark.parametrize("bad", [
    {"port": 0}, {"port": 70000}, {"max_upload_mb": 0}, {"log_level": "VERBOSE"},
    {"theme": "material"},
])
def test_app_config_validation(bad):
    with pytest.raises(ValueError):
        hb.AppConfig(**bad)


# ---------------------------------------------------------------------------
# App / AppState 
# ---------------------------------------------------------------------------

def test_app_state_is_not_a_dict():
    """AppState is a param.Parameterized with `data`/`model` attributes."""
    state = AppState()
    assert state.data is None
    assert state.model is None
    state.data = pd.DataFrame({"y": [1.0]})
    assert state.data is not None


def test_app_no_panel_needed_to_construct():
    """Constructing App works without a running server (no pn.serve() call)."""
    app = App(app_config=hb.AppConfig(port=9999, open_browser=False))
    assert app.state.data is None
    assert app.state.model is None


def test_app_view_returns_template():
    app = App(app_config=hb.AppConfig(port=9998, open_browser=False))
    template = app.view()
    assert isinstance(template, pn.template.FastListTemplate)


def test_app_view_respects_show_sidebar():
    app = App(app_config=hb.AppConfig(port=9997, open_browser=False, show_sidebar=False))
    template = app.view()
    assert len(template.sidebar) == 0


def test_launch_app_is_callable():
    """v1 no longer raises NotImplementedError — it's a thin App(...).serve() wrapper."""
    assert callable(hb.launch_app)
    assert not isinstance(hb.launch_app, type(NotImplemented))


# ---------------------------------------------------------------------------
# Architectural contract
# ---------------------------------------------------------------------------

def test_no_bambi_in_app():
    for path in APP_DIR.rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import bambi|from bambi)", src, re.M), path
        assert not re.search(r"^\s*(import pymc|from pymc)", src, re.M), path
        assert "bmb.Model(" not in src, path


def test_results_tab_uses_estimate_areas():
    src = (APP_DIR / "tabs" / "results_tab.py").read_text(encoding="utf-8")
    assert "posterior_predictive" not in src
    assert "estimate_areas" in src
    assert "check_convergence" in src
    assert "az.summary" not in src
    assert "az.plot_trace" not in src


def test_model_tab_uses_tier3_dispatch():
    src = (APP_DIR / "tabs" / "model_tab.py").read_text(encoding="utf-8")
    assert "hbm_gaussian" in src and "hbm_beta" in src and "hbm_binomial" in src
    assert "def _build_bambi_model" not in src
    assert "def _validate_build" not in src
    assert "threading.Thread" not in src


# ---------------------------------------------------------------------------
# Family/link/extra-param registry consistency (guards against drift)
# ---------------------------------------------------------------------------

def test_dispatch_covers_every_family():
    assert set(_HBM_DISPATCH) == set(hb.list_families())


@pytest.mark.parametrize("family", hb.list_families())
def test_link_options_match_spec(family):
    spec = hb.get_family_spec(family)
    state = AppState()
    tab = ModelTab(state=state)
    tab._family_sel.value = family
    assert set(tab._link_sel.options) == set(spec.supported_links)
    assert tab._link_sel.value == spec.default_link


@pytest.mark.parametrize("family", hb.list_families())
def test_extra_params_match_spec(family):
    spec = hb.get_family_spec(family)
    state = AppState()
    tab = ModelTab(state=state)
    tab._family_sel.value = family
    assert set(tab._extra_widgets) == set(spec.user_params.values()) - {"link"}


# ---------------------------------------------------------------------------
# ModelTab build contract
# ---------------------------------------------------------------------------

def test_build_model_returns_unfitted_basemodel(data_gaussian: pd.DataFrame):
    state = AppState()
    state.data = data_gaussian
    tab = ModelTab(state=state)
    tab._response_sel.value = "y"
    tab._predictors_sel.value = ["x1", "x2"]
    tab._family_sel.value = "gaussian"

    model = tab._build_model()

    assert isinstance(model, hb.BaseModel)
    assert model.is_fitted is False
    with pytest.raises(hb.ModelNotFittedError):
        _ = model.result


def test_formula_preview_updates_on_selection(data_gaussian: pd.DataFrame):
    state = AppState()
    state.data = data_gaussian
    tab = ModelTab(state=state)
    tab._response_sel.value = "y"
    tab._predictors_sel.value = ["x1", "x2"]
    assert tab._model_draft is not None
    assert "y ~" in tab._model_draft.formula


def test_binomial_requires_trials_widget(data_binomial: pd.DataFrame):
    state = AppState()
    state.data = data_binomial
    tab = ModelTab(state=state)
    tab._family_sel.value = "binomial"
    tab._response_sel.value = "y"
    tab._predictors_sel.value = ["x1", "x2"]
    assert tab._model_draft is None

    tab._extra_widgets["trials"].value = "n"
    tab._update_preview()
    assert tab._model_draft is not None
    assert tab._model_draft.is_fitted is False


# ---------------------------------------------------------------------------
# Tab construction smoke tests 
# ---------------------------------------------------------------------------

def test_all_tabs_construct_and_render():
    state = AppState()
    for cls in (DataTab, ExploreTab, ModelTab, ResultsTab, UpdateModelTab):
        tab = cls(state=state)
        assert tab.panel() is not None


# ---------------------------------------------------------------------------
# UpdateModelTab 
# ---------------------------------------------------------------------------

def test_update_tab_requires_fitted_model():
    """No fitted model yet -> clicking Update Model must not call update_model()."""
    import asyncio

    state = AppState()
    tab = UpdateModelTab(state=state)
    asyncio.run(tab._on_update(None))
    assert "No fitted model" in tab._update_status.object


@pytest.mark.slow
def test_update_tab_reacts_to_first_fit(data_gaussian: pd.DataFrame):
    """Fitting in ModelTab (not this tab) must refresh the summary via the
    state.model watcher — no manual tab switch/refresh should be required."""
    state = AppState()
    state.data = data_gaussian
    tab = UpdateModelTab(state=state)
    assert "No fitted model" in tab._current_info.object

    mt = ModelTab(state=state)
    mt._response_sel.value = "y"
    mt._predictors_sel.value = ["x1", "x2"]
    mt._draws_in.value = 20
    mt._tune_in.value = 20
    mt._chains_in.value = 1
    mt._cores_in.value = 1
    model = mt._build_model()
    model.fit()
    state.model = model

    assert "No fitted model" not in tab._current_info.object
    assert model.formula in tab._current_info.object


@pytest.mark.slow
def test_update_tab_refits_in_place(data_gaussian: pd.DataFrame):
    import asyncio

    state = AppState()
    state.data = data_gaussian

    mt = ModelTab(state=state)
    mt._response_sel.value = "y"
    mt._predictors_sel.value = ["x1", "x2"]
    mt._group_sel.value = "group"
    mt._family_sel.value = "gaussian"
    mt._draws_in.value = 50
    mt._tune_in.value = 50
    mt._chains_in.value = 1
    mt._cores_in.value = 1
    asyncio.run(mt._on_fit_model(None))
    assert state.model is not None and state.model.is_fitted

    model_before = state.model
    original_target_accept = model_before.config.target_accept

    ut = UpdateModelTab(state=state)
    ut._target_accept_cb.value = True
    ut._target_accept_in.value = 0.95
    asyncio.run(ut._on_update(None))

    assert "refit complete" in ut._update_status.object.lower()
    assert state.model is model_before  
    assert state.model.config.target_accept == 0.95
    assert state.model.config.target_accept != original_target_accept


# ---------------------------------------------------------------------------
# End-to-end: DataTab -> ModelTab -> ResultsTab, one real fit
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_end_to_end_fit_and_sae(data_gaussian: pd.DataFrame):
    state = AppState()
    state.data = data_gaussian

    mt = ModelTab(state=state)
    mt._response_sel.value = "y"
    mt._predictors_sel.value = ["x1", "x2"]
    mt._group_sel.value = "group"
    mt._family_sel.value = "gaussian"
    mt._draws_in.value = 100
    mt._tune_in.value = 100
    mt._chains_in.value = 2
    mt._cores_in.value = 1

    model = mt._build_model()
    assert model.is_fitted is False

    fitted = mt._fit_blocking(model)
    assert fitted.is_fitted is True
    state.model = fitted

    rt = ResultsTab(state=state)
    sae = rt._sae_blocking(state.model)
    assert set(sae.result_table.columns) >= {
        "mean", "sd", "ci_lower", "ci_upper", "rse_pct", "mse", "rmse",
    }

    conv, warnings_ = rt._convergence_blocking(state.model)
    assert conv.rhat_ess is not None
    assert isinstance(warnings_, list)


# ---------------------------------------------------------------------------
# Code export (`ModelTab.to_code()`) — GUI to CLI code correctness
# ---------------------------------------------------------------------------

def test_to_code_disabled_without_valid_model_draft(data_binomial: pd.DataFrame):
    """No `_model_draft` (e.g. binomial missing its required `trials`
    widget) -> nothing safe to export yet."""
    state = AppState()
    state.data = data_binomial
    tab = ModelTab(state=state)
    tab._family_sel.value = "binomial"
    tab._response_sel.value = "y"
    tab._predictors_sel.value = ["x1", "x2"]

    assert tab._model_draft is None
    assert tab._code_view.value == ""
    assert tab._code_download_btn.disabled is True


@pytest.mark.parametrize("family", hb.list_families())
def test_to_code_reproduces_gui_model_draft(
    family, data_gaussian, data_beta, data_binomial,
):
    """Exec the generated snippet and check it builds the *same* formula as
    the model the GUI itself built (`_model_draft`) — the actual proof that
    the exported code matches GUI behaviour, not just that it looks plausible.
    """
    data = {"gaussian": data_gaussian, "beta": data_beta, "binomial": data_binomial}[family]

    state = AppState()
    state.data = data
    tab = ModelTab(state=state)
    tab._family_sel.value = family
    tab._response_sel.value = "y"
    tab._predictors_sel.value = ["x1", "x2"]
    tab._group_sel.value = "group"

    if family == "binomial":
        tab._extra_widgets["trials"].value = "n"
        tab._update_preview()
    elif family == "beta":
        tab._extra_widgets["n"].value = "n"
        tab._extra_widgets["deff"].value = "deff"
        tab._update_preview()

    assert tab._model_draft is not None, f"GUI could not build a {family} model draft"
    assert tab._code_download_btn.disabled is False

    code = tab.to_code(include_fit=False)
    code = code.replace('data = pd.read_csv("your_data.csv")', "pass")

    namespace: dict[str, Any] = {"data": data}
    exec(code, namespace)  
    model_from_code = namespace["model"]

    assert model_from_code.formula == tab._model_draft.formula
    assert model_from_code.is_fitted is False


def test_to_code_matches_build_kwargs_single_source_of_truth(data_gaussian: pd.DataFrame):
    """`to_code()` must read from `_collect_build_kwargs()` — the same dict
    `_build_model()` uses — never a hand-duplicated copy of it. This guards
    against the two silently drifting apart after a future edit.
    """
    state = AppState()
    state.data = data_gaussian
    tab = ModelTab(state=state)
    tab._response_sel.value = "y"
    tab._predictors_sel.value = ["x1", "x2"]
    tab._family_sel.value = "gaussian"

    for key, value in tab._collect_build_kwargs().items():
        if key == "config":
            continue
        assert f"{key}={value!r}" in tab.to_code(), f"{key} missing/mismatched in generated code"


def test_to_code_includes_estimate_areas_by_default(data_gaussian: pd.DataFrame):
    """Default export mirrors the full GUI workflow: build -> fit -> estimate_areas."""
    state = AppState()
    state.data = data_gaussian
    tab = ModelTab(state=state)
    tab._response_sel.value = "y"
    tab._predictors_sel.value = ["x1", "x2"]

    code = tab.to_code()
    assert "model.fit()" in code
    assert "estimate_areas(model, ci_prob=0.95)" in code

    code_no_fit = tab.to_code(include_fit=False)
    assert "model.fit()" not in code_no_fit
    assert "estimate_areas" not in code_no_fit


def test_code_download_callback_returns_matching_bytes(data_gaussian: pd.DataFrame):
    """`FileDownload`'s callback (what the user actually saves to disk) must
    return the same content shown in the code viewer."""
    state = AppState()
    state.data = data_gaussian
    tab = ModelTab(state=state)
    tab._response_sel.value = "y"
    tab._predictors_sel.value = ["x1", "x2"]

    downloaded = tab._get_code_bytes().getvalue().decode("utf-8")
    assert downloaded == tab._code_view.value == tab.to_code()
