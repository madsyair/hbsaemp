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
    """`<=`, not `==` — a future family without a hbm_<family> shortcut
    should degrade gracefully (hidden from the selector, S1) rather than
    fail this test; `_UNSUPPORTED_FAMILIES` is where that gets logged."""
    assert set(_HBM_DISPATCH) <= set(hb.list_families())


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
    sae = rt._sae_blocking(state.model, 0.95, None)
    assert set(sae.result_table.columns) >= {
        "mean", "sd", "ci_lower", "ci_upper", "rse_pct", "mse", "rmse",
    }

    conv = rt._convergence_blocking(state.model)
    assert conv.rhat_ess is not None


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


def test_to_code_uses_result_table_not_estimates(data_gaussian: pd.DataFrame):
    """Regression test: the generated script used to reference
    `result.estimates`, which doesn't exist on `AreaEstimatesResult`
    (`result_table` does) — the generated code would crash the moment
    anyone actually ran it. Caught by running it for real, not just
    checking the string."""
    state = AppState()
    state.data = data_gaussian
    tab = ModelTab(state=state)
    tab._response_sel.value = "y"
    tab._predictors_sel.value = ["x1", "x2"]

    code = tab.to_code()
    assert "result.estimates" not in code
    assert "result.result_table" in code


# ---------------------------------------------------------------------------
# matplotlib.use("Agg")
# ---------------------------------------------------------------------------

def test_matplotlib_use_agg_not_at_tab_import_time():
    """Setting the matplotlib backend at tab-import time would silently
    hijack it for anything that imports these modules (e.g. a notebook).
    It belongs in App.serve(), scoped to an actually-running dashboard."""
    for name in ("model_tab.py", "explore_tab.py"):
        src = (APP_DIR / "tabs" / name).read_text(encoding="utf-8")
        assert 'matplotlib.use("Agg")' not in src, name

    app_src = APP_DIR.joinpath("_app.py").read_text(encoding="utf-8")
    assert 'matplotlib.use("Agg")' in app_src


# ---------------------------------------------------------------------------
# Save Model + Model Comparison 
# ---------------------------------------------------------------------------

def test_save_button_disabled_until_fit(data_gaussian: pd.DataFrame):
    state = AppState()
    state.data = data_gaussian
    tab = ModelTab(state=state)
    tab._response_sel.value = "y"
    tab._predictors_sel.value = ["x1", "x2"]
    assert tab._save_btn.disabled is True


@pytest.mark.slow
def test_save_model_stores_independent_snapshot(data_gaussian: pd.DataFrame):
    """A saved model must not be affected by a later refit of the active
    model — `update_model()` mutates `state.model`'s attributes in place,
    so the saved entry has to be a separate object, not the same
    reference."""
    import asyncio

    state = AppState()
    state.data = data_gaussian
    mt = ModelTab(state=state)
    mt._response_sel.value = "y"
    mt._predictors_sel.value = ["x1", "x2"]
    mt._draws_in.value = 200
    mt._tune_in.value = 200
    mt._chains_in.value = 2
    mt._cores_in.value = 1
    asyncio.run(mt._on_fit_model(None))
    assert mt._save_btn.disabled is False

    mt._save_name_in.value = "Model A"
    asyncio.run(mt._on_save_model(None))

    assert "Model A" in state.saved_models
    saved = state.saved_models["Model A"]
    assert saved.model is not state.model  # independent object, not a shared reference
    original_target_accept = saved.model.config.target_accept

    from hbsaemp import update_model as _update_model
    _update_model(state.model, target_accept=0.5, draws=50, tune=50)

    assert saved.model.config.target_accept == original_target_accept
    assert saved.model.config.target_accept != state.model.config.target_accept


def test_sampler_config_change_reflected_in_saved_model(data_gaussian: pd.DataFrame):
    """Regression test: sampler widgets touched *after* the formula
    preview already rendered must still reach `_model_draft` — this was
    silently dropped once before (draws/chains stayed at the widget
    defaults no matter what was typed in afterward)."""
    state = AppState()
    state.data = data_gaussian
    tab = ModelTab(state=state)
    tab._response_sel.value = "y"
    tab._predictors_sel.value = ["x1", "x2"]
    assert tab._model_draft.config.draws == 1000  # still the widget default

    tab._draws_in.value = 77
    tab._chains_in.value = 1
    assert tab._model_draft.config.draws == 77
    assert tab._model_draft.config.chains == 1


def test_model_comparison_table_reflects_saved_models():
    from hbsaemp.app._app import SavedModel

    state = AppState()
    rt = ResultsTab(state=state)
    assert rt._compare_table.value.empty

    state.saved_models = {
        "Model A": SavedModel(model=object()),
        "Model B": SavedModel(model=object()),
    }
    table = rt._compare_table.value
    assert list(table["Model"]) == ["Model A", "Model B"]
    assert "Converged" not in table.columns  # no verdict — user reads Convergence Evaluation themselves
    assert rt._use_model_sel.options == [None, "Model A", "Model B"]


def test_compare_table_model_column_is_never_numeric_dtype():
    """Regression test: an empty pd.DataFrame({"Model": []}) defaults to
    float64 (pandas' rule for a column with nothing to infer a type
    from). Tabulator.js then treats "Model" as a numeric column, and a
    later string value renders as NaN in the browser instead of the
    model's name — this only shows up in the rendered widget, not in
    plain `.value` access, so it's checked via the Bokeh model's actual
    ColumnDataSource here rather than just `.value.dtypes`."""
    from hbsaemp.app._app import SavedModel

    state = AppState()
    rt = ResultsTab(state=state)
    bokeh_model = rt._compare_table.get_root()

    assert rt._compare_table.value["Model"].dtype == object
    assert bokeh_model.source.data["Model"].dtype == object

    state.saved_models = {"Model 1": SavedModel(model=object())}
    assert rt._compare_table.value["Model"].dtype == object
    assert bokeh_model.source.data["Model"].dtype == object
    assert list(bokeh_model.source.data["Model"]) == ["Model 1"]


def test_compare_requires_at_least_one_model():
    """`compare_models(single_model, metrics=["bf"])` works fine on the
    backend (single-model LOO/Bayes Factor is a real, supported use
    case), so the GUI no longer forces 2+ just to reach it. Zero
    selected is still rejected."""
    state = AppState()
    rt = ResultsTab(state=state)
    rt._compare_table.selection = []
    import asyncio
    asyncio.run(rt._on_compare_click(None))
    assert "select a model" in rt._compare_status.object.lower()


@pytest.mark.slow
def test_compare_allows_single_model_for_its_own_diagnostics(data_gaussian: pd.DataFrame):
    """A single saved model can be "compared" against nothing — this
    just shows its own LOO/pp_check/params plots (and Bayes Factor, if
    enabled), with no ranking table/plot (those need 2+, and stay None)."""
    import asyncio

    state = AppState()
    state.data = data_gaussian
    mt = ModelTab(state=state)
    rt = ResultsTab(state=state)
    mt._response_sel.value = "y"
    mt._predictors_sel.value = ["x1"]
    mt._draws_in.value = 200
    mt._tune_in.value = 200
    mt._chains_in.value = 2
    mt._cores_in.value = 1
    asyncio.run(mt._on_fit_model(None))
    mt._save_name_in.value = "Solo"
    asyncio.run(mt._on_save_model(None))

    rt._compare_table.selection = [0]
    rt._compare_bf_cb.value = True
    asyncio.run(rt._on_compare_click(None))

    titles = [o.title for o in rt._compare_result_pane.objects if isinstance(o, pn.Card)]
    assert "Bayes Factor" in titles
    assert "Ranking (LOO / ELPD)" not in titles  # needs 2+, correctly absent


@pytest.mark.slow
def test_use_model_button_sets_independent_copy(data_gaussian: pd.DataFrame):
    import asyncio

    from hbsaemp.app._app import SavedModel

    state = AppState()
    state.data = data_gaussian
    mt = ModelTab(state=state)
    mt._response_sel.value = "y"
    mt._predictors_sel.value = ["x1", "x2"]
    mt._draws_in.value = 200
    mt._tune_in.value = 200
    mt._chains_in.value = 2
    mt._cores_in.value = 1
    asyncio.run(mt._on_fit_model(None))

    state.saved_models = {"A": SavedModel(model=mt.state.model)}
    rt = ResultsTab(state=state)
    rt._use_model_sel.value = "A"
    rt._on_use_model_click(None)

    assert state.model is not state.saved_models["A"].model  # fresh copy, not shared
    assert state.model.formula == state.saved_models["A"].model.formula


# ---------------------------------------------------------------------------
# ResultsTab: stale-result clearing, ci_prob, error mapping 
# ---------------------------------------------------------------------------

def test_results_clear_on_model_change():
    from types import SimpleNamespace

    state = AppState()
    rt = ResultsTab(state=state)
    rt._rhat_ess_table.value = pd.DataFrame({"a": [1]})
    rt._sae_table.value = pd.DataFrame({"b": [2]})
    rt._sae_download_btn.disabled = False
    rt._plots_download_btn.disabled = False

    # A minimal stand-in satisfying what both ResultsTab's own watcher and
    # the embedded UpdateModelTab's watcher need (is_fitted/formula/family/
    # config/data) — any change to state.model must fire both.
    fake_model = SimpleNamespace(
        is_fitted=True, formula="y ~ x1", family="gaussian",
        config=SimpleNamespace(draws=1, tune=1, chains=1, cores=1, target_accept=0.8),
        data=pd.DataFrame({"y": [1]}),
    )
    state.model = fake_model

    assert rt._rhat_ess_table.value.empty
    assert rt._sae_table.value.empty
    assert rt._sae_download_btn.disabled is True
    assert rt._plots_download_btn.disabled is True
    assert "changed" in rt._sae_status.object.lower()


def test_describe_error_maps_value_error():
    from hbsaemp.app.tabs.results_tab import _describe_error

    title, body = _describe_error(ValueError("ci_prob must be in (0, 1)"))
    assert title == "Invalid setting"
    assert "ci_prob" in body


def test_sae_blocking_passes_ci_prob_and_new_data(data_gaussian: pd.DataFrame, monkeypatch):
    captured = {}

    def fake_estimate_areas(model, *, new_data=None, ci_prob=0.95):
        captured["new_data"] = new_data
        captured["ci_prob"] = ci_prob
        class _Result:
            result_table = pd.DataFrame({"mean": [1.0]})
            mean_rse = 1.0
            mean_mse = 1.0
        return _Result()

    monkeypatch.setattr("hbsaemp.app.tabs.results_tab.estimate_areas", fake_estimate_areas)

    state = AppState()
    rt = ResultsTab(state=state)
    rt._sae_blocking(object(), 0.90, data_gaussian)
    assert captured["ci_prob"] == 0.90
    assert captured["new_data"] is data_gaussian


# ---------------------------------------------------------------------------
# Convergence plot titles 
# ---------------------------------------------------------------------------

def test_rhat_plot_title_is_distribution_not_forest():
    from hbsaemp.app.tabs.results_tab import _PLOT_TITLES

    assert _PLOT_TITLES["rhat"] == "R-hat Distribution Plot"
    assert "forest" not in _PLOT_TITLES["rhat"].lower()


@pytest.mark.slow
def test_comparison_table_uses_saved_names_not_generic_labels(data_gaussian: pd.DataFrame):
    """Regression test: compare_models() always labels models "model_0",
    "model_1", ... by list position (comparison.py has no way to accept
    custom names) — the rendered table/badges must translate those back
    to the names the user actually saved, not leak the generic labels."""
    import asyncio

    state = AppState()
    state.data = data_gaussian
    mt = ModelTab(state=state)
    rt = ResultsTab(state=state)

    mt._response_sel.value = "y"
    mt._predictors_sel.value = ["x1", "x2"]
    mt._draws_in.value = 800
    mt._tune_in.value = 800
    mt._chains_in.value = 4
    mt._cores_in.value = 1
    asyncio.run(mt._on_fit_model(None))
    mt._save_name_in.value = "Model 2"
    asyncio.run(mt._on_save_model(None))

    mt._predictors_sel.value = ["x1"]
    asyncio.run(mt._on_fit_model(None))
    mt._save_name_in.value = "Model 3"
    asyncio.run(mt._on_save_model(None))

    rt._compare_table.selection = [0, 1]
    asyncio.run(rt._on_compare_click(None))

    ranking_table = next(
        obj.objects[0] for obj in rt._compare_result_pane.objects
        if isinstance(obj, pn.Card) and obj.title == "Ranking (LOO / ELPD)"
    )
    labels = ranking_table.value["Model"].astype(str).tolist()
    assert set(labels) == {"Model 2", "Model 3"}
    assert not any("model_" in label for label in labels)


def test_bayes_factor_option_defaults_off():
    """Off by default — not because it's crash-prone (the backend's
    _bf_frame() correctly handles arviz>=1.2.0's xarray.Dataset shape now),
    but because it's an extra statistic most comparisons don't need, and
    it costs another prior_predictive_idata() sampling pass per model."""
    state = AppState()
    rt = ResultsTab(state=state)
    assert rt._compare_bf_cb.value is False


@pytest.mark.slow
def test_bayes_factor_rendered_with_saved_names_when_enabled(data_gaussian: pd.DataFrame):
    """`metrics=["loo", "bf"]` exercises the backend's `_bf_frame()`, which
    reshapes `az.bayes_factor()`'s `xarray.Dataset` return (arviz-stats
    >= 1.2.0, this project's pinned floor) into a BF10/BF01 table. This
    test checks the GUI's rendering (saved names, not "model_0"/"model_1")
    on top of that — it isn't a substitute for the backend's own tests of
    `_bf_frame()` itself."""
    import asyncio

    state = AppState()
    state.data = data_gaussian
    mt = ModelTab(state=state)
    rt = ResultsTab(state=state)

    mt._response_sel.value = "y"
    mt._predictors_sel.value = ["x1", "x2"]
    mt._draws_in.value = 800
    mt._tune_in.value = 800
    mt._chains_in.value = 4
    mt._cores_in.value = 1
    asyncio.run(mt._on_fit_model(None))
    mt._save_name_in.value = "Model 2"
    asyncio.run(mt._on_save_model(None))

    mt._predictors_sel.value = ["x1"]
    asyncio.run(mt._on_fit_model(None))
    mt._save_name_in.value = "Model 3"
    asyncio.run(mt._on_save_model(None))

    rt._compare_table.selection = [0, 1]
    rt._compare_bf_cb.value = True
    asyncio.run(rt._on_compare_click(None))

    bf_card = next(
        (obj for obj in rt._compare_result_pane.objects
         if isinstance(obj, pn.Card) and obj.title == "Bayes Factor"),
        None,
    )
    assert bf_card is not None, rt._compare_status.object

    rendered_names = set()
    for section in bf_card.objects[0].objects:
        if isinstance(section, pn.pane.Markdown):
            continue
        for child in section.objects:
            if isinstance(child, pn.pane.Markdown):
                rendered_names.add(child.object.strip("*"))
    assert rendered_names == {"Model 2", "Model 3"}
    assert not any("model_" in n for n in rendered_names)


# ---------------------------------------------------------------------------
# Update Model: max_treedepth, formula template, UserWarning capture
# ---------------------------------------------------------------------------

def test_update_tab_has_treedepth_override():
    state = AppState()
    tab = UpdateModelTab(state=state)
    tab._treedepth_cb.value = True
    tab._treedepth_in.value = 14
    assert tab._collect_overrides()["max_treedepth"] == 14


def test_update_tab_formula_override_included_when_set():
    state = AppState()
    tab = UpdateModelTab(state=state)
    assert "formula" not in tab._collect_overrides()
    tab._formula_tpl.value = ". ~ . + x3"
    assert tab._collect_overrides()["formula"] == ". ~ . + x3"


@pytest.mark.slow
def test_update_blocking_captures_user_warning(data_gaussian: pd.DataFrame, monkeypatch):
    import warnings as warnings_mod

    from hbsaemp.app._app import AppState as _AppState
    from hbsaemp.app.tabs import ModelTab as _ModelTab

    state = _AppState()
    state.data = data_gaussian
    mt = _ModelTab(state=state)
    mt._response_sel.value = "y"
    mt._predictors_sel.value = ["x1", "x2"]
    mt._draws_in.value = 50
    mt._tune_in.value = 50
    mt._chains_in.value = 1
    mt._cores_in.value = 1
    import asyncio
    asyncio.run(mt._on_fit_model(None))

    def fake_update_model(model, **kwargs):
        warnings_mod.warn("design column copied from original data", UserWarning, stacklevel=2)

    monkeypatch.setattr("hbsaemp.app.tabs.update_tab.update_model", fake_update_model)

    ut = UpdateModelTab(state=state)
    fitted, notes = ut._update_blocking(state.model, None, {})
    assert any("copied" in n for n in notes)


# ---------------------------------------------------------------------------
# Fit/refit progress feedback (progress bar + elapsed-time ticker)
# ---------------------------------------------------------------------------

def test_sampling_progress_callback_updates_progress_bar():
    from hbsaemp.app.tabs.model_tab import _make_sampling_progress_callback

    progress = pn.indicators.Progress(max=100, value=0)
    callback = _make_sampling_progress_callback(progress, total_draws=10)

    class _FakeDraw:
        chain = 0

    for _ in range(10):
        callback(None, _FakeDraw())

    assert progress.value == 100


@pytest.mark.slow
def test_fit_progress_bar_hidden_before_and_after(data_gaussian: pd.DataFrame):
    """The bar only shows *during* a fit — not before it starts, and not
    left dangling at 100% once it's done (the status text already
    confirms completion)."""
    import asyncio

    state = AppState()
    state.data = data_gaussian
    mt = ModelTab(state=state)
    mt._response_sel.value = "y"
    mt._predictors_sel.value = ["x1", "x2"]
    mt._draws_in.value = 50
    mt._tune_in.value = 50
    mt._chains_in.value = 1
    mt._cores_in.value = 1

    assert mt._fit_progress.visible is False
    asyncio.run(mt._on_fit_model(None))
    assert mt._fit_progress.visible is False
    assert mt._fit_progress.value == 100


@pytest.mark.slow
def test_update_progress_bar_hidden_before_and_after(data_gaussian: pd.DataFrame):
    import asyncio

    state = AppState()
    state.data = data_gaussian
    mt = ModelTab(state=state)
    mt._response_sel.value = "y"
    mt._predictors_sel.value = ["x1", "x2"]
    mt._draws_in.value = 50
    mt._tune_in.value = 50
    mt._chains_in.value = 1
    mt._cores_in.value = 1
    asyncio.run(mt._on_fit_model(None))

    ut = UpdateModelTab(state=state)
    assert ut._update_progress.visible is False
    asyncio.run(ut._on_update(None))
    assert ut._update_progress.visible is False
    assert ut._update_progress.value == 100


# ---------------------------------------------------------------------------
# fixed_params pinning
# ---------------------------------------------------------------------------

def test_pin_widgets_hidden_for_binomial(data_gaussian: pd.DataFrame):
    """pinnable_params == {} for binomial (p is the estimand, can't be
    pinned) — no pin widgets should render at all, not a control that
    always errors if used."""
    state = AppState()
    state.data = data_gaussian
    tab = ModelTab(state=state)
    tab._family_sel.value = "binomial"
    assert tab._pin_cbs == {}
    assert tab._pin_params_pane.objects == []


def test_pin_widgets_shown_for_gaussian_and_beta(data_gaussian: pd.DataFrame):
    state = AppState()
    state.data = data_gaussian
    tab = ModelTab(state=state)
    tab._family_sel.value = "gaussian"
    assert set(tab._pin_cbs) == {"sigma"}
    tab._family_sel.value = "beta"
    assert set(tab._pin_cbs) == {"kappa"}


def test_pinning_sigma_disables_sampling_var(data_gaussian: pd.DataFrame):
    """Backend rejects pinning a parameter that's also supplied via
    sampling_var/n+deff with ValueError ("pinned twice") — the GUI
    disables the other source instead of letting the user hit that."""
    state = AppState()
    state.data = data_gaussian
    tab = ModelTab(state=state)
    tab._family_sel.value = "gaussian"
    assert tab._extra_widgets["sampling_var"].disabled is False

    tab._pin_cbs["sigma"].value = True
    assert tab._extra_widgets["sampling_var"].disabled is True

    tab._pin_cbs["sigma"].value = False
    assert tab._extra_widgets["sampling_var"].disabled is False


def test_collect_family_kwargs_includes_fixed_params_only_when_pinned(
    data_gaussian: pd.DataFrame,
):
    state = AppState()
    state.data = data_gaussian
    tab = ModelTab(state=state)
    tab._family_sel.value = "gaussian"
    assert "fixed_params" not in tab._collect_family_kwargs()

    tab._pin_cbs["sigma"].value = True
    tab._pin_mode["sigma"].value = "Fixed value"
    tab._pin_val["sigma"].value = 2.0
    assert tab._collect_family_kwargs()["fixed_params"] == {"sigma": 2.0}

    tab._pin_mode["sigma"].value = "Column"
    tab._pin_col["sigma"].value = "D"
    assert tab._collect_family_kwargs()["fixed_params"] == {"sigma": "D"}


@pytest.mark.slow
def test_pinned_model_builds_and_fits(data_gaussian: pd.DataFrame):
    import asyncio

    state = AppState()
    state.data = data_gaussian
    tab = ModelTab(state=state)
    tab._response_sel.value = "y"
    tab._predictors_sel.value = ["x1"]
    tab._pin_cbs["sigma"].value = True
    tab._pin_mode["sigma"].value = "Fixed value"
    tab._pin_val["sigma"].value = 2.0
    tab._draws_in.value = 50
    tab._tune_in.value = 50
    tab._chains_in.value = 1
    tab._cores_in.value = 1

    asyncio.run(tab._on_fit_model(None))
    assert state.model is not None and state.model.is_fitted
    assert "fixed_params" in tab.to_code()


def test_switching_dataset_with_different_columns_does_not_crash():
    """Regression test: loading a second dataset whose numeric
    columns differ from the first used to KeyError inside
    _transformed_xy() — one of the four selector widgets (_x_var/_y_var)
    could still hold a column name from the previous dataframe when a
    value-watcher fired mid-update-loop."""
    from hbsaemp.app.tabs import ExploreTab

    state = AppState()
    tab = ExploreTab(state=state)

    state.data = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    state.data = pd.DataFrame({"c": [1, 2, 3, 4], "d": [4, 3, 2, 1]})  # must not raise

    assert tab._x_var.value == "c"
    assert tab._y_var.value == "c"
    assert len(tab._corr_table.value) == 4  # scatter/corr still renders correctly after


def test_default_predictors_start_empty(data_gaussian: pd.DataFrame):
    """Regression test: auto-selecting *every* remaining numeric column
    as a predictor silently includes columns that should never be
    ordinary predictors — simulation ground-truth (e.g. data_fhnorm's
    theta_true/u), design columns meant for a different parameter (the
    sampling-variance column, which belongs in sampling_var=), or the
    area/group column doing double duty. check_data() doesn't catch any
    of this, so the GUI no longer guesses — predictors start empty and
    the response is the only auto-picked field."""
    state = AppState()
    state.data = data_gaussian
    tab = ModelTab(state=state)
    assert tab._predictors_sel.value == []
    assert tab._response_sel.value == data_gaussian.select_dtypes(include="number").columns[0]


# ---------------------------------------------------------------------------
# Prior Sensitivity in Model Comparison
# ---------------------------------------------------------------------------

def test_prior_sensitivity_option_defaults_off():
    state = AppState()
    rt = ResultsTab(state=state)
    assert rt._compare_psense_cb.value is False


@pytest.mark.slow
def test_prior_sensitivity_rendered_with_saved_names_when_enabled(data_gaussian: pd.DataFrame):
    import asyncio

    state = AppState()
    state.data = data_gaussian
    mt = ModelTab(state=state)
    rt = ResultsTab(state=state)

    mt._response_sel.value = "y"
    mt._predictors_sel.value = ["x1"]
    mt._draws_in.value = 200
    mt._tune_in.value = 200
    mt._chains_in.value = 2
    mt._cores_in.value = 1
    asyncio.run(mt._on_fit_model(None))
    mt._save_name_in.value = "Model A"
    asyncio.run(mt._on_save_model(None))

    rt._compare_table.selection = [0]
    rt._compare_psense_cb.value = True
    asyncio.run(rt._on_compare_click(None))

    psense_card = next(
        (obj for obj in rt._compare_result_pane.objects
         if isinstance(obj, pn.Card) and obj.title == "Prior Sensitivity"),
        None,
    )
    assert psense_card is not None, rt._compare_status.object

    rendered_names = {
        section.objects[0].object.strip("*")
        for section in psense_card.objects[0].objects
        if isinstance(section, pn.Column)
    }
    assert rendered_names == {"Model A"}