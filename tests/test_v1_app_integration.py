"""Integration tests for the GUI: several tabs around ONE shared ``AppState``.

Every test builds more than one tab on the same ``AppState`` and drives them
through their real callbacks, the way a user moves through the app:

    Data Upload -> Data Exploration -> Modeling (build, prior predictive
    check, fit, save, posterior predictive check) -> Results (convergence,
    model comparison, SAE estimation)

The fast tests check that the tabs stay in sync through ``AppState``. The
slow tests run a real (small) MCMC fit.

    pytest tests/test_v1_app_integration.py -m "not slow"   # fast
    pytest tests/test_v1_app_integration.py                 # full workflow
"""
from __future__ import annotations

import matplotlib

# The GUI draws its figures from worker threads, and the macOS backend forbids
# that. launch_app() selects Agg for real sessions; tests bypass launch_app().
matplotlib.use("Agg")

import asyncio

import pandas as pd
import pytest

import hbsaemp as hb

pn = pytest.importorskip("panel")
pytestmark = pytest.mark.gui

from hbsaemp.app._app import App, AppState  # noqa: E402
from hbsaemp.app.tabs import DataTab, ExploreTab, ModelTab, ResultsTab, UpdateModelTab  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def _upload(tab: DataTab, df: pd.DataFrame, name: str = "areas.csv") -> None:
    tab._file_input.filename = name
    tab._file_input.value = df.to_csv(index=False).encode("utf-8")


def _configure_gaussian_model(mt: ModelTab, *, draws: int = 100) -> None:
    mt._draws_in.value = draws
    mt._tune_in.value = draws
    mt._chains_in.value = 2
    mt._cores_in.value = 1
    mt._response_sel.value = "y"
    mt._predictors_sel.value = ["x1", "x2"]
    mt._group_sel.value = "group"
    mt._family_sel.value = "gaussian"
    mt._update_preview()


# ---------------------------------------------------------------------------
# Fast integration: tabs stay in sync through AppState
# ---------------------------------------------------------------------------
def test_uploaded_data_reaches_every_tab(data_gaussian: pd.DataFrame):
    state = AppState()
    data_tab = DataTab(state=state)
    explore_tab = ExploreTab(state=state)
    model_tab = ModelTab(state=state)

    _upload(data_tab, data_gaussian)

    numeric = data_gaussian.select_dtypes(include="number").columns.tolist()
    assert state.data is not None
    assert list(explore_tab._hist_var.options) == numeric
    assert list(model_tab._response_sel.options) == numeric
    assert list(model_tab._group_sel.options) == [None, *data_gaussian.columns]
    # The model tab never guesses predictors; the user must choose them.
    assert model_tab._predictors_sel.value == []


def test_loading_a_second_dataset_replaces_the_first_in_all_tabs(data_gaussian: pd.DataFrame):
    state = AppState()
    data_tab = DataTab(state=state)
    explore_tab = ExploreTab(state=state)
    model_tab = ModelTab(state=state)

    _upload(data_tab, data_gaussian)
    data_tab._dataset_sel.value = "data_fhnorm"
    data_tab._on_load_dataset(None)

    expected = state.data.select_dtypes(include="number").columns.tolist()
    assert len(state.data) == 30
    assert list(explore_tab._x_var.options) == expected
    assert list(model_tab._response_sel.options) == expected


def test_build_and_prior_check_use_the_uploaded_data(data_gaussian: pd.DataFrame):
    state = AppState()
    data_tab = DataTab(state=state)
    model_tab = ModelTab(state=state)
    _upload(data_tab, data_gaussian)
    _configure_gaussian_model(model_tab)
    model_tab._prior_n_draws.value = 20

    model_tab._on_build(None)
    assert "Model built and data validated successfully" in model_tab._build_status.object

    run(model_tab._on_prior_check(None))
    assert "Prior predictive check complete" in model_tab._prior_status.object
    # The draft is checked, not fitted, and it is not published as the active model.
    assert model_tab._model_draft.is_fitted is False
    assert state.model is None


def test_results_tab_stays_empty_until_a_model_is_fitted(data_gaussian: pd.DataFrame):
    state = AppState()
    _upload(DataTab(state=state), data_gaussian)
    results_tab = ResultsTab(state=state)

    run(results_tab._on_load_convergence(None))
    run(results_tab._on_run_sae_estimation(None))

    assert "Model has not been fitted" in results_tab._conv_status.object
    assert "Model has not been fitted" in results_tab._sae_status.object
    assert results_tab._sae_table.value.empty


# ---------------------------------------------------------------------------
# Slow integration: the whole workflow, one real fit
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_full_workflow_through_gui_callbacks(data_gaussian: pd.DataFrame):
    state = AppState()
    data_tab = DataTab(state=state)
    ExploreTab(state=state)
    model_tab = ModelTab(state=state)
    results_tab = ResultsTab(state=state)

    # a. Load data
    _upload(data_tab, data_gaussian)
    assert state.data is not None

    # b. Specify and build the model
    _configure_gaussian_model(model_tab)
    model_tab._on_build(None)
    assert "Model built and data validated successfully" in model_tab._build_status.object

    # c. Prior predictive check
    model_tab._prior_n_draws.value = 20
    run(model_tab._on_prior_check(None))
    assert "Prior predictive check complete" in model_tab._prior_status.object
    assert state.model is None

    # d. Fit, then save a named snapshot
    run(model_tab._on_fit_model(None))
    assert "MCMC sampling has completed" in model_tab._fit_status.object
    assert state.model is not None and state.model.is_fitted
    assert model_tab._save_btn.disabled is False

    model_tab._save_name_in.value = "Model A"
    run(model_tab._on_save_model(None))
    assert "Saved as" in model_tab._save_status.object
    assert list(state.saved_models) == ["Model A"]

    # e. Convergence
    run(results_tab._on_load_convergence(None))
    assert "Diagnostics computed" in results_tab._conv_status.object
    assert not results_tab._rhat_ess_table.value.empty

    # f. Posterior predictive check, then model comparison
    run(model_tab._on_posterior_check(None))
    assert "Model has not been fitted" not in model_tab._postpc_status.object
    assert "Posterior predictive check complete" in model_tab._postpc_status.object

    assert list(results_tab._compare_table.value["Model"]) == ["Model A"]
    results_tab._compare_table.selection = [0]
    run(results_tab._on_compare_click(None))
    assert "Compared 1 model(s): Model A" in results_tab._compare_status.object

    # h. Estimate the areas
    run(results_tab._on_run_sae_estimation(None))
    assert "SAE estimation complete" in results_tab._sae_status.object
    table = results_tab._sae_table.value
    assert not table.empty
    assert {"mean", "sd", "ci_lower", "ci_upper", "rse_pct"} <= set(table.columns)


# ---------------------------------------------------------------------------
# Component/assembly tests moved from the former test_v1_app.py
# ---------------------------------------------------------------------------


def test_app_view_returns_template():
    app = App(app_config=hb.AppConfig(port=9998, open_browser=False))
    template = app.view()
    assert isinstance(template, pn.template.FastListTemplate)


def test_app_view_respects_show_sidebar():
    app = App(app_config=hb.AppConfig(port=9997, open_browser=False, show_sidebar=False))
    template = app.view()
    assert len(template.sidebar) == 0


def test_all_tabs_construct_and_render():
    state = AppState()
    for cls in (DataTab, ExploreTab, ModelTab, ResultsTab, UpdateModelTab):
        tab = cls(state=state)
        assert tab.panel() is not None


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

    ranking_card = next(
        obj for obj in rt._compare_result_pane.objects
        if isinstance(obj, pn.Card) and obj.title == "Ranking (LOO / ELPD)"
    )
    # Find the table wherever it sits inside the card; the card may wrap it in
    # a Column, so indexing objects[0] depends on layout details.
    tables = ranking_card.select(pn.widgets.Tabulator)
    assert tables, "no ranking table found inside the Ranking card"
    labels = tables[0].value["Model"].astype(str).tolist()
    assert set(labels) == {"Model 2", "Model 3"}
    assert not any("model_" in label for label in labels)


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

    # Collect every text pane in the card instead of walking a fixed layout.
    rendered_text = {
        pane.object.strip("*").strip()
        for pane in bf_card.select(pn.pane.Markdown)
        if isinstance(pane.object, str)
    }
    assert {"Model 2", "Model 3"} <= rendered_text
    assert not any("model_" in t for t in rendered_text)


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

    rendered_text = {
        pane.object.strip("*").strip()
        for pane in psense_card.select(pn.pane.Markdown)
        if isinstance(pane.object, str)
    }
    assert "Model A" in rendered_text
    assert not any("model_" in t for t in rendered_text)