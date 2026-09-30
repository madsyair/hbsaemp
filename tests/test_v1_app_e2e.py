"""End-to-end tests: a real browser drives the real dashboard.

Unlike ``test_v1_app_unit.py`` and ``test_v1_app_integration.py`` (which call
tab callbacks directly), these tests start the Panel server, open it in
Chromium through Playwright, and act the way a user does: click tabs, pick
options, upload a file, press buttons, and read what appears on screen.

Setup (once):
    pip install playwright
    playwright install chromium

Run:
    pytest tests/test_v1_app_e2e.py -m "gui and not slow"    # quick checks
    pytest tests/test_v1_app_e2e.py                          # full workflow

The tests skip themselves when Playwright or Chromium is not installed.
"""
from __future__ import annotations

import matplotlib

# The GUI draws its figures from worker threads, and the macOS backend forbids
# that. launch_app() selects Agg for real sessions; tests bypass launch_app().
matplotlib.use("Agg")

import re
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pn = pytest.importorskip("panel")
sync_api = pytest.importorskip("playwright.sync_api")

import hbsaemp as hb  # noqa: E402
from hbsaemp.app._app import App  # noqa: E402

pytestmark = pytest.mark.gui

FAST_MS = 20_000
MCMC_MS = 240_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # Chromium not installed
            pytest.skip(f"Chromium is not available for Playwright: {exc}")
        yield b
        b.close()


@pytest.fixture()
def dashboard():
    """A fresh app on a free port. Yields the base URL."""
    port = _free_port()
    app = App(app_config=hb.AppConfig(port=port, open_browser=False))
    server = pn.serve(app.view, port=port, show=False, threaded=True)
    time.sleep(1.5)
    try:
        yield f"http://localhost:{port}/"
    finally:
        server.stop()


@pytest.fixture()
def page(browser, dashboard):
    pg = browser.new_page(viewport={"width": 1500, "height": 1100})
    pg.set_default_timeout(FAST_MS)
    pg.goto(dashboard)
    pg.wait_for_selector("text=Load Dataset")
    yield pg
    pg.close()


@pytest.fixture(scope="module")
def csv_file(tmp_path_factory) -> Path:
    rng = np.random.default_rng(7)
    n = 100
    group = np.repeat(np.arange(1, 11), n // 10)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 1.0 + 0.3 * x1 - 0.2 * x2 + rng.normal(0, 0.5, 10)[group - 1] + rng.normal(0, 0.5, n)
    path = tmp_path_factory.mktemp("e2e") / "areas.csv"
    pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": group}).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Helpers: the vocabulary of a user session
# ---------------------------------------------------------------------------

def open_tab(pg, name: str) -> None:
    pg.locator(".bk-tab", has_text=name).first.click()


def click(pg, name: str) -> None:
    """Click a Panel button by its exact label (card headers also render as buttons)."""
    pg.locator("button.bk-btn").filter(has_text=re.compile(rf"^\s*{re.escape(name)}\s*$")).click()


def check_option(pg, label: str) -> None:
    """Tick the checkbox whose visible label is *label*.

    Panel renders the label in a shadow root next to the input, so the
    accessible name is empty and get_by_label cannot find it. The label text
    is read from the input's sibling instead.
    """
    boxes = pg.get_by_role("checkbox")
    for i in range(boxes.count()):
        text = boxes.nth(i).evaluate(
            "e => { const l = e.parentNode.querySelector('.bk-label');"
            " return l ? l.textContent.trim() : ''; }"
        )
        if text == label:
            boxes.nth(i).check()
            return
    raise AssertionError(f"No checkbox labelled {label!r} on the page")


def load_builtin_dataset(pg, dataset: str = "data_fhnorm") -> None:
    pg.locator("select").first.select_option(dataset)
    click(pg, "Load Dataset")
    pg.wait_for_selector("text=successfully loaded")


def upload_csv(pg, path: Path) -> None:
    pg.locator("input[type=file]").first.set_input_files(str(path))
    pg.wait_for_selector("text=successfully loaded")


def specify_gaussian_model(pg, *, draws: int = 100) -> None:
    """Modeling > Model Building: y ~ x1 + x2 with a group effect, small sampler."""
    open_tab(pg, "Modeling")
    open_tab(pg, "Model Building")
    pg.get_by_label("Response Variable (y)").select_option("y")
    pg.get_by_label("Area / Group Variable").select_option("group")
    for var in ("x1", "x2"):
        check_option(pg, var)
    for label, value in (("draws", draws), ("tune", draws), ("chains", 2)):
        box = pg.get_by_label(label, exact=True)
        box.fill(str(value))
        box.press("Enter")


def build_model(pg) -> None:
    click(pg, "Build Model")
    pg.wait_for_selector("text=Model built and data validated successfully")


# ---------------------------------------------------------------------------
# Quick end-to-end checks
# ---------------------------------------------------------------------------

def test_app_opens_with_the_four_workflow_tabs(page):
    assert page.title() == "HBSAEMP APP"
    tabs = [t.inner_text() for t in page.locator(".bk-tab").all() if t.inner_text()]
    assert tabs[:4] == ["Data Upload", "Data Exploration", "Modeling", "Results"]


def test_modeling_subtabs_follow_the_workflow_order(page):
    open_tab(page, "Modeling")
    subtabs = [t.inner_text() for t in page.locator(".bk-tab").all() if t.inner_text()][4:]
    assert subtabs == [
        "Overview", "Model Building", "Prior Predictive Check",
        "Fit Model", "Posterior Predictive Check",
    ]


def test_loading_a_builtin_dataset_shows_summary_and_variables(page):
    load_builtin_dataset(page)
    assert page.get_by_text("Total Rows:").first.is_visible()
    assert page.get_by_text("theta_true", exact=True).first.is_visible()


def test_upload_csv_through_the_file_input(page, csv_file):
    upload_csv(page, csv_file)
    assert page.get_by_text("areas.csv").first.is_visible()


def test_unreadable_file_shows_an_error_and_no_data(page, tmp_path):
    bad = tmp_path / "broken.csv"
    bad.write_bytes(b"\xff\xfe\xfd\xfc\xfb")
    page.locator("input[type=file]").first.set_input_files(str(bad))
    page.wait_for_selector("text=Could not read CSV file")
    assert page.get_by_text("successfully loaded").count() == 0


def test_load_button_without_a_selection_shows_an_error(page):
    click(page, "Load Dataset")
    page.wait_for_selector("text=select a built-in dataset")


def test_loaded_data_appears_in_data_exploration(page):
    load_builtin_dataset(page)
    open_tab(page, "Data Exploration")
    page.wait_for_selector("text=Variable")
    assert page.get_by_text("theta_true").first.is_visible()


def test_loaded_data_appears_in_model_building(page):
    load_builtin_dataset(page)
    open_tab(page, "Modeling")
    open_tab(page, "Model Building")
    options = page.get_by_label("Response Variable (y)").locator("option").all_inner_texts()
    assert "y" in options and "x1" in options


def test_build_model_before_choosing_predictors_is_blocked_or_valid(page):
    """Build Model must answer with a message either way, never a blank page."""
    load_builtin_dataset(page)
    open_tab(page, "Modeling")
    open_tab(page, "Model Building")
    click(page, "Build Model")
    page.wait_for_selector("text=/Model built|Cannot build model/")


def test_prior_predictive_check_runs_from_the_browser(page, csv_file):
    upload_csv(page, csv_file)
    specify_gaussian_model(page)
    build_model(page)

    open_tab(page, "Prior Predictive Check")
    box = page.get_by_label("n_draws", exact=True)
    box.fill("20")
    box.press("Enter")
    click(page, "Run Prior Predictive Check")
    page.wait_for_selector("text=Prior predictive check complete", timeout=MCMC_MS)
    assert page.get_by_text("Prior Summary").first.is_visible()


def test_results_tab_asks_for_a_fitted_model(page):
    open_tab(page, "Results")
    click(page, "Load Convergence Diagnostics")
    page.wait_for_selector("text=Model has not been fitted")


def test_fit_without_a_model_draft_shows_an_error(page):
    open_tab(page, "Modeling")
    open_tab(page, "Fit Model")
    click(page, "Fit Model")
    page.wait_for_selector("text=Cannot fit model")


# ---------------------------------------------------------------------------
# Full workflow (one real MCMC fit)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_full_workflow_in_the_browser(page, csv_file):
    # a. Load data
    upload_csv(page, csv_file)

    # b. Specify and build the model
    specify_gaussian_model(page)
    build_model(page)

    # c. Prior predictive check
    open_tab(page, "Prior Predictive Check")
    box = page.get_by_label("n_draws", exact=True)
    box.fill("20")
    box.press("Enter")
    click(page, "Run Prior Predictive Check")
    page.wait_for_selector("text=Prior predictive check complete", timeout=MCMC_MS)

    # d. Fit, then save a named snapshot
    open_tab(page, "Fit Model")
    click(page, "Fit Model")
    page.wait_for_selector("text=The MCMC sampling has completed", timeout=MCMC_MS)
    name = page.get_by_label("Model name")
    name.fill("Model A")
    name.press("Enter")
    click(page, "Save Model")
    page.wait_for_selector("text=Saved as")

    # f. Posterior predictive check
    open_tab(page, "Posterior Predictive Check")
    click(page, "Run Posterior Predictive Check")
    page.wait_for_selector("text=Posterior predictive check complete", timeout=MCMC_MS)

    # e. Convergence
    open_tab(page, "Results")
    click(page, "Load Convergence Diagnostics")
    page.wait_for_selector("text=Diagnostics computed", timeout=MCMC_MS)

    # h. Estimate the areas
    open_tab(page, "SAE Estimation")
    click(page, "Run SAE Estimation")
    page.wait_for_selector("text=SAE estimation complete", timeout=MCMC_MS)
    assert page.get_by_text("Mean RSE").first.is_visible()