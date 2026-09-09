"""Sphinx configuration for the hbsaemp documentation."""

import os

import hbsaemp

# -- Project --------------------------------------------------------------
project = "hbsaemp"
author = "hbsaemp Contributors"
copyright = "2026, hbsaemp Contributors"

# The site version follows the package. There is deliberately no second place
# to write it down, so it can never fall behind.
version = hbsaemp.__version__
release = version

language = "en"
master_doc = "index"

# -- Extensions -----------------------------------------------------------
extensions = [
    "myst_nb",  # loads myst_parser as well; never list both
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_copybutton",
]

# Narrative pages are Markdown; only the API reference files are reStructuredText,
# because `autosummary` is an rST directive. Tutorials are notebooks, executed on
# CI only (see below).
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- MyST -----------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "substitution",
]
myst_heading_anchors = 3  # direct links to sub-headings

# -- Notebook execution ---------------------------------------------------
# Executed on CI, never on a local build: a local `make html` stays fast, while
# every push proves the tutorials still run against the current package.
#
# "force", not "cache", on purpose. A cache makes CI quick but hides code that
# has already broken, which is exactly what this check exists to catch.
# Set CI=1 locally to reproduce what CI does.
on_ci = bool(os.environ.get("CI", ""))
nb_execution_mode = "force" if on_ci else "off"
nb_execution_allow_errors = False
nb_execution_raise_on_error = True
nb_execution_timeout = 900
nb_kernel_rgx_aliases = {".*": "python3"}

# -- Autodoc / autosummary ------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Doctest --------------------------------------------------------------
# Only blocks that explicitly ask to be tested (`.. testcode::`, `.. doctest::`)
# are run. Plain `>>>` blocks are left alone.
#
# Without this, `-b doctest` also collects the `Examples:` sections of the
# package's own docstrings, which are illustrative fragments: they use
# `create_model`, `ModelConfig` and `df` without importing or defining them, so
# they can only ever fail. Making them runnable is a change to the package, not
# to the documentation, and is tracked separately.
doctest_test_doctest_blocks = ""

# Every source module is underscore-private (_base.py, _factory.py, ...).
# Without this, signatures render as `hbsaemp.models._base.BaseModel` instead
# of the path users actually import from.
add_module_names = False

# -- Intersphinx ----------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
    "arviz": ("https://python.arviz.org/en/stable/", None),
    "bambi": ("https://bambinos.github.io/bambi/", None),
}

# -- HTML -----------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "hbsaemp"
html_static_path = ["_static"]

# Without this, custom.css is copied into the output but never linked, and the
# stub markers lose their styling silently.
html_css_files = ["custom.css"]

# `pygments_style` is deliberately unset: pydata-sphinx-theme ships a matched
# pair of light and dark syntax themes, and forcing a single one breaks dark mode.

html_theme_options = {
    "show_nav_level": 1,
    "navigation_depth": 3,
    "show_toc_level": 2,
    "search_bar_text": "Search the hbsaemp docs...",
    # `html_context` below supplies the repository coordinates, but the theme
    # only renders the "Edit this page" link when this switch is on as well.
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/madsyair/hbsaemp",
            "icon": "fa-brands fa-github",
        },
    ],
}

# Drives the "Edit this page" link, so contributors never have to guess where
# a page lives in the repository.
html_context = {
    "github_user": "madsyair",
    "github_repo": "hbsaemp",
    "github_version": "main",
    "doc_path": "docs/source/",
    "default_mode": "light",
}
