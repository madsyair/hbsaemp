"""Sphinx configuration for the hbsaemp documentation."""

import os

import hbsaemp

# -- Project --------------------------------------------------------------
project = "hbsaemp"
author = "Achmad Syahrul Choir, M. Ihsan Silmi Kaffah, Rahmadika Kemala Salsabiela"
copyright = f"2026, {author}"

# The site version follows the package. There is deliberately no second place
# to write it down, so it can never fall behind.
version = hbsaemp.__version__
release = version

master_doc = "index"

# -- Languages ------------------------------------------------------------
# English is the source; the Indonesian site is built from the same sources
# with DOCS_LANG=id (docs/i18n.py does both). An environment variable rather
# than `-D language=id`, because strings set in this file, such as the search
# placeholder below, have to follow the language too.
#
# Translations live in locale/id/LC_MESSAGES/, one .po per page
# (gettext_compact = False keeps them reviewable). A sentence with no
# translation falls back to English, which is how the API reference and the
# GUI track, deliberately left untranslated, still appear on the Indonesian
# site. locale/id/LC_MESSAGES/sphinx.po also covers the theme's own strings,
# because pydata-sphinx-theme ships no Indonesian catalogue.
language = os.environ.get("DOCS_LANG", "en")
locale_dirs = ["locale/"]
gettext_compact = False

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
# because `autosummary` is an rST directive. The tutorial and the how-to pages that
# sample are notebooks, executed on every build (see below).
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
# Notebooks — the tutorial and the how-to pages that sample — are executed so
# their outputs are always real, through a cache that both language builds
# share: the English build runs each notebook, the Indonesian build reuses the
# result, since translation touches the prose and never the code.
#
# A notebook re-runs when its code changes. CI also keeps the cache between
# runs, keyed on the package source, its dependencies and the week, so a change
# to the package empties the cache and every notebook runs again: code that has
# broken still fails the build, while a change to the prose alone runs nothing.
nb_execution_mode = "cache"
nb_execution_cache_path = os.path.join(os.path.dirname(__file__), "..", "_build", ".jupyter_cache")
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

# The how-to pages that sample are notebooks, executed at build time. A
# `testcode` block that samples elsewhere is still tested, on a light budget:
# under `-b doctest` every fit runs 100 draws after 300 tuning steps on 2
# chains, with no progress bar (it writes to stdout, which doctest compares)
# and seed 0 when none is set.
# The patch sits on `ModelConfig.to_sampler_kwargs`, the one place every fit
# passes through, so the code on the page runs exactly as written and only the
# sampler budget shrinks. That proves each call works, not that it converges.
#
# Tuning stays at 300 rather than 100 on purpose: with 100 the sampler never
# adapts, every transition hits the maximum tree depth, and the job takes three
# times as long. Sphinx runs this before every test group, hence the guard.
doctest_global_setup = """
from hbsaemp import ModelConfig as _ModelConfig

if not hasattr(_ModelConfig, "_full_sampler_kwargs"):
    _ModelConfig._full_sampler_kwargs = _ModelConfig.to_sampler_kwargs

    def _light_sampler_kwargs(self):
        kwargs = self._full_sampler_kwargs()
        seed = kwargs["random_seed"]
        kwargs.update(draws=100, tune=300, chains=2, progressbar=False,
                      random_seed=0 if seed is None else seed)
        return kwargs

    _ModelConfig.to_sampler_kwargs = _light_sampler_kwargs
"""

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
html_js_files = ["theme-toggle.js"]

# `pygments_style` is deliberately unset: pydata-sphinx-theme ships a matched
# pair of light and dark syntax themes, and forcing a single one breaks dark mode.

html_theme_options = {
    "show_nav_level": 1,
    "navigation_depth": 3,
    "show_toc_level": 2,
    "search_bar_text": {
        "en": "Search the hbsaemp docs...",
        "id": "Cari di dokumentasi hbsaemp...",
    }[language],
    # `html_context` below supplies the repository coordinates, but the theme
    # only renders the "Edit this page" link when this switch is on as well.
    "use_edit_page_button": True,
    # A link to the same page in the other language (_templates/
    # language-switcher.html), and a two-state light/dark button
    # (_templates/theme-toggle.html) in place of the theme's light/dark/system
    # menu; light stays the default below.
    "navbar_end": ["language-switcher", "theme-toggle", "navbar-icon-links"],
    # Credits the developers and links the project pages under about/.
    "footer_start": ["developed-by"],
    "footer_end": ["sphinx-version", "theme-version"],
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
    # Named in the footer (_templates/developed-by.html); `author` is the one
    # place the names are written.
    "developers": author,
}
