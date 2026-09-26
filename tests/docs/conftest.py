"""Fixtures for the documentation tests.

Two kinds, one folder each:
    unit/         one artifact at a time in isolation: a page, a docstring or a
                  translation catalogue. No build.
    integration/  artifacts that must agree with each other: pages and the pages
                  they link to, the package and its reference, pages and their
                  catalogues; and the built site, where Sphinx, the executed
                  notebooks and the translation build meet.

Two markers, by what a test needs:
    docs       only the sources; fast.
    docs_site  the built site in docs/_build/html; run after
               `python docs/i18n.py build`, skipped when no build exists.

Each page, notebook and catalogue is its own test case, so a report counts them
one by one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.docs._pages import HTML


@pytest.fixture(scope="session")
def site() -> Path:
    """The built site; the test is skipped when it has not been built."""
    if not (HTML / "index.html").is_file() or not (HTML / "id" / "index.html").is_file():
        pytest.skip("site not built: run `python docs/i18n.py build` first")
    return HTML
