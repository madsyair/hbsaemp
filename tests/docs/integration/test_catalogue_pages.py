"""Integration tests, no build: the pages and their Indonesian catalogues match."""
from __future__ import annotations

import pytest

from tests.docs._pages import LOCALE, catalogues, is_translated, source_pages

pytestmark = pytest.mark.docs

PAGES = source_pages()


@pytest.mark.parametrize("docname", sorted(d for d in PAGES if is_translated(d)))
def test_translated_page_has_a_catalogue(docname: str) -> None:
    assert (LOCALE / f"{docname}.po").is_file(), "run `python docs/i18n.py update`"


@pytest.mark.parametrize("catalogue", [c for c in catalogues() if c != "sphinx"])
def test_catalogue_belongs_to_an_existing_page(catalogue: str) -> None:
    """A renamed or deleted page leaves its catalogue behind."""
    assert catalogue in PAGES, f"no page {catalogue}; move or delete the .po"
