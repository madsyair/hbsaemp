"""Integration tests on the built site: every page in both languages, notebook
outputs and links. They exercise the whole pipeline together: Sphinx, the
notebooks executed against hbsaemp, the translation build and the templates.

Reads docs/_build/html as `python docs/i18n.py build` leaves it. A local build
can hold pages that no longer exist; only the current pages are checked.
"""
from __future__ import annotations

from functools import cache
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from tests.docs._pages import is_executed, is_translated, source_pages

bs4 = pytest.importorskip("bs4", reason="BeautifulSoup ships with the docs theme")

pytestmark = pytest.mark.docs_site

PAGES = source_pages()
LANGS = {"en": "", "id": "id/"}
CASES = [pytest.param(lang, doc, id=f"{lang}:{doc}") for lang in LANGS for doc in sorted(PAGES)]
EXECUTED = [pytest.param(lang, doc, id=f"{lang}:{doc}")
            for lang in LANGS for doc in sorted(PAGES) if is_executed(PAGES[doc])]

# Figures each executed page drew when these tests were written. Fewer means a
# plot failed to render; the diagnostics record such failures instead of raising.
MIN_FIGURES = {
    "python/how-to/03-check-priors": 2,
    "python/how-to/05-check-convergence": 10,
    "python/how-to/07-check-fit-and-compare": 2,
    "python/tutorials/case-study-papua": 10,
}


@cache
def _soup(path: Path):
    return bs4.BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")


def _page(site: Path, lang: str, docname: str) -> Path:
    return site / LANGS[lang] / f"{docname}.html"


def _local_target(page: Path, url: str) -> Path | None:
    """The file a relative URL points to, or None for external and in-page links."""
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or not parts.path or url.startswith(("#", "//")):
        return None
    target = (page.parent / unquote(parts.path)).resolve()
    return target / "index.html" if parts.path.endswith("/") else target


@pytest.mark.parametrize(("lang", "docname"), CASES)
def test_page_is_built(site: Path, lang: str, docname: str) -> None:
    assert _page(site, lang, docname).is_file()


@pytest.mark.parametrize(("lang", "docname"), CASES)
def test_page_declares_its_language(site: Path, lang: str, docname: str) -> None:
    assert _soup(_page(site, lang, docname)).html.get("lang") == lang


@pytest.mark.parametrize(("lang", "docname"), CASES)
def test_language_switcher_opens_the_same_page(site: Path, lang: str, docname: str) -> None:
    page = _page(site, lang, docname)
    link = _soup(page).select_one("a.language-switcher")
    assert link is not None, "no language switcher in the navbar"
    other = "id" if lang == "en" else "en"
    assert link.get("hreflang") == other
    assert _local_target(page, link["href"]) == _page(site, other, docname).resolve()


@pytest.mark.parametrize(("lang", "docname"), CASES)
def test_internal_links_resolve(site: Path, lang: str, docname: str) -> None:
    page = _page(site, lang, docname)
    soup = _soup(page)
    urls = [a["href"] for a in soup.select("a[href]")] + [i["src"] for i in soup.select("img[src]")]
    broken = sorted({u for u in urls if (t := _local_target(page, u)) and not t.exists()})
    assert not broken, f"{len(broken)} broken link(s), first: {broken[0]}"


@pytest.mark.parametrize(("lang", "docname"), EXECUTED)
def test_notebook_shows_its_outputs(site: Path, lang: str, docname: str) -> None:
    outputs = _soup(_page(site, lang, docname)).select("div.cell_output")
    assert outputs, "the notebook was rendered without outputs"
    text = "\n".join(o.get_text() for o in outputs)
    assert "Traceback (most recent call last)" not in text
    figures = sum(len(o.select("img")) for o in outputs)
    assert figures >= MIN_FIGURES.get(docname, 0), f"{figures} figure(s) rendered"


@pytest.mark.parametrize("docname", sorted(d for d in PAGES if is_translated(d)))
def test_indonesian_page_is_translated(site: Path, docname: str) -> None:
    def paragraphs(lang: str) -> list[str]:
        article = _soup(_page(site, lang, docname)).select_one("article")
        return [p.get_text(" ", strip=True) for p in article.select("p")]

    en, id_ = paragraphs("en"), paragraphs("id")
    changed = sum(a != b for a, b in zip(en, id_, strict=False))
    assert changed, f"0 of {len(en)} paragraphs differ from the English page"


def test_indonesian_site_keeps_diataxis_names(site: Path) -> None:
    cards = _soup(site / "id" / "python" / "index.html").select(".sd-card-title")
    assert [c.get_text(strip=True) for c in cards] == [
        "Tutorials", "How-to guides", "Reference", "Explanation",
    ]


@pytest.mark.parametrize("lang", LANGS)
def test_search_index_is_built(site: Path, lang: str) -> None:
    assert (site / LANGS[lang] / "searchindex.js").is_file()
