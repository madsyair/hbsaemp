"""Unit tests: one source page at a time, read in isolation, no build."""
from __future__ import annotations

import pytest

from tests.docs._pages import cells, is_executed, source_pages

pytestmark = pytest.mark.docs

PAGES = source_pages()
EXECUTED = [d for d, p in PAGES.items() if is_executed(p)]


@pytest.mark.parametrize("docname", EXECUTED)
def test_executed_page_declares_a_python_kernel(docname: str) -> None:
    page = PAGES[docname]
    text = page.read_text(encoding="utf-8")
    if page.suffix == ".ipynb":
        assert '"kernelspec"' in text and '"python3"' in text
    else:
        front = text.split("---", 2)[1]
        assert "kernelspec:" in front and "name: python3" in front
        assert "language_info:" in front, "without language_info the code is not highlighted"
        assert "{code-cell}" in text


@pytest.mark.parametrize("docname", EXECUTED)
def test_executed_page_hides_the_progress_bar(docname: str) -> None:
    """A progress bar in an executed page prints a Rich warning into the output."""
    code = cells(PAGES[docname])[1]
    if ".fit(" in code or "update_model(" in code:
        assert "progressbar=False" in code
