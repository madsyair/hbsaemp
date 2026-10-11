"""Integration tests, no build: the pages of the Python track agree with each other and
with the package.

A section index lists every page of its section, the how-to index follows the stage
numbers, and the migration page names only functions that exist.
"""
from __future__ import annotations

import re

import pytest

from tests.docs._pages import SOURCE, source_pages

pytestmark = pytest.mark.docs

PAGES = source_pages()
SECTIONS = ["tutorials", "how-to", "reference", "explanation"]

_MD_TOCTREE = re.compile(r"```\{toctree\}\n(.*?)```", re.S)
_RST_TOCTREE = re.compile(r"^\.\. toctree::\n((?:[ \t]+.*\n|[ \t]*\n)+)", re.M)


def _toctree(docname: str) -> list[str]:
    """Docnames listed in the toctrees of a page, resolved against the page's folder."""
    path = PAGES[docname]
    text = path.read_text(encoding="utf-8")
    entries = []
    for block in _MD_TOCTREE.findall(text) + _RST_TOCTREE.findall(text):
        for line in block.splitlines():
            entry = line.strip()
            if entry and not entry.startswith(":"):
                angle = re.search(r"<([^>]+)>\s*$", entry)
                entries.append(angle.group(1) if angle else entry)
    base = docname.rsplit("/", 1)[0]
    return [e.lstrip("/") if e.startswith("/") else f"{base}/{e}" for e in entries]


def test_python_index_links_the_four_diataxis_sections() -> None:
    listed = _toctree("python/index")
    for section in SECTIONS:
        assert f"python/{section}/index" in listed, f"python/index.md does not list {section}/index"


@pytest.mark.parametrize("section", SECTIONS)
def test_section_index_lists_every_page_of_its_section(section: str) -> None:
    """A page that no index lists is orphaned: it builds, but the reader never finds it."""
    prefix = f"python/{section}/"
    pages = {d for d in PAGES if d.startswith(prefix) and d != f"{prefix}index"}
    assert pages, f"python/{section}/ has no pages"
    listed = set(_toctree(f"{prefix}index"))
    missing = sorted(pages - listed)
    assert not missing, f"python/{section}/index does not list: {missing}"
    stale = sorted(listed - pages)
    assert not stale, f"python/{section}/index lists pages that do not exist: {stale}"


def _howto_numbers() -> list[int]:
    return sorted(int(m.group(1)) for d in PAGES
                  if (m := re.fullmatch(r"python/how-to/(\d\d)-.+", d)))


def test_howto_stage_numbers_are_contiguous() -> None:
    numbers = _howto_numbers()
    assert numbers == list(range(1, len(numbers) + 1)), f"stage numbers are {numbers}"


def test_howto_index_follows_the_stage_order() -> None:
    """The index table and toctree list the stages once each, in ascending order."""
    text = PAGES["python/how-to/index"].read_text(encoding="utf-8")
    in_table = re.findall(r"^\| \{doc\}`(\d\d)-[^`]+`", text, re.M)
    in_toctree = [int(m.group(1)) for e in _toctree("python/how-to/index")
                  if (m := re.search(r"/(\d\d)-", e))]
    assert [int(n) for n in in_table] == _howto_numbers(), f"index table lists stages {in_table}"
    assert in_toctree == _howto_numbers(), f"index toctree lists stages {in_toctree}"


def test_migration_page_names_only_existing_functions() -> None:
    """Every Python name in the migration table is exported by hbsaemp."""
    import hbsaemp

    exports = set(hbsaemp.__all__)
    text = (SOURCE / "python" / "how-to" / "migrate-from-hbsaems.md").read_text(encoding="utf-8")
    python_cells = [row.split("|")[2] for row in text.splitlines()
                    if row.startswith("| `") and row.count("|") >= 3]
    names = {n for cell in python_cells for n in re.findall(r"`([A-Za-z_]\w*)\(", cell)}
    assert names, "no function names found in the migration table"
    missing = sorted(names - exports)
    assert not missing, f"migration table names functions hbsaemp does not export: {missing}"
