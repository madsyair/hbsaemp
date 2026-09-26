"""Integration tests, no build: pages agree with the pages they link to, and the
package agrees with its API reference."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.docs._pages import SOURCE, cells, public_names, source_pages

pytestmark = pytest.mark.docs

PAGES = source_pages()

_DOC_ROLE = re.compile(r"(?:\{doc\}|:doc:)`([^`]+)`")
_MD_TOCTREE = re.compile(r"```\{toctree\}\n(.*?)```", re.S)
_RST_TOCTREE = re.compile(r"^\.\. toctree::\n((?:[ \t]+.*\n|[ \t]*\n)+)", re.M)


def _targets(prose: str) -> list[str]:
    """Documents a page links to through {doc}/:doc: roles and toctrees."""
    found = []
    for body in _DOC_ROLE.findall(prose):
        angle = re.search(r"<([^>]+)>\s*$", body)
        found.append(angle.group(1) if angle else body)
    for block in _MD_TOCTREE.findall(prose) + _RST_TOCTREE.findall(prose):
        for line in block.splitlines():
            entry = line.strip()
            if entry and not entry.startswith(":"):
                angle = re.search(r"<([^>]+)>\s*$", entry)
                found.append(angle.group(1) if angle else entry)
    return found


def _resolves(page: Path, target: str) -> bool:
    base = SOURCE / target.lstrip("/") if target.startswith("/") else page.parent / target
    return any(base.with_name(base.name + suffix).is_file() for suffix in (".md", ".rst", ".ipynb"))


@pytest.mark.parametrize("docname", sorted(PAGES))
def test_cross_references_resolve(docname: str) -> None:
    page = PAGES[docname]
    missing = [t for t in _targets(cells(page)[0]) if not _resolves(page, t)]
    assert not missing, f"{docname} links to missing documents: {missing}"


_REFERENCE = "\n".join(p.read_text(encoding="utf-8")
                       for p in (SOURCE / "python" / "reference").glob("*.rst"))

# Exported, but deliberately left out of the Python reference, which keeps its
# five sections (models, diagnostics, estimation, datasets, errors).
NOT_IN_REFERENCE = {"configure_logging"}


@pytest.mark.parametrize("name", [n for n in public_names() if n not in NOT_IN_REFERENCE])
def test_public_name_is_in_the_reference(name: str) -> None:
    listed = re.search(
        rf"^\s+{name}\s*$|^\.\. data:: {name}\b|``{name}``|:(?:func|class):`{name}`",
        _REFERENCE, re.M,
    )
    assert listed, f"{name} is exported by hbsaemp but missing from python/reference/*.rst"
