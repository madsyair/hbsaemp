"""Paths and page inventory shared by the documentation tests."""
from __future__ import annotations

import ast
import fnmatch
import json
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "docs" / "source"
HTML = REPO / "docs" / "_build" / "html"
LOCALE = SOURCE / "locale" / "id" / "LC_MESSAGES"

_SKIP_DIRS = {"generated", "_templates", "_static", "locale", ".ipynb_checkpoints"}
_SUFFIXES = (".md", ".rst", ".ipynb")


@cache
def source_pages() -> dict[str, Path]:
    """Every page of the site, keyed by docname (path without suffix)."""
    pages = {}
    for path in sorted(SOURCE.rglob("*")):
        if path.suffix in _SUFFIXES and not _SKIP_DIRS & set(path.relative_to(SOURCE).parts):
            pages[path.relative_to(SOURCE).with_suffix("").as_posix()] = path
    return pages


@cache
def cells(path: Path) -> tuple[str, str]:
    """(prose, code) of a page. Plain pages are all prose."""
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".ipynb":
        return text, text
    nb = json.loads(text)
    prose = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    return prose, code


def is_executed(path: Path) -> bool:
    """Whether myst-nb executes the page at build time."""
    return path.suffix == ".ipynb" or "file_format: mystnb" in path.read_text(encoding="utf-8")[:300]


@cache
def translated_patterns() -> tuple[str, ...]:
    """The TRANSLATED list of docs/i18n.py, read without importing it."""
    tree = ast.parse((REPO / "docs" / "i18n.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "TRANSLATED":
            return tuple(ast.literal_eval(node.value))
    raise LookupError("TRANSLATED not found in docs/i18n.py")


def is_translated(docname: str) -> bool:
    return any(fnmatch.fnmatch(docname, pattern) for pattern in translated_patterns())


def python_track(docname: str) -> bool:
    """The Python pages and the home page, the part translated into Indonesian."""
    return docname == "index" or docname.startswith("python/")


@cache
def catalogues() -> tuple[str, ...]:
    """Every Indonesian catalogue, by docname ("sphinx" holds the theme strings)."""
    return tuple(sorted(p.relative_to(LOCALE).with_suffix("").as_posix()
                        for p in LOCALE.rglob("*.po")))


def messages(catalogue: str) -> list:
    """The messages of one catalogue, header excluded."""
    from babel.messages.pofile import read_po  # ships with Sphinx

    with (LOCALE / f"{catalogue}.po").open("rb") as fh:
        return [m for m in read_po(fh, locale="id") if m.id]


@cache
def public_names() -> tuple[str, ...]:
    """Names hbsaemp exports, without the GUI ones.

    The GUI names resolve lazily through PEP 562, so touching them would import
    Panel; they are documented in the GUI track, not in the Python reference.
    """
    import hbsaemp

    gui = set(getattr(hbsaemp, "_LAZY_GUI", ()))
    return tuple(sorted(n for n in hbsaemp.__all__ if not n.startswith("__") and n not in gui))
