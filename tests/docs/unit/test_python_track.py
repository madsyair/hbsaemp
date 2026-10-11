"""Unit tests for the Python track of the documentation: one page at a time, no build.

The Python track is the ``docs/source/python/`` tree. These tests pin the conventions
that its pages already follow: how-to guides numbered by stage with a fixed set of
sections, code that uses only names the package exports, and a tutorial that follows
the CRISP-DM phases.
"""
from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

from tests.docs._pages import cells, source_pages

pytestmark = pytest.mark.docs

PAGES = {d: p for d, p in source_pages().items() if d.startswith("python/")}
HOWTO = {d: p for d, p in PAGES.items() if re.fullmatch(r"python/how-to/\d\d-.+", d)}
TUTORIALS = {d: p for d, p in PAGES.items()
             if d.startswith("python/tutorials/") and d != "python/tutorials/index"}

HOWTO_SECTIONS = ["Before you start", "Do it", "Check it worked", "If it fails", "Related"]
CRISP_DM = ["Business understanding", "Data understanding", "Data preparation",
            "Modeling", "Evaluation", "Deployment"]

_FENCE = re.compile(r"^```\{?(?P<kind>code-cell|testcode|python)\}?[^\n]*\n(?P<body>.*?)^```",
                    re.S | re.M)


def headings(path: Path) -> list[tuple[int, str]]:
    """(level, text) of every heading, ignoring lines inside code fences."""
    if path.suffix == ".ipynb":
        lines = cells(path)[0].splitlines()
    else:
        lines, fenced = [], False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("```"):
                fenced = not fenced
            elif not fenced:
                lines.append(line)
    found = []
    for line in lines:
        m = re.match(r"^(#{1,6}) (.+?)\s*$", line)
        if m:
            found.append((len(m.group(1)), m.group(2)))
    return found


def code_of(path: Path) -> str:
    """Python code of a page: notebook code cells, or code-cell/testcode/python fences."""
    if path.suffix == ".ipynb":
        return cells(path)[1]
    blocks = []
    for m in _FENCE.finditer(path.read_text(encoding="utf-8")):
        body = [ln for ln in m.group("body").splitlines() if not ln.startswith(":")]
        blocks.append("\n".join(body))
    return "\n".join(blocks)


def _plain_python(code: str) -> str:
    """Drop IPython magics and shell escapes, which ast cannot parse."""
    return "\n".join(ln for ln in code.splitlines() if not ln.lstrip().startswith(("%", "!")))


# Pages without Python code (indexes, the autosummary reference, prose-only explanations)
# have nothing for the code test to check, so they are left out of its parameters.
PAGES_WITH_CODE = {d: p for d, p in PAGES.items() if _plain_python(code_of(p)).strip()}


# how-to guides 
@pytest.mark.parametrize("docname", sorted(HOWTO))
def test_howto_title_carries_the_page_number(docname: str) -> None:
    """The title number is the stage number, and it matches the file name."""
    number = int(re.search(r"/(\d\d)-", docname).group(1))
    title = next(text for level, text in headings(HOWTO[docname]) if level == 1)
    m = re.match(r"^(\d+) · .+", title)
    assert m, f"{docname}: title {title!r} should read 'N · Title'"
    assert int(m.group(1)) == number, f"{docname}: title says stage {m.group(1)}, file says {number}"


@pytest.mark.parametrize("docname", sorted(HOWTO))
def test_howto_has_the_standard_sections(docname: str) -> None:
    """Every guide has the same sections, in the same order (others may sit between)."""
    h2 = [text for level, text in headings(HOWTO[docname]) if level == 2]
    position = -1
    for section in HOWTO_SECTIONS:
        assert section in h2[position + 1:], f"{docname}: missing or misplaced section {section!r}"
        position = h2.index(section, position + 1)


@pytest.mark.parametrize("docname", sorted(HOWTO))
def test_howto_related_section_links_somewhere(docname: str) -> None:
    text = HOWTO[docname].read_text(encoding="utf-8")
    related = text.split("## Related", 1)[1]
    assert re.search(r"\{(?:doc|ref|func|class|meth)\}`|\]\(", related), \
        f"{docname}: the Related section has no link"


# code on every page of the track 

def _exports() -> set[str]:
    import hbsaemp

    return set(hbsaemp.__all__) | {"__version__"}


@pytest.mark.parametrize("docname", sorted(PAGES_WITH_CODE))
def test_code_parses_and_uses_exported_names(docname: str) -> None:
    """Page code is valid Python and imports only what hbsaemp exports."""
    code = _plain_python(code_of(PAGES[docname]))
    tree = ast.parse(code)  # SyntaxError fails the test, with its line number
    exports = _exports()
    aliases: set[str] = set()
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "hbsaemp":
                    aliases.add(alias.asname or "hbsaemp")
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "hbsaemp":
            if node.module == "hbsaemp":
                problems += [f"hbsaemp.{a.name}" for a in node.names
                             if a.name != "*" and a.name not in exports]
            else:
                module = importlib.import_module(node.module)
                problems += [f"{node.module}.{a.name}" for a in node.names
                             if a.name != "*" and not hasattr(module, a.name)]
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in aliases and node.attr not in exports):
            problems.append(f"{node.value.id}.{node.attr}")
    assert not problems, f"{docname} uses names hbsaemp does not export: {sorted(set(problems))}"


# tutorials

@pytest.mark.parametrize("docname", sorted(TUTORIALS))
def test_tutorial_follows_the_crisp_dm_phases(docname: str) -> None:
    """The case study is organised as the six CRISP-DM phases, numbered 1 to 6, in order."""
    h2 = [text for level, text in headings(TUTORIALS[docname]) if level == 2]
    phases = [re.match(r"^(\d+) · (.+)$", t) for t in h2]
    phases = [(int(m.group(1)), m.group(2)) for m in phases if m]
    assert [n for n, _ in phases] == list(range(1, 7)), f"{docname}: phases are {phases}"
    for (_, name), expected in zip(phases, CRISP_DM):
        assert name.lower() == expected.lower(), f"{docname}: phase {name!r}, expected {expected!r}"
