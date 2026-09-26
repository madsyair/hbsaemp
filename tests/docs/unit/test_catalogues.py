"""Unit tests: one Indonesian catalogue (.po) at a time, read in isolation."""
from __future__ import annotations

import re

import pytest

from tests.docs._pages import catalogues, messages, python_track

pytest.importorskip("babel.messages.pofile", reason="Babel ships with Sphinx")

pytestmark = pytest.mark.docs

CATALOGUES = catalogues()
# Page catalogues of the Python track and the home page ("sphinx" is the theme).
PYTHON_TRACK = [c for c in CATALOGUES if c != "sphinx" and python_track(c)]

# Section names of the Diataxis framework; they stay English in every language.
_QUADRANT = re.compile(r"\b(Tutorials|How-to guides|Reference|Explanation)\b")

# Markup a translation must copy unchanged, or Sphinx reports
# "inconsistent references" and the -W build fails.
_ROLE = re.compile(r"(\{[a-z:-]+\}|:[a-z:-]+:)`([^`]*)`")
_LITERAL = re.compile(r"(?<![\}:])`{1,2}([^`]+)`{1,2}")
_URL = re.compile(r"https?://[^\s>)\]]+")
_MATH = re.compile(r"\$[^$]+\$")
_FORMAT = re.compile(r"%\([a-zA-Z_]+\)s")


def _markup(text: str) -> dict[str, list[str]]:
    roles = []
    for role, body in _ROLE.findall(text):
        angle = re.search(r"<([^>]+)>\s*$", body)
        roles.append(f"{role}{angle.group(1) if angle else body}")
    return {
        "roles": sorted(roles),
        "code": sorted(_LITERAL.findall(_ROLE.sub("", text))),
        "urls": sorted(_URL.findall(text)),
        "math": sorted(_MATH.findall(text)),
        "format": sorted(_FORMAT.findall(text)),
    }


@pytest.mark.parametrize("catalogue", CATALOGUES)
def test_catalogue_is_fully_translated(catalogue: str) -> None:
    msgs = messages(catalogue)
    done = sum(1 for m in msgs if m.string and not m.fuzzy)
    fuzzy = sum(1 for m in msgs if m.fuzzy)
    assert done == len(msgs), (
        f"{done}/{len(msgs)} translated ({100 * done / len(msgs):.0f}%), {fuzzy} fuzzy"
    )


@pytest.mark.parametrize("catalogue", CATALOGUES)
def test_catalogue_keeps_markup_intact(catalogue: str) -> None:
    broken = []
    for m in messages(catalogue):
        if not m.string:
            continue
        source, target = _markup(m.id), _markup(m.string)
        kinds = [k for k in source if source[k] != target[k]]
        if kinds:
            broken.append(f"{kinds}: {m.id[:60]!r}")
    assert not broken, f"{len(broken)} message(s), first: {broken[0]}"


@pytest.mark.parametrize("catalogue", PYTHON_TRACK)
def test_diataxis_names_stay_english(catalogue: str) -> None:
    renamed = [
        m.id[:60] for m in messages(catalogue)
        if m.string and any(name not in m.string for name in _QUADRANT.findall(m.id))
    ]
    assert not renamed, f"{len(renamed)} message(s) translate a Diataxis name, first: {renamed[0]!r}"
