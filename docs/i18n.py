"""Maintain and build the Indonesian translation of the documentation.

    python docs/i18n.py update   # refresh the .po catalogues after English edits
    python docs/i18n.py build    # build English and Indonesian, as CI does

`update` extracts every translatable sentence (the gettext builder), then merges
it into locale/id/LC_MESSAGES/<page>.po for the pages in TRANSLATED only. A
sentence whose English changed is marked fuzzy and shows in English until it is
translated again; the site never breaks for want of a translation.

`build` writes the English site to docs/_build/html and the Indonesian one to
its id/ folder, which is the layout the navbar's language switcher links across.
Run both from the repository root.
"""
from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pydata_sphinx_theme
from babel.messages.catalog import Catalog
from babel.messages.pofile import read_po, write_po

DOCS = Path(__file__).resolve().parent
SOURCE = DOCS / "source"
BUILD = DOCS / "_build"
LOCALE = SOURCE / "locale" / "id" / "LC_MESSAGES"

# Pages translated into Indonesian: the narrative Python track, the home page,
# the project pages, and the theme/template strings ("sphinx"). The API
# reference and the GUI track stay English on purpose.
TRANSLATED = [
    "sphinx",
    "index",
    "about/*",
    "python/index",
    "python/tutorials/*",
    "python/how-to/*",
    "python/explanation/*",
]


def _sphinx(builder: str, outdir: Path, *extra: str, lang: str = "en") -> None:
    env = {**os.environ, "DOCS_LANG": lang}
    cmd = [sys.executable, "-m", "sphinx", "-b", builder, *extra, str(SOURCE), str(outdir)]
    subprocess.run(cmd, env=env, check=True)


def update() -> None:
    """Merge freshly extracted sentences into the Indonesian catalogues."""
    pot_dir = BUILD / "gettext"
    # Only the text is needed, not the notebook outputs.
    _sphinx("gettext", pot_dir, "-q", "-D", "nb_execution_mode=off")

    for pot in sorted(pot_dir.rglob("*.pot")):
        page = pot.relative_to(pot_dir).with_suffix("").as_posix()
        if not any(fnmatch.fnmatch(page, pattern) for pattern in TRANSLATED):
            continue
        with pot.open("rb") as fh:
            template = read_po(fh)
        if page == "sphinx":
            # The theme's own strings ("On this page", ...) are not extracted
            # from this project, and the theme ships no Indonesian catalogue.
            # Merging its template keeps them from being dropped as obsolete.
            with (Path(pydata_sphinx_theme.__file__).parent / "locale" / "sphinx.pot").open("rb") as fh:
                for message in read_po(fh):
                    if message.id and message.id not in template:
                        template.add(message.id, locations=message.locations)
        po_path = LOCALE / f"{page}.po"
        if po_path.exists():
            with po_path.open("rb") as fh:
                catalog = read_po(fh, locale="id")
        else:
            catalog = Catalog(locale="id", fuzzy=False)
        catalog.project, catalog.version = "hbsaemp", template.version
        catalog.copyright_holder = "the hbsaemp authors"
        catalog.update(template)
        po_path.parent.mkdir(parents=True, exist_ok=True)
        # Sentences that left the English text are dropped rather than kept
        # as "#~" entries, which would only clutter the file under review.
        with po_path.open("wb") as fh:
            write_po(fh, catalog, width=0, omit_header=False, ignore_obsolete=True)

        messages = [m for m in catalog if m.id]
        untranslated = sum(1 for m in messages if not m.string)
        fuzzy = sum(1 for m in messages if m.fuzzy)
        print(f"{page:<45} {len(messages):4d} sentences, "
              f"{untranslated:4d} untranslated, {fuzzy:4d} fuzzy")


def build() -> None:
    """Build both languages under -W and nest the Indonesian site in id/."""
    html, html_id = BUILD / "html", BUILD / "html_id"
    _sphinx("html", html, "-W", lang="en")
    _sphinx("html", html_id, "-W", lang="id")
    shutil.rmtree(html / "id", ignore_errors=True)
    shutil.copytree(html_id, html / "id")
    print(f"English: {html / 'index.html'}\nIndonesian: {html / 'id' / 'index.html'}")


if __name__ == "__main__":
    commands = {"update": update, "build": build}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(f"usage: python {Path(__file__).name} {{{'|'.join(commands)}}}")
    commands[sys.argv[1]]()
