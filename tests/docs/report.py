"""Summarise a JUnit report of tests/docs as one table row per test.

    py -3.14 -m pytest tests/docs -m "docs or docs_site" --junitxml=docs/_build/docs-tests.xml
    py -3.14 tests/docs/report.py docs/_build/docs-tests.xml
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import Counter

# What each test checks, as it appears in the table.
TARGETS = {
    "test_cross_references_resolve": "Rujukan {doc} dan toctree di setiap halaman sumber",
    "test_executed_page_declares_a_python_kernel": "Notebook: kernel Python dan highlight kode",
    "test_executed_page_hides_the_progress_bar": "Notebook: progress bar dimatikan",
    "test_public_name_is_in_the_reference": "Nama publik paket tercantum di Reference",
    "test_public_object_has_a_summary_line": "Fungsi/kelas publik punya docstring (ringkasan Reference)",
    "test_translated_page_has_a_catalogue": "Halaman terjemahan punya katalog .po",
    "test_catalogue_belongs_to_an_existing_page": "Katalog .po milik halaman yang masih ada",
    "test_catalogue_is_fully_translated": "Katalog terjemahan lengkap (tanpa kosong/fuzzy)",
    "test_catalogue_keeps_markup_intact": "Markup utuh di terjemahan",
    "test_diataxis_names_stay_english": "Nama Diataxis tidak diterjemahkan",
    "test_page_is_built": "Halaman terbangun (en dan id)",
    "test_page_declares_its_language": "Atribut bahasa halaman (lang)",
    "test_language_switcher_opens_the_same_page": "Pemilih bahasa membuka halaman yang sama",
    "test_internal_links_resolve": "Tautan internal dan gambar tidak putus",
    "test_notebook_shows_its_outputs": "Notebook menampilkan output, tanpa traceback",
    "test_indonesian_page_is_translated": "Halaman /id/ berisi teks terjemahan",
    "test_indonesian_site_keeps_diataxis_names": "Kartu Diataxis di situs Indonesia",
    "test_search_index_is_built": "Indeks pencarian per bahasa",
}


KINDS = {"unit": "Unit", "integration": "Integrasi"}


def main(path: str) -> None:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root[0]
    cases, passed, failed, skipped = Counter(), Counter(), Counter(), Counter()
    order: list[tuple[str, str, str]] = []
    for case in suite.iter("testcase"):
        # classname: tests.docs.<unit|integration>.<test module>
        *_, folder, module = case.get("classname").split(".")
        key = (folder, f"{folder}/{module}.py", case.get("name").split("[")[0])
        if key not in cases:
            order.append(key)
        cases[key] += 1
        if case.find("failure") is not None or case.find("error") is not None:
            failed[key] += 1
        elif case.find("skipped") is not None:
            skipped[key] += 1
        else:
            passed[key] += 1

    def counts(keys: list) -> list[str]:
        return [str(sum(c[k] for k in keys)) for c in (cases, passed, failed, skipped)]

    header = ("Jenis", "Berkas test", "Unit sasaran", "Kasus uji", "Lolos", "Gagal", "Dilewati")
    rows: list[tuple[str, ...] | None] = [header, None]
    for folder in KINDS:  # unit first, then integration; tests in run order
        keys = [k for k in order if k[0] == folder]
        for key in keys:
            rows.append((KINDS[folder], key[1], TARGETS.get(key[2], key[2]),
                         *map(str, (cases[key], passed[key], failed[key], skipped[key]))))
        if keys:
            rows += [None, ("", "", f"Subtotal {KINDS[folder]}", *counts(keys)), None]
    rows.append(("TOTAL", "", "", *counts(order)))

    widths = [max(len(r[i]) for r in rows if r) for i in range(len(header))]
    for row in rows:
        if row is None:
            print("  ".join("-" * w for w in widths))
        else:
            print("  ".join(c.rjust(w) if j >= 3 else c.ljust(w)
                            for j, (c, w) in enumerate(zip(row, widths, strict=True))))
    sys.exit(1 if sum(failed.values()) else 0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/_build/docs-tests.xml")
