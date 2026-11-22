"""Tests that loop over the cases of a SQL fixture, and running only some cases.

Several of sqlglot's slowest tests are loops over independent fixture cases read
through `tests.helpers.load_sql_fixture_pairs` (one case = one (sql, expected)
pair) or `load_sql_fixtures` (one case = one statement): the TPC-H/TPC-DS
executor and optimizer tests, every `check_file` optimizer test, the
annotate_types/annotate_functions tests, identity/pretty transpile tests. Each
case is parsed, transformed and compared on its own, so a test whose cases all
pass on the patched tree passes, and a case that is unchanged and does not run
any changed code behaves exactly as at the pinned commit (where it passes).

At decision time we therefore run such a test restricted to the cases that can
differ: the cases whose fixture text the patch changed, plus the cases whose
own coverage (traced per case at build time, see tools/trace_cases.py) touches
changed code. The restriction is a wrapper around the two loaders that only
yields the chosen case ordinals; everything else in the test (fixtures,
assertions after the loop, other reads of the file) runs unchanged.

Which tests loop over which fixture is read from maps/cases.json (built from
the pinned commit's test sources by tools/build_cases.py and reviewed by hand).
"""

from __future__ import annotations

import os

# fixture path relative to tests/fixtures -> frozenset of case ordinals to keep.
# Fixtures not listed are not restricted. Set in the verification child only.
ALLOWED: dict = {}


def _universal(text: str) -> str:
    # open(..., encoding="utf-8") in the helpers reads with universal newlines
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _filter_comments(s: str) -> str:
    # same as tests/helpers.py::_filter_comments
    return "\n".join([line for line in s.splitlines() if line and not line.startswith("--")])


def pair_cases(text: str) -> list:
    """Raw (sql, expected) statements of each case, as load_sql_fixture_pairs splits them."""
    statements = _filter_comments(_universal(text)).split(";")
    size = len(statements)
    return [(statements[i].strip(), statements[i + 1].strip()) for i in range(0, size, 2) if i + 1 < size]


def line_cases(text: str) -> list:
    """Each statement load_sql_fixtures yields (leading '#' meta lines included)."""
    out, lines = [], []
    for line in _filter_comments(_universal(text)).splitlines():
        lines.append(line)
        if not line.startswith("#"):
            out.append("\n".join(lines))
            lines.clear()
    return out


def changed_cases(old_text: str, new_text: str, kind: str) -> set[int]:
    """Ordinals of the cases whose text differs (every case past the shorter end
    counts as changed: it was added, removed or shifted)."""
    split = pair_cases if kind == "pairs" else line_cases
    old, new = split(old_text), split(new_text)
    changed = {i for i, (a, b) in enumerate(zip(old, new)) if a != b}
    changed |= set(range(min(len(old), len(new)), max(len(old), len(new))))
    return changed


def changed_new_cases(old_text: str, new_text: str, kind: str) -> set[int]:
    """Ordinals, in the new file, of the cases that are new or whose text changed.

    Cases are aligned by content, so a case that only moved (because one was
    inserted or removed before it) keeps its text and is not counted: it runs
    exactly as before. Removed cases need no run at all."""
    import difflib

    split = pair_cases if kind == "pairs" else line_cases
    old, new = split(old_text), split(new_text)
    out = set()
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            out.update(range(j1, j2))
    return out


def install(helpers) -> None:
    """Wrap the fixture loaders of the (overlay's) tests.helpers module so they
    only yield the ordinals listed in ALLOWED. Must run before test modules
    import the loaders by name."""
    if getattr(helpers, "_br_restricted", False):
        return
    original_pairs = getattr(helpers, "load_sql_fixture_pairs", None)
    original_lines = getattr(helpers, "load_sql_fixtures", None)

    def restrict(original):
        def loader(filename, *args, **kwargs):
            keep = ALLOWED.get(os.path.normpath(filename)) if isinstance(filename, str) else None
            for i, item in enumerate(original(filename, *args, **kwargs)):
                if keep is None or i in keep:
                    yield item

        loader.__name__ = original.__name__
        loader.__qualname__ = original.__qualname__
        loader.__doc__ = original.__doc__
        return loader

    if callable(original_pairs):
        helpers.load_sql_fixture_pairs = restrict(original_pairs)
    if callable(original_lines):
        helpers.load_sql_fixtures = restrict(original_lines)
    helpers._br_restricted = True
