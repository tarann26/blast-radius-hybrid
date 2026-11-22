"""Builds maps/keymap.bin from the table entry trace (keytrace.py) of the pinned tree.

usage: python build_keymap.py TRACE_DIR [TRACE_DIR ...] TESTS_JSON OUT_FILE
(several trace directories, runs with different hash seeds, are merged by union)

The map holds, for every traced table (a dict, set or frozenset that a sqlglot module global or
class attribute holds) that no test changes: for each entry key, the tests that looked it up,
and the tests that used the table as a whole (WHOLE), its size (LEN) or its keys (KEYS).
Reads during the import of a test file count for every test of that file. Tests whose reads
are not known (they fail or time out on the pinned tree here, or start other processes) are
listed as unknown and are never cut. Built from the pinned commit only.
"""
from __future__ import annotations

import glob
import gzip
import json
import marshal
import os
import sys
import zlib
from array import array

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import keymap  # noqa: E402


def main():
    *trace_dirs, tests_path, out_path = sys.argv[1:]
    tests = json.load(open(tests_path))["tests"]
    by_file = {}
    for t in tests:
        by_file.setdefault(t["file"], []).append(t["number"])
    info = None
    excluded = {}
    for d in trace_dirs:
        tj = json.load(open(os.path.join(d, "tables.json")))
        if tj["refcount_bad"]:
            raise SystemExit("%s: the swap missed references: %r" % (d, tj["refcount_bad"][:5]))
        if info is None:
            info = tj["info"]
        for name, v in tj["info"].items():
            if v["left_out"]:
                excluded.setdefault(name, "left out: " + v["left_out"])
            if name not in info or info[name]["holders"] != v["holders"]:
                excluded.setdefault(name, "holders differ between runs")
        for name in set(info) - set(tj["info"]):
            excluded.setdefault(name, "not in every run")
    reads = {}  # test -> {table: set of keys}
    traced = {}  # test -> number of runs that traced it
    unknown = set()
    files_seen = {}
    for d in trace_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json.gz"))):
            rec = json.load(gzip.open(path))
            files_seen[rec["file"]] = files_seen.get(rec["file"], 0) + 1
            nums = by_file.get(rec["file"], [])
            for name in rec.get("import_changed", ()):
                excluded.setdefault(name, "changed by a test file import")
            if not rec.get("import_ok", True):
                unknown.update(nums)
                continue
            at_import = rec.get("import", {})
            for n in nums:
                m = rec["tests"].get(str(n))
                if m is not None:
                    for name in m.get("changed", ()):
                        excluded.setdefault(name, "changed by a test")
                if m is None or "reads" not in m or not m.get("ok") or m.get("spawned"):
                    unknown.add(n)
                    continue
                traced[n] = traced.get(n, 0) + 1
                r = reads.setdefault(n, {})
                for src in (m["reads"], at_import):
                    for name, keys in src.items():
                        r.setdefault(name, set()).update(keys)
    runs = len(trace_dirs)
    for n in list(traced):
        if traced[n] != runs:
            unknown.add(n)  # traced in some runs only
    unknown |= {t["number"] for t in tests} - set(traced)
    # invert
    tables = {}  # name -> key -> set of tests
    for n, r in reads.items():
        if n in unknown:
            continue
        for name, keys in r.items():
            if name in excluded:
                continue
            tk = tables.setdefault(name, {})
            for k in keys:
                tk.setdefault(k, set()).add(n)
    # tables that no traced test reads are kept (with no entries): a change there reaches no
    # traced test
    for name in info:
        if name not in excluded:
            tables.setdefault(name, {})
    names = sorted(tables)
    tid = {name: i for i, name in enumerate(names)}
    sets, sid = [], {}
    out_tables = []
    n_keys = 0
    for name in names:
        tk = tables[name]
        whole = tk.get(keymap.WHOLE, set())
        d = {}
        for k, nums in sorted(tk.items()):
            if k != keymap.WHOLE:
                nums = nums - whole  # the whole readers are added for every change anyway
            if not nums:
                continue
            b = array("H", sorted(nums)).tobytes()
            i = sid.get(b)
            if i is None:
                i = sid[b] = len(sets)
                sets.append(b)
            d[k] = i
            n_keys += 1
        out_tables.append(d)
    holders = {}
    for name in names:
        for h in info[name]["holders"]:
            holders[h] = tid[name]
    out = {
        "version": 1,
        "holders": holders,
        "tables": out_tables,
        "sets": sets,
        "unknown": array("H", sorted(unknown)).tobytes(),
    }
    data = zlib.compress(marshal.dumps(out), 9)
    with open(out_path, "wb") as f:
        f.write(data)
    with open(out_path + ".excluded.json", "w") as f:
        json.dump(excluded, f, indent=0, sort_keys=True)
    missing_files = sorted(set(by_file) - set(files_seen))
    print("keymap: %d tables (%d excluded), %d holders, %d keys, %d sets, %d tests traced, %d unknown, %d bytes; "
          "files without a trace: %s" % (len(names), len(excluded), len(holders), n_keys, len(sets),
                                         len(set(traced) - unknown), len(unknown), len(data), missing_files[:5]))


if __name__ == "__main__":
    main()
