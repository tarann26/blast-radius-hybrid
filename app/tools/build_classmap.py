"""Builds maps/classmap.bin from the class attribute trace (classtrace.py) of the pinned tree.

The map answers: which tests read attribute NAME through an object or class object R of the
tokenizer, parser, generator or dialect trees? A class constant defined by class K reaches a
test only through a receiver R whose MRO holds K. Built from the pinned commit only.

usage: python build_classmap.py TRACE_DIR [TRACE_DIR ...] TESTS_JSON OUT_FILE
(several trace directories, e.g. runs with different hash seeds, are merged by union)
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


def main():
    *trace_dirs, tests_path, out_path = sys.argv[1:]
    tests = json.load(open(tests_path))["tests"]
    by_file = {}
    for t in tests:
        by_file.setdefault(t["file"], []).append(t["number"])
    classes = {}
    for d in trace_dirs:
        for k, v in json.load(open(os.path.join(d, "classes.json"))).items():
            classes.setdefault(k, v)
    # names that some class of the trees defines as a function, method or property: a changed
    # key with such a name is never cut by this map (their reads are not kept, to stay small)
    fnames = set()
    for v in classes.values():
        fnames.update(v["fn"])
    mro = {k: v["mro"] for k, v in classes.items()}
    reads = {}  # test -> set of (receiver, name)
    traced = set()
    unknown = set()
    for d in trace_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json.gz"))):
            rec = json.load(gzip.open(path))
            for k, v in rec.get("mro", {}).items():
                mro.setdefault(k, v)
            nums = by_file.get(rec["file"], [])
            if not rec.get("import_ok", True):
                unknown.update(nums)
                continue
            at_import = {tuple(p) for p in rec.get("import", ())}
            for n in nums:
                m = rec["tests"].get(str(n))
                if m is None or "pairs" not in m or not m.get("ok") or m.get("spawned"):
                    # not traced to its end (it failed, timed out, or crashed on the pinned
                    # tree), or it started another process, whose reads are not seen: its
                    # reads are not known, so it is never cut
                    unknown.add(n)
                    continue
                traced.add(n)
                s = reads.setdefault(n, set())
                s.update(tuple(p) for p in m["pairs"])
                s.update(at_import)  # the import of its test file runs before it
    unknown |= {t["number"] for t in tests} - traced
    unknown -= set()  # (kept explicit: a test is either traced or unknown)
    # invert: (receiver, name) -> tests
    inv = {}
    for n, pairs in reads.items():
        if n in unknown:
            continue
        for r, name in pairs:
            if name in fnames:
                continue
            inv.setdefault(name, {}).setdefault(r, []).append(n)
    receivers = sorted({r for d in inv.values() for r in d})
    rid = {r: i for i, r in enumerate(receivers)}
    class_keys = sorted(set(receivers) | set(classes) | {b for v in mro.values() for b in v})
    cid = {k: i for i, k in enumerate(class_keys)}
    sets = []
    sid = {}
    names = {}
    for name, d in inv.items():
        arr = array("I")
        for r, nums in sorted(d.items()):
            b = array("H", sorted(set(nums))).tobytes()
            i = sid.get(b)
            if i is None:
                i = sid[b] = len(sets)
                sets.append(b)
            arr.extend((rid[r], i))
        names[name] = arr.tobytes()
    out = {
        "version": 1,
        "classes": class_keys,  # class keys "module:path"
        "hooked": array("I", sorted(cid[k] for k in classes)).tobytes(),  # pinned classes of the trees
        "receivers": [array("I", [cid[b] for b in mro.get(r, [r])]).tobytes() for r in receivers],  # MRO of each
        "names": names,
        "sets": sets,
        "fnames": sorted(fnames),
        "unknown": array("H", sorted(unknown)).tobytes(),
    }
    data = zlib.compress(marshal.dumps(out), 9)
    with open(out_path, "wb") as f:
        f.write(data)
    print("classmap: %d tests traced, %d unknown, %d names, %d receivers, %d sets, %d bytes"
          % (len(traced - unknown), len(unknown), len(names), len(receivers), len(sets), len(data)))


if __name__ == "__main__":
    main()
