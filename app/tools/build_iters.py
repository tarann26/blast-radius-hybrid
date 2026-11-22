"""Builds MAPS/iters.bin from the cold iteration traces (itercold.py).

usage (inside br-label): python build_iters.py REPO TESTS_JSON COLD_DIR OUT_DIR (writes OUT_DIR/<test>.bin, OUT_DIR/index.bin)
iters.bin = zlib(marshal({test: (mode, edits, stmts)})):
  mode "stmt" (the test has statement units: the trace ran the statement alone) or "all"
  (the trace ran the whole test);
  edits {site: (line, col, end line, end col)} = the iterable of each loop (UTF-8 offsets);
  stmts {statement: (sites, own, paths, cov)}: own {file: lines} = lines that the statement runs
  outside its iterations; paths = iteration paths; cov {file: {line: bit mask over paths}} =
  lines that an iteration runs beyond the part that leads to it.
"""
from __future__ import annotations

import ast
import glob
import gzip
import json
import marshal
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import iterred
import reduce
from itertrace import method_node


def main():
    root, tests_path, cold_dir, out = sys.argv[1:5]
    tests = {t["number"]: t for t in json.load(open(tests_path))["tests"]}
    res = {}
    for f in sorted(glob.glob(os.path.join(cold_dir, "*.json.gz"))):
        rec = json.load(gzip.open(f))
        n = rec["n"]
        if rec.get("err") or not rec.get("stmts"):
            continue
        rel, cls_name, meth = tests[n]["id"].split("::")
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        func = method_node(src, cls_name, meth)
        shape = reduce.method_shape(func)
        sites = iterred.method_sites(func, [s[2] for s in shape[2]])
        edits = {k: (lp.iter.lineno, lp.iter.col_offset, lp.iter.end_lineno, lp.iter.end_col_offset) for k, _, lp in sites}
        stmts = {}
        for s, srec in rec["stmts"].items():
            if "err" in srec or not srec.get("paths"):
                continue
            paths = [tuple(tuple(x) for x in p) for p in srec["paths"]]
            cov = {}
            for i, c in enumerate(srec["cov"]):
                for fn, lines in c.items():
                    d = cov.setdefault(fn, {})
                    for ln in lines:
                        d[ln] = d.get(ln, 0) | (1 << i)
            own = {fn: frozenset(v) for fn, v in srec["own"].items()}
            stmts[int(s)] = (tuple(srec["sites"]), own, paths, cov)
        if stmts:
            res[n] = (rec["mode"], edits, stmts)
    os.makedirs(out, exist_ok=True)
    total = 0
    for n, v in res.items():
        blob = zlib.compress(marshal.dumps(v), 9)
        total += len(blob)
        with open(os.path.join(out, "%d.bin" % n), "wb") as f:
            f.write(blob)
    with open(os.path.join(out, "index.bin"), "wb") as f:
        f.write(marshal.dumps(frozenset(res)))
    print("tests", len(res), "bytes", total)


if __name__ == "__main__":
    main()
