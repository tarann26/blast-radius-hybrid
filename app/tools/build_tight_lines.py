"""Builds maps/tight_lines.bin: for each of the long tests, every sqlglot line that the test
runs, from exact traces of the pinned tree (tight1106/xt: xmon.py, sitecustomize.py,
xtplugin.py, xreplay.py, xrun.py). The selector uses it in plan_python_file: such a test is a
candidate for an edit inside a function body only when it runs a line that holds a changed
instruction (engine.changed_lines).

The traces cover every process the test uses, under several hash seeds:
  b<test>_s<seed>/  pytest running the test alone; the pytest process and each spawn worker of
                    the test's process pool record their lines (one bucket per pool task)
  c<test>_s<seed>_pool<k>.json  each captured pool task replayed as the first task of a fresh
                    worker (cold), and all tasks in one worker in order (warm)
  a_s<seed>_l<lazy>.json  the selector's own runner (in-process pool), cold and warm runs, and
                    the shared class fixtures that its parent builds
  d_full_s<seed>/   pytest running the whole suite: the test's own phases, its class's setUpClass
                    and class cleanup, and the pool tasks that the workers took while the test ran

usage (inside the pinned image): python build_tight_lines.py TRACES TESTS_JSON MAPS_DIR N[,N...]
"""
import glob
import json
import marshal
import os
import re
import sys
import zlib
from array import array


def main():
    tr, tests_path, out_dir, numbers = sys.argv[1:5]
    numbers = [int(x) for x in numbers.split(",")]
    tests = json.load(open(tests_path))["tests"]
    ident = {n: tests[n]["id"] for n in numbers}
    cls_of = {n: ident[n].rsplit("::", 1)[0] for n in numbers}
    lines = {n: {} for n in numbers}
    count = {n: {} for n in numbers}

    def add(n, src, rows):
        out = lines[n]
        for f, ln, _ in rows:
            out.setdefault(f, set()).add(ln)
        count[n][src] = count[n].get(src, 0) + len(rows)

    for d in glob.glob(os.path.join(tr, "b*_s*")):
        m = re.match(r"b(\d+)_s(\d+)$", os.path.basename(d))
        if not m or int(m.group(1)) not in lines:
            continue
        n = int(m.group(1))
        for p in glob.glob(os.path.join(d, "*.json")):
            r = json.load(open(p))
            for _, rows in r["buckets"]:
                add(n, "B", rows)  # every bucket: import, collection, the test, its workers
    for p in glob.glob(os.path.join(tr, "c*_s*_pool*.json")):
        m = re.match(r"c(\d+)_s(\d+)_pool", os.path.basename(p))
        if not m or int(m.group(1)) not in lines:
            continue
        n = int(m.group(1))
        r = json.load(open(p))
        for t in r["cold"].values():
            if t.get("ok") is not True:
                raise SystemExit("replayed task did not finish: %s %s" % (p, t.get("ok")))
            add(n, "C", t.get("lines", []))
        for _, rows in r["buckets"]:
            add(n, "C", rows)
    for p in glob.glob(os.path.join(tr, "a_s*_l*.json")):
        r = json.load(open(p))
        fixtures = [rows for k, rows in r["parent"] if k == "fixtures"]
        for ns, x in r["tests"].items():
            n = int(ns)
            if n not in lines:
                continue
            if not x.get("ok") or not x["ok"][0]:
                raise SystemExit("runner trace failed: %s %s" % (p, n))
            for _, rows in x.get("buckets", []):
                add(n, "A", rows)
            for rows in fixtures:
                add(n, "A", rows)
    for d in glob.glob(os.path.join(tr, "d_full_s*")):
        if not os.path.isdir(d) or d.endswith("_cap"):
            continue
        par = None
        workers = []
        for p in glob.glob(os.path.join(d, "*.json")):
            r = json.load(open(p))
            if r["worker"]:
                workers.append(r)
            elif any(k == "pre" for k, _ in r["buckets"]) and (par is None or len(r["buckets"]) > len(par["buckets"])):
                par = r
        order = sorted((a, b, k) for k, v in par["windows"] if isinstance(k, str) and "|" in k for a, b in v)
        bk = {k: v for k, v in par["buckets"] if isinstance(k, str)}
        for n in numbers:
            for ph in ("setup", "call", "teardown"):
                add(n, "D", bk.get(ident[n] + "|" + ph, []))
            mine = [x for x in order if x[2].startswith(cls_of[n] + "::")]
            if mine:
                add(n, "D", bk.get(mine[0][2], []))  # setUpClass runs in the first test's setup
                add(n, "D", bk.get(mine[-1][2], []))  # class cleanup in the last test's teardown
        for w in workers:
            tasks = [(k, rows) for k, rows in w["buckets"] if k != "wstart"]
            start = [rows for k, rows in w["buckets"] if k == "wstart"]
            for k, rows in tasks:
                for a, b, name in order:
                    if a <= k[3] <= b:
                        for n in numbers:
                            if name.startswith(ident[n] + "|"):
                                add(n, "D", rows)
                                if start:
                                    add(n, "D", start[0])
    out = {}
    for n in numbers:
        if not all(count[n].get(s) for s in ("B", "A", "D")):
            raise SystemExit("missing traces for test %d: %s" % (n, count[n]))
        files = {}
        for f, ls in lines[n].items():
            files[f] = array("I", sorted(ls)).tobytes()
        out[n] = files
        print(n, ident[n], "lines", sum(len(v) for v in lines[n].values()), "files", len(files), count[n])
    with open(os.path.join(out_dir, "tight_lines.bin"), "wb") as f:
        f.write(zlib.compress(marshal.dumps(out), 9))


if __name__ == "__main__":
    main()
