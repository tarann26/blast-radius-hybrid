"""Cold per-iteration coverage (pinned commit only): for each eligible loop iteration of a
test statement, the sqlglot lines that a run of that statement with only that iteration
(and the iterations that lead to it) runs. Same method as trace_units.py (fork per run).

usage (inside br-label): python itercold.py REPO TESTS_JSON WARM_DIR OUT_DIR NUMBERS...
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine
import iterred
import preload
import reduce
import runner
from itertrace import Ctx, prepare
from reduce import STATE

CO_OPTIMIZED = 0x1
TOOL = 4


def child(ctx, number, test_id, keep_stmts, keep_all, keep_prefix, wfd, rewrite=None):
    out = {"ok": False}
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        if keep_stmts is not None or rewrite is not None:
            sites = prepare(ctx, test_id, keep_stmts, rewrite)
            if sites is None:
                out["err"] = "prepare"
                os.write(wfd, json.dumps(out).encode())
                return
        root = ctx.root
        cut = len(root) + 1
        cov = {}
        mon = sys.monitoring
        mon.use_tool_id(TOOL, "cold")

        def on_line(code, line):
            if code.co_flags & CO_OPTIMIZED:
                fn = code.co_filename
                if fn.startswith(root) and fn[cut:cut + 8] == "sqlglot/":
                    cov.setdefault(fn[cut:], set()).add(line)
            return mon.DISABLE

        def on_start(code, offset):
            return on_line(code, code.co_firstlineno)

        mon.register_callback(TOOL, mon.events.LINE, on_line)
        mon.register_callback(TOOL, mon.events.PY_START, on_start)
        case = runner.load_case(test_id, number)
        files_keep = {} if keep_stmts is not None and not keep_stmts else None
        tasks_keep = set() if keep_stmts is not None and not keep_stmts else None

        class Result(unittest.TestResult):
            def startTest(self, test):
                super().startTest(test)
                STATE.reset(files_keep, tasks_keep)
                STATE.in_test = True
                iterred.LOOPS.reset(keep_all, keep_prefix)

            def stopTest(self, test):
                super().stopTest(test)
                STATE.in_test = False

        result = Result()
        result.failfast = False
        mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)
        unittest.TestSuite([case]).run(result)
        mon.set_events(TOOL, 0)
        out["ok"] = not (result.failures or result.errors or result.unexpectedSuccesses)
        if not out["ok"] and os.environ.get("BR_IDEBUG"):
            out["err"] = [t for _, t in (result.failures + result.errors)][:2]
        out["cov"] = {f: sorted(v) for f, v in cov.items()}
        os.write(wfd, json.dumps(out).encode())
    except BaseException as e:  # noqa: BLE001
        out["err"] = repr(e)[:300]
        try:
            os.write(wfd, json.dumps(out).encode())
        except OSError:
            pass


def measure(ctx, number, test_id, keep_stmts, keep_all=None, keep_prefix=frozenset(), rewrite=None):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            child(ctx, number, test_id, keep_stmts, keep_all, keep_prefix, w, rewrite)
        finally:
            os._exit(0)
    os.close(w)
    buf = b""
    while True:
        chunk = os.read(r, 1 << 20)
        if not chunk:
            break
        buf += chunk
    os.close(r)
    os.waitpid(pid, 0)
    m = json.loads(buf) if buf else {"ok": False, "err": "no output"}
    m["cov"] = {f: set(v) for f, v in (m.get("cov") or {}).items()}
    return m


def minus(a, b):
    out = {}
    for f, v in a.items():
        left = v - b.get(f, set())
        if left:
            out[f] = sorted(left)
    return out


def trace(ctx, n, test_id, warm, mode):
    rec = {"n": n, "stmts": {}, "mode": mode}
    sites = {k: s for k, s, _ in warm["sites"]}
    bad = set(warm.get("bad") or [])
    paths = []
    for p, cpu, _ in warm["tags"]:
        p = tuple(tuple(x) for x in p)
        if p and not any(k in bad for k, _ in p):
            paths.append((p, cpu))
    by_stmt = {}
    for p, cpu in paths:
        by_stmt.setdefault(sites[p[0][0]], []).append((p, cpu))
    if mode == "stmt":
        base = measure(ctx, n, test_id, set())
        if not base.get("ok"):
            rec["err"] = "base failed %r" % base.get("err")
            return rec
        bcov = base["cov"]
    else:
        bcov = {}
    for s, plist in sorted(by_stmt.items()):
        plist.sort()
        keep = {s} if mode == "stmt" else None
        own = measure(ctx, n, test_id, keep, frozenset(), frozenset(), {s})
        if not own.get("ok"):
            rec["stmts"][s] = {"err": "own failed %r" % own.get("err")}
            continue
        covs = {(): own["cov"]}
        srec = {"own": minus(own["cov"], bcov), "paths": [], "cpu": [], "cov": [], "sites": sorted(k for k, v in sites.items() if v == s and k not in bad)}
        failed = None
        for p, cpu in plist:
            pre = frozenset(p[:k] for k in range(1, len(p) + 1))
            m = measure(ctx, n, test_id, keep, frozenset(), pre, {s})
            if not m.get("ok"):
                failed = (p, m.get("err"))
                break
            covs[p] = m["cov"]
            parent = covs.get(p[:-1])
            if parent is None:
                failed = (p, "parent missing")
                break
            srec["paths"].append([list(x) for x in p])
            srec["cpu"].append(cpu)
            srec["cov"].append(minus(m["cov"], parent))
        if failed:
            rec["stmts"][s] = {"err": "path failed %r" % (failed,)}
            continue
        rec["stmts"][s] = srec
    return rec


def main():
    root, tests_path, warm_dir, out_dir = sys.argv[1:5]
    numbers = [int(x) for x in sys.argv[5:]]
    root = os.path.abspath(root)
    tests = json.load(open(tests_path))["tests"]
    byn = {t["number"]: t for t in tests}
    preload.install_environment(root)
    preload.import_sqlglot()
    mods = sorted({t["file"][:-3].replace("/", ".") for t in tests})
    preload.import_tests(mods)
    assert reduce.install_loaders()
    preload.build_class_fixtures(set(preload.SHARED_FIXTURES))
    iterred.install()
    ctx = Ctx(root)
    ctx.index = engine.FunctionIndex(root)
    for n in numbers:
        path = os.path.join(out_dir, "%d.json.gz" % n)
        if os.path.exists(path):
            continue
        warm = json.load(gzip.open(os.path.join(warm_dir, "%d.json.gz" % n)))
        t0 = time.time()
        if not warm.get("ok") or not warm.get("sites"):
            rec = {"n": n, "err": "no warm"}
        else:
            mode = "stmt" if os.path.exists("/maps/t/%d.bin" % n) else "all"
            rec = trace(ctx, n, byn[n]["id"], warm, mode)
        with gzip.open(path + ".tmp", "wt") as f:
            json.dump(rec, f)
        os.replace(path + ".tmp", path)
        print(n, {s: (len(v.get("paths", [])) if "err" not in v else v["err"][:80]) for s, v in rec.get("stmts", {}).items()}, rec.get("err", ""), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
