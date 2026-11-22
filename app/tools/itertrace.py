"""Warm per-iteration coverage of tests (estimate tool; pinned commit only).

usage (inside br-label): python itertrace.py REPO TESTS_JSON OUT_DIR MODE NUMBERS...
MODE warm: one run per test with every iteration tagged.
"""
from __future__ import annotations

import ast
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
import stmtmap
from reduce import STATE

CO_OPTIMIZED = 0x1
TOOL = 4


def method_node(src, cls_name, meth):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == meth:
                    return item
    return None


class Ctx:
    def __init__(self, root):
        self.root = root
        self.cache = {}
        self.index = None

    def info(self, rel):
        v = self.cache.get(rel)
        if v is None:
            src = open(os.path.join(self.root, rel), encoding="utf-8").read()
            units, flags, inames = stmtmap.build_units(src)
            v = self.cache[rel] = (src, src.split("\n"), reduce.file_shapes(src), flags, inames)
        return v


def prepare(ctx, test_id, keep_stmts=None, rewrite=None):
    """Swaps in the rewritten method. Returns the site list or None."""
    rel, cls_name, meth = test_id.split("::")
    src, lines, shapes, flags, inames = ctx.info(rel)
    shape = shapes.get((cls_name, meth))
    func = method_node(src, cls_name, meth)
    if shape is None or func is None:
        return None
    first, last, stmts = shape
    sites = iterred.method_sites(func, [s[2] for s in stmts])
    if not sites:
        return []
    full = os.path.join(ctx.root, rel)
    old_lines = lines[first - 1:last]
    base_lines = old_lines if keep_stmts is None else reduce.reduced_lines(lines, shape, keep_stmts)
    if rewrite is None:
        rewrite = set(range(len(stmts))) if keep_stmts is None else keep_stmts
    new_lines = iterred.rewrite_lines(base_lines, first, [s for s in sites if s[1] in rewrite])
    old_mod = engine.compile_lines(old_lines, first, (cls_name,), full, flags, inames)
    new_mod = engine.compile_lines(new_lines, first, (cls_name,), full, flags, inames)
    old = reduce.find_code(old_mod, meth, first)
    new = reduce.find_code(new_mod, meth, first)
    live = ctx.index.live(old)
    if len(live) != 1 or len(old.co_freevars) != len(new.co_freevars):
        return None
    live[0].__code__ = new
    return sites


def child(ctx, number, test_id, wfd):
    out = {"n": number, "ok": False}
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        sites = prepare(ctx, test_id)
        if sites is None:
            out["err"] = "no prepare"
            os.write(wfd, json.dumps(out).encode())
            return
        out["sites"] = [[k, i, lp.lineno] for k, i, lp in sites]
        root = ctx.root
        cut = len(root) + 1
        units = {}
        cpu = {}
        cur = [()]
        t_last = [time.process_time()]
        bad = set()
        mon = sys.monitoring
        mon.use_tool_id(TOOL, "iters")

        def on_line(code, line):
            if code.co_flags & CO_OPTIMIZED:
                fn = code.co_filename
                if fn.startswith(root) and fn[cut:cut + 8] == "sqlglot/":
                    units.setdefault(cur[0], {}).setdefault(fn[cut:], set()).add(line)
            return mon.DISABLE

        def on_start(code, offset):
            return on_line(code, code.co_firstlineno)

        def hook(p):
            t = time.process_time()
            cpu[cur[0]] = cpu.get(cur[0], 0.0) + t - t_last[0]
            cur[0] = p
            mon.restart_events()
            t_last[0] = time.process_time()

        def on_enter(tag):
            if iterred.LOOPS.path:
                bad.update(s for s, _ in iterred.LOOPS.path)

        def on_submit(index):
            if iterred.LOOPS.path:
                bad.update(s for s, _ in iterred.LOOPS.path)

        STATE.on_enter = on_enter
        STATE.on_submit = on_submit
        iterred.LOOPS.hook = hook
        mon.register_callback(TOOL, mon.events.LINE, on_line)
        mon.register_callback(TOOL, mon.events.PY_START, on_start)
        case = runner.load_case(test_id, number)

        class Result(unittest.TestResult):
            def startTest(self, test):
                super().startTest(test)
                STATE.reset(None, None)
                STATE.in_test = True
                iterred.LOOPS.reset()

            def stopTest(self, test):
                super().stopTest(test)
                STATE.in_test = False

        result = Result()
        result.failfast = False
        t0 = time.process_time()
        mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)
        unittest.TestSuite([case]).run(result)
        mon.set_events(TOOL, 0)
        hook(())
        out["cpu"] = round(time.process_time() - t0, 4)
        out["ok"] = not (result.failures or result.errors or result.unexpectedSuccesses)
        out["bad"] = sorted(bad)
        out["tags"] = [[[list(x) for x in k], round(cpu.get(k, 0.0), 5), {f: sorted(v) for f, v in d.items()}]
                       for k, d in units.items()]
        for k, c in cpu.items():
            if k not in units:
                out["tags"].append([[list(x) for x in k], round(c, 5), {}])
        os.write(wfd, json.dumps(out).encode())
    except BaseException as e:  # noqa: BLE001
        out["err"] = repr(e)[:300]
        try:
            os.write(wfd, json.dumps(out).encode())
        except OSError:
            pass


def measure(ctx, number, test_id):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            child(ctx, number, test_id, w)
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
    return json.loads(buf) if buf else {"n": number, "ok": False, "err": "no output"}


def main():
    root, tests_path, out_dir, mode = sys.argv[1:5]
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
        t0 = time.time()
        rec = measure(ctx, n, byn[n]["id"])
        with gzip.open(path + ".tmp", "wt") as f:
            json.dump(rec, f)
        os.replace(path + ".tmp", path)
        print(n, rec.get("ok"), rec.get("err", ""), len(rec.get("tags") or []), round(time.time() - t0, 2), flush=True)


if __name__ == "__main__":
    main()
