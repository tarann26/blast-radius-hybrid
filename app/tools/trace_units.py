"""Offline tracer (pinned commit only): for each test, which lines does each independent part run?

Each measurement forks a child from a parent that is in the same state as the selector's
parent at run time (sqlglot and the test modules imported, nothing run). The child runs the
test with exactly one part kept, so its coverage is that of a cold start.

usage (inside the pinned image):
    python trace_units.py REPO TESTS_JSON OUT_DIR SHARD NSHARDS
"""
from __future__ import annotations

import gzip
import json
import os
import resource
import select
import signal
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine
import preload
import reduce
import runner
import stmtmap
from reduce import STATE

CO_OPTIMIZED = 0x1
TOOL = 3


def cpu_now():
    r = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime + c.ru_utime + c.ru_stime


class Ctx:
    def __init__(self, root, tests):
        self.root = root
        self.tests = tests
        self.index = None
        self.file_cache = {}

    def file_info(self, rel):
        info = self.file_cache.get(rel)
        if info is None:
            src = open(os.path.join(self.root, rel), encoding="utf-8").read()
            units, flags, inames = stmtmap.build_units(src)
            info = self.file_cache[rel] = (src.split("\n"), reduce.file_shapes(src), flags, inames)
        return info


def apply_reduction(ctx: Ctx, test_id: str, keep_stmts):
    """Replaces the test method's code by the reduced one. Returns False when not possible."""
    rel, cls_name, meth = test_id.split("::")
    lines, shapes, flags, inames = ctx.file_info(rel)
    shape = shapes.get((cls_name, meth))
    if shape is None:
        return False
    first = shape[0]
    full = os.path.join(ctx.root, rel)
    old_mod = engine.compile_lines(lines[first - 1:shape[1]], first, (cls_name,), full, flags, inames)
    new_mod = engine.compile_lines(reduce.reduced_lines(lines, shape, keep_stmts), first, (cls_name,), full, flags, inames)
    old = reduce.find_code(old_mod, meth, first)
    new = reduce.find_code(new_mod, meth, first)
    if old is None or new is None:
        return False
    live = ctx.index.live(old)
    if len(live) != 1 or len(ctx.index.any_at(old)) != 1:
        return False
    if len(old.co_freevars) != len(new.co_freevars):
        return False
    live[0].__code__ = new
    return True


def init_codes():
    """The __init__ code objects of every class defined in sqlglot (to see instantiations)."""
    codes = set()
    seen = set()

    def visit(cls):
        if cls in seen:
            return
        seen.add(cls)
        f = cls.__dict__.get("__init__")
        if isinstance(f, type(init_codes)):
            codes.add(f.__code__)
        for v in list(cls.__dict__.values()):
            if isinstance(v, type) and (v.__module__ or "").startswith("sqlglot"):
                visit(v)

    for name, m in list(sys.modules.items()):
        if m is not None and (name == "sqlglot" or name.startswith("sqlglot.")):
            for v in list(vars(m).values()):
                if isinstance(v, type) and (v.__module__ or "").startswith("sqlglot"):
                    visit(v)
    return codes


def child_classes(ctx, number, test_id, keep_stmts, keep_files, keep_tasks, wfd):
    """Runs in the forked child: which sqlglot classes does this run instantiate?"""
    out = {"ok": False, "reduced": True}
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        if keep_stmts is not None and not apply_reduction(ctx, test_id, keep_stmts):
            out["reduced"] = False
            os.write(wfd, json.dumps(out).encode())
            return
        seen = set()
        mon = sys.monitoring
        mon.use_tool_id(TOOL, "classes")

        def on_start(code, offset):
            obj = sys._getframe(1).f_locals.get(code.co_varnames[0])
            seen.add(type(obj))

        mon.register_callback(TOOL, mon.events.PY_START, on_start)
        for code in ctx.init_codes:
            mon.set_local_events(TOOL, code, mon.events.PY_START)
        case = runner.load_case(test_id, number)

        class Result(unittest.TestResult):
            def startTest(self, test):
                super().startTest(test)
                STATE.reset(keep_files, keep_tasks)
                STATE.in_test = True

            def stopTest(self, test):
                super().stopTest(test)
                STATE.in_test = False

        result = Result()
        unittest.TestSuite([case]).run(result)
        out["ok"] = not (result.failures or result.errors or result.unexpectedSuccesses)
        out["classes"] = sorted("%s.%s" % (c.__module__, c.__qualname__) for c in seen if (c.__module__ or "").startswith("sqlglot"))
        os.write(wfd, json.dumps(out).encode())
    except BaseException as e:  # noqa: BLE001
        out["err"] = repr(e)[:300]
        try:
            os.write(wfd, json.dumps(out).encode())
        except OSError:
            pass


def child_classes_full(ctx, number, test_id, wfd):
    """Runs in the forked child (defgap): the whole test, classes recorded like the first tracer
    (trace/trace_one.py on_start): at the start of every sqlglot function with an argument, the
    class of its first argument (the argument itself when it is a class). The pool is the
    in-process one (rt/pool.py), so the work of pool tasks counts for this test."""
    out = {"ok": False}
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        seen = set()
        mon = sys.monitoring
        mon.use_tool_id(TOOL, "classesfull")
        prefix = ctx.root + "/sqlglot/"

        def on_start(code, offset):
            if code.co_argcount == 0 or not code.co_filename.startswith(prefix):
                return mon.DISABLE
            try:
                obj = sys._getframe(1).f_locals.get(code.co_varnames[0])
            except Exception:
                return None
            seen.add(obj if isinstance(obj, type) else type(obj))
            return None

        mon.register_callback(TOOL, mon.events.PY_START, on_start)
        STATE.reset(None, None)
        case = runner.load_case(test_id, number)

        class Result(unittest.TestResult):
            def startTest(self, test):
                super().startTest(test)
                STATE.reset(None, None)
                STATE.in_test = True

            def stopTest(self, test):
                super().stopTest(test)
                STATE.in_test = False

        result = Result()
        mon.set_events(TOOL, mon.events.PY_START)
        unittest.TestSuite([case]).run(result)
        mon.set_events(TOOL, 0)
        out["ok"] = not (result.failures or result.errors or result.unexpectedSuccesses)
        names = set()
        for c in seen:
            m = getattr(c, "__module__", "") or ""
            if m.startswith("sqlglot") or m.startswith("tests"):
                names.add(m + "." + c.__qualname__)
        out["classes"] = sorted(names)
        out["tasks"] = STATE.tasks_seen
        os.write(wfd, json.dumps(out).encode())
    except BaseException as e:  # noqa: BLE001
        out["err"] = repr(e)[:300]
        try:
            os.write(wfd, json.dumps(out).encode())
        except OSError:
            pass


def child_run(ctx, number, test_id, keep_stmts, keep_files, keep_tasks, wfd, by_unit):
    """Runs in the forked child. Writes one JSON object to wfd."""
    out = {"ok": False, "reduced": True}
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        if keep_stmts is not None:
            if not apply_reduction(ctx, test_id, keep_stmts):
                out["reduced"] = False
                os.write(wfd, json.dumps(out).encode())
                return
        STATE.reset(keep_files, keep_tasks)
        root = ctx.root
        cut = len(root) + 1
        cov = {}
        units = {}
        cur = [None]
        mon = sys.monitoring
        mon.use_tool_id(TOOL, "units")

        def on_line(code, line):
            if code.co_flags & CO_OPTIMIZED:
                fn = code.co_filename
                if fn.startswith(root) and fn[cut:cut + 8] == "sqlglot/":
                    if by_unit:
                        units.setdefault(cur[0], {}).setdefault(fn[cut:], []).append(line)
                    else:
                        cov.setdefault(fn[cut:], []).append(line)
            return mon.DISABLE

        def on_start(code, offset):
            # a one-line lambda or function gives no LINE event: count its entry as its first line
            return on_line(code, code.co_firstlineno)

        if by_unit:
            def on_enter(tag):
                cur[0] = tag
                mon.restart_events()

            def on_submit(index):
                # lines since the last event prepared this task's arguments
                pending = units.pop(cur[0], None) if (cur[0] is None or cur[0][0] == "p") else None
                if pending:
                    bucket = units.setdefault(("w", index), {})
                    for f, ls in pending.items():
                        bucket.setdefault(f, []).extend(ls)
                cur[0] = ("p", index)
                mon.restart_events()

            STATE.on_enter = on_enter
            STATE.on_submit = on_submit
        mon.register_callback(TOOL, mon.events.LINE, on_line)
        mon.register_callback(TOOL, mon.events.PY_START, on_start)
        case = runner.load_case(test_id, number)

        class Result(unittest.TestResult):
            def startTest(self, test):
                super().startTest(test)
                STATE.reset(keep_files, keep_tasks)
                STATE.in_test = True

            def stopTest(self, test):
                super().stopTest(test)
                STATE.in_test = False

        result = Result()
        result.failfast = False
        suite = unittest.TestSuite([case])
        c0 = cpu_now()
        mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)
        suite.run(result)
        mon.set_events(TOOL, 0)
        out["cpu"] = round(cpu_now() - c0, 4)
        out["ok"] = not (result.failures or result.errors or result.unexpectedSuccesses)
        out["files"] = STATE.files_seen
        out["tasks"] = STATE.tasks_seen
        STATE.in_test = False
        if by_unit:
            out["units"] = [[list(k) if k else None, v] for k, v in units.items()]
        else:
            out["cov"] = cov
        os.write(wfd, json.dumps(out).encode())
    except BaseException as e:  # noqa: BLE001
        out["err"] = repr(e)[:300]
        try:
            os.write(wfd, json.dumps(out).encode())
        except OSError:
            pass


def measure(ctx, number, test_id, keep_stmts, keep_files, keep_tasks, by_unit=False, timeout=2400):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            os.setpgid(0, 0)
        except OSError:
            pass
        try:
            if by_unit == "classesfull":
                child_classes_full(ctx, number, test_id, w)
            elif by_unit == "classes":
                child_classes(ctx, number, test_id, keep_stmts, keep_files, keep_tasks, w)
            else:
                child_run(ctx, number, test_id, keep_stmts, keep_files, keep_tasks, w, by_unit)
        finally:
            os._exit(0)
    os.close(w)
    buf = b""
    deadline = time.monotonic() + timeout
    timed_out = False
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            timed_out = True
            break
        ready, _, _ = select.select([r], [], [], min(left, 2.0))
        if ready:
            chunk = os.read(r, 1 << 20)
            if not chunk:
                break
            buf += chunk
    os.close(r)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    os.waitpid(pid, 0)
    if timed_out or not buf:
        return {"ok": False, "timeout": timed_out, "reduced": True}
    return json.loads(buf)


def minus(cov, base):
    out = {}
    for f, lines in cov.items():
        b = base.get(f)
        left = sorted(set(lines) - b) if b else sorted(set(lines))
        if left:
            out[f] = left
    return out


def trace_test(ctx, number, test_id):
    rel, cls_name, meth = test_id.split("::")
    lines, shapes, flags, inames = ctx.file_info(rel)
    shape = shapes.get((cls_name, meth))
    rec = {"number": number, "id": test_id, "reducible": False, "units": []}
    full = measure(ctx, number, test_id, None, None, None)
    rec["full_ok"] = full.get("ok")
    rec["full_cpu"] = full.get("cpu")
    rec["full_cov_size"] = sum(len(v) for v in full.get("cov", {}).values())
    if shape is None or not full.get("ok"):
        rec["why"] = "no shape" if shape is None else "full run failed"
        return rec
    stmts = shape[2]
    droppable = [i for i, s in enumerate(stmts) if s[2]]
    base = measure(ctx, number, test_id, set(), {}, set())
    if not base.get("reduced") or not base.get("ok"):
        rec["why"] = "base run failed: %r" % ({k: base.get(k) for k in ("ok", "reduced", "err", "timeout")},)
        return rec
    base_cov = {f: set(v) for f, v in base["cov"].items()}
    rec["base"] = {f: sorted(v) for f, v in base_cov.items()}
    rec["base_cpu"] = base["cpu"]
    rec["n_stmts"] = len(stmts)
    all_ok = True
    seen_files = {}
    seen_tasks = 0

    def add(key, parent, m):
        nonlocal all_ok
        ok = bool(m.get("ok")) and m.get("reduced", True)
        if not ok:
            all_ok = False
        rec["units"].append({"key": key, "parent": parent, "ok": ok, "cpu": m.get("cpu"),
                             "cov": minus(m.get("cov", {}), base_cov)})

    # entries and tasks that run even with every droppable statement removed
    for name, count in (base.get("files") or {}).items():
        seen_files[name] = count
        for i in range(count):
            add(["f", name, i], None, measure(ctx, number, test_id, set(), {name: {i}}, set()))
    for i in range(base.get("tasks") or 0):
        add(["t", i], None, measure(ctx, number, test_id, set(), {}, {i}))
        seen_tasks = max(seen_tasks, i + 1)
    for k in droppable:
        whole = measure(ctx, number, test_id, {k}, None, None)
        files = {n: c for n, c in (whole.get("files") or {}).items() if n not in seen_files}
        tasks = (whole.get("tasks") or 0) - seen_tasks
        if not whole.get("ok") or not whole.get("reduced", True):
            add(["s", k], None, whole)
            continue
        if not files and tasks <= 0:
            add(["s", k], None, whole)
            continue
        own = measure(ctx, number, test_id, {k}, {}, set())
        add(["s", k], None, own)
        for name, count in files.items():
            for i in range(count):
                add(["f", name, i], k, measure(ctx, number, test_id, {k}, {name: {i}}, set()))
        for i in range(seen_tasks, seen_tasks + max(tasks, 0)):
            add(["t", i], k, measure(ctx, number, test_id, {k}, {}, {i}))
    rec["reducible"] = all_ok
    warm = measure(ctx, number, test_id, None, None, None, by_unit=True)
    rec["warm_ok"] = warm.get("ok")
    rec["warm"] = [[k, minus(v, base_cov)] for k, v in (warm.get("units") or [])]
    return rec


def main():
    root, tests_path, out_dir, shard, nshards = sys.argv[1:6]
    shard, nshards = int(shard), int(nshards)
    root = os.path.abspath(root)
    tests = json.load(open(tests_path))["tests"]
    only = set(int(x) for x in os.environ["BR_ONLY_TESTS"].split(",")) if os.environ.get("BR_ONLY_TESTS") else None
    preload.install_environment(root)
    preload.import_sqlglot()
    mods = sorted({t["file"][:-3].replace("/", ".") for t in tests})
    preload.import_tests(mods)
    ok = reduce.install_loaders()
    assert ok
    # the same shared class fixtures as the selector's parent process
    preload.build_class_fixtures(set(preload.SHARED_FIXTURES))
    ctx = Ctx(root, tests)
    ctx.index = engine.FunctionIndex(root)
    ctx.init_codes = init_codes()
    os.makedirs(out_dir, exist_ok=True)
    # big tests first inside each shard so that the shards end at similar times
    todo = [t for t in tests if t["number"] % nshards == shard and (only is None or t["number"] in only)]
    todo.sort(key=lambda t: -t["seconds"])
    for t in todo:
        path = os.path.join(out_dir, "%d.json.gz" % t["number"])
        if os.path.exists(path):
            continue
        t0 = time.time()
        if os.environ.get("BR_CLASSES"):
            # classes instantiated by the always-run part and by each part (same runs as the line trace)
            src = os.path.join(os.environ["BR_CLASSES"], "%d.json.gz" % t["number"])
            rec = json.load(gzip.open(src)) if os.path.exists(src) else {}
            out = {"number": t["number"], "base": None, "units": []}
            if rec.get("reducible"):
                has_child = {u["parent"] for u in rec["units"] if u["key"][0] in ("f", "t") and u.get("parent") is not None}
                m = measure(ctx, t["number"], t["id"], set(), {}, set(), by_unit="classes")
                out["base"] = m.get("classes")
                for u in rec["units"]:
                    k = u["key"]
                    parent = u.get("parent")
                    if k[0] == "s":
                        args = ({k[1]}, {}, set()) if k[1] in has_child else ({k[1]}, None, None)
                    elif k[0] == "f":
                        args = (set() if parent is None else {parent}, {k[1]: {k[2]}}, set())
                    else:
                        args = (set() if parent is None else {parent}, {}, {k[1]})
                    m = measure(ctx, t["number"], t["id"], args[0], args[1], args[2], by_unit="classes")
                    out["units"].append(m.get("classes") if m.get("ok") else None)
            with gzip.open(path + ".tmp", "wt") as f:
                json.dump(out, f)
            os.replace(path + ".tmp", path)
            print(t["number"], "classes", len(out["units"]), round(time.time() - t0, 2), flush=True)
            continue
        if os.environ.get("BR_CLASSES_FULL"):
            m = measure(ctx, t["number"], t["id"], None, None, None, by_unit="classesfull")
            with gzip.open(path + ".tmp", "wt") as f:
                json.dump({"number": t["number"], "ok": m.get("ok"), "classes": m.get("classes"),
                           "tasks": m.get("tasks"), "err": m.get("err"), "timeout": m.get("timeout")}, f)
            os.replace(path + ".tmp", path)
            print(t["number"], "classesfull", m.get("ok"), len(m.get("classes") or ()), round(time.time() - t0, 2), flush=True)
            continue
        if os.environ.get("BR_FULL_ONLY"):
            m = measure(ctx, t["number"], t["id"], None, None, None)
            with gzip.open(path + ".tmp", "wt") as f:
                json.dump({"number": t["number"], "ok": m.get("ok"), "cov": m.get("cov", {})}, f)
            os.replace(path + ".tmp", path)
            print(t["number"], "full", m.get("ok"), round(time.time() - t0, 2), flush=True)
            continue
        try:
            rec = trace_test(ctx, t["number"], t["id"])
        except Exception as e:  # noqa: BLE001
            rec = {"number": t["number"], "id": t["id"], "reducible": False, "units": [], "why": "tracer error %r" % (e,)}
        rec["wall"] = round(time.time() - t0, 2)
        with gzip.open(path + ".tmp", "wt") as f:
            json.dump(rec, f)
        os.replace(path + ".tmp", path)
        print(t["number"], t["id"], "reducible", rec["reducible"], "units", len(rec["units"]), "wall", rec["wall"], rec.get("why", ""), flush=True)


if __name__ == "__main__":
    main()
