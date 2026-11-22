"""Builds maps/memo.bin and maps/memo.dat for memo.py (pinned commit only).

usage (pinned interpreter, PYTHONHASHSEED=0, as the selector runs):
    python build_memo.py REPO STAGES_BIN OUT_DIR [JOBS]

For every query that test_execute_tpcds executes, a forked child of a process that has
imported sqlglot and the test module and read the TPC-DS tables (as setUpClass does) runs
the test's task function (tests/test_executor.py mp_execute) twice with a line monitor. A
copy of PythonExecutor._execute (memo.execute's loop, with every step run) records per step
of the canonical order: the chained plan key, the content hash of the output, its CPU time,
the sqlglot lines of its dynamic extent (both runs), and for the steps that feed an
Aggregate, Sort or SetOperation step, the root, and every small output, the output itself.
The lines that only the first run executes inside steps (first use in a process) go to
"first": a commit that changes one of them gets no memo. The expression, tables and rows
fingerprints must equal those of stages.bin (the test as the selector records it)."""
import gzip
import marshal
import os
import pickle
import select
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import memo  # noqa: E402
import stages  # noqa: E402

TEST_ID = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
TOOL = 4
CO_OPTIMIZED = 0x1
SMALL = 64 << 10
BIG = 16 << 20
STORE_CPU = 0.02  # CPU s of a step whose output is stored whatever its dependents are
# The step costs come from a fixed file (maps/memo_fixed.bin: per query the step count, the CPU s of
# every step, the steps whose output is stored for their cost, the optimize() CPU s), recorded once on
# the pinned commit, so that the store and the costs in memo.bin do not depend on the build machine's
# speed or load. A query whose step count differs from the file falls back to the measured CPU.
FIXED = {}


def load_fixed(path):
    try:
        with open(path, "rb") as f:
            return marshal.loads(zlib.decompress(f.read()))
    except Exception as e:  # noqa: BLE001
        print("memo: no fixed step costs (%s); measured CPU decides" % e, file=sys.stderr)
        return {}


def fixed_of(q, n):
    fx = FIXED.get(q)
    return fx if fx is not None and fx.get("n") == n else None


def setup(root):
    os.chdir(root)
    sys.path.insert(0, root)
    import ast
    import csv

    import tests.test_executor as te
    from tests.helpers import TPCDS_SCHEMA

    tables = {}
    for table, columns in TPCDS_SCHEMA.items():
        rows, ctypes = [], []
        reader = csv.reader(te.open_file(te.DIR_TPCDS + table + ".csv.gz"), delimiter="|")
        next(reader)
        for row in reader:
            if not ctypes:
                for v in row:
                    try:
                        ctypes.append(type(ast.literal_eval(v)))
                    except (ValueError, SyntaxError):
                        ctypes.append(str)
            rows.append(tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row)))
        tables[table] = te.Table(columns=columns, rows=rows)
    sqls = list(te.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
    return te, TPCDS_SCHEMA, tables, sqls


def text_lines(root, te, sql):
    """The sqlglot lines that the stage check's transpile(sql, write="duckdb") runs in a fresh
    fork of the parent (cold: first-use lines too) and the hash of its text."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        res = None
        try:
            cut = len(root) + 1
            seen = set()
            mon = sys.monitoring

            def on_line(code, line):
                fn = code.co_filename
                if code.co_flags & CO_OPTIMIZED and fn.startswith(root) and fn[cut:cut + 8] == "sqlglot/":
                    seen.add((fn[cut:], line))
                return mon.DISABLE

            mon.use_tool_id(TOOL, "memo-text")
            mon.register_callback(TOOL, mon.events.LINE, on_line)
            mon.register_callback(TOOL, mon.events.PY_START, lambda code, off: on_line(code, code.co_firstlineno))
            mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)
            text = te.transpile(sql, write="duckdb")[0]
            mon.set_events(TOOL, 0)
            res = (sorted(seen), stages.hashlib.blake2b(text.encode(), digest_size=16).digest())
        except BaseException:  # noqa: BLE001
            res = None
        finally:
            data = marshal.dumps(res)
            while data:
                data = data[os.write(w, data):]
            os._exit(0)
    os.close(w)
    chunks = []
    while True:
        c = os.read(r, 1 << 20)
        if not c:
            break
        chunks.append(c)
    os.close(r)
    os.waitpid(pid, 0)
    return marshal.loads(b"".join(chunks))


def child(root, q, te, schema, tables, sqls, want, wfd):
    out = {"ok": False, "q": q}
    try:
        cold = text_lines(root, te, sqls[q][1])  # (before anything of this child runs)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        from sqlglot import planner, parse_one
        from sqlglot.executor import python as PX
        from sqlglot.executor import table as TB

        mods = (planner, TB.Table, TB.RowReader, TB.RangeReader)
        cut = len(root) + 1
        lines = {}
        bucket = [None]
        mon = sys.monitoring

        def on_line(code, line):
            b = bucket[0]
            if b is not None and code.co_flags & CO_OPTIMIZED:
                fn = code.co_filename
                if fn.startswith(root) and fn[cut:cut + 8] == "sqlglot/":
                    lines.setdefault(b, set()).add((fn[cut:], line))
            return mon.DISABLE

        def switch(b):
            bucket[0] = b
            mon.restart_events()

        mon.use_tool_id(TOOL, "memo")
        mon.register_callback(TOOL, mon.events.LINE, on_line)
        mon.register_callback(TOOL, mon.events.PY_START, lambda code, off: on_line(code, code.co_firstlineno))
        real_execute = PX.PythonExecutor._execute
        depth = [0]
        runs = []
        cur = [0]  # the run (0 or 1)

        def rec_execute(self, plan):
            if depth[0]:
                return real_execute(self, plan)
            depth[0] += 1
            try:
                return record(self, plan)
            finally:
                depth[0] -= 1

        def record(self, plan):
            r = len(runs)
            co = memo.canon_order(plan)
            if co is None:
                raise RuntimeError("plan shape")
            order, idx = co
            keys = memo.plan_keys(order, idx)
            if keys is None:
                raise RuntimeError("plan key")
            steps = [None] * len(order)
            fx = fixed_of(q, len(order))
            hot = set(fx["store"]) if fx is not None else None
            run = {"keys": keys, "steps": steps, "order": [type(s).__name__ for s in order],
                   "feeds": [any(isinstance(d, (planner.Aggregate, planner.Sort, planner.SetOperation)) for d in s.dependents) for s in order],
                   "root": idx[id(plan.root)]}
            runs.append(run)
            finished, contexts = set(), {}
            queue = set(plan.leaves)
            while queue:
                node = queue.pop()
                i = idx[id(node)]
                context = self.context({name: table for dep in node.dependencies for name, table in contexts[dep].tables.items()})
                switch((r, i))
                t0 = time.process_time()
                if isinstance(node, planner.Scan):
                    o = self.scan(node, context)
                elif isinstance(node, planner.Aggregate):
                    o = self.aggregate(node, context)
                elif isinstance(node, planner.Join):
                    o = self.join(node, context)
                elif isinstance(node, planner.Sort):
                    o = self.sort(node, context)
                elif isinstance(node, planner.SetOperation):
                    o = self.set_operation(node, context)
                else:
                    raise NotImplementedError
                if node.offset:
                    tb = o.tables[node.name]
                    tb.rows = tb.rows[node.offset:]
                dt = time.process_time() - t0
                switch((r, "frame"))
                mine = memo._objs(o)
                if any(mine & memo._objs(contexts[d]) for d in node.dependencies):
                    run.setdefault("alias", set()).add(i)  # (returns an input object)
                st = memo.out_state(o, *mods[1:])
                if st is None:
                    raise RuntimeError("output state of step %d" % i)
                h = memo.state_hash(st)
                blob = None
                if r == 1:
                    data = zlib.compress(pickle.dumps(st, 4), 6)
                    # (also a step that costs STORE_CPU or more: a step that runs in the stage check,
                    # because it is dirty or follows a step with another output, takes such an
                    # input from here instead of running it)
                    costly = (i in hot) if hot is not None else dt >= STORE_CPU
                    if len(data) <= SMALL or ((run["feeds"][i] or i == run["root"] or costly) and len(data) <= BIG):
                        blob = data
                        if memo.state_hash(pickle.loads(zlib.decompress(data))) != h:
                            raise RuntimeError("stored output of step %d does not hash back" % i)
                steps[i] = (h, dt, blob, memo.content_hash(st))
                if len(node.dependents) > 1:
                    run.setdefault("cont", {})[i] = memo.content_hash(st)
                contexts[node] = o
                finished.add(node)
                for dep in node.dependents:
                    if all(d in contexts for d in dep.dependencies):
                        queue.add(dep)
                # every live shared output (not only the direct inputs: an alias can reach one)
                # must keep its content
                for dep, ctx in contexts.items():
                    if dep is not node and len(dep.dependents) > 1:
                        s2 = memo.out_state(ctx, *mods[1:])
                        if s2 is None or memo.content_hash(s2) != run["cont"][idx[id(dep)]]:
                            raise RuntimeError("a step changed a shared output")
                for dep in node.dependencies:
                    if all(d in finished for d in dep.dependents):
                        contexts.pop(dep)
            run["subq"] = bool(self._subquery_plans)
            rootn = plan.root
            run["rname"] = rootn.name
            return contexts[rootn].tables[rootn.name]

        PX.PythonExecutor._execute = rec_execute
        g = te.execute.__globals__
        real_plan = g["Plan"]
        fps = []

        def plan_hook(expression):
            fps.append(stages.fingerprint(expression))
            if cur[0] == 1:
                switch(None)
                mtext.append(expression.sql(pretty=True))  # (the text of execute()'s logger.debug line)
            switch((cur[0], "plan"))
            try:
                return real_plan(expression)
            finally:
                switch((cur[0], "mid"))

        mtext = []
        g["Plan"] = plan_hook
        real_px = g["PythonExecutor"]

        class PXHook(real_px):
            def __init__(self, *a, **k):
                fps.append(stages.fingerprint(k.get("tables"), stages.table_leaves(tables)))
                switch((cur[0], "frame"))
                super().__init__(*a, **k)

            def execute(self, *a, **k):
                switch((cur[0], "frame"))
                try:
                    return super().execute(*a, **k)
                finally:
                    switch((cur[0], "post"))

        g["PythonExecutor"] = PXHook
        real_opt = g["optimize"]
        opt = {}

        def opt_hook(*a, **k):
            r = cur[0]
            fp_in = stages.fingerprint(a[0]) if a else None
            switch((r, "opt"))
            t0 = time.process_time()
            try:
                return_value = real_opt(*a, **k)
            finally:
                dt = time.process_time() - t0
                switch((r, "mid"))
            opt.setdefault("cpu", []).append(dt)
            opt.setdefault("in", []).append(fp_in)
            opt.setdefault("call", []).append((len(a), sorted(k.items())))
            if r == 1:
                fp_out = stages.fingerprint(return_value)
                data = memo.dump_expr(return_value)
                back = memo.load_expr(data)
                kfp = memo._key_fp(return_value)
                # the loaded expression equals the computed one but for string identity (a loaded
                # string is not the interned one of the code): compare() checks it with kfp
                if kfp is not None and memo._key_fp(back) == kfp:
                    opt["blob"], opt["out"], opt["kfp"] = data, fp_out, kfp  # (else no record: optimize() runs)
            return return_value

        g["optimize"] = opt_hook
        te.initializer(schema, tables)
        meta, sql = sqls[q][:2]
        mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)
        results = []
        for r in range(2):
            cur[0] = r
            switch((r, "pre"))
            t0 = time.process_time()
            results.append(te.mp_execute(pickle.loads(pickle.dumps(parse_one(sql))), meta))  # (as the pool hands it over)
            out.setdefault("cpu", []).append(time.process_time() - t0)
        switch((1, "text"))
        warm_text = te.transpile(sql, write="duckdb")[0]
        switch(None)
        mon.set_events(TOOL, 0)
        a, b = runs
        text_ok = (cold is not None and cold[1] == want[2]
                   and stages.hashlib.blake2b(warm_text.encode(), digest_size=16).digest() == want[2])
        out["text"] = sorted(set(map(tuple, cold[0])) | lines.get((1, "text"), set())) if text_ok else None
        out["sqlh"] = stages.hashlib.blake2b(sql.encode("utf-8", "surrogatepass"), digest_size=16).digest()
        out["mtext"] = mtext[0] if len(mtext) == 1 else None
        rootn = runs[1]["rname"]
        out["rname"] = rootn
        if a["keys"] != b["keys"]:
            raise RuntimeError("runs differ: keys")
        if [s[3] for s in a["steps"]] != [s[3] for s in b["steps"]]:
            raise RuntimeError("runs differ: content")
        if a.get("alias", set()) != b.get("alias", set()):
            raise RuntimeError("runs differ: alias")
        # a step whose readers were left at another row in the two runs: its content counts
        outs = [y[0] if x[0] == y[0] else b"c" + y[3] for x, y in zip(a["steps"], b["steps"])]
        if fps[0] != want[0] or fps[2] != want[0] or fps[1] != want[1] or fps[3] != want[1]:
            raise RuntimeError("input fingerprints differ from stages.bin")
        if any(stages.rows_fingerprint(t) != want[3] for t in results):
            raise RuntimeError("rows differ from stages.bin")
        n = len(b["keys"])
        cover = []
        for i in range(n):
            cover.append(sorted(lines.get((0, i), set()) | lines.get((1, i), set())))
        step0 = set().union(*[lines.get((0, i), set()) for i in range(n)])
        all1 = set().union(*[v for k, v in lines.items() if k[0] == 1])
        def lines_of(*kinds):
            u = set()
            for r in range(2):
                for kd in kinds:
                    u |= lines.get((r, kd), set())
            return sorted(u)

        steps_all = set()
        for i in range(n):
            steps_all.update(cover[i])
        good_opt = (len(opt.get("in", ())) == 2 and opt["in"][0] == opt["in"][1] and opt["in"][0] is not None
                    and opt["call"][0] == opt["call"][1] == (2, [("dialect", None), ("leave_tables_isolated", True)])
                    and opt.get("out") == want[0])
        out.update(frame=lines_of("frame"), planl=lines_of("plan"), anyl=sorted(steps_all | set(lines_of("frame", "plan", "post"))),
                   early=lines_of("pre", "opt"), mid=lines_of("mid"),
                   opt={"in": opt["in"][0], "blob": opt["blob"], "cpu": round(min(opt["cpu"]), 5), "kfp": opt["kfp"]} if good_opt else None)
        out.update(ok=True, keys=b["keys"], outs=outs, cpu=[round(min(x[1], y[1]), 5) for x, y in zip(a["steps"], b["steps"])],
                   blobs={i: s[2] for i, s in enumerate(b["steps"]) if s[2] is not None}, cover=cover,
                   first=sorted(step0 - all1), subq=a["subq"] or b["subq"], order=b["order"], cont=b.get("cont", {}),
                   alias=sorted(b.get("alias", ())), chash=[s[3] for s in b["steps"]])
    except BaseException as e:  # noqa: BLE001
        import traceback

        out["err"] = traceback.format_exc()[-1500:]
    finally:
        data = marshal.dumps(out)
        while data:
            data = data[os.write(wfd, data):]
        os._exit(0)


def main():
    root, stages_bin, out_dir = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3]
    jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    record = stages.load_map(stages_bin)[TEST_ID]
    FIXED.update(load_fixed(os.path.join(HERE, "maps", "memo_fixed.bin")))
    te, schema, tables, sqls = setup(root)
    import sqlglot.executor.python  # noqa: F401  (imported before the forks)

    pending = sorted(record["tasks"])
    live, done = {}, {}
    while pending or live:
        while pending and len(live) < jobs:
            q = pending.pop(0)
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(r)
                child(root, q, te, schema, tables, sqls, record["tasks"][q], w)
            os.close(w)
            live[r] = (q, pid, [])
        ready, _, _ = select.select(list(live), [], [])
        for r in ready:
            q, pid, chunks = live[r]
            chunk = os.read(r, 1 << 20)
            if chunk:
                chunks.append(chunk)
                continue
            os.close(r)
            os.waitpid(pid, 0)
            del live[r]
            done[q] = marshal.loads(b"".join(chunks))
    queries, by_line, first, dat, skipped, t_line = {}, {}, {}, [], {}, {}
    off = 0
    for q in sorted(done):
        o = done[q]
        if not o.get("ok"):
            skipped[q] = o.get("err", "?")[-300:]
            continue
        n = len(o["keys"])
        fx = fixed_of(q, n)
        if fx is not None:
            o["cpu"] = list(fx["cpu"])
            if o["opt"] is not None and fx["opt"] is not None:
                o["opt"]["cpu"] = fx["opt"]
        cover = o["cover"]
        if o["subq"]:
            u = sorted(set().union(*map(set, cover)))
            cover = [u] * n
        for i, ls in enumerate(cover):
            for path, ln in ls:
                by_line.setdefault((path, ln), set()).add("%d:%d" % (q, i))
        for path, ln in o["first"]:
            first.setdefault(path, set()).add(ln)
        store = {}
        for i, blob in sorted(o["blobs"].items()):
            store[i] = (off, len(blob))
            dat.append(blob)
            off += len(blob)
        for kind, ls in (("F", o["frame"]), ("A", o["anyl"]), ("E", o["early"]), ("M", o["mid"]), ("P", o["planl"]), ("T", o["text"] or ())):
            for path, ln in ls:
                t_line.setdefault((path, ln), set()).add((kind, q))
        opt = None
        if o["opt"] is not None:
            opt = {"in": o["opt"]["in"], "blob": (off, len(o["opt"]["blob"])), "cpu": o["opt"]["cpu"], "kfp": o["opt"]["kfp"]}
            dat.append(o["opt"]["blob"])
            off += len(o["opt"]["blob"])
        queries[q] = {"keys": o["keys"], "outs": o["outs"], "cpu": o["cpu"], "store": store, "opt": opt,
                      "late": round(sum(o["cpu"]), 4), "order": o["order"], "subq": o["subq"], "cont": o["cont"],
                      "alias": o["alias"], "chash": o["chash"], "planl": True,
                      "text": o["text"] is not None, "sqlh": o["sqlh"], "mtext": o["mtext"], "rname": o["rname"]}
    sets, ids, cov = [], {}, {}
    for (path, ln), users in sorted(by_line.items()):
        key = tuple(sorted(users))
        sid = ids.get(key)
        if sid is None:
            sid = ids[key] = len(sets)
            sets.append(key)
        cov.setdefault(path, {})[ln] = sid
    tsets, tids, tcov = [], {}, {}
    for (path, ln), users in sorted(t_line.items()):
        key = tuple(sorted(users))
        sid = tids.get(key)
        if sid is None:
            sid = tids[key] = len(tsets)
            tsets.append(key)
        tcov.setdefault(path, {})[ln] = sid
    # the row content of every TPC-DS table (stages.StageCheck.rows_ok)
    rowh = stages.row_hashes(tables)
    index = {"version": memo.MEMO_VERSION, "probe": stages.hash_probe(), "queries": queries, "cover": cov, "rowh": rowh,
             "sets": sets, "first": {p: sorted(v) for p, v in first.items()}, "tcov": tcov, "tsets": tsets}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "memo.dat.tmp"), "wb") as f:
        for blob in dat:
            f.write(blob)
    os.replace(os.path.join(out_dir, "memo.dat.tmp"), os.path.join(out_dir, "memo.dat"))
    with open(os.path.join(out_dir, "memo.bin.tmp"), "wb") as f:
        f.write(zlib.compress(marshal.dumps(index, 2), 9))
    os.replace(os.path.join(out_dir, "memo.bin.tmp"), os.path.join(out_dir, "memo.bin"))
    print("memo: queries %d (with optimize record %d, unstable steps %d, alias steps %d) skipped %r steps %d stored %d (%.1f MB) first-use lines %d" % (
        len(queries), sum(1 for v in queries.values() if v["opt"] is not None), sum(1 for v in queries.values() for x in v["outs"] if len(x) == 17),
        sum(len(v["alias"]) for v in queries.values()), skipped, sum(len(v["keys"]) for v in queries.values()),
        sum(len(v["store"]) for v in queries.values()), off / 1e6, sum(len(v) for v in first.values())), file=sys.stderr)


if __name__ == "__main__":
    main()
