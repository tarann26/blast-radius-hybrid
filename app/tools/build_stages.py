"""Builds maps/stages.bin and maps/stage_cost.bin for stages.py (pinned commit only).

usage (pinned interpreter and packages, as the selector runs):
    PYTHONHASHSEED=0 python build_stages.py REPO TESTS_JSON OUT_DIR [JOBS]

For the test of stages.py it runs the pinned test in forked children of a process that is set
up like the selector's parent (same stand-in pool, same lazy class fixture, same DuckDB answer
replay), and records for every task (query) that the test executes:
  - the fingerprints of the optimized expression, of the tables, of the DuckDB SQL text and of
    the rows that the test receives (two runs of the whole test, a traced run and a run of
    the task alone must all agree);
  - the sqlglot lines that run after that point (plan, executor, result, comparison), in a
    run of the whole test and in a run of that task alone (a cold start), plus every line of
    sqlglot.executor.execute() itself (the hook of stages.py sits inside that function);
  - the module state that the test writes (stages.state_ok compares with it).
That is stages.bin: it depends on the pinned tree only, and a second build gives the same
bytes. stage_cost.bin holds measurements of this machine (CPU seconds of each task before and
after that point), which the selector uses for the budget of a check, not for its verdict.
At the end the check of stages.py runs once against the new map: on the pinned tree every
task must stop with the pinned input and the test must pass.
Lines are counted as trace_units.py counts them: function code under sqlglot/, a function
entry counts as its first line."""
import hashlib
import json
import marshal
import os
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import preload
import reduce
import runner
import stages
from reduce import STATE

# test id -> (class attribute with the queries, class attribute with the tables, module name of the schema)
TESTS = {"tests/test_executor.py::TestExecutor::test_execute_tpcds": ("tpcds_sqls", "tpcds_tables", "TPCDS_SCHEMA")}
CO_OPTIMIZED = 0x1
TOOL = 3


class Recorder(stages.StageCheck):
    """Runs every task to its end: records the fingerprints, the CPU time before and after the
    point where the plan is built and, with trace on, the lines after that point."""

    def __init__(self, record, root, trace):
        super().__init__(record, set(), 0.0)
        self.found = {}
        self.times = {}
        self.lines = {}  # task -> {path: set of lines} after the point
        self.bucket = None
        self.root = root
        self.trace = trace
        self.t0 = None
        self.late_from = None

    def switch(self, bucket):
        self.bucket = bucket
        if self.trace:
            sys.monitoring.restart_events()

    def on_line(self, code, line):
        b = self.bucket
        if b is not None and code.co_flags & CO_OPTIMIZED:
            fn = code.co_filename
            cut = len(self.root) + 1
            if fn.startswith(self.root) and fn[cut:cut + 8] == "sqlglot/":
                self.lines.setdefault(b, {}).setdefault(fn[cut:], set()).add(line)
        return sys.monitoring.DISABLE

    def close_task(self):
        if self.late_from is not None:
            index, t = self.late_from
            self.times[index][1] = time.process_time() - t
            self.late_from = None
        self.switch(None)

    def run_task(self, index, fn, args, kwargs):
        self.close_task()
        sqls = getattr(self.cls, self.record["sqls"])
        if not sqls[index][0].get("execute"):
            return fn(*args, **kwargs)
        self.t0 = time.process_time()
        text = self.module.transpile(sqls[index][1], write="duckdb")[0]
        self.found[index] = [None, None, hashlib.blake2b(text.encode(), digest_size=16).digest(), None]
        self.task = index
        try:
            return fn(*args, **kwargs)
        finally:
            self.task = None

    def at_executor(self, args, kwargs, plan_input):
        plan, executor = self.real
        index = self.task
        assert not args and list(kwargs) == ["tables"], (args, kwargs)
        self.found[index][0] = stages.fingerprint(plan_input.expression)
        self.found[index][1] = stages.fingerprint(kwargs["tables"], stages.table_leaves(self.module._tables))
        now = time.process_time()
        self.times[index] = [now - self.t0, 0.0]
        self.late_from = (index, now)
        self.switch(index)
        return executor(*args, **kwargs).execute(plan(plan_input.expression))

    def on_result(self, index, table):
        self.found[index][3] = stages.rows_fingerprint(table)
        return False  # the pinned test goes on with its comparison


def run_one(case, number, test_id, keep_tasks):
    """One run of the test as the selector runs it; True when it passed."""
    out = []
    keeps = None if keep_tasks is None else {number: (None, keep_tasks)}
    runner.run_tests([(number, test_id)], lambda n, failed: out.append(failed), False, keeps)
    return out == [False]


def child(root, number, test_id, attr, keep_tasks, trace, record, wfd):
    out = {"ok": False}
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        case = runner.load_case(test_id, number)
        module = sys.modules[type(case).__module__]
        cls = type(case)
        if trace == "check":
            # the check of stages.py itself, against the new map
            check = stages.StageCheck(record, set(), 0.0)
            check.slack = None  # (this run only measures: other load on the machine must not end it)
            before = stages.state_before(record["effect"])
            assert check.install(module, cls)
            STATE.task_hook = check.run_task
            t0 = time.process_time()
            try:
                passed = check.probe(case) and run_one(case, number, test_id, None)
            finally:
                STATE.task_hook = None
                check.remove()
            out["cpu"] = time.process_time() - t0
            ok, changed = stages.state_ok(record["effect"], before)
            out["same"] = sorted(check.same)
            out["state"] = [list(k) for k in changed]
            out["ok"] = passed and ok and not check.ran and not check.over
            return
        before = None
        if trace == "state":
            # What module state does the pinned test write (see stages.state_ok)? The tables
            # that it fills when they lack an entry (keep_tasks names them here: the dialect
            # registry, the generators' dispatch tables) start empty, so that every entry
            # that the test can add shows.
            trace = False
            import statefp

            # The dialect registry is one of them: the selector's child empties it as the
            # full run does (child_reporters.started), and a lookup fills it again.
            sys.modules["sqlglot.dialects.dialect"].Dialect._classes.clear()
            objects = statefp.snapshot()[1].objects
            for key in keep_tasks or ():
                objects[key].clear()
            keep_tasks = None
            before = stages.state_before({}, None, True)
        if isinstance(keep_tasks, set):
            # a run of one task alone: with the dialect registry of a process that has only
            # imported the test files (as a worker of the real pool starts), so that the
            # lines of a first lookup count too
            try:
                with open(os.path.join(os.environ.get("BR_MAPS", os.path.join(HERE, "maps")), "registry_start.json")) as f:
                    keep = set(json.load(f))
            except (OSError, ValueError):
                keep = set()
            dialect = sys.modules["sqlglot.dialects.dialect"]
            classes = dialect.Dialect._classes
            for key in [k for k in classes if k in dialect.DIALECT_MODULE_NAMES and k not in keep]:
                del classes[key]
        t_load = time.process_time()
        if trace == "load":
            # the libraries that the comparison with DuckDB loads at its first use, timed alone
            trace = False
            import lazymods

            lazymods.load()
            lazymods.load_one("duckdb")
        t_load = time.process_time() - t_load
        rec = Recorder({"sqls": attr, "tasks": {}, "cpu": {}}, root, trace)
        assert rec.install(module, cls)
        real_transpile = module.transpile

        def transpile(*args, **kwargs):
            # the DuckDB SQL text is compared by itself: its lines are not lines after the point
            bucket = rec.bucket
            rec.switch(None)
            try:
                return real_transpile(*args, **kwargs)
            finally:
                rec.switch(bucket)

        module.transpile = transpile
        STATE.task_hook = rec.run_task
        # a task starts with the pickling of its arguments: not later work of the task before
        STATE.on_enter = lambda tag: rec.close_task()
        mon = sys.monitoring
        if trace:
            mon.use_tool_id(TOOL, "stages")
            mon.register_callback(TOOL, mon.events.LINE, rec.on_line)
            mon.register_callback(TOOL, mon.events.PY_START, lambda code, offset: rec.on_line(code, code.co_firstlineno))
            mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)
        t0 = time.process_time()
        passed = run_one(case, number, test_id, keep_tasks)
        rec.close_task()
        out["cpu"] = time.process_time() - t0
        if trace:
            mon.set_events(TOOL, 0)
        STATE.task_hook = None
        STATE.on_enter = None
        out["ok"] = passed
        out["found"] = rec.found
        out["times"] = rec.times
        out["load"] = t_load
        out["lines"] = {i: {p: sorted(s) for p, s in d.items()} for i, d in rec.lines.items()}
        out["tasks"] = len(getattr(cls, attr))
        code = module.execute.__code__
        out["hook"] = (os.path.relpath(code.co_filename, root),
                       sorted({code.co_firstlineno} | {ln for _, _, ln in code.co_lines() if ln}))
        if before is not None:
            rec.remove()
            out["effect"] = stages.state_effect(before)
    except BaseException as e:  # noqa: BLE001
        out["err"] = repr(e)[:500]
    finally:
        data = marshal.dumps(out)
        while data:
            data = data[os.write(wfd, data):]


class Jobs:
    """Forked children, at most 'limit' at a time; each result is read from a pipe."""

    def __init__(self, limit):
        self.limit = limit
        self.live = {}  # read fd -> (key, pid, chunks)
        self.done = {}

    def start(self, key, *args):
        while len(self.live) >= self.limit:
            self.wait_one()
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            try:
                child(*args, w)
            finally:
                os._exit(0)
        os.close(w)
        self.live[r] = (key, pid, [])

    def wait_one(self):
        import select

        ready, _, _ = select.select(list(self.live), [], [])
        for r in ready:
            key, pid, chunks = self.live[r]
            chunk = os.read(r, 1 << 20)
            if chunk:
                chunks.append(chunk)
                continue
            os.close(r)
            os.waitpid(pid, 0)
            del self.live[r]
            out = marshal.loads(b"".join(chunks))
            if not out.get("ok"):
                raise SystemExit("stage run %r failed: %r" % (key, {k: out.get(k) for k in ("err", "state", "same")}))
            self.done[key] = out

    def get(self, key):
        while key not in self.done:
            self.wait_one()
        return self.done[key]


def parent_setup(root, tests):
    """The state of the selector's parent before it forks the child of a commit that changes
    function bodies (selector.run): stand-in pool, lazy numpy / pandas / duckdb with the DuckDB
    answer replay when its recording is there, sqlglot and the test modules, the fixture
    loaders of reduce.py, and the lazy class fixture of TestExecutor."""
    preload.install_environment(root)
    replay = False
    try:
        import duckreplay
        import lazymods

        recording = duckreplay.read_file(duckreplay.default_path())
        if recording is not None and lazymods.install():
            replay = duckreplay.install(recording)
    except Exception:  # noqa: BLE001
        replay = False
    preload.import_sqlglot()
    mods = sorted({t["file"][:-3].replace("/", ".") for t in tests if t["id"] in TESTS})
    preload.import_tests(mods)
    assert reduce.install_loaders()
    wanted = {(t["id"].split("::")[0][:-3].replace("/", "."), t["id"].split("::")[1]) for t in tests if t["id"] in TESTS}
    preload.build_class_fixtures(wanted, exec_lazy=True, exec_prebuild=("tpch_sqls", "tpcds_sqls"))
    return replay


def main():
    root, tests_path, out_dir = sys.argv[1:4]
    jobs = Jobs(int(sys.argv[4]) if len(sys.argv) > 4 else (os.cpu_count() or 1))
    root = os.path.abspath(root)
    with open(tests_path) as f:
        tests = json.load(f)["tests"]
    replay = parent_setup(root, tests)
    print("DuckDB answer replay:", replay, file=sys.stderr)
    result, costs = {}, {}
    for number, t in enumerate(tests):
        if t["id"] not in TESTS:
            continue
        (attr, tables_attr, schema_name), test_id = TESTS[t["id"]], t["id"]
        base = (root, number, test_id, attr)
        jobs.start("a", *base, None, False, None)
        jobs.start("warm", *base, None, True, None)
        jobs.start("state", *base, None, "state", None)
        a, warm = jobs.get("a"), jobs.get("warm")
        if any(None in v for v in a["found"].values()):
            raise SystemExit("a task without a fingerprint")
        if warm["found"] != a["found"]:
            raise SystemExit("fingerprints differ under the tracer")
        filled = sorted(k for k, e in jobs.get("state")["effect"].items() if e[0] == "entries")
        jobs.start("state2", *base, filled, "state", None)
        for index in sorted(a["found"]):
            jobs.start(("cold", index), *base, {index}, True, None)
        effect = jobs.get("state2")["effect"]
        if sorted(k for k, e in effect.items() if e[0] == "entries") != filled:
            raise SystemExit("the tables that the test fills are not stable")
        print("module state that the test writes:", {k: (e[0], sorted(e[1]) if e[0] == "entries" else e[1])
                                                     for k, e in sorted(effect.items())}, file=sys.stderr)
        by_line = {}
        hook_path, hook_lines = a["hook"]
        for index in sorted(a["found"]):
            cold = jobs.get(("cold", index))
            if cold["found"] != {index: a["found"][index]}:
                raise SystemExit("task %d alone gives other fingerprints" % index)
            for src in (warm["lines"].get(index, {}), cold["lines"].get(index, {}), {hook_path: hook_lines}):
                for path, lines in src.items():
                    for ln in lines:
                        by_line.setdefault((path, ln), set()).add(index)
        sets, ids, late = [], {}, {}
        for (path, ln), users in sorted(by_line.items()):
            key = tuple(sorted(users))
            sid = ids.get(key)
            if sid is None:
                sid = ids[key] = len(sets)
                sets.append(key)
            late.setdefault(path, {})[ln] = sid
        record = result[test_id] = {
            "sqls": attr,
            "tables": tables_attr,
            "schema": schema_name,
            "count": a["tasks"],
            "tasks": {i: tuple(v) for i, v in sorted(a["found"].items())},
            "late": {p: dict(sorted(d.items())) for p, d in sorted(late.items())},
            "sets": sets,
            "effect": {k: (e[0], dict(sorted(e[1].items()))) if e[0] == "entries" else e for k, e in sorted(effect.items())},
            "probe": stages.hash_probe(),
        }
        # The measurements, one run at a time with nothing beside it: the whole test with the
        # libraries loaded before it (so the tasks are timed alone), then the check itself.
        while jobs.live:
            jobs.wait_one()
        # Each is taken three times and the smallest value counts (other load on the machine
        # only ever adds).
        runs = []
        for k in range(3):
            jobs.start(("b", k), *base, None, "load", None)
            runs.append(jobs.get(("b", k)))
            if a["found"] != runs[-1]["found"]:
                raise SystemExit("fingerprints are not stable")
        b = {"cpu": min(r["cpu"] for r in runs), "load": min(r["load"] for r in runs)}
        cpu = {i: (round(min(r["times"][i][0] for r in runs), 4), round(min(r["times"][i][1] for r in runs), 4))
               for i in sorted(a["found"])}
        record_cpu = dict(record, cpu=cpu)
        checks = []
        for k in range(3):
            jobs.start(("check", k), *base, None, "check", record_cpu)
            checks.append(jobs.get(("check", k)))
        check = dict(checks[0], cpu=min(r["cpu"] for r in checks))
        if check["same"] != sorted(record["tasks"]):
            raise SystemExit("the check does not stop every task on the pinned tree")
        costs[test_id] = {
            "cpu": cpu,
            "load": round(b["load"], 3),  # numpy, pandas and duckdb, at the first comparison
            "check": round(check["cpu"], 3),  # a check in which every task stops at the point
            "whole": round(b["cpu"], 3),  # the whole test
            "replay": replay,
        }
        print(test_id, "tasks", len(a["found"]), "late lines", len(by_line), "files", len(late),
              "cpu before %.2f after %.2f whole %.2f check %.2f load %.2f" % (
                  sum(v[0] for v in cpu.values()), sum(v[1] for v in cpu.values()), b["cpu"], check["cpu"], b["load"]),
              file=sys.stderr)
    os.makedirs(out_dir, exist_ok=True)
    for name, data in (("stages.bin", zlib.compress(marshal.dumps(result, 2), 9)), ("stage_cost.bin", marshal.dumps(costs, 2))):
        path = os.path.join(out_dir, name)
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
    print("stages.bin sha256", hashlib.sha256(open(os.path.join(out_dir, "stages.bin"), "rb").read()).hexdigest(), file=sys.stderr)


if __name__ == "__main__":
    main()
