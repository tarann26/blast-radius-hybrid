"""Writes maps/cost_fix.json: corrections of cost.bin for the tests whose whole run, in a child
of this program as it is set up today, costs clearly more or less CPU than cost.bin says.

cost.bin was measured with an earlier set-up of the parent (class fixtures built in the parent,
DuckDB with worker threads). Today the TestExecutor tests build their class attributes on
demand and DuckDB runs on the calling thread, which changes the CPU seconds of some tests by
a factor of two to four. This script runs every test of the pinned commit once, in order, in
forked children of a parent that is set up like the selector's (same steps as selector.run),
and measures the CPU seconds of each test alone (from the start of the test to its verdict,
without class set-up). It does so in two ways: all tests in order in one child, and one child
per test class. A test that pays a first-use cost (a module or a table that loads on demand)
is slow in only one of the two, and gets no correction. The times are scaled like
build_costs.py scales them (to the cheap tests of cost.bin: the measuring machine may be
loaded). An entry is written for a test whose scaled time is, in both ways of measuring,
outside 0.7 .. 1.4 times its cost.bin value on the same side, and differs from it by more
than 10 ms; the entry is the measurement nearer to cost.bin. Nothing here comes from patches
or labels.

usage: python build_cost_fix.py REPO TESTS_JSON MAPS_DIR [ROUNDS]   (BR_EXEC_CACHE as at run time)"""
import gc, json, marshal, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preload, reduce, runner  # noqa: E402


def measure(tests, per_class):
    """{test number: [CPU seconds of the test alone, failed]}: from one child that runs every
    test in order, or (per_class) from one child per test class."""
    if not per_class:
        return measure_child(tests)
    out = {}
    by_class = {}
    for t in tests:
        by_class.setdefault(t["id"].rsplit("::", 1)[0], []).append(t)
    for members in by_class.values():
        out.update(measure_child(members))
    return out


def measure_child(tests):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        mark = [0.0]
        out = {}

        def started(n):
            mark[0] = time.process_time()

        def emit(n, failed):
            out[n] = [round(time.process_time() - mark[0], 6), int(bool(failed))]

        try:
            import lazymods

            lazymods.load()
        except Exception:  # noqa: BLE001
            pass
        runner.run_tests([(t["number"], t["id"]) for t in tests], emit, False, None, started)
        data = json.dumps(out).encode()
        while data:
            data = data[os.write(w, data):]
        os._exit(0)
    os.close(w)
    buf = b""
    while True:
        b = os.read(r, 1 << 20)
        if not b:
            break
        buf += b
    os.waitpid(pid, 0)
    return {int(k): v for k, v in json.loads(buf).items()}


def main():
    root = os.path.abspath(sys.argv[1])
    tests = json.load(open(sys.argv[2]))["tests"]
    maps_dir = sys.argv[3]
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    with open(os.path.join(maps_dir, "cost.bin"), "rb") as f:
        cost = marshal.load(f)
    # the parent, as selector.run sets it up for function-body commits
    preload.install_environment(root)
    import duckdb  # noqa: F401

    try:
        import lazymods

        lazymods.install()
    except Exception:  # noqa: BLE001
        pass
    preload.import_sqlglot()
    preload.import_tests(sorted({t["file"][:-3].replace("/", ".") for t in tests}))
    reduce.install_loaders()
    preload.build_class_fixtures(set(preload.SHARED_FIXTURES), exec_lazy=True, exec_prebuild=["tpch_sqls", "tpcds_sqls"])
    gc.freeze()
    scaled = []  # per way of measuring: {test: seconds in the units of cost.bin}
    failed = set()
    factors = []
    for per_class in (False, True):
        runs = [measure(tests, per_class) for _ in range(rounds)]
        failed.update(n for run in runs for n, v in run.items() if v[1])
        times = {n: statistics.median(run[n][0] for run in runs if n in run) for n in runs[0]}
        small = [n for n in times if 0.005 <= cost[n] < 0.5]
        factor = statistics.median(times[n] / cost[n] for n in small)
        factors.append(round(factor, 3))
        scaled.append({n: t / factor for n, t in times.items()})
    fix = {}
    for n in sorted(scaled[0]):
        if n in failed or n not in scaled[1]:
            continue
        a, b = scaled[0][n], scaled[1][n]
        if min(a, b) > 1.4 * cost[n]:
            value = min(a, b)
        elif max(a, b) < 0.7 * cost[n]:
            value = max(a, b)
        else:
            continue
        if abs(value - cost[n]) > 0.01:
            fix[str(n)] = round(value, 5)
    with open(os.path.join(maps_dir, "cost_fix.json"), "w") as f:
        json.dump({"load": factors, "cost": fix}, f, separators=(",", ":"))
    sys.stderr.write("load factors %s, %d tests, %d failed here, %d corrections, sum cost.bin %.1f -> %.1f\n" % (
        factors, len(scaled[0]), len(failed), len(fix), sum(cost), sum(fix.get(str(n), cost[n]) for n in range(len(cost)))))
    if os.environ.get("BR_COST_FIX_DUMP"):
        json.dump({"scaled": scaled, "failed": sorted(failed), "factors": factors}, open(os.environ["BR_COST_FIX_DUMP"], "w"))


if __name__ == "__main__":
    main()
