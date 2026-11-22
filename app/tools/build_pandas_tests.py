"""Which tests of the two files that import pandas and duckdb really load numpy, pandas or
duckdb, in the pinned tree (the selector imports them in the parent only when a batch needs
them; see lazymods.py). Each test runs once in a forked child, as the selector's hot children
run tests.
Usage: python build_pandas_tests.py REPO OUT.json [DUCK_REPLAY_FILE]
OUT.json: "tests" = the tests that load numpy or pandas when every DuckDB query runs.
With a recording of build_duck_replay.py, also for the DuckDB answer replay (duckreplay.py):
"replay_tests" = the tests that still load numpy or pandas, "replay_duckdb" = the tests that
load duckdb itself."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FILES = ("tests/test_executor.py", "tests/test_optimizer.py")


def survey(repo, replay_file):
    """{test id: [numpy loaded, pandas loaded, duckdb loaded, [failed]]} or None per test."""
    import preload

    preload.install_environment(repo)
    import lazymods

    assert lazymods.install()
    if replay_file:
        import duckreplay

        assert duckreplay.install(duckreplay.read_file(replay_file)), "no valid recording"
    else:
        import duckdb  # noqa: F401

    preload.import_sqlglot()
    with open(os.path.join(repo, "..", "tests.json") if os.path.exists(os.path.join(repo, "..", "tests.json")) else "/data/tests.json") as f:
        tests = json.load(f)["tests"]
    preload.import_tests([f[:-3].replace("/", ".") for f in FILES])
    import reduce

    reduce.install_loaders()
    wanted = {(f[:-3].replace("/", "."), c) for f in FILES for c in ("TestExecutor", "TestOptimizer")}
    preload.build_class_fixtures(wanted, exec_lazy=True, exec_prebuild=["tpch_sqls", "tpcds_sqls"])
    assert not lazymods.loaded("pandas") and not lazymods.loaded("numpy"), "fixtures load pandas"
    assert not replay_file or not lazymods.loaded("duckdb"), "fixtures load duckdb"
    import runner

    rows = {}
    for n, t in enumerate(tests):
        if t["file"] not in FILES:
            continue
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            verdict = []
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            runner.run_tests([(n, t["id"])], lambda number, failed: verdict.append(failed))
            os.write(w, json.dumps([lazymods.loaded("numpy"), lazymods.loaded("pandas"), lazymods.loaded("duckdb"), verdict]).encode())
            os._exit(0)
        os.close(w)
        data = b""
        while True:
            chunk = os.read(r, 65536)
            if not chunk:
                break
            data += chunk
        os.close(r)
        os.waitpid(pid, 0)
        rows[t["id"]] = json.loads(data) if data else None
    return rows


def survey_in_child(repo, replay_file):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            with os.fdopen(w, "w") as f:
                json.dump(survey(repo, replay_file), f)
        finally:
            os._exit(0)
    os.close(w)
    with os.fdopen(r) as f:
        rows = json.load(f)
    os.waitpid(pid, 0)
    return rows


def main(repo, out, replay_file=None):
    rows = survey_in_child(repo, None)
    result = {"tests": sorted(tid for tid, v in rows.items() if v is None or v[0] or v[1])}
    failed = sorted(tid for tid, v in rows.items() if v is not None and any(x for x in v[3]))
    print("pandas users:", len(result["tests"]), "of", len(rows), "tests; failing on the pinned tree:", failed, file=sys.stderr)
    if replay_file:
        again = survey_in_child(repo, replay_file)
        result["replay_tests"] = sorted(tid for tid, v in again.items() if v is None or v[0] or v[1])
        result["replay_duckdb"] = sorted(tid for tid, v in again.items() if v is None or v[2])
        failed = sorted(tid for tid, v in again.items() if v is not None and any(x for x in v[3]))
        differ = sorted(tid for tid in rows if (rows[tid] or [0, 0, 0, None])[3] != (again[tid] or [0, 0, 0, None])[3])
        print("with the replay: pandas users", len(result["replay_tests"]), "duckdb users", len(result["replay_duckdb"]),
              "; failing:", failed, "; verdicts that differ from the run without it:", differ, file=sys.stderr)
    with open(out, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    t0 = time.time()
    main(*sys.argv[1:4])
    print("took %.1f s" % (time.time() - t0), file=sys.stderr)
