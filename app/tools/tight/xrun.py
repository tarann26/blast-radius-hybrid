
"""Exact trace of chosen tests under the selector's runner environment (in-process pool).

The parent loads what the selector's parent loads (preload.install_environment, every sqlglot
module, the test modules, the shared class fixtures: traced as bucket 'fixtures'). For each
test a forked child runs it twice: 'cold' (first run after the fork) and 'warm' (again, same
process). With XT_LAZY=1 the TestExecutor fixture is the lazy one of the selector's run time.

usage: python xrun.py REPO TESTS_JSON OUT_JSON NUMBERS
"""
import os, sys, json, unittest
sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xmon

root, tests_path, out_path, numbers = sys.argv[1:5]
root = os.path.abspath(root)
numbers = [int(x) for x in numbers.split(",")]
tests = json.load(open(tests_path))["tests"]
import preload, reduce, runner
from reduce import STATE
xmon.start(root, "import")
preload.install_environment(root)
preload.import_sqlglot()
mods = sorted({t["file"][:-3].replace("/", ".") for t in tests})
preload.import_tests(mods)
assert reduce.install_loaders()
xmon.switch("fixtures")
lazy = os.environ.get("XT_LAZY") == "1"
preload.build_class_fixtures(set(preload.SHARED_FIXTURES), exec_lazy=lazy)
xmon.switch("idle")
results = {}
for n in numbers:
    tid = tests[n]["id"]
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            xmon.buckets.clear()
            oks = []
            for phase in ("cold", "warm"):
                xmon.switch(phase)
                case = runner.load_case(tid, n)

                class Result(unittest.TestResult):
                    def startTest(self, test):
                        super().startTest(test)
                        STATE.reset(None, None)
                        STATE.in_test = True

                    def stopTest(self, test):
                        super().stopTest(test)
                        STATE.in_test = False

                res = Result()
                unittest.TestSuite([case]).run(res)
                oks.append(not (res.failures or res.errors))
            xmon.switch("end")
            data = json.dumps({"ok": oks, "buckets": [[k, sorted(v)] for k, v in xmon.buckets.items() if k in ("cold", "warm")]}).encode()
            os.write(w, data)
        finally:
            os._exit(0)
    os.close(w)
    buf = b""
    while True:
        c = os.read(r, 1 << 20)
        if not c:
            break
        buf += c
    os.waitpid(pid, 0)
    results[str(n)] = json.loads(buf) if buf else {"err": "no output"}
    print(n, tid, results[str(n)].get("ok"), flush=True)
xmon.stop()
out = {"parent": [[k, sorted(v)] for k, v in xmon.buckets.items()], "tests": results, "seed": os.environ.get("PYTHONHASHSEED"), "lazy": lazy}
json.dump(out, open(out_path, "w"))
