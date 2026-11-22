"""Measures the CPU seconds of each test under the direct runner (pinned commit), in one forked
child of a parent in the selector's run-time state. usage: python time_all.py REPO TESTS_JSON OUT_JSON [SKIP_IDS...]"""
import gc, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preload, reduce, runner

root = os.path.abspath(sys.argv[1])
tests = json.load(open(sys.argv[2]))["tests"]
skip = set(sys.argv[4:])
preload.install_environment(root)
t0 = time.process_time()
preload.import_sqlglot()
t1 = time.process_time()
preload.import_tests(sorted({t["file"][:-3].replace("/", ".") for t in tests}))
t2 = time.process_time()
reduce.install_loaders()
preload.build_class_fixtures(set(preload.SHARED_FIXTURES))
t3 = time.process_time()
gc.collect(); gc.freeze()
print("import sqlglot", round(t1 - t0, 3), "import tests", round(t2 - t1, 3), "fixtures", round(t3 - t2, 3), file=sys.stderr, flush=True)
r, w = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(r)
    devnull = os.open(os.devnull, os.O_WRONLY); os.dup2(devnull, 1); os.dup2(devnull, 2)
    last = [time.process_time()]
    out = {}
    def emit(n, failed):
        now = time.process_time()
        out[n] = [round(now - last[0], 5), int(failed)]
        last[0] = now
    items = [(t["number"], t["id"]) for t in tests if t["id"] not in skip]
    runner.run_tests(items, emit)
    os.write(w, json.dumps(out).encode())
    os._exit(0)
os.close(w)
buf = b""
while True:
    b = os.read(r, 1 << 20)
    if not b: break
    buf += b
os.waitpid(pid, 0)
res = json.loads(buf)
json.dump({"times": res, "preload": [t1 - t0, t2 - t1, t3 - t2]}, open(sys.argv[3], "w"))
bad = [n for n, v in res.items() if v[1]]
print("ran", len(res), "failed", bad, "sum cpu", round(sum(v[0] for v in res.values()), 2), file=sys.stderr)
