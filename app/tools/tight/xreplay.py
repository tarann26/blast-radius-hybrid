
"""Per-task cold replay of captured pool tasks (sitecustomize.py with XT_CAP).

A spawn worker starts from nothing: it imports what the pickled initializer needs and runs
it ('wstart'), then runs tasks. Here each task runs in its own fork of such a started worker,
so every task is traced as the first task of a fresh worker. Then all tasks run in one more
worker in order ('warm_<i>').

usage: python xreplay.py REPO POOL_DIR OUT_JSON
"""
import os, sys, json, pickle, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
root, pool_dir, out_path = sys.argv[1:4]
sys.path.insert(0, os.path.abspath(root))
os.chdir(root)
import xmon
xmon.start(root, "wstart")
with open(os.path.join(pool_dir, "init.pkl"), "rb") as f:
    initializer, initargs, where = pickle.load(f)
if initializer is not None:
    initializer(*initargs)
xmon.switch("idle")
tasks = sorted(glob.glob(os.path.join(pool_dir, "task*.pkl")))


def run_task(p):
    with open(p, "rb") as f:
        fn, args, kwargs, owner = pickle.load(f)
    try:
        pickle.dumps(fn(*args, **kwargs))
        return owner, True
    except BaseException as e:  # noqa: BLE001
        return owner, repr(e)[:200]


res = {}
for i, p in enumerate(tasks):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            xmon.buckets.clear()
            xmon.switch("cold")
            owner, ok = run_task(p)
            xmon.switch("end")
            os.write(w, json.dumps({"owner": owner, "ok": ok, "lines": sorted(xmon.buckets.get("cold", ()))}).encode())
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
    res[os.path.basename(p)] = json.loads(buf) if buf else {"err": "no output"}
warm = {}
for i, p in enumerate(tasks):
    xmon.switch(("warm", os.path.basename(p)))
    run_task(p)
xmon.switch("end")
xmon.stop()
out = {"cold": res, "buckets": [[k, sorted(v)] for k, v in xmon.buckets.items()], "seed": os.environ.get("PYTHONHASHSEED")}
json.dump(out, open(out_path, "w"))
print("tasks", len(tasks))
