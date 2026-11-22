
"""Line recorder for the exact traces of the long tests (sys.monitoring LINE + PY_START).

Records (file, line, in_function) for code under ROOT/sqlglot/ in named buckets. Each bucket
switch re-arms every location (restart_events), so a bucket holds every line that ran while
it was current, not only lines that are new to the process."""
import os, sys, json, time

CO_OPTIMIZED = 0x1
TOOL = 4
_ROOT = None
_cut = 0
buckets = {}
_cur = [None]
_win = {}


def _on_line(code, line):
    fn = code.co_filename
    if fn.startswith(_ROOT):
        b = buckets.get(_cur[0])
        if b is None:
            b = buckets[_cur[0]] = set()
        b.add((fn[_cut:], line, code.co_flags & CO_OPTIMIZED))
    return sys.monitoring.DISABLE


def _on_start(code, offset):
    return _on_line(code, code.co_firstlineno)


def start(root, bucket):
    global _ROOT, _cut
    _ROOT = os.path.join(os.path.abspath(root), "sqlglot") + os.sep
    _cut = len(os.path.abspath(root)) + 1
    mon = sys.monitoring
    mon.use_tool_id(TOOL, "xtrace")
    mon.register_callback(TOOL, mon.events.LINE, _on_line)
    mon.register_callback(TOOL, mon.events.PY_START, _on_start)
    switch(bucket)
    mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)


def switch(bucket):
    now = time.time()
    if _cur[0] is not None:
        _win.setdefault(_cur[0], []).append([_win.pop(("_open", _cur[0]), now), now])
    _cur[0] = bucket
    _win[("_open", bucket)] = now
    sys.monitoring.restart_events()


def stop():
    sys.monitoring.set_events(TOOL, 0)


def dump(path, extra=None):
    now = time.time()
    if _cur[0] is not None and ("_open", _cur[0]) in _win:
        _win.setdefault(_cur[0], []).append([_win.pop(("_open", _cur[0])), now])
    out = {"pid": os.getpid(), "ppid": os.getppid(), "seed": os.environ.get("PYTHONHASHSEED"),
           "buckets": [[k, sorted(v)] for k, v in buckets.items()],
           "windows": [[k, v] for k, v in _win.items() if not (isinstance(k, tuple) and k and k[0] == "_open")]}
    if extra:
        out.update(extra)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)
