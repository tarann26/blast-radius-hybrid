
"""Installed through PYTHONPATH for the pytest-based traces: every process (the pytest parent
and the spawn workers of the process pools) records the sqlglot lines it runs (xmon.py).
Spawn workers record one bucket per task. With XT_CAP set, the parent also saves each pool's
initializer and every submitted task (pickled), for the per-task cold replays (xreplay.py)."""
import os, sys
_out = os.environ.get("XT_OUT")
_tok = os.environ.get("XT_TOK")
if _tok and not _out:
    # tokenize inputs only (tokrec.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tokrec, atexit, time
    _w = "--multiprocessing-fork" in sys.argv
    _tag = ("w" if _w else "p") + "_%d_%d" % (os.getpid(), int(time.time() * 1000))
    atexit.register(lambda: tokrec.dump(os.path.join(_tok, _tag + ".json")))
    if _w:
        import concurrent.futures.process as _cfp
        _orig = _cfp._process_worker

        class _Q:
            def __init__(self, q):
                self._q = q

            def get(self, *a, **k):
                item = self._q.get(*a, **k)
                tokrec.install()
                return item

            def __getattr__(self, name):
                return getattr(self._q, name)

        def _process_worker(call_queue, *args, **kwargs):
            try:
                return _orig(_Q(call_queue), *args, **kwargs)
            finally:
                tokrec.dump(os.path.join(_tok, _tag + ".json"))

        _cfp._process_worker = _process_worker
if _out:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import xmon, atexit, time
    root = os.environ.get("XT_ROOT", "/repo")
    worker = "--multiprocessing-fork" in sys.argv
    xmon.start(root, "wstart" if worker else "pre")
    tag = ("w" if worker else "p") + "_%d_%d" % (os.getpid(), int(time.time() * 1000))

    def _dump():
        try:
            xmon.stop()
            xmon.dump(os.path.join(_out, tag + ".json"), {"worker": worker})
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("xmon dump failed %r\n" % (e,))

    atexit.register(_dump)
    if worker:
        import concurrent.futures.process as _cfp
        _orig = _cfp._process_worker

        class _Q:
            def __init__(self, q):
                self._q = q
                self._n = 0

            def get(self, *a, **k):
                item = self._q.get(*a, **k)
                if item is not None:
                    self._n += 1
                    fn = getattr(item, "fn", None)
                    xmon.switch(("task", self._n, getattr(fn, "__name__", "?"), time.time()))
                return item

            def __getattr__(self, name):
                return getattr(self._q, name)

        def _process_worker(call_queue, *args, **kwargs):
            try:
                return _orig(_Q(call_queue), *args, **kwargs)
            finally:
                _dump()

        _cfp._process_worker = _process_worker
    else:
        cap = os.environ.get("XT_CAP")
        if cap:
            import pickle
            import concurrent.futures as _cf
            _PPE = _cf.ProcessPoolExecutor
            _init = _PPE.__init__
            _submit = _PPE.submit
            _cnt = [0]

            def __init__(self, *a, **k):
                _cnt[0] += 1
                self._xt_id = _cnt[0]
                self._xt_n = 0
                d = os.path.join(cap, "pool%d" % self._xt_id)
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "init.pkl"), "wb") as f:
                    pickle.dump((k.get("initializer"), k.get("initargs", ()), str(xmon._cur[0])), f)
                return _init(self, *a, **k)

            def submit(self, fn, /, *args, **kwargs):
                self._xt_n += 1
                d = os.path.join(cap, "pool%d" % self._xt_id)
                with open(os.path.join(d, "task%05d.pkl" % self._xt_n), "wb") as f:
                    pickle.dump((fn, args, kwargs, str(xmon._cur[0])), f)
                return _submit(self, fn, *args, **kwargs)

            _PPE.__init__ = __init__
            _PPE.submit = submit
