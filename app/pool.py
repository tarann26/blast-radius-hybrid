"""An in-process stand-in for the process pools that two sqlglot test classes create.

The real tests start one "spawn" worker per CPU. Each worker imports sqlglot, pandas and duckdb
again, which costs CPU time, and it reads the code from disk, so it would not see a patch
that lives only in memory. This pool runs each task in the calling process when its result
is asked for. It keeps what the tests can observe from a real pool: arguments and results
go through pickle, an exception of the task comes out of Future.result(), and a pool whose
worker would have died is broken for every later task."""
from __future__ import annotations

import pickle
from concurrent.futures import Future

from reduce import STATE

def BrokenProcessPool(*args):  # noqa: N802
    """concurrent.futures.process.BrokenProcessPool, imported when a pool breaks: that module
    loads multiprocessing.connection and more (about 25 ms of CPU), which no test needs
    otherwise."""
    from concurrent.futures.process import BrokenProcessPool as cls

    return cls(*args)

_FINISHED = "FINISHED"


class LazyFuture(Future):
    """Reports itself as finished; computes on the first result() or exception() call."""

    def __init__(self, pool, fn, args, kwargs):
        super().__init__()
        self._pool = pool
        self._task = (fn, args, kwargs)
        self._state = _FINISHED
        self._computed = False
        # the part of the test that this task belongs to (see reduce.py)
        self._tag = STATE.current
        self._index = STATE.task_counter
        STATE.task_counter += 1
        STATE.tasks_seen = STATE.task_counter
        if STATE.on_submit is not None and self._tag is None:
            STATE.on_submit(self._index)

    def _compute(self):
        if self._computed:
            return
        self._computed = True
        pool = self._pool
        fn, args, kwargs = self._task
        self._task = None
        if self._tag is not None:
            STATE.enter(self._tag)
        else:
            keep = STATE.keep_tasks
            if keep is not None and self._index not in keep:
                self._result = None  # a task that the patch cannot change: the test skips None
                return
            STATE.enter(("t", self._index))
        if pool._broken:
            self._exception = BrokenProcessPool("A process in the process pool was terminated abruptly")
            return
        if not pool._start_worker():
            self._exception = BrokenProcessPool("A process in the process pool was terminated abruptly")
            return
        try:
            payload = pickle.dumps((fn, args, kwargs))
        except BaseException as e:  # the real pool reports this on the future and lives on
            self._exception = e
            return
        try:
            fn, args, kwargs = pickle.loads(payload)
        except BaseException:  # the real worker dies here
            pool._broken = True
            if pool._class_level:
                InlinePool.broke_any = True
            self._exception = BrokenProcessPool("A process in the process pool was terminated abruptly")
            return
        try:
            hook = STATE.task_hook if self._tag is None else None
            # stages.py: a task of the long executor test may end where its input is known
            # to equal the pinned one
            value = fn(*args, **kwargs) if hook is None else hook(self._index, fn, args, kwargs)
        except SystemExit:
            pool._broken = True
            if pool._class_level:
                InlinePool.broke_any = True
            self._exception = BrokenProcessPool("A process in the process pool was terminated abruptly")
            return
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            self._exception = e
            return
        try:
            payload = pickle.dumps(value)
        except BaseException as e:
            self._exception = e
            return
        try:
            self._result = pickle.loads(payload)
        except BaseException:
            pool._broken = True
            if pool._class_level:
                InlinePool.broke_any = True
            self._exception = BrokenProcessPool("A process in the process pool was terminated abruptly")

    def result(self, timeout=None):
        self._compute()
        if self._exception is not None:
            raise self._exception
        return self._result

    def exception(self, timeout=None):
        self._compute()
        return self._exception

    def cancel(self):
        return False


class InlinePool:
    broke_any = False  # set when any pool of this process broke; the caller reads it

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kwargs):
        # a pool made outside a test (class fixture) lives on for the later tests of the class
        self._class_level = not STATE.in_test
        self._initializer = initializer
        self._initargs = initargs
        self._started = False
        self._broken = False
        self._shutdown = False

    def _start_worker(self) -> bool:
        if self._started:
            return not self._broken
        self._started = True
        if self._initializer is not None:
            try:
                self._initializer(*self._initargs)
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
                self._broken = True
                if self._class_level:
                    InlinePool.broke_any = True
                return False
        return True

    def submit(self, fn, /, *args, **kwargs):
        if self._broken:
            raise BrokenProcessPool("A child process terminated abruptly, the process pool is not usable anymore")
        if self._shutdown:
            raise RuntimeError("cannot schedule new futures after shutdown")
        return LazyFuture(self, fn, args, kwargs)

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        futures = [self.submit(fn, *args) for args in zip(*iterables)]

        def results():
            for f in futures:
                yield f.result()

        return results()

    def shutdown(self, wait=True, *, cancel_futures=False):
        self._shutdown = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


def install():
    """Must run before the test modules are imported: they bind the name at import."""
    import concurrent.futures

    concurrent.futures.ProcessPoolExecutor = InlinePool
