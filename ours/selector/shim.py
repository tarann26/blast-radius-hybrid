"""In-process stand-in for ProcessPoolExecutor.

sqlglot's optimizer and executor tests hand their work to a spawn-based process
pool. Spawned workers would (a) escape the coverage tracer and (b) cost a fresh
interpreter each when we verify a patch at decision time. This executor runs each
task synchronously in the calling process, and pickles the function, arguments,
initializer arguments and result on the way through, so a change that breaks
pickling still fails the way it would across a real process boundary.
"""

import concurrent.futures as _cf
import pickle as _pickle


class SyncProcessPoolExecutor:
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_):
        self._initializer = initializer
        self._initargs = initargs
        self._initialized = False

    def _ensure_initialized(self):
        if self._initializer is not None and not self._initialized:
            self._initialized = True
            args = _pickle.loads(_pickle.dumps(self._initargs))
            self._initializer(*args)

    def submit(self, fn, /, *args, **kwargs):
        future = _cf.Future()
        try:
            self._ensure_initialized()
            fn, args, kwargs = _pickle.loads(_pickle.dumps((fn, args, kwargs)))
            result = fn(*args, **kwargs)
            result = _pickle.loads(_pickle.dumps(result))
        except BaseException as error:  # noqa: BLE001 - mirrored into the future
            if getattr(error, "_br_abort", False):
                raise  # a stage check decided: stop the whole test (selector/stage.py)
            future.set_exception(error)
        else:
            future.set_result(result)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1, buffersize=None):
        futures = [self.submit(fn, *args) for args in zip(*iterables)]
        return (future.result() for future in futures)

    def shutdown(self, wait=True, *, cancel_futures=False):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


def install():
    _cf.ProcessPoolExecutor = SyncProcessPoolExecutor
    try:
        import concurrent.futures.process as _process

        _process.ProcessPoolExecutor = SyncProcessPoolExecutor
    except Exception:  # pragma: no cover
        pass
