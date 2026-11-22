"""DuckDB answer replay.

Many tests of tests/test_optimizer.py fill an in-memory DuckDB database, send it SQL text that
sqlglot generated, and compare the results; three tests of tests/test_executor.py do the same
with views on the TPC-H / TPC-DS csv files. DuckDB is not part of sqlglot and no patch changes
it: a database that was filled by the same statements answers the same SELECT text with the
same result as in the pinned run. build_duck_replay.py runs the pinned tests and records, for
each database state (how the connection was opened and the statements that changed it, in
order), the result of every SELECT text that the tests fetched. This module answers such a
text from the recording, without DuckDB and without pandas.

What makes the answers exact:
- A state is known only as "connect arguments + these statements, in this order", and a
  result only as "this state + this text + this fetch method". Anything else (a text that
  was not recorded, parameters, another method, a statement that raised in the pinned run)
  makes the connection real on the spot: DuckDB is loaded, a real connection is opened, the
  statements of the state run on it in order, and from then on every call goes to it.
- Recorded reads are single SELECT statements that use no function whose value depends on
  anything but its arguments and the tables (see the builder), recorded in three different
  orders with equal results. So leaving them out does not change what a later statement sees.
- A fetched list of rows is rebuilt as an equal new object. For the connections of
  TestExecutor (views on the TPC csv files; the recording is only valid while those files
  have their recorded sizes, see read_file) a DataFrame result is rebuilt from its pickle:
  that test class works on the frames with pandas anyway. For the other connections a
  DataFrame result is a stand-in that carries the number of its "frame class" (frames with
  exactly equal content, dtypes, column names and index). The stand-in of pandas.testing.assert_frame_equal
  (lazymods.py) passes at once only for two untouched stand-ins whose pair of classes passed
  the real assert_frame_equal at build time. Every other use of a stand-in makes the
  connection real first; the real frames of the stand-ins that are still alive are then
  fetched in the order in which the tests asked for them, before the new statement runs.
- A child whose patch changes anything but sqlglot/ code and .sql fixture entries turns the
  replay off (patch_ok): the test code and the data that it loads are then not the pinned ones.
"""
from __future__ import annotations

import marshal
import os
import sys
import weakref

FILE_VERSION = 2
ACTIVE = False  # False: every connection becomes real at its next use
STARTS = {}  # repr of the connect arguments -> state
STEPS = {}  # (state, text of a statement that changes the database) -> next state
READS = {}  # (state, SELECT text) -> {fetch method: (kind, data)}
PAIRS = frozenset()  # (frame class, frame class) for which the real assert_frame_equal passed
STATS = {"answers": 0, "frames": 0, "real": 0}  # of this process, for the logs
_REAL = [None]  # the real duckdb.connect (one thread, see preload.py), once duckdb has run


def default_path() -> str:
    return os.path.join(os.environ.get("BR_EXEC_CACHE", "/app/cache"), "duck_replay.bin")


def read_file(path):
    """(starts, steps, reads, pairs) of a recording, or None when there is no valid one (no
    file, another format version, or a data file of another size than at recording time)."""
    try:
        with open(path, "rb") as f:
            data = marshal.loads(f.read())  # much faster than marshal.load(f)
        if data[0] != FILE_VERSION:
            return None
        starts, steps, reads, pairs, files = data[2:7]
        if not (type(starts) is dict and type(steps) is dict and type(reads) is dict and type(pairs) is frozenset):
            return None
        for name, size in files.items():
            # the data files that recorded statements read (the TPC tables of TestExecutor)
            if os.path.getsize(name) != size:
                return None
        return starts, steps, reads, pairs
    except (OSError, ValueError, EOFError, TypeError, IndexError, AttributeError):
        return None


def real_connect():
    """The real duckdb.connect. Loads duckdb when it has not run yet."""
    if _REAL[0] is None:
        import lazymods

        lazymods.load_one("duckdb")  # runs _adopt below
        if _REAL[0] is None:
            import duckdb

            _adopt(duckdb)
    return _REAL[0]


def _adopt(module) -> None:
    """Called with the real duckdb module: keeps its connect, and leaves ours in its place."""
    import preload

    real = module.connect
    if real is not connect:
        _REAL[0] = preload.single_thread_connect(real)
        module.connect = connect


def connect(*args, **kwargs):
    """Stands for duckdb.connect: a replaying connection when the recording knows what a
    connection opened with these arguments is, else a real one."""
    if ACTIVE:
        try:
            state = STARTS.get(repr((args, sorted(kwargs.items()))))
        except Exception:  # noqa: BLE001  (arguments without an order or a repr: a real connection)
            state = None
        if state is not None:
            return Conn(args, kwargs, state)
    return real_connect()(*args, **kwargs)


connect._br = True  # for preload._single_thread_duckdb: nothing to wrap here


class Conn:
    """A DuckDB connection in a recorded state. See the module text."""

    def __init__(self, args, kwargs, state):
        self._br_args = (args, kwargs)
        self._br_state = state
        self._br_real = None
        # in the order of the calls: the text of each statement that changed the database
        # (str), and a weak reference to each DataFrame stand-in that was handed out
        self._br_hist = []
        self._br_pending = None  # (text, answers): the last statement is a recorded read, not fetched yet
        self._br_last = None  # (text, method): the last statement is a recorded read, fetched with method
        self._br_closed = False

    def _br_go_real(self):
        real = self._br_real
        if real is None:
            real = real_connect()(*self._br_args[0], **self._br_args[1])
            for item in self._br_hist:
                if type(item) is str:
                    real.execute(item)
                else:
                    frame = item()
                    if frame is not None and frame._br_frame is None:
                        object.__setattr__(frame, "_br_frame", getattr(real.execute(frame._br_text), frame._br_method)())
            # the result that the connection holds after the last statement, as in a real run
            if self._br_pending is not None:
                real.execute(self._br_pending[0])
            elif self._br_last is not None:
                getattr(real.execute(self._br_last[0]), self._br_last[1])()
            if self._br_closed:
                real.close()
            self._br_real = real
            self._br_hist = self._br_pending = self._br_last = None
            STATS["real"] += 1
        return real

    def execute(self, query, *args, **kwargs):
        if self._br_real is None and ACTIVE and not args and not kwargs and type(query) is str and not self._br_closed:
            key = (self._br_state, query)
            state = STEPS.get(key)
            if state is not None:
                self._br_hist.append(query)
                self._br_state = state
                self._br_pending = self._br_last = None
                return self
            answers = READS.get(key)
            if answers is not None:
                self._br_pending = (query, answers)
                self._br_last = None
                return self
        real = self._br_go_real()
        out = real.execute(query, *args, **kwargs)
        return self if out is real else out  # the real execute returns its connection

    def _br_fetch(self, method, args, kwargs):
        pending = self._br_pending
        if pending is not None and self._br_real is None and ACTIVE and not args and not kwargs:
            answer = pending[1].get(method)
            if answer is not None:
                self._br_pending = None
                self._br_last = (pending[0], method)
                kind, data = answer
                if kind == "m":
                    STATS["answers"] += 1
                    return marshal.loads(data)  # a new list of row tuples, as fetchall makes one
                if kind == "p":
                    import pickle

                    STATS["answers"] += 1
                    return pickle.loads(data)
                if kind == "d":
                    # a real DataFrame, for tests that work on the frame itself (TestExecutor
                    # compares it with frames that it builds with pandas): a new, equal frame
                    # from its pickle, as fetchdf makes a new one each time
                    import pickle

                    import lazymods

                    lazymods.load()
                    STATS["answers"] += 1
                    return pickle.loads(data)
                frame = Frame(self, pending[0], method, data)
                hist = self._br_hist
                if len(hist) > 256:  # drop the references to stand-ins that are gone
                    hist[:] = [h for h in hist if type(h) is str or h() is not None]
                hist.append(weakref.ref(frame))
                STATS["frames"] += 1
                return frame
        return getattr(self._br_go_real(), method)(*args, **kwargs)

    def fetchall(self, *args, **kwargs):
        return self._br_fetch("fetchall", args, kwargs)

    def df(self, *args, **kwargs):
        return self._br_fetch("df", args, kwargs)

    def fetchdf(self, *args, **kwargs):
        return self._br_fetch("fetchdf", args, kwargs)

    def close(self):
        if self._br_real is not None:
            return self._br_real.close()
        self._br_closed = True  # a later use opens the real connection and closes it first
        return None

    def __enter__(self):
        return self  # as the real connection does

    def __exit__(self, exc_type, exc, tb):
        self.close()  # as the real connection does
        return None

    def __getattr__(self, name):
        if name.startswith("_br_"):
            raise AttributeError(name)
        return getattr(self._br_go_real(), name)


_FRAME_SLOTS = ("_br_conn", "_br_text", "_br_method", "_br_class", "_br_frame")


class Frame:
    """Stands for the DataFrame of a recorded read. Only assert_frame_equal of two such
    objects is answered without the frame; any other use fetches the real frame first."""

    __slots__ = _FRAME_SLOTS + ("__weakref__",)

    def __init__(self, conn, text, method, frame_class):
        set_ = object.__setattr__
        set_(self, "_br_conn", conn)
        set_(self, "_br_text", text)
        set_(self, "_br_method", method)
        set_(self, "_br_class", frame_class)
        set_(self, "_br_frame", None)

    def _br_get(self):
        """The real frame. The first call makes the connection real, which fetches the frames
        of all its living stand-ins in the order of the test's calls."""
        if self._br_frame is None:
            self._br_conn._br_go_real()
        return self._br_frame

    def __getattr__(self, name):
        if name.startswith("_br_"):
            raise AttributeError(name)
        return getattr(self._br_get(), name)

    def __setattr__(self, name, value):
        setattr(self._br_get(), name, value)

    @property
    def __class__(self):  # isinstance() and friends see the real frame
        return type(self._br_get())


def _forward(name):
    def method(self, *args, **kwargs):
        return getattr(self._br_get(), name)(*args, **kwargs)

    method.__name__ = name
    return method


# the special methods that Python looks up on the type: each one works on the real frame
for _name in (
    "__len__", "__iter__", "__getitem__", "__setitem__", "__delitem__", "__contains__", "__bool__",
    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "__repr__", "__str__", "__array__",
    "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__", "__pow__", "__matmul__",
    "__radd__", "__rsub__", "__rmul__", "__rtruediv__", "__and__", "__or__", "__xor__", "__invert__",
    "__neg__", "__pos__", "__abs__", "__round__", "__copy__", "__deepcopy__", "__reduce_ex__", "__reduce__",
    "__dataframe__", "__finalize__", "__sizeof__", "__dir__", "__format__",
):
    setattr(Frame, _name, _forward(_name))
Frame.__hash__ = None  # a DataFrame cannot be hashed


def frame_hook(args, kwargs):
    """For lazymods.assert_frame_equal. None: the comparison is known to pass (two untouched
    stand-ins, default options, and the real assert passed for this pair of frame classes at
    build time). Otherwise (args, kwargs) for the real assert, with the real frames."""
    if len(args) == 2 and not kwargs:
        a, b = args
        if (type(a) is Frame and type(b) is Frame and a._br_frame is None and b._br_frame is None
                and (a._br_class, b._br_class) in PAIRS):
            return None
    args = tuple(x._br_get() if type(x) is Frame else x for x in args)
    for k, v in kwargs.items():
        if type(v) is Frame:
            kwargs[k] = v._br_get()
    return args, kwargs


def patch_ok(paths) -> bool:
    """True when a patch that changes these files leaves the recording valid: only sqlglot/
    code and .sql fixtures change. The tests' own code, and with it every statement that
    fills a database, is then the pinned one; a changed fixture entry gives a text that the
    recording does not know, which goes to the real connection."""
    for p in paths:
        if p.startswith("sqlglot/"):
            continue
        if p.startswith("tests/fixtures/") and p.endswith(".sql"):
            continue
        return False
    return True


def disable() -> None:
    """From now on in this process: every connection is real (the ones that exist become
    real at their next use), and assert_frame_equal is the real one for real frames."""
    global ACTIVE
    ACTIVE = False


def install(recording=None) -> bool:
    """Puts connect in the place of duckdb.connect, with the recording (the result of
    read_file; read from the default path when not given). With duckdb not imported yet, a
    lazy stand-in module takes its place (lazymods.py): duckdb itself runs in the first
    process that needs a real connection or another attribute of the module.
    False (and nothing changed) without a valid recording or without the lazy pandas."""
    global ACTIVE, STARTS, STEPS, READS, PAIRS
    import lazymods

    if ACTIVE:
        return True
    if not lazymods._STATE["installed"]:
        return False  # assert_frame_equal must be the stand-in of lazymods
    data = recording if recording is not None else read_file(default_path())
    if data is None:
        return False
    if "duckdb" in sys.modules:
        _adopt(sys.modules["duckdb"])
    elif not lazymods.install_one("duckdb", {"connect": connect}, _adopt):
        return False
    STARTS, STEPS, READS, PAIRS = data
    lazymods.FRAME_HOOK = frame_hook
    ACTIVE = True
    return True
