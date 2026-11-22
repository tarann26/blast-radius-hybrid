"""Records the DuckDB answers that rt/duckreplay.py replays (run at image build time).

usage: python build_duck_replay.py REPO OUT_FILE [--check]
       python build_duck_replay.py REPO OUT_FILE --from FILE [--check]   (FILE: a recording this
           script made on the pinned tree, committed as a plain file; it is copied to OUT_FILE and
           --check rehearses it as it does a fresh one, with every DuckDB test passing as the
           pinned verdict)
       python build_duck_replay.py --rehearse REPO FILE VERDICTS.json   (used by --check)

The tests of the pinned tests/test_optimizer.py that can reach a DuckDB connection run twice,
in two processes (in file order and in reverse order), with the real duckdb, pandas and
assert_frame_equal. A recording connection notes:
- a state for each connection: the connect arguments, then the text of every statement that
  changes the database, in order (CREATE / INSERT / ... given as plain text without parameters);
- for each state, each single SELECT statement that was fetched with fetchall, df or fetchdf,
  and its result;
- each pair of DataFrame results that the test gave to assert_frame_equal(a, b), when it passed.
A connection stops recording for good at the first thing that is not of that kind (a
statement that raises, parameters, any other method, a fetch without a read before it).

A text is recorded only when no word in it names something whose result can depend on more
than the text and the tables (and no result with a fraction that the C math library computed,
see MATH_WORDS): the functions that DuckDB's own catalog does not mark as
CONSISTENT, the functions that can be called without arguments (clock, session, random), age
(reads the clock with one argument), the macros built on those, and the table functions that
read files, catalogs or settings. A SELECT with a sampling keyword is recorded only when
SAMPLE_REPEATS more runs of it give the same result each time (a sample as large as its table
is the whole table; a real random sample differs between runs).

Then, in this process: every state is built again on a new connection and its reads run in a
shuffled order; a read is kept only when all three runs gave the same result (rows: equal
values of equal types; frames: equal column names, dtypes, index and values). Frames with
equal content form a class. A pair of classes is kept when the real assert_frame_equal passes
for it here again, and (c, c) when it passes for two frames of class c from separate queries.

The two connections of TestExecutor (views on the TPC-H / TPC-DS csv.gz files) are recorded
without running its tests, which take long in the Python executor: the views are made with the
statement text of setUpClass (preload.exec_view_sql), and the texts that the pinned tests send
are computed as the tests compute them (cached_execute: transpile(sql, write="duckdb")[0] for
the TPC-H queries and the TPC-DS queries marked for execution; test_optimized_tpch: the same
for the optimized TPC-H queries). A text that the tests would not send costs nothing: it is
never asked for. Three processes run the texts in three orders on new connections; a result
is kept when all three are equal, as a pickled DataFrame that gives back an equal frame. The
sizes of the csv files go into the file; the selector uses the recording only while they hold.

File: marshal of (version, [duckdb, pandas, numpy versions], starts, steps, reads, pairs,
files), see duckreplay.py. --check then runs the optimizer tests and the two TPC-H tests of
the executor once more in a new process with the replay on (as the selector runs them) and
fails unless every verdict equals the one of the pinned run (--check-all: test_execute_tpcds
too, which takes longer)."""
import marshal
import os
import pickle
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TEST_MODULE = "tests.test_optimizer"
READ_METHODS = ("fetchall", "df", "fetchdf")
STEP_TYPES = ("CREATE", "INSERT", "UPDATE", "DELETE", "ALTER", "DROP", "SELECT")
# functions that can be called without arguments and are constants all the same
CONSTANT_FUNCTIONS = ("pi", "map", "list_value", "list_pack")
# table functions that compute their rows from their arguments alone
PURE_TABLE_FUNCTIONS = ("range", "generate_series", "unnest", "repeat", "repeat_row", "json_each", "json_tree")
# words that the catalog does not list as functions (keywords for the clock and the session,
# sampling, statements that reach outside the database)
OTHER_WORDS = (
    "age", "current_time", "current_timestamp", "localtime", "localtimestamp", "current_localtime",
    "current_localtimestamp", "yesterday", "tomorrow", "pragma", "copy", "attach",
    "detach", "install", "load", "export", "import", "call", "set", "reset", "checkpoint", "vacuum",
    "prepare", "execute", "deallocate", "begin", "commit", "rollback", "transaction", "sequence", "temp",
    "temporary", "secret", "macro", "function", "getenv",
)
# functions that DuckDB computes with the C math library: the last digit of a result that is
# not a whole number can depend on the CPU model (the library picks its code by CPU), and the
# image may be built on another machine than it runs on. +, -, *, / and SQRT do not.
MATH_WORDS = frozenset((
    "pow", "power", "exp", "ln", "log", "log2", "log10", "sin", "cos", "tan", "cot", "asin", "acos", "atan",
    "atan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "cbrt", "gamma", "lgamma", "degrees", "radians",
    "skewness", "kurtosis", "kurtosis_pop", "entropy", "geomean", "geometric_mean", "approx_quantile",
    "approx_count_distinct", "corr", "regr_r2",
))
SAMPLE_WORDS = frozenset(("sample", "tablesample"))
SAMPLE_REPEATS = 40
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Unsupported(Exception):
    pass


def words_of(text):
    return {w.lower() for w in WORD.findall(text)}


def unsafe_words(conn):
    """The words that keep a text out of the recording (see the module text)."""
    rows = conn.execute(
        "SELECT function_name, function_type, stability, parameters, varargs, macro_definition FROM duckdb_functions()"
    ).fetchall()
    bad = set(OTHER_WORDS)
    macros = {}
    for name, kind, stability, parameters, varargs, definition in rows:
        name = name.lower()
        if kind in ("scalar", "aggregate", "window") and stability != "CONSISTENT":
            bad.add(name)
        if kind in ("scalar", "macro") and not parameters and varargs is None and name not in CONSTANT_FUNCTIONS:
            bad.add(name)
        if kind in ("table", "pragma", "table_macro") and name not in PURE_TABLE_FUNCTIONS:
            bad.add(name)
        if kind in ("macro", "table_macro"):
            macros.setdefault(name, set()).update(words_of(definition or ""))
    changed = True
    while changed:  # a macro whose definition uses an unsafe word is unsafe
        changed = False
        for name, words in macros.items():
            if name not in bad and words & bad:
                bad.add(name)
                changed = True
    return frozenset(bad)


# --- canonical text of a result -------------------------------------------------------

def value_code(v, out):
    """Appends a text that is equal for two values exactly when they have the same types and
    the same content. Raises Unsupported for a type that is not listed here."""
    import datetime
    import decimal
    import uuid

    t = type(v)
    if v is None:
        out.append("N;")
    elif t is bool:
        out.append("b%d;" % v)
    elif t is int:
        out.append("i%d;" % v)
    elif t is float:
        out.append("fnan;" if v != v else "f%r;" % v)
    elif t is str:
        out.append("s%d:%s;" % (len(v), v))
    elif t is bytes or t is bytearray:
        out.append("y%s:%s;" % (t.__name__, bytes(v).hex()))
    elif t is list or t is tuple:
        out.append("[" if t is list else "(")
        for x in v:
            value_code(x, out)
        out.append("]" if t is list else ")")
    elif t is dict:
        out.append("{")
        for k, x in v.items():
            value_code(k, out)
            value_code(x, out)
        out.append("}")
    elif t is decimal.Decimal or t is uuid.UUID or t is datetime.date or t is datetime.timedelta:
        out.append("o%r;" % (v,))
    elif t is datetime.datetime or t is datetime.time:
        if v.tzinfo is not None:
            raise Unsupported("time zone object")
        out.append("o%r;" % (v,))
    else:
        raise Unsupported(t.__name__)


def frame_value_code(v, out):
    """value_code plus what a DataFrame cell can hold besides."""
    import numpy as np
    import pandas as pd

    t = type(v)
    if v is pd.NaT:
        out.append("NaT;")
    elif v is pd.NA:
        out.append("NA;")
    elif t is pd.Timestamp or t is pd.Timedelta:
        out.append("P%r;" % (v,))
    elif t is np.ndarray:
        out.append("a%s%r" % (v.dtype.str, v.shape))
        if v.dtype.kind == "O":
            for x in v.ravel().tolist():
                frame_value_code(x, out)
        elif v.dtype.kind in "biufmM":
            array_code(v, out)
        else:
            raise Unsupported("array of " + v.dtype.str)
        out.append(";")
    elif isinstance(v, np.generic):
        if v.dtype.kind not in "biufmM":
            raise Unsupported(t.__name__)
        out.append("g%s" % v.dtype.str)
        array_code(np.asarray(v).reshape(1), out)
        out.append(";")
    elif t is list or t is tuple:
        out.append("[" if t is list else "(")
        for x in v:
            frame_value_code(x, out)
        out.append("]" if t is list else ")")
    elif t is dict:
        out.append("{")
        for k, x in v.items():
            frame_value_code(k, out)
            frame_value_code(x, out)
        out.append("}")
    else:
        value_code(v, out)


def array_code(a, out):
    """The bytes of a numeric array; every NaN counts as the same value."""
    import numpy as np

    a = np.ascontiguousarray(a)
    if a.dtype.kind == "f":
        nan = np.isnan(a)
        out.append(nan.tobytes().hex())
        a = np.where(nan, 0, a)
    out.append(a.tobytes().hex())


def frame_code(df) -> bytes:
    """Equal for two frames exactly when their column names, dtypes, index and values are
    equal (and every type in them is one that this code knows)."""
    import numpy as np
    import pandas as pd

    if type(df) is not pd.DataFrame or df.attrs or not df.flags.allows_duplicate_labels:
        raise Unsupported("frame kind")
    out = []
    cols = df.columns
    out.append("C%s|%s|%r|" % (type(cols).__name__, cols.dtype, list(cols.names)))
    for c in cols.tolist():
        value_code(c, out)
    idx = df.index
    if type(idx) is not pd.RangeIndex:
        raise Unsupported("index kind")
    out.append("R%d,%d,%d|%r|%r" % (idx.start, idx.stop, idx.step, idx.name, df.shape))
    for i in range(df.shape[1]):
        col = df.iloc[:, i]
        dt = col.dtype
        out.append("|D%s:%s|" % (type(dt).__name__, dt))
        if isinstance(dt, np.dtype) and dt.kind in "biufmM":
            array_code(col.to_numpy(), out)
        else:
            for v in col.tolist():
                frame_value_code(v, out)
    return "".join(out).encode("utf-8", "surrogatepass")


def has_fraction(v) -> bool:
    """True when v holds a finite float that is not a whole number."""
    t = type(v)
    if t is float:
        return v == v and v not in (float("inf"), float("-inf")) and v != int(v)
    if t is list or t is tuple:
        return any(has_fraction(x) for x in v)
    if t is dict:
        return any(has_fraction(x) for x in v.values())
    if t is str or t is int or v is None or t is bool or t is bytes:
        return False
    try:
        import numpy as np

        if isinstance(v, (np.ndarray, np.generic)):
            if v.dtype.kind == "f":
                a = np.asarray(v)
                a = a[np.isfinite(a)]
                return bool((a != np.floor(a)).any())
            if v.dtype.kind == "O":
                return any(has_fraction(x) for x in np.asarray(v).ravel().tolist())
    except ImportError:
        pass
    return False


def frame_has_fraction(df) -> bool:
    return any(has_fraction(df.iloc[:, i].to_numpy()) for i in range(df.shape[1]))


def rows_code(value) -> bytes:
    out = []
    value_code(value, out)
    return "".join(out).encode("utf-8", "surrogatepass")


def rows_blob(value):
    """(kind, bytes) from which duckreplay.py builds an equal new value."""
    try:
        blob = ("m", marshal.dumps(value, 4))
        back = marshal.loads(blob[1])
    except ValueError:
        blob = ("p", pickle.dumps(value, 4))
        back = pickle.loads(blob[1])
    if rows_code(back) != rows_code(value):
        raise Unsupported("the stored rows differ")
    return blob


# --- the recording run (in a forked child) --------------------------------------------

class Recorder:
    def __init__(self, real_connect, unsafe):
        self.real_connect = real_connect
        self.unsafe = unsafe
        self.starts = {}  # connect key -> (args, kwargs)
        self.steps = set()  # (state, text)
        self.reads = {}  # (state, text, method) -> code, or None when two results differ
        self.rows = {}  # (state, text, method) -> the fetched rows
        self.frames = {}  # id(frame) -> (frame, code)
        self.pairs = set()  # (code, code) that passed assert_frame_equal
        self.dropped = {}  # reason -> count
        self.calls = 0

    def drop(self, why):
        self.dropped[why] = self.dropped.get(why, 0) + 1

    def connect(self, *args, **kwargs):
        real = self.real_connect(*args, **kwargs)
        try:
            key = repr((args, sorted(kwargs.items())))
            pickle.dumps((args, kwargs))
        except Exception:  # noqa: BLE001
            return real
        self.starts[key] = (args, kwargs)
        return RecConn(self, real, (key,))

    def kind_of(self, real, text):
        """"read", "step" or None for a statement text that ran without an error."""
        if words_of(text) & self.unsafe:
            self.drop("unsafe word")
            return None
        try:
            types = [s.type.name for s in real.extract_statements(text)]
        except Exception:  # noqa: BLE001
            self.drop("not parsed")
            return None
        if types == ["SELECT"]:
            return "read"
        if types and all(t in STEP_TYPES for t in types) and not (words_of(text) & SAMPLE_WORDS):
            return "step"
        self.drop("statement type")
        return None

    def assert_frame_equal(self, real_assert):
        def wrapper(*args, **kwargs):
            real_assert(*args, **kwargs)
            if len(args) == 2 and not kwargs:
                a, b = self.frames.get(id(args[0])), self.frames.get(id(args[1]))
                if a is not None and b is not None and a[0] is args[0] and b[0] is args[1]:
                    self.pairs.add((a[1], b[1]))

        return wrapper


class RecConn:
    def __init__(self, rec, real, state):
        self._rec, self._real, self._state = rec, real, state
        self._ok = True  # False: nothing more is recorded for this connection
        self._pending = None

    def execute(self, query, *args, **kwargs):
        rec = self._rec
        rec.calls += 1
        self._pending = None
        kind = None
        if self._ok and not args and not kwargs and type(query) is str:
            kind = rec.kind_of(self._real, query)
        try:
            out = self._real.execute(query, *args, **kwargs)
        except BaseException:
            self._ok = False
            raise
        if kind == "step":
            rec.steps.add((self._state, query))
            self._state = self._state + (query,)
        elif kind == "read":
            self._pending = query
        else:
            self._ok = False
        return self if out is self._real else out

    def _fetch(self, method, args, kwargs):
        text, self._pending = self._pending, None
        value = getattr(self._real, method)(*args, **kwargs)
        rec = self._rec
        if not self._ok or text is None or args or kwargs:
            self._ok = False
            return value
        key = (self._state, text, method)
        try:
            if words_of(text) & MATH_WORDS and (has_fraction(value) if method == "fetchall" else frame_has_fraction(value)):
                raise Unsupported("fraction from the math library")
            if method == "fetchall":
                code = rows_code(value)
                rows_blob(value)
                rec.rows[key] = value
            else:
                code = frame_code(value)
                rec.frames[id(value)] = (value, code)
        except Unsupported as e:
            rec.drop("result type: %s" % e)
            code = None
        if key in rec.reads and rec.reads[key] != code:
            code = None
        rec.reads[key] = code
        return value

    def fetchall(self, *args, **kwargs):
        return self._fetch("fetchall", args, kwargs)

    def df(self, *args, **kwargs):
        return self._fetch("df", args, kwargs)

    def fetchdf(self, *args, **kwargs):
        return self._fetch("fetchdf", args, kwargs)

    def close(self):
        self._ok = False
        return self._real.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return None

    def __getattr__(self, name):
        self._ok = False
        return getattr(self._real, name)


def code_names(fn, seen):
    """The names that the code of fn (and of the functions nested in it) uses."""
    fn = getattr(fn, "__wrapped__", fn)
    fn = getattr(fn, "__func__", fn)
    code = getattr(fn, "__code__", None)
    names = set()
    stack = [code] if code is not None else []
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        names.update(c.co_names)
        stack.extend(k for k in c.co_consts if isinstance(k, type(code)))
    return names


def duck_tests(mod):
    """[test id] of the tests in mod whose code can reach a DuckDB connection: the method, or
    a function of the module or a method of the class that it names (followed on), uses the
    name duckdb or the name conn (the connection of the class fixture). The other tests are
    not run here; what they would send to DuckDB is then simply not recorded."""
    import types
    import unittest

    out = []
    for cls_name, cls in sorted(vars(mod).items()):
        if not (isinstance(cls, type) and issubclass(cls, unittest.TestCase) and cls.__module__ == mod.__name__):
            continue
        for meth in sorted(n for n in dir(cls) if n.startswith("test") and callable(getattr(cls, n))):
            seen = set()
            todo = [getattr(cls, meth)]
            done = set()
            hit = False
            while todo and not hit:
                names = code_names(todo.pop(), seen)
                if "duckdb" in names or "conn" in names:
                    hit = True
                for n in names - done:
                    done.add(n)
                    for v in (vars(mod).get(n), getattr(cls, n, None)):
                        if isinstance(v, (types.FunctionType, types.MethodType)) or hasattr(v, "__wrapped__"):
                            if getattr(getattr(v, "__wrapped__", v), "__module__", None) == mod.__name__:
                                todo.append(v)
            if hit:
                out.append("%s::%s::%s" % (mod.__name__.replace(".", "/") + ".py", cls_name, meth))
    return out


# --- TestExecutor: the two connections on the TPC csv files ---------------------------

EXEC_MODULE = "tests.test_executor"
EXEC_TESTS = ("test_execute_tpch", "test_optimized_tpch")  # rehearsed by --check
EXEC_LONG_TESTS = ("test_execute_tpcds",)  # and by --check-all


def executor_texts(mod):
    """{"tpch": [text], "tpcds": [text]}: what the pinned TestExecutor sends to its connections."""
    import preload

    out = {}
    for which, fixture in preload.EXEC_SQL_FIXTURES.items():
        texts = []
        for meta, sql, optimized in mod.load_sql_fixture_pairs(fixture):
            queries = (sql, optimized) if which == "tpch" else ((sql,) if meta.get("execute") else ())
            for query in queries:
                try:
                    texts.append(mod.transpile(query, write="duckdb")[0])
                except Exception:  # noqa: BLE001  (the test fails there before it reaches duckdb)
                    pass
        out[which] = list(dict.fromkeys(texts))
    return out


def executor_views(mod, which):
    """[(file name, statement text)] of the views of one connection, in the order of setUpClass."""
    import preload

    schema, directory = preload._exec_schema(mod, which)
    return [(f"{directory}{table}.csv.gz", preload.exec_view_sql(f"{directory}{table}.csv.gz", table, columns))
            for table, columns in schema.items()]


def executor_run(mod, texts, order, unsafe):
    """One run of every text on new connections. order: "file", "reverse" or "shuffled".
    Returns {(which, text): (frame code, pickled frame)}; a connection stops at the first
    statement that raises."""
    import duckdb
    from pandas.testing import assert_frame_equal

    out = {}
    for which in sorted(texts):
        conn = duckdb.connect()
        for _, view in executor_views(mod, which):
            conn.execute(view)
        items = list(texts[which])
        if order == "reverse":
            items.reverse()
        elif order == "shuffled":
            random.Random(7).shuffle(items)
        for text in items:
            if words_of(text) & unsafe:
                continue
            try:
                if [s.type.name for s in conn.extract_statements(text)] != ["SELECT"]:
                    continue
                frame = conn.execute(text).fetchdf()
            except Exception:  # noqa: BLE001
                break
            try:
                if words_of(text) & MATH_WORDS and frame_has_fraction(frame):
                    raise Unsupported("fraction from the math library")
                code = frame_code(frame)
                blob = pickle.dumps(frame, 4)
                back = pickle.loads(blob)
                if frame_code(back) != code:
                    raise Unsupported("the stored frame differs")
                assert_frame_equal(frame, back, check_exact=True)
            except (Unsupported, AssertionError):
                continue
            out[(which, text)] = (code, blob)
        conn.close()
    return out


def record_executor(unsafe, connect_key):
    """(kept, files): kept = {(state, text, "fetchdf"): ("pickle", blob)} for the results that
    are equal in three runs; files = {csv file: size}."""
    import importlib

    import preload

    preload.import_tests([EXEC_MODULE])
    mod = importlib.import_module(EXEC_MODULE)
    texts = executor_texts(mod)
    runs = [in_child(lambda order=order: executor_run(mod, texts, order, unsafe)) for order in ("file", "reverse", "shuffled")]
    a, b, c = [r() for r in runs]
    kept, files = {}, {}
    for which in sorted(texts):
        views = executor_views(mod, which)
        state = (connect_key,) + tuple(view for _, view in views)
        n = 0
        for text in texts[which]:
            got = a.get((which, text))
            if got is not None and b.get((which, text), (None,))[0] == got[0] and c.get((which, text), (None,))[0] == got[0]:
                kept[(state, text, "fetchdf")] = ("pickle", got[1])
                n += 1
        if n:
            files.update({name: os.path.getsize(name) for name, _ in views})
        print("executor %s: %d texts, %d kept, %d bytes" % (which, len(texts[which]), n, sum(
            len(v[1]) for k, v in kept.items() if k[0] == state)), file=sys.stderr)
    return kept, files


def setup_real(repo):
    """The pinned tree with the real duckdb and pandas, as the selector's parent loads it."""
    import preload

    preload.install_environment(repo)
    import duckdb  # noqa: F401
    import pandas  # noqa: F401

    preload.import_sqlglot()
    preload.import_tests([TEST_MODULE])
    import reduce

    reduce.install_loaders()
    return sys.modules[TEST_MODULE]


def record_run(mod, ids, unsafe):
    """Runs the tests with a recording connection; returns what it saw."""
    import duckdb

    import runner

    rec = Recorder(duckdb.connect, unsafe)
    duckdb.connect = rec.connect
    mod.assert_frame_equal = rec.assert_frame_equal(mod.assert_frame_equal)
    verdicts = {}
    runner.run_tests(list(enumerate(ids)), lambda number, failed: verdicts.__setitem__(ids[number], bool(failed)))
    return {
        "starts": rec.starts, "steps": rec.steps, "reads": rec.reads, "rows": rec.rows, "pairs": rec.pairs,
        "dropped": rec.dropped, "verdicts": verdicts, "calls": rec.calls,
    }


def in_child(fn):
    """Starts fn() in a forked child; returns a function that waits for its (pickled) result."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        code = 1
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            data = pickle.dumps(fn(), 4)
            with os.fdopen(w, "wb") as f:
                f.write(data)
            code = 0
        finally:
            os._exit(code)
    os.close(w)

    def result():
        with os.fdopen(r, "rb") as f:
            data = f.read()
        os.waitpid(pid, 0)
        return pickle.loads(data)

    return result


def build(repo, out_path):
    mod = setup_real(repo)
    import duckdb
    import numpy
    import pandas
    from pandas.testing import assert_frame_equal

    connect = duckdb.connect  # one thread (preload.import_tests), as at run time
    unsafe = unsafe_words(connect())
    ids = duck_tests(mod)
    print("tests that can reach duckdb:", len(ids), file=sys.stderr)
    first = in_child(lambda: record_run(mod, ids, unsafe))
    second = in_child(lambda: record_run(mod, ids[::-1], unsafe))
    a, b = first(), second()
    failed = sorted(t for t, f in a["verdicts"].items() if f) + sorted(t for t, f in b["verdicts"].items() if f)
    if failed or len(a["verdicts"]) != len(ids) or len(b["verdicts"]) != len(ids):
        raise SystemExit("pinned tests fail or did not run: %s" % failed[:5])
    steps = a["steps"] & b["steps"]
    reads = {k: c for k, c in a["reads"].items() if c is not None and b["reads"].get(k) == c}
    print("execute calls %d; steps %d; reads seen %d / %d, equal in both runs %d; dropped %s" % (
        a["calls"], len(steps), len(a["reads"]), len(b["reads"]), len(reads), a["dropped"]), file=sys.stderr)

    # third run, here: each state on a new connection, its reads in a shuffled order
    by_state = {}
    for (state, text, method), code in reads.items():
        by_state.setdefault(state, []).append((text, method, code))
    rng = random.Random(7)
    kept = {}  # (state, text, method) -> ("rows", value) or ("frame", code)
    frames = {}  # code -> a frame of that class
    second_frames = {}  # code -> a frame of that class from another query run
    differ = unreachable = 0
    for state, items in sorted(by_state.items()):
        if state[0] not in a["starts"] or any((state[:i], state[i]) not in steps for i in range(1, len(state))):
            unreachable += len(items)
            continue
        args, kwargs = a["starts"][state[0]]
        conn = connect(*args, **kwargs)
        for text in state[1:]:
            conn.execute(text)
        rng.shuffle(items)
        for text, method, code in items:
            value = getattr(conn.execute(text), method)()
            try:
                now = rows_code(value) if method == "fetchall" else frame_code(value)
            except Unsupported:
                now = None
            if now == code and words_of(text) & SAMPLE_WORDS:
                for _ in range(SAMPLE_REPEATS):
                    again = getattr(conn.execute(text), method)()
                    if (rows_code(again) if method == "fetchall" else frame_code(again)) != code:
                        now = None
                        break
            if now != code:
                differ += 1
                continue
            if method == "fetchall":
                kept[(state, text, method)] = ("rows", value)
            else:
                kept[(state, text, method)] = ("frame", code)
                if code not in frames:
                    frames[code] = value
                    second_frames[code] = getattr(conn.execute(text), method)()
        conn.close()
    print("third run: kept %d, differ %d, state not recorded %d" % (len(kept), differ, unreachable), file=sys.stderr)

    # frame classes and the pairs that pass the real assert
    class_of = {code: i for i, code in enumerate(sorted(frames))}
    pairs = set()
    bad_pairs = 0
    for code in frames:
        try:
            assert_frame_equal(frames[code], second_frames[code])
            pairs.add((class_of[code], class_of[code]))
        except AssertionError:
            bad_pairs += 1
    for x, y in a["pairs"] & b["pairs"]:
        if x in frames and y in frames and x != y:
            try:
                assert_frame_equal(frames[x], frames[y])
                pairs.add((class_of[x], class_of[y]))
            except AssertionError:
                bad_pairs += 1
    print("frame classes %d, pairs that pass %d (of them unequal classes %d), pairs that fail here %d" % (
        len(frames), len(pairs), sum(1 for x, y in pairs if x != y), bad_pairs), file=sys.stderr)

    # states as numbers; only the states and steps that lead to a kept read
    numbers = {}

    def number(state):
        if state not in numbers:
            numbers[state] = len(numbers) + 1
        return numbers[state]

    # the connections of TestExecutor (a connect() without arguments, like the others)
    files = {}
    if "((), [])" in a["starts"]:
        exec_kept, files = record_executor(unsafe, "((), [])")
        kept.update(exec_kept)

    out_starts, out_steps, out_reads = {}, {}, {}
    for (state, text, method), (kind, data) in sorted(kept.items(), key=lambda kv: kv[0]):
        for i in range(1, len(state)):
            out_steps[(number(state[:i]), state[i])] = number(state[:i + 1])
        out_starts[state[0]] = number(state[:1])
        if kind == "rows":
            answer = rows_blob(data)
        elif kind == "pickle":
            answer = ("d", data)
        else:
            answer = ("f", class_of[data])
        out_reads.setdefault((number(state), text), {})[method] = answer
    versions = [duckdb.__version__, pandas.__version__, numpy.__version__]
    import duckreplay

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path + ".tmp", "wb") as f:
        marshal.dump((duckreplay.FILE_VERSION, versions, out_starts, out_steps, out_reads, frozenset(pairs), files), f)
    os.replace(out_path + ".tmp", out_path)
    print("%s: %d bytes; states %d, steps %d, reads %d" % (
        out_path, os.path.getsize(out_path), len(numbers), len(out_steps), len(out_reads)), file=sys.stderr)
    return ids, a["verdicts"]


def rehearse(repo, path, verdict_path, long_tests=False):
    """The same tests with the replay on, set up as in the selector's parent, and the TPC
    tests of the executor (which pass on the pinned tree). Prints what was answered from the
    recording; exit code 1 when a verdict differs from the pinned run."""
    import json

    import preload

    preload.install_environment(repo)
    import lazymods

    import duckreplay

    assert lazymods.install(), "lazy pandas"
    assert duckreplay.install(duckreplay.read_file(path)), "no valid recording"
    preload.import_sqlglot()
    preload.import_tests([TEST_MODULE, EXEC_MODULE])
    import reduce

    reduce.install_loaders()
    preload.build_class_fixtures({(TEST_MODULE, "TestOptimizer"), preload.EXEC_KEY}, exec_lazy=True,
                                 exec_prebuild=["tpch_sqls", "tpcds_sqls"])
    assert not lazymods.loaded("duckdb") and not lazymods.loaded("pandas"), "the class fixture loads duckdb"
    import runner

    with open(verdict_path) as f:
        want = json.load(f)
    for name in EXEC_TESTS + (EXEC_LONG_TESTS if long_tests else ()):
        want["%s::%s::%s" % (EXEC_MODULE.replace(".", "/") + ".py", preload.EXEC_KEY[1], name)] = False
    ids = sorted(want)
    got = {}
    real = [0]

    def emit(number, failed):
        got[ids[number]] = bool(failed)
        if duckreplay.STATS["real"] != real[0]:
            print("  real connections: %d in %s (duckdb loaded %s, pandas loaded %s)" % (
                duckreplay.STATS["real"] - real[0], ids[number].split("::")[-1], lazymods.loaded("duckdb"),
                lazymods.loaded("pandas")), file=sys.stderr)
            real[0] = duckreplay.STATS["real"]

    runner.run_tests(list(enumerate(ids)), emit)
    print("rehearsal:", duckreplay.STATS, "duckdb loaded:", lazymods.loaded("duckdb"), "pandas loaded:",
          lazymods.loaded("pandas"), file=sys.stderr)
    bad = sorted(t for t in ids if got.get(t) != want[t])
    if bad:
        print("verdicts differ with the replay:", bad, file=sys.stderr)
    sys.exit(1 if bad else 0)


def main():
    if sys.argv[1] == "--rehearse":
        rehearse(sys.argv[2], sys.argv[3], sys.argv[4], "--check-all" in sys.argv)
        return
    repo, out_path = sys.argv[1], sys.argv[2]
    if "--from" in sys.argv:
        import shutil

        src = sys.argv[sys.argv.index("--from") + 1]
        ids = in_child(lambda: duck_tests(setup_real(repo)))()
        if not ids:
            raise SystemExit("no DuckDB tests found")
        verdicts = {t: False for t in ids}
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        shutil.copyfile(src, out_path + ".tmp")
        os.replace(out_path + ".tmp", out_path)
        print("%s: copied from %s (%d bytes); tests that can reach duckdb: %d" % (
            out_path, src, os.path.getsize(out_path), len(ids)), file=sys.stderr)
    else:
        ids, verdicts = build(repo, out_path)
    if "--check" in sys.argv or "--check-all" in sys.argv:
        import json
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(verdicts, f)
        code = subprocess.call([sys.executable, os.path.abspath(__file__), "--rehearse", repo, out_path, f.name]
                               + (["--check-all"] if "--check-all" in sys.argv else []))
        os.unlink(f.name)
        if code != 0:
            os.unlink(out_path)
            raise SystemExit("rehearsal failed: no recording")


if __name__ == "__main__":
    main()
