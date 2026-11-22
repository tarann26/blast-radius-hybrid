"""What the long-lived parent process loads before it forks one child per commit."""
from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import sys


class _StrictLogHandler(logging.Handler):
    """pytest installs log handlers that re-raise formatting errors, so a broken
    logging call fails the test there. This handler gives a direct run the same behaviour."""

    def emit(self, record):
        record.getMessage()

    def handleError(self, record):
        raise


def _single_thread_duckdb() -> None:
    """Every duckdb connection of the tests runs on the calling thread only. A connection
    without worker threads still works in a forked child, so the parent can build the class
    fixtures once. It also avoids one thread per CPU of the host."""
    try:
        import duckdb
    except ImportError:
        return
    real = duckdb.connect
    if getattr(real, "_br", False):
        return  # done already, or the replaying connect of duckreplay.py (which calls the function below)
    duckdb.connect = single_thread_connect(real)


def single_thread_connect(real):
    def connect(*args, **kwargs):
        config = dict(kwargs.pop("config", None) or {})
        config.setdefault("threads", 1)
        return real(*args, config=config, **kwargs)

    connect._br = True
    return connect


def install_environment(root: str) -> None:
    # sqlglot checks at import whether a test framework is loaded (expressions/core.py UNITTEST)
    # and then validates expression arguments strictly, as it does under pytest
    import unittest  # noqa: F401

    import pool

    os.environ["SKIP_INTEGRATION"] = "0"
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    pool.install()
    logging.getLogger().addHandler(_StrictLogHandler(level=0))


def import_sqlglot() -> None:
    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        if info.name.endswith("__main__"):
            continue
        importlib.import_module(info.name)


def broken_modules(test_modules=()) -> list:
    """Imports every sqlglot module and the given test modules; returns the names of those
    that raise. A test file that cannot be imported fails all of its tests."""
    bad = []
    try:
        import sqlglot
    except BaseException as e:  # noqa: BLE001
        if isinstance(e, KeyboardInterrupt):
            raise
        return ["sqlglot"]
    for info in pkgutil.walk_packages(sqlglot.__path__, "sqlglot.", onerror=lambda name: bad.append(name)):
        if info.name.endswith("__main__"):
            continue
        try:
            importlib.import_module(info.name)
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            bad.append(info.name)
    bad.extend(broken_test_modules(test_modules))
    return sorted(set(bad))


def broken_test_modules(names) -> list:
    """Imports the given test modules; returns the names of those that raise. A test file
    that cannot be imported fails all of its tests."""
    import unittest

    bad = []
    for name in names:
        try:
            importlib.import_module(name)
        except unittest.SkipTest:
            pass  # pytest reports the tests of such a file as skipped, which is not a failure
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            bad.append(name)
    return bad


def import_tests(modules) -> None:
    for name in modules:
        importlib.import_module(name)
    _single_thread_duckdb()


# Test classes whose setUpClass is costly and reads nothing that a test changes.
SHARED_FIXTURES = (
    ("tests.test_optimizer", "TestOptimizer"),
    ("tests.test_executor", "TestExecutor"),
)


# --- TestExecutor: lazy class fixture -----------------------------------------
# TestExecutor.setUpClass (2.3 s) parses the TPC-H / TPC-DS csv.gz files into sqlglot Tables
# and creates one duckdb view per table. Only a few tests read those attributes. The parent
# installs, in place of the call, one descriptor per attribute; the first test that reads one
# builds it exactly like setUpClass does (same code for the duckdb views and the Table
# objects; the parsed csv rows come from a cache made at image build time from the same
# files, see build_exec_cache.py). Without the cache, the real setUpClass runs as before.

EXEC_KEY = ("tests.test_executor", "TestExecutor")
EXEC_DATA_PREFIXES = ("tests/fixtures/optimizer/tpc-h/", "tests/fixtures/optimizer/tpc-ds/")
EXEC_CACHE = os.environ.get("BR_EXEC_CACHE", "/app/cache")
EXEC_STATS = {"built": []}  # what was built lazily in this process (for logs)
# the .sql fixture of each data set, as setUpClass loads it into <which>_sqls
EXEC_SQL_FIXTURES = {"tpch": "optimizer/tpc-h/tpc-h.sql", "tpcds": "optimizer/tpc-ds/tpc-ds.sql"}


def exec_rows_from_csv(mod, file_name):
    """The row list of TestExecutor.setUpClass.setup() for one csv file (same code)."""
    import ast
    import csv

    reader = csv.reader(mod.open_file(file_name), delimiter="|")
    rows = []
    ctypes = []
    next(reader)
    for row in reader:
        if not ctypes:
            for i, v in enumerate(row):
                try:
                    ctypes.append(type(ast.literal_eval(v)))
                except (ValueError, SyntaxError):
                    ctypes.append(str)
        rows.append(tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row)))
    return rows


def _exec_schema(mod, which):
    if which == "tpch":
        return mod.TPCH_SCHEMA, mod.DIR_TPCH
    return mod.TPCDS_SCHEMA, mod.DIR_TPCDS


def _exec_cached_rows(mod, which):
    """{table: marshalled rows} from the build-time cache, or None when there is no valid
    cache. The rows of one table are unpacked only when that table is read (_LazyTables)."""
    import marshal

    path = os.path.join(EXEC_CACHE, "exec_%s.marshal" % which)
    try:
        with open(path, "rb") as f:
            # loads(read()), not load(f): load() asks the file object for every single value,
            # which is about 9 times slower on these files (1.9 s against 0.2 s for TPC-DS)
            sizes, data = marshal.loads(f.read())
    except (OSError, ValueError, EOFError, TypeError):
        return None
    schema, directory = _exec_schema(mod, which)
    if list(data) != list(schema) or not all(type(v) is bytes for v in data.values()):
        return None
    for table in schema:
        try:
            if os.path.getsize(f"{directory}{table}.csv.gz") != sizes.get(table):
                return None
        except OSError:
            return None
    return data


def exec_view_sql(file_name, table, columns) -> str:
    """The statement with which TestExecutor.setUpClass makes the duckdb view of one table
    (the same text, character by character: build_duck_replay.py records under it)."""
    return f"""
                CREATE VIEW {table} AS
                SELECT *
                FROM READ_CSV('{file_name}', delim='|', header=True, columns={columns})
                """


def _exec_build(cls, name):
    mod = sys.modules[cls.__module__]
    which = name.split("_", 1)[0]
    schema, directory = _exec_schema(mod, which)
    if name.endswith("_conn"):
        conn = mod.duckdb.connect()
        for table, columns in schema.items():
            conn.execute(exec_view_sql(f"{directory}{table}.csv.gz", table, columns))
        return conn
    if name.endswith("_tables"):
        import marshal

        cached = _exec_cached_rows(mod, which)
        if cached is not None:
            lazy = _LazyTables(schema, lambda table: mod.Table(columns=schema[table], rows=marshal.loads(cached[table])))
            if EXEC_TABLES_STRICT:
                lazy.load_all()
                return dict(lazy)
            return lazy
        tables = {}
        for table, columns in schema.items():
            tables[table] = mod.Table(columns=columns, rows=exec_rows_from_csv(mod, f"{directory}{table}.csv.gz"))
        return tables
    # tpch_sqls / tpcds_sqls: setUpClass loads every entry (no reduced run applies here)
    fixture = EXEC_SQL_FIXTURES[which]
    try:
        from reduce import STATE
    except ImportError:
        STATE = None
    saved = STATE.in_test if STATE is not None else None
    if STATE is not None:
        STATE.in_test = False
    try:
        return list(mod.load_sql_fixture_pairs(fixture))
    finally:
        if STATE is not None:
            STATE.in_test = saved


class _LazyTables(dict):
    """The {table: Table} dict that TestExecutor.setUpClass builds, filled on demand.

    The tests hand this dict to the worker pool, and a worker reads only the tables of the
    query that it executes (tests/test_executor.py mp_execute: _tables[name]); the TPC-H
    queries execute nothing, so none of its tables is read. A read of one table by its name
    unpacks that table. Every other use of the dict (iteration, length, membership, get,
    comparison, copy, pickle, changes) first unpacks all tables, in the order of setUpClass,
    so it sees the same dict as before. A commit that changes anything but sqlglot code or
    .sql fixtures gets the plain dict (exec_tables_strict)."""

    __slots__ = ("_names", "_build", "_full")

    def __init__(self, names, build):
        super().__init__()
        self._names = tuple(names)
        self._build = build
        self._full = False

    def __missing__(self, key):
        if self._full or key not in self._names:
            raise KeyError(key)
        value = self._build(key)
        dict.__setitem__(self, key, value)
        return value

    def load_all(self):
        if not self._full:
            have = {k: dict.__getitem__(self, k) for k in dict.keys(self)}
            dict.clear(self)
            for name in self._names:  # the order of setUpClass
                dict.__setitem__(self, name, have[name] if name in have else self._build(name))
            self._full = True
        return self

    def _all(name):  # noqa: N805
        plain = getattr(dict, name)

        def method(self, *args, **kwargs):
            self.load_all()
            return plain(self, *args, **kwargs)

        method.__name__ = name
        return method

    for _name in ("__iter__", "__len__", "__contains__", "__eq__", "__ne__", "__repr__", "__reversed__",
                  "__or__", "__ror__", "__ior__", "__setitem__", "__delitem__", "__sizeof__",
                  "keys", "values", "items", "get", "copy", "pop", "popitem", "setdefault", "update", "clear"):
        locals()[_name] = _all(_name)
    del _name, _all

    def __reduce_ex__(self, protocol):
        return (dict, (dict(self.load_all()),))


EXEC_TABLES_STRICT = False  # the tables as a plain, complete dict (see exec_tables_strict)


def exec_tables_strict():
    """For a commit that changes more than sqlglot code and .sql fixtures (test code can then
    use the tables in any way): the table dicts are plain and complete from the start."""
    global EXEC_TABLES_STRICT
    EXEC_TABLES_STRICT = True
    cls = getattr(sys.modules.get(EXEC_KEY[0]), EXEC_KEY[1], None)
    for name in ("tpch_tables", "tpcds_tables"):
        value = None if cls is None else cls.__dict__.get(name)
        if isinstance(value, _LazyTables):
            setattr(cls, name, dict(value.load_all()))


class _LazyClassAttr:
    """Non-data descriptor: the first read through an instance builds the value and stores
    it on the class, which replaces the descriptor."""

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, owner):
        import time

        t0 = time.process_time()
        value = _exec_build(owner, self.name)
        setattr(owner, self.name, value)
        EXEC_STATS["built"].append("%s:%.3f" % (self.name, time.process_time() - t0))
        return value


EXEC_LAZY = ("tpch_conn", "tpcds_conn", "tpch_tables", "tpcds_tables", "tpch_sqls", "tpcds_sqls")


def exec_cache_present() -> bool:
    return all(os.path.exists(os.path.join(EXEC_CACHE, "exec_%s.marshal" % w)) for w in EXEC_SQL_FIXTURES)


def setup_exec_lazy(cls, prebuild=()):
    """Replaces the work of TestExecutor.setUpClass by lazy attributes. The files that it
    would read are recorded as for an eager setUpClass (see refresh_class_fixtures)."""
    mod = sys.modules[cls.__module__]
    cls.cache = {}
    for name in EXEC_LAZY:
        setattr(cls, name, _LazyClassAttr(name))
    for name in prebuild:
        setattr(cls, name, _exec_build(cls, name))
    reads = set()
    attrs = {}
    for which, fixture in EXEC_SQL_FIXTURES.items():
        schema, directory = _exec_schema(mod, which)
        reads.update(os.path.abspath(f"{directory}{table}.csv.gz") for table in schema)
        reads.add(os.path.abspath(os.path.join(mod.FIXTURES_DIR, fixture)))
        attrs[which + "_sqls"] = (fixture, "load_sql_fixture_pairs")
    cls._br_reads = reads
    cls._br_attrs = attrs


def exec_attribute_users(cls) -> dict:
    """{test method name: lazy attributes that its code reads (self.<name>)}."""
    out = {}
    for meth, fn in vars(cls).items():
        code = getattr(fn, "__code__", None)
        if not meth.startswith("test") or code is None:
            continue
        names = set()
        stack = [code]
        while stack:
            c = stack.pop()
            names.update(c.co_names)
            stack.extend(k for k in c.co_consts if isinstance(k, type(code)))
        used = names.intersection(EXEC_LAZY)
        if used:
            out[meth] = used
    return out


def exec_fresh_ok(changed_paths) -> bool:
    """In a child that imports the patched tree again (fresh mode), TestExecutor.setUpClass
    still does the same work when the patch changes none of: the test files (tests/*.py), the
    TPC data files, and the sqlglot code that setUpClass runs (sqlglot/executor/table.py:
    Table, RowReader, RangeReader; the rest is duckdb, csv, ast). Then the lazy attributes
    give the same objects (the .sql fixtures are read through the open() overlay)."""
    if not exec_cache_present():
        return False
    for p in changed_paths:
        if p.startswith("tests/") and p.endswith(".py"):
            return False
        if p.startswith(EXEC_DATA_PREFIXES) or p.startswith("sqlglot/executor/"):
            return False
    return True


def exec_fresh_hook(mod):
    """runner.ON_IMPORT hook for tests.test_executor in a fresh child."""
    cls = getattr(mod, EXEC_KEY[1], None)
    if cls is None or "setUpClass" not in cls.__dict__:
        return
    setup_exec_lazy(cls)
    cls.setUpClass = classmethod(lambda c: None)
    cls.tearDownClass = classmethod(lambda c: None)  # it would build the connections to close them


def _reads_during(fn) -> set:
    """Runs fn(); returns the absolute paths of the files that it opened."""
    import builtins
    import io

    real = builtins.open
    seen = set()

    def recording_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            seen.add(os.path.abspath(os.fspath(file)))
        return real(file, *args, **kwargs)

    builtins.open = io.open = recording_open
    try:
        fn()
    finally:
        builtins.open = io.open = real
    return seen


def _fixture_attributes(cls, reads) -> dict:
    """{class attribute: (fixture name, loader name)} for the attributes that setUpClass filled
    with the entries of a .sql fixture (list(load_sql_fixture_pairs(name)) or load_sql_fixtures)."""
    helpers = sys.modules.get("tests.helpers")
    if helpers is None:
        return {}
    base = os.path.abspath(helpers.FIXTURES_DIR) + os.sep
    names = [p[len(base):] for p in reads if p.startswith(base) and p.endswith(".sql")]
    out = {}
    for attr, value in list(cls.__dict__.items()):
        if not isinstance(value, list) or not value:
            continue
        for name in names:
            for loader in ("load_sql_fixture_pairs", "load_sql_fixtures"):
                try:
                    if list(getattr(helpers, loader)(name)) == value:
                        out[attr] = (name, loader)
                except Exception:  # noqa: BLE001
                    pass
    return out


def refresh_class_fixtures(root: str, files) -> list:
    """Child side, after the open() overlay: a shared class fixture that the parent built from
    a file that this commit changes. Attributes that hold the entries of a .sql fixture are
    loaded again from the patched file. If that fails, or setUpClass read the file in another
    way, setUpClass raises, so every test of the class fails (as in a full run when setUpClass
    raises; otherwise a costly but safe answer). The parent's objects stay referenced: a
    forked child that drops the last reference to a duckdb connection of the parent can die.
    Returns [(module of the class, True when its setUpClass now raises)]."""
    changed = {os.path.join(root, rel) for rel in files}
    helpers = sys.modules.get("tests.helpers")
    done = []
    for mod_name, cls_name in SHARED_FIXTURES:
        mod = sys.modules.get(mod_name)
        cls = getattr(mod, cls_name, None) if mod else None
        reads = getattr(cls, "_br_reads", None) if cls is not None else None
        if not reads or not (reads & changed):
            continue
        attrs = cls._br_attrs
        base = os.path.abspath(helpers.FIXTURES_DIR) + os.sep
        known = {base + name for name, _ in attrs.values()}
        error = None
        if (reads & changed) - known:
            error = RuntimeError("class fixture read a changed file: " + ", ".join(sorted((reads & changed) - known)))
        cls._br_parent_state = dict(vars(cls))
        for attr, (name, loader) in attrs.items():
            if base + name in changed and error is None:
                try:
                    setattr(cls, attr, list(getattr(helpers, loader)(name)))
                except Exception as e:  # noqa: BLE001
                    error = e
        if error is not None:
            def fail(c, error=error):
                raise error

            cls.setUpClass = classmethod(fail)
        done.append((mod_name, error is not None))
    return done


def reduce_class_fixtures(changed) -> None:
    """Child side, for a commit that changes nothing but data files. changed: {(kind, fixture
    name): indexes of the entries that the pinned file does not have} (reduce.changed_entries).
    A class attribute that holds the entries of such a fixture keeps only those entries. A
    test that walks the list handles each entry by itself, and an entry that equals a pinned
    entry gives the pinned result, so only the new entries can make the test fail."""
    for mod_name, cls_name in SHARED_FIXTURES:
        mod = sys.modules.get(mod_name)
        cls = getattr(mod, cls_name, None) if mod else None
        attrs = getattr(cls, "_br_attrs", None) if cls is not None else None
        for attr, (name, loader) in (attrs or {}).items():
            keep = changed.get(("p" if loader == "load_sql_fixture_pairs" else "x", name))
            if keep is not None:
                setattr(cls, attr, [e for i, e in enumerate(getattr(cls, attr)) if i in keep])


def shared_class_info(root: str) -> list:
    """[(test id prefix, class, {repo-relative fixture path: class attribute that holds its
    entries}, repo-relative paths that setUpClass read in another way)] for the classes whose
    fixture this process built."""
    out = []
    helpers = sys.modules.get("tests.helpers")
    for mod_name, cls_name in SHARED_FIXTURES:
        mod = sys.modules.get(mod_name)
        cls = getattr(mod, cls_name, None) if mod else None
        reads = getattr(cls, "_br_reads", None) if cls is not None else None
        if not reads or helpers is None:
            continue
        base = os.path.abspath(helpers.FIXTURES_DIR) + os.sep
        by_file = {os.path.relpath(base + name, root): attr for attr, (name, _) in cls._br_attrs.items()}
        other = {os.path.relpath(p, root) for p in reads} - set(by_file)
        out.append((mod_name.replace(".", "/") + ".py::" + cls_name + "::", cls, by_file, other))
    return out


def build_class_fixtures(wanted, exec_lazy=False, exec_prebuild=()) -> list:
    """Runs setUpClass once, here in the parent, for the classes in 'wanted' (a set of
    (module, class) names). The forked children inherit the result and skip the call.
    The files that setUpClass read are recorded for refresh_class_fixtures.
    With exec_lazy (and the build-time cache), TestExecutor gets lazy attributes instead (see
    above); exec_prebuild names the ones to build here all the same. build_maps.py calls this
    without exec_lazy: its map of the lines that the parent runs must keep every line of the
    real setUpClass (a commit that changes one of them then runs in fresh mode).
    Returns the classes that were set up."""
    done = []
    for mod_name, cls_name in SHARED_FIXTURES:
        if (mod_name, cls_name) not in wanted:
            continue
        mod = sys.modules.get(mod_name)
        cls = getattr(mod, cls_name, None) if mod else None
        if cls is None:
            continue
        if exec_lazy and (mod_name, cls_name) == EXEC_KEY and exec_cache_present():
            setup_exec_lazy(cls, exec_prebuild)
        else:
            reads = _reads_during(cls.setUpClass)
            cls._br_reads = reads
            cls._br_attrs = _fixture_attributes(cls, reads)
        cls.setUpClass = classmethod(lambda c: None)
        # a child must not close what it shares with the parent and with the other children
        cls.tearDownClass = classmethod(lambda c: None)
        done.append((mod_name, cls_name))
    return done
