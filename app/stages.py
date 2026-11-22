"""Stage check for the long executor test (tests/test_executor.py test_execute_tpcds).

That test hands one task per TPC-DS query to a process pool. A task runs
sqlglot.executor.execute(): it optimizes the parsed query, builds a plan from the optimized
expression, and runs the plan with the pure-Python executor over the TPC-DS tables; the test
then compares the rows with what DuckDB returns for the query's DuckDB SQL text. Nearly all of
the CPU time is the executor's row work. The work from the plan on starts from two values
only: the optimized expression that execute() passes to Plan(), and the tables that it
passes to PythonExecutor(). The comparison needs the task's rows and the SQL text that
transpile() makes for DuckDB.

build_stages.py records, on the pinned commit, for every query: a fingerprint of those two
values, of the DuckDB SQL text and of the rows that the test receives, and the sqlglot lines
that run after the point where the plan is built ("late lines": plan, executor, result). At
run time the test runs once with the patched code, with a hook on its tasks:

- The patched code makes another DuckDB SQL text for a query than the pinned code: DuckDB's
  answer is not known, the check ends without a verdict.
- A task whose two input fingerprints equal the pinned ones, and whose late lines (as recorded
  on the pinned commit) hold no line that the commit changes and no line that reads a name
  whose import-time value the commit changes, stops where the plan would be built. Its later
  work would get the same input and run the same code as on the pinned commit, so it would
  give the pinned rows.
- Every other task (another input, or late lines that the commit touches) runs to its end
  right there, as in an ordinary run of the test, and the test receives its rows. When they
  equal the pinned rows (same columns, same values of the same types, same order), the
  comparison with DuckDB's answer to the same SQL text gets the same two inputs as on the
  pinned commit, where it passes; it is left out. Other rows: no verdict, the check ends.
- The runs to the end have a budget (the build-time CPU seconds of their later work; the
  selector sets it from the seconds that the test would add to the answer). When the budget
  is used up, the check ends without a verdict.

Without a verdict the selector treats the test as before the check (selected, not run). An
error in the test or in a task fails the test, as in a full run. When the test passes in the
check with every task proven in one of the two ways, and the module state after the check
differs from the state before it only as the pinned test changes it (state_ok), the test is
known to pass on the patched code."""
from __future__ import annotations

import enum
import hashlib
import io
import marshal
import os
import pickle
import sys
import time
import types
import zlib

MAP_VERSION = 7
_MISSING = "\x00unset"
_OPAQUE_TYPES = (types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.ModuleType,
                 types.CodeType, types.GeneratorType, types.MethodDescriptorType, types.WrapperDescriptorType,
                 staticmethod, classmethod, property)
_BASES = ((dict, lambda o: list(dict.items(o))), (list, lambda o: list(list.__iter__(o))),
          (tuple, lambda o: list(tuple.__iter__(o))), (set, lambda o: list(set.__iter__(o))),
          (frozenset, lambda o: list(frozenset.__iter__(o))), (str, lambda o: str.__add__(o, "")),
          (bytes, lambda o: bytes.__add__(o, b"")), (float, lambda o: float.__add__(o, 0.0)),
          (int, lambda o: int.__add__(o, 0)))


def _describe(t):
    """How to write an object of type t: (kind, text, slot names, has a dict, base value)."""
    name = "%s.%s" % (t.__module__, t.__qualname__)
    if issubclass(t, type):
        return (1, name, (), False, None)
    if issubclass(t, enum.Enum):
        return (2, name, (), False, None)
    if issubclass(t, _OPAQUE_TYPES):
        return (3, name, (), False, None)
    if not (t.__module__ or "").startswith("sqlglot"):
        return (4, name, (), False, None)  # an object of another library: pickle writes its value
    slots = []
    for klass in reversed(t.__mro__):
        s = klass.__dict__.get("__slots__", ())
        for n in ((s,) if isinstance(s, str) else s):
            if n not in ("__dict__", "__weakref__") and n not in slots:
                slots.append(n)
    base = None
    for b, read in _BASES:
        if issubclass(t, b):
            base = read
            break
    return (0, "O:" + name, tuple(slots), bool(t.__dictoffset__), base)


class _Writer(pickle.Pickler):
    """Writes an object graph as bytes that depend only on its content and shape: every object
    with its exact class, every slot and attribute (for an expression: args, parent, arg_key,
    index, comments, type, meta, cached hash), containers in their iteration order, and each
    repeated reference as a reference to the first one (so shared and cyclic structure counts).
    It calls no sqlglot code: an object of a sqlglot class is written from its slots and its
    __dict__, not through its __reduce__. Functions and modules have no value here: a graph
    that holds one gets no fingerprint."""

    def __init__(self, file, leaves):
        super().__init__(file, 4)
        self.leaves = leaves
        self.info = {}
        self.opaque = False

    def reducer_override(self, obj):
        t = type(obj)
        info = self.info.get(t)
        if info is None:
            info = self.info[t] = _describe(t)
        kind = info[0]
        if kind == 0:
            if self.leaves:
                leaf = self.leaves.get(id(obj))
                if leaf is not None:
                    return (str, (leaf,))
            state = [getattr(obj, n, _MISSING) for n in info[2]]
            if info[3]:
                state.append(obj.__dict__)
            if info[4] is not None:
                state.append(info[4](obj))
            return (str, (info[1],), tuple(state))
        if kind == 4:
            return NotImplemented
        if kind == 1:
            if not (obj.__module__ or "").startswith("sqlglot"):
                return NotImplemented  # a class of another library: pickle writes its name
            return (str, ("C:%s.%s" % (obj.__module__, obj.__qualname__),))
        if kind == 2:
            return (str, ("E:%s.%s" % (info[1], obj.name),))
        self.opaque = True
        return (str, ("?",))


def fingerprint(obj, leaves=None):
    """16 bytes, or None when the object graph has a part without a value to compare."""
    buf = io.BytesIO()
    w = _Writer(buf, leaves)
    try:
        w.dump(obj)
    except Exception:  # noqa: BLE001  (an object that pickle cannot write)
        return None
    if w.opaque:
        return None
    return hashlib.blake2b(buf.getvalue(), digest_size=16).digest()


def table_leaves(tables):
    """{id(table): text} for the tables of the class fixture that are built (the fixture
    builds a table when a task first reads it, see preload._LazyTables). A task gets the
    table objects as they are; the text holds what the executor reads from one besides the
    values of its rows: the columns, the column range, the reader's column positions and the
    number of rows. None when a table does not look like one."""
    out = {}
    try:
        items = list(dict.items(tables)) if isinstance(tables, dict) else []
        for name, table in items:
            reader = sorted(table.reader.columns.items())
            out[id(table)] = "T:%s:%r" % (name, (table.columns, table.column_range, reader, len(table.rows)))
    except Exception:  # noqa: BLE001
        return None
    return out


def row_hashes(tables):
    """{name: hash of the rows} of the tables of the class fixture that are built (dict.items:
    a table that is not built yet is not read by anyone), or None. hash(tuple(rows)) hashes
    every value of every row (str, int, float, None in the TPC-DS tables); it is valid under
    one hash seed (hash_probe)."""
    try:
        if not isinstance(tables, dict):
            return None
        return {name: hash(tuple(table.rows)) for name, table in dict.items(tables)}
    except Exception:  # noqa: BLE001
        return None


def rows_fingerprint(table):
    """Fingerprint of the result that the test receives from a task: the columns and the rows
    (values with their types, in order). None for anything but a plain result table."""
    try:
        columns, rows = table.columns, table.rows
        if type(columns) is not tuple or type(rows) is not list:
            return None
    except Exception:  # noqa: BLE001
        return None
    return fingerprint((columns, rows))


def hash_probe():
    """The fingerprints hold string hashes, and the recorded module state holds hashes of
    code: both are valid under one hash seed and one Python version only."""
    return (hash("stage map"), hash(("G", "sqlglot", 1, 2.5, b"x", None)), sys.version_info[:3], MAP_VERSION)


def load_map(path):
    """{test id: record} written by build_stages.py; {} when there is no map, or when this
    process hashes in another way than the process that built it."""
    try:
        with open(path, "rb") as f:
            records = marshal.loads(zlib.decompress(f.read()))
    except (OSError, ValueError, EOFError, TypeError, zlib.error):
        return {}
    if type(records) is not dict:
        return {}
    probe = hash_probe()
    return {k: r for k, r in records.items() if type(r) is dict and r.get("probe") == probe}


def late_tasks(record, lines):
    """The tasks whose later work runs a changed line on the pinned commit.
    lines: {repo-relative path: set of lines} of the commit (the changed statements, and the
    statements that read a name whose import-time value the commit changes)."""
    out = set()
    for rel, lns in lines.items():
        by_line = record["late"].get(rel)
        if not by_line or not lns:
            continue
        for ln in lns:
            sid = by_line.get(ln)
            if sid is not None:
                out.update(record["sets"][sid])
    return out


class _Same(BaseException):
    """Raised where the plan would be built: the task ends there."""


class _PlanInput:
    """Stands for the plan between the two calls of execute(): Plan(expression) and
    PythonExecutor(tables=...).execute(plan)."""

    def __init__(self, expression):
        self.expression = expression


class _Quiet:
    """Stands for the pinned optimized expression between optimize() and Plan() in execute():
    sql(pretty=True) gives the pinned text without running the generator. Any other use reaches
    the expression itself and marks it as seen (compare() then checks its content)."""

    __slots__ = ("_expr", "_text", "_seen")

    def __init__(self, expr, text):
        object.__setattr__(self, "_expr", expr)
        object.__setattr__(self, "_text", text)
        object.__setattr__(self, "_seen", False)

    def sql(self, *args, **kwargs):
        if not args and kwargs == {"pretty": True}:
            return self._text
        object.__setattr__(self, "_seen", True)
        return self._expr.sql(*args, **kwargs)

    def __getattr__(self, name):
        object.__setattr__(self, "_seen", True)
        return getattr(self._expr, name)


class StageCheck:
    """One run of the test with the hook run_task on its tasks (see the module text). After
    it: 'same' holds the tasks that stopped with the pinned input, 'ran' the tasks that ran
    to their end, 'proved' those of them that gave the pinned rows; 'over' is True when the
    check ended without a verdict ('why' says where)."""

    slack = 5.0  # CPU s over twice the build-time cost before a check counts as slow (None: never)

    def __init__(self, record, later, budget, active=None):
        self.record = record
        self.tasks = record["tasks"]  # {task index: (expression, tables, duckdb sql, rows) fingerprints}
        self.cpu = record["cpu"]  # {task index: (CPU s up to the plan, CPU s after it)} at build time
        self.later = later  # tasks whose later work runs changed code: they run to their end
        self.budget = budget  # build-time CPU s that the runs to the end may cost
        self.active = active  # the tasks that the test runs at all (None = all), see reduce.py
        self.spent = 0.0
        self.expected = 0.0  # build-time CPU s of the tasks so far (to the plan, or to the end)
        self.clock = None  # CPU time of this process at the first task
        self.over = False
        self.why = None
        self.same = []
        self.ran = []  # (task, why it ran to its end)
        self.proved = []
        self.seq = []  # one letter per task, in the order of the test (for the logs)
        self.task = None
        self.module = None
        self.globals = None
        self.real = None
        self.cls = None
        self.helper = None
        self.memo = None  # (memo.Store, dirty step ids) when the step memo applies (memo.py)
        self.memo_task = False
        self.memo_used = False
        self.memo_stats = [0, 0, 0, 0]  # steps taken from the record, run, restored; tasks that fell back
        self.memo_full = set()  # tasks that run to their end although the memo has a record (memo.prepare)
        self.noopt = set()  # tasks whose optimize() may give the pinned optimized expression
        self.notext = set()  # tasks whose DuckDB SQL text is the pinned one when the SQL string is (memo.prepare)
        self.nomid = set()  # tasks whose logger text in execute() is the pinned one (memo.prepare)
        self.hook_changed = True  # the commit changes execute() itself (memo.prepare)
        self.fast_stats = [0, 0, 0, 0]  # texts from the record, roots without a plan, logger texts, shared schemas
        # The row content of the fixture tables (table_leaves holds only their shape): the pinned
        # hashes (build_memo.py, when the memo is there), the hashes at the start of the check, and
        # of each table that compare() first sees after the start. rows_ok() checks them at the end.
        self.rowh = None
        self.rows0 = {}
        self.rows_seen = {}
        self.rows_cpu = 0.0  # texts taken from the record, roots restored without a plan, logger texts
        self.opt_stats = [0, 0]  # optimize() calls answered from the record, calls that ran

    # -- the two names that execute() looks up in its module, and the test's comparison ----

    def install(self, test_module, cls):
        """Returns False when the test does not look like the pinned one from outside."""
        fn = getattr(test_module, "execute", None)
        g = getattr(fn, "__globals__", None)
        if not isinstance(fn, types.FunctionType) or g is None or g.get("__name__") != "sqlglot.executor":
            return False
        plan, executor = g.get("Plan"), g.get("PythonExecutor")
        if not isinstance(plan, type) or not isinstance(executor, type):
            return False
        if not isinstance(getattr(test_module, "transpile", None), types.FunctionType):
            return False
        helper = cls.__dict__.get("subtestHelper")
        if not isinstance(helper, types.FunctionType):
            return False
        self.module, self.cls, self.globals, self.real = test_module, cls, g, (plan, executor)
        self.helper = helper
        check = self

        def plan_stub(*args, **kwargs):
            if check.task is None or len(args) != 1 or kwargs:
                return plan(*args, **kwargs)
            return _PlanInput(args[0])

        class ExecutorStub:
            def __init__(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs

            def execute(self, plan_input):
                return check.at_executor(self.args, self.kwargs, plan_input)

        def subtest_helper(case, i, table, tpch=True):
            # the test calls this with the rows of a task that ran to its end
            if tpch or not check.on_result(i, table):
                return helper(case, i, table, tpch=tpch)
            return None

        self.real_opt = None
        if (self.noopt or self.nomid) and self.memo is not None and os.environ.get("BR_MEMO_OPT", "1") == "1":
            real_opt = g.get("optimize")
            if isinstance(real_opt, types.FunctionType):
                self.real_opt = real_opt
                g["optimize"] = self.optimize_stub
                real_schema = g.get("ensure_schema")
                if self.noopt and not self.hook_changed and isinstance(real_schema, types.FunctionType):
                    self.real_schema = real_schema
                    self.schema_cache = {}
                    g["ensure_schema"] = self.schema_stub
        g["Plan"], g["PythonExecutor"] = plan_stub, ExecutorStub
        cls.subtestHelper = subtest_helper
        return True

    def schema_stub(self, *args, **kwargs):
        """ensure_schema() as execute() calls it in a task whose lines before and inside
        optimize() hold no changed function (noopt): pinned code, so the same schema dict (by
        fingerprint) gives an equal schema. One object serves those tasks; execute() reads only
        its supported_table_args, and optimize_stub answers optimize() from the record or hands
        the real optimize() a schema of its own (made then, by the same pinned code)."""
        index = self.task
        if index is not None and index in self.noopt and len(args) == 1 and kwargs == {"dialect": None} and type(args[0]) is dict:
            key = fingerprint(args[0])
            if key is not None:
                hit = self.schema_cache.get(key)
                if hit is None:
                    hit = self.schema_cache[key] = (self.real_schema(*args, **kwargs), args[0], key)
                self.fast_stats[3] += 1
                self.schema_given = (index, hit[0], args[0])
                return hit[0]
        return self.real_schema(*args, **kwargs)

    def optimize_stub(self, *args, **kwargs):
        """optimize() as execute() calls it in a task: the pinned optimized expression when no
        line that ran before or inside optimize() for this task on the pinned commit is in a
        changed function (memo.prepare) and the input expression is the pinned one. The result
        is checked like the computed one (compare(): the expression fingerprint)."""
        import memo

        index = self.task
        given_schema, self.schema_given = getattr(self, "schema_given", None), None
        if index is not None and index in self.noopt and len(args) == 2 and kwargs == {"leave_tables_isolated": True, "dialect": None}:
            rec = self.memo[0].index["queries"].get(index)
            o = rec.get("opt") if rec is not None else None
            if o is not None and fingerprint(args[0]) == o["in"]:
                quiet = index in self.nomid and rec.get("mtext") is not None
                try:
                    blob = self.memo[0].raw(o["blob"])
                    value = memo.load_expr_checked(blob) if quiet else memo.load_expr(blob)
                except Exception:  # noqa: BLE001  (no record: optimize() runs)
                    value = None
                if value is not None and o.get("kfp") is not None:
                    self.opt_stats[0] += 1
                    # (compare() takes this object when its identity-free fingerprint is kfp)
                    if quiet:
                        # execute() is the pinned one and hands the expression to one call before
                        # Plan(): expression.sql(pretty=True) for its logger, whose lines hold no
                        # changed function (memo.prepare). The proxy answers that call with the
                        # pinned text; no code sees the loaded expression before compare()
                        proxy = _Quiet(value, rec["mtext"])
                        self.opt_given = (index, value, o["kfp"], proxy)
                        return proxy
                    self.opt_given = (index, value, o["kfp"], None)
                    return value
        self.opt_stats[1] += 1
        if given_schema is not None and len(args) >= 2 and args[1] is given_schema[1]:
            # (the shared schema of schema_stub: the real optimize() gets one of its own)
            args = (args[0], self.real_schema(given_schema[2], dialect=None)) + tuple(args[2:])
        value = self.real_opt(*args, **kwargs)
        if index is not None and index in self.nomid and index == self.task and index in self.tasks:
            # optimize() ran: when its result is the pinned optimized expression (the fingerprint
            # that compare() would take), the logger call before Plan() gets the pinned text as above
            rec = self.memo[0].index["queries"].get(index) if self.memo is not None else None
            if rec is not None and rec.get("mtext") is not None and fingerprint(value) == self.tasks[index][0]:
                proxy = _Quiet(value, rec["mtext"])
                self.opt_given = (index, value, None, proxy)
                return proxy
        return value

    def remove(self):
        if self.globals is not None:
            self.globals["Plan"], self.globals["PythonExecutor"] = self.real
            if getattr(self, "real_opt", None) is not None:
                self.globals["optimize"] = self.real_opt
            if getattr(self, "real_schema", None) is not None:
                self.globals["ensure_schema"] = self.real_schema
            self.cls.subtestHelper = self.helper
            self.globals = None

    def stop(self, why):
        """Ends the check without a verdict: the tasks that are left do not run."""
        self.over = True
        if self.why is None:
            self.why = why
        # (the tasks that are left are not even handed over: see pool.LazyFuture)
        from reduce import STATE

        STATE.keep_tasks = frozenset()

    def affordable(self, index, whole):
        """Counts the build-time cost of running this task to its end; False (and the check is
        over) when the budget does not hold it."""
        early, late = self.cpu.get(index, (0.0, 0.0))
        cost = late + (early if whole else 0.0)
        if cost > self.budget - self.spent:
            self.stop("budget")
            return False
        self.spent += cost
        self.expected += late
        return True

    def compare(self, index, args, kwargs, plan_input):
        """None when the task's input at this point equals the pinned one, else what differs."""
        want = self.tasks[index]
        if args or list(kwargs) != ["tables"]:
            return "call"
        given, self.opt_given = getattr(self, "opt_given", None), None
        if given is not None and given[0] == index and given[3] is not None and given[3] is plan_input.expression:
            # the proxy of optimize_stub: its expression was loaded from the record with checked
            # class names (memo.load_expr_checked), the same graph as at build time, where its
            # fingerprint is kfp; no code saw it unless the proxy says so
            proxy = plan_input.expression
            plan_input.expression = proxy._expr
            if proxy._seen:
                import memo

                if given[2] is None:
                    if fingerprint(plan_input.expression) != want[0]:
                        return "expression"
                elif memo._key_fp(plan_input.expression) != given[2]:
                    return "expression"
            else:
                self.fast_stats[2] += 1
        elif given is not None and given[0] == index and given[1] is plan_input.expression:
            # the pinned optimized expression from the record (optimize_stub): its content must be
            # the recorded one (its fingerprint with string identity differs by construction)
            import memo

            if memo._key_fp(plan_input.expression) != given[2]:
                return "expression"
        elif fingerprint(plan_input.expression) != want[0]:
            # the pinned optimized expression from the memo's record (optimize_stub) differs from
            # the computed one in string identity only (memo.load_expr): its content must be the
            # recorded one
            import memo

            if not (given is not None and given[0] == index and given[1] is plan_input.expression
                    and memo._key_fp(plan_input.expression) == given[2]):
                return "expression"
        fixture = getattr(self.module, "_tables", None)
        if isinstance(fixture, dict):
            for name, table in dict.items(fixture):
                if name not in self.rows0 and name not in self.rows_seen:
                    t0 = time.process_time()
                    try:
                        self.rows_seen[name] = hash(tuple(table.rows))
                    except Exception:  # noqa: BLE001
                        return "tables"
                    self.rows_cpu += time.process_time() - t0
        if fingerprint(kwargs["tables"], table_leaves(fixture)) != want[1]:
            return "tables"
        return None

    def at_executor(self, args, kwargs, plan_input):
        plan, executor = self.real
        index = self.task
        if not isinstance(plan_input, _PlanInput):
            # execute() made its plan in another way than the pinned one (or this is not a
            # task): nothing to compare, the real executor runs
            if index is not None:
                if not self.affordable(index, False):
                    self.seq.append("o")
                    raise _Same()
                self.ran.append((index, "call"))
                self.seq.append("d")
                self.task = None
            return executor(*args, **kwargs).execute(plan_input)
        what = self.compare(index, args, kwargs, plan_input)
        if what is None and self.memo_task:
            return self.memo_execute(index, args, kwargs, plan_input)
        if what is None:
            self.same.append(index)
            self.seq.append("s")
            raise _Same()
        # another input than on the pinned commit: the task goes on to its end, when the
        # budget allows
        if not self.affordable(index, False):
            self.seq.append("o")
            raise _Same()
        self.ran.append((index, what))
        self.seq.append("d")
        self.task = None
        return executor(*args, **kwargs).execute(plan(plan_input.expression))

    def spend(self, cost):
        if cost > self.budget - self.spent:
            self.stop("budget")
            return False
        self.spent += cost
        return True

    def memo_execute(self, index, args, kwargs, plan_input):
        """A "later" task with the pinned input: the patched planner and steps, with the pinned
        outputs of the steps that run no changed line (memo.py)."""
        import memo

        plan, executor = self.real
        store, dirty = self.memo
        rec = store.index["queries"][index]
        self.task = None
        root = self.pinned_root(index, rec, store, dirty)
        if root is not None:
            self.memo_used = True
            self.ran.append((index, "memo"))
            self.seq.append("m")
            return root
        try:
            tables = getattr(self.module, "_tables", None)
            guard = set()
            for t in (dict.values(tables) if isinstance(tables, dict) else ()):
                guard.add(id(t))
                guard.add(id(getattr(t, "rows", None)))
            mods = (sys.modules["sqlglot.planner"],) + tuple(getattr(sys.modules["sqlglot.executor.table"], n) for n in ("Table", "RowReader", "RangeReader"))
            rec["_stats"] = [0, 0, 0]
            value = memo.execute(executor(*args, **kwargs), plan(plan_input.expression), index, rec, dirty, store, self.spend, guard, mods)
            for k in range(3):
                self.memo_stats[k] += rec["_stats"][k]
            self.memo_used = True
            self.ran.append((index, "memo"))
            self.seq.append("m")
            if memo._DBG:
                sys.stderr.write("memo dbg %r\n" % (sorted(memo._DBG_COUNTS.items()),))
            return value
        except memo.Fallback as e:
            if memo._DBG:
                sys.stderr.write("memo dbg fallback %d %s\n" % (index, e))
            if self.over:
                raise _Same()
            if os.environ.get("BR_DEBUG"):
                sys.stderr.write("memo task %d: %s\n" % (index, e))
            self.memo_stats[3] += 1
        except Exception as e:  # noqa: BLE001  (an error in a step that runs: no verdict)
            self.stop("memo task %d: %r" % (index, e))
            raise _Same()
        # as before the memo: the task runs on to its end
        if not self.spend(self.cpu.get(index, (0.0, 0.0))[1]):
            raise _Same()
        self.ran.append((index, "memo-fallback"))
        self.seq.append("f")
        return executor(*args, **kwargs).execute(plan(plan_input.expression))

    def pinned_root(self, index, rec, store, dirty):
        """The root's table from the record, without a plan, when nothing of the task's later work
        differs from the pinned run: no step of the query is dirty, its plan is built by pinned
        code from the pinned expression (no "k:" mark, memo.prepare), the frame is pinned (the task
        is not in memo_full), and the root's output is stored and may be restored (not an alias
        step, not a content-only match). Else None (memo.execute runs)."""
        import memo

        try:
            n = len(rec["keys"])
            if n == 0 or rec.get("planl") is not True or ("k:%d" % index) in dirty or rec.get("rname") is None:
                return None
            if index in self.memo_full or any(("%d:%d" % (index, i)) in dirty for i in range(n)):
                return None
            i = n - 1  # (the root is the last step of the canonical post order)
            span = rec["store"].get(i)
            if span is None or i in set(rec.get("alias") or ()) or (len(rec["outs"][i]) == 17 and rec["outs"][i][:1] == b"c"):
                return None
            if rec.get("order") is None or len(rec["order"]) != n:
                return None
            table_mod = sys.modules["sqlglot.executor.table"]
            out = memo.restore(store.blob(span), table_mod.Table, table_mod.RowReader, table_mod.RangeReader)
            value = out.tables[rec["rname"]]
        except Exception:  # noqa: BLE001  (memo.execute runs)
            return None
        self.fast_stats[1] += 1
        st = rec.setdefault("_stats", [0, 0, 0])
        self.memo_stats[0] += n - 1
        self.memo_stats[2] += 1
        return value

    def rows_start(self):
        """At the start of the check (the fixture handed to the initializer): the hashes of the
        built tables; False when one differs from the pinned hash."""
        t0 = time.process_time()
        h = row_hashes(getattr(self.module, "_tables", None))
        self.rows_cpu += time.process_time() - t0
        if h is None:
            return False
        self.rows0 = h
        return all(self.rowh is None or self.rowh.get(k) == v for k, v in h.items())

    def rows_ok(self):
        """At the end: every built table of the fixture has the pinned row content (else the
        tasks that read it after a task changed it in place do not count as proven)."""
        t0 = time.process_time()
        h = row_hashes(getattr(self.module, "_tables", None))
        self.rows_cpu += time.process_time() - t0
        if h is None:
            return False
        for k, v in h.items():
            want = self.rowh.get(k) if self.rowh is not None else None
            if want is None:
                want = self.rows0.get(k, self.rows_seen.get(k))
            if want is None or want != v:
                return False
            if k in self.rows0 and self.rows0[k] != v:
                return False
        return True

    def on_result(self, index, table):
        """The test received the rows of a task that ran to its end. True: the comparison
        with DuckDB's answer is left out (the rows are the pinned rows and the comparison
        passes, or the check is over and has no verdict anyway)."""
        want = self.tasks.get(index)
        if want is not None and not self.over and rows_fingerprint(table) == want[3]:
            self.proved.append(index)
        else:
            self.stop("rows of task %d" % index)
        return True

    def sql_text_differs(self, index):
        """The DuckDB side: the same SQL text gives the same rows. True when the patched code
        makes another text for this query than the pinned code (or raises)."""
        try:
            sql = getattr(self.cls, self.record["sqls"])[index][1]
            if index in self.notext and self.memo is not None and type(sql) is str:
                # no line that transpile() runs for this query on the pinned commit is in a
                # function that the commit changes (memo.prepare), and the string is the pinned
                # one: the pinned text
                rec = self.memo[0].index["queries"].get(index)
                import sqlglot

                if (rec is not None and rec.get("text") is True and self.module.transpile is getattr(sqlglot, "transpile", None)
                        and hashlib.blake2b(sql.encode("utf-8", "surrogatepass"), digest_size=16).digest() == rec.get("sqlh")):
                    self.fast_stats[0] += 1
                    return False
            text = self.module.transpile(sql, write="duckdb")[0]
            return hashlib.blake2b(text.encode(), digest_size=16).digest() != self.tasks[index][2]
        except Exception:  # noqa: BLE001
            return True

    # -- pool.LazyFuture calls this in place of the task ---------------------------------

    # Mid-task form of the "slow" rule of run_task: a task that the check lets run cannot run
    # past clock + 2 * (expected, its own later part included) + slack CPU seconds. Past that
    # point the rule would end the check when the next task starts anyway; a task that does not
    # end (patched code in a loop) used to run on until the CPU guard of the check (up to
    # STAGE_GAIN times the seconds that it saves). The timer is ITIMER_PROF, whose SIGPROF ends
    # the child as the CPU guard does (the parent: "stage check stopped", no verdict, the test
    # is selected as without the check). Between tasks the guard's own limit holds again.
    _guard_at = None  # process CPU time at which the guard of started() fires (inf: none)
    _armed = False

    def _arm(self, index):
        if self.slack is None or os.environ.get("BR_NO_TASK_TIMER") == "1":
            return
        import signal

        now = time.process_time()
        if self._guard_at is None:
            left = signal.getitimer(signal.ITIMER_PROF)[0]
            self._guard_at = now + left if left > 0 else float("inf")
        limit = self.clock + 2.0 * (self.expected + self.cpu.get(index, (0.0, 0.0))[1]) + self.slack
        at = min(limit, self._guard_at)
        if at == float("inf"):
            return
        signal.setitimer(signal.ITIMER_PROF, max(at - now, 0.001))
        self._armed = True

    def _disarm(self):
        if not self._armed:
            return
        import signal

        self._armed = False
        if self._guard_at == float("inf"):
            signal.setitimer(signal.ITIMER_PROF, 0)
        else:
            signal.setitimer(signal.ITIMER_PROF, max(self._guard_at - time.process_time(), 0.001))

    def run_task(self, index, fn, args, kwargs):
        try:
            return self._run_task(index, fn, args, kwargs)
        finally:
            self._disarm()

    def _run_task(self, index, fn, args, kwargs):
        want = self.tasks.get(index)
        if want is None:
            # a query that the pinned test does not execute: the task is short and runs as it is
            return fn(*args, **kwargs)
        if self.over:
            return None  # the test skips a task without a result
        # A check that costs far more CPU than the pinned tasks did (the patched code is slow)
        # ends here: it would cost more than it can save.
        now = time.process_time()
        if self.clock is None:
            self.clock = now
        elif self.slack is not None and now - self.clock > 2.0 * self.expected + self.slack:
            self.seq.append("t")
            self.stop("slow")
            return None
        self.expected += self.cpu[index][0] + 0.01
        if self.sql_text_differs(index):
            self.seq.append("q")
            self.stop("sql text of task %d" % index)
            return None
        self._arm(index)
        if index in self.later and self.memo is not None and index in self.memo[0].index["queries"] and index not in self.memo_full:
            if not self.spend(self.cpu.get(index, (0.0, 0.0))[0]):
                self.seq.append("o")
                return None
            self.task = index
            self.memo_task = True
            try:
                value = fn(*args, **kwargs)
            except _Same:
                return None
            finally:
                reached = self.task is None
                self.task = None
                self.memo_task = False
            if not reached:
                self.ran.append((index, "path"))
                self.seq.append("p")
            if value is None and not self.over and any(i == index for i, _ in self.ran):
                self.proved.append(index)
            return value
        if index in self.later:
            if not self.affordable(index, True):
                self.seq.append("o")
                return None
            self.ran.append((index, "later"))
            self.seq.append("l")
            value = fn(*args, **kwargs)
            if value is None and not self.over:
                # the test compares only a result that is not None (_mp_execute): nothing to fail
                self.proved.append(index)
            return value
        self.task = index
        try:
            value = fn(*args, **kwargs)
        except _Same:
            return None
        finally:
            reached = self.task is None
            self.task = None
        if not reached:
            # the task ended without asking for the executor: its value is what the test gets
            self.ran.append((index, "path"))
            self.seq.append("p")
        if value is None and not self.over and any(i == index for i, _ in self.ran):
            # the test compares only a result that is not None (_mp_execute): nothing to fail
            self.proved.append(index)
        return value

    def probe(self, case):
        """Before the test runs: its first task by hand, as the test and its pool would run it
        (parse the query, hand it over through pickle, the worker's initializer, the task with
        the hook, the result back through pickle). Most commits that break the test break
        this task, and then the check is over before the test has parsed its other queries.
        Returns False when the check is over. Nothing of the probe counts for the verdict:
        the test runs afterwards with every task, this one too."""
        module = self.module
        index = min((i for i in self.tasks if self.active is None or i in self.active), default=None)
        if index is None:
            return True
        try:
            meta, sql = getattr(self.cls, self.record["sqls"])[index][:2]
            expression = module.parse_one(sql)
            fn, args, kwargs = pickle.loads(pickle.dumps((module.mp_execute, (expression, meta), {})))
            module.initializer(getattr(module, self.record["schema"]), getattr(case, self.record["tables"]))
            if not self.rows_start():
                self.stop("fixture rows")
                return False
            value = self.run_task(index, fn, args, kwargs)
            if value is not None:
                self.on_result(index, pickle.loads(pickle.dumps(value)))
        except Exception:  # noqa: BLE001  (the test will very likely fail in the same way)
            self.stop("probe: error in task %d" % index)
        alive = not self.over
        if alive:
            del self.same[:], self.ran[:], self.proved[:], self.seq[:]
            self.spent = self.expected = 0.0
            self.clock = None
        return alive

    def complete(self):
        """True when every task that ran to its end gave the pinned rows. (A task that ran
        and gave the test nothing to compare, where the pinned task gives rows, is not proven.)"""
        return not self.over and sorted(self.proved) == sorted(i for i, _ in self.ran)


# --- module state around the check ---------------------------------------------------
# The later work also reads module state (module globals, class attributes). The pinned test
# writes a little of it itself: the dialect registry fills when a dialect is first looked up,
# the generator's dispatch table of a generator class is made at its first use, and pickle
# stores the slot names of a class that it writes. build_stages.py records those writes
# ("effect"). After a check, every key of the state that differs from before must be one of
# them, with the pinned value; in a table (dict), every new or changed entry must be an entry
# that the pinned run adds, with the pinned value.


def code_alias(pairs):
    """{hash of new code: hash of old code} for statefp, from (old code, new code) pairs of
    functions whose body the commit changes. A function then counts as the pinned function:
    which of its lines changed is what late_tasks looks at."""
    import statefp

    f = statefp.Fingerprinter()
    alias = {}
    stack = list(pairs)
    while stack:
        old, new = stack.pop()
        if f.code(new) != f.code(old):
            alias[f.code(new)] = f.code(old)
        oc = [k for k in old.co_consts if isinstance(k, types.CodeType)]
        nc = [k for k in new.co_consts if isinstance(k, types.CodeType)]
        if len(oc) == len(nc):
            stack.extend(zip(oc, nc))
    return alias


def _entry_name(x):
    """The name of a table entry's key: a class by the plain name of classes (not through a
    __repr__ that patched code may define), anything else by its repr."""
    return type.__repr__(x) if isinstance(x, type) else repr(x)


def state_before(effect, alias=None, every_table=False):
    """The state now, and for the tables that the pinned test fills (for every table when
    build_stages.py asks, which finds out which ones those are) the fingerprint of every entry
    under the name of its key (not the values: they can change in place)."""
    import statefp

    state, f = statefp.snapshot(alias)
    keys = list(f.objects) if every_table else [k for k, e in effect.items() if e[0] == "entries"]
    tables = {k: {_entry_name(x): f.fp(v) for x, v in f.objects[k].items()} for k in keys if type(f.objects.get(k)) is dict}
    return state, tables


def hold_state(effect, alias=None):
    """Keeps the module state of this moment without computing its snapshot: a forked copy
    of this process waits, and computes state_before() only when held_state() asks for it.
    Most checks end without a verdict (a task with other rows, an error) and never ask: they
    save the snapshot. Returns the handle for held_state() / release()."""
    ask_r, ask_w = os.pipe()
    ans_r, ans_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(ask_w)
            os.close(ans_r)
            if os.read(ask_r, 1):
                view = memoryview(marshal.dumps(state_before(effect, alias)))
                while view:
                    view = view[os.write(ans_w, view):]
        except BaseException as e:  # noqa: BLE001  (no answer: the check gives no verdict)
            if os.environ.get("BR_DEBUG"):
                sys.stderr.write("stage check: no snapshot of the state before: %r\n" % (e,))
        finally:
            os._exit(0)
    os.close(ask_r)
    os.close(ans_w)
    return [pid, ask_w, ans_r]


def release(holder):
    """Ends the waiting copy (also after held_state)."""
    if holder and holder[0]:
        pid, ask_w, ans_r = holder
        holder[0] = 0
        for fd in (ask_w, ans_r):
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.waitpid(pid, 0)
        except OSError:
            pass


def held_state(holder):
    """The state_before() of the moment of hold_state(), or None when the copy gave none."""
    chunks = []
    try:
        os.write(holder[1], b"s")
        while True:
            chunk = os.read(holder[2], 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return marshal.loads(b"".join(chunks))
    except (OSError, ValueError, EOFError, TypeError):
        return None
    finally:
        release(holder)


def state_effect(before, alias=None):
    """What changed since state_before(): {key: ("value", fingerprint) or ("entries", {name of
    the entry's key: fingerprint}) or ("gone",)}. A table (dict) that only gained or changed
    entries gives "entries" when its entries were kept."""
    import statefp

    state0, tables0 = before
    state1, f = statefp.snapshot(alias)
    out = {}
    for k in statefp.diff(state0, state1):
        if k not in state1:
            out[k] = ("gone",)
            continue
        old, new = tables0.get(k), f.objects.get(k)
        now = {_entry_name(x): f.fp(v) for x, v in new.items()} if old is not None and type(new) is dict else None
        if now is not None and all(x in now for x in old):
            out[k] = ("entries", {x: h for x, h in now.items() if x not in old or old[x] != h})
        else:
            out[k] = ("value", state1[k])
    return out


def state_ok(effect, before, alias=None):
    """(True, []) when the state changed only as the pinned test changes it."""
    bad = []
    for k, got in state_effect(before, alias).items():
        want = effect.get(k)
        if want is None or want[0] != got[0]:
            bad.append(k)
        elif got[0] == "value" and got[1] != want[1]:
            bad.append(k)
        elif got[0] == "entries" and any(want[1].get(x) != v for x, v in got[1].items()):
            bad.append(k)
    return not bad, bad
