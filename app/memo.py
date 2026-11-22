"""Step memo for the stage check of test_execute_tpcds (see stages.py).

A task whose later work runs changed code ("later" task) used to run to its end. With the
memo it runs as follows, inside StageCheck.at_executor, when its optimized expression and
tables equal the pinned ones: the PATCHED planner builds the plan and a copy of
PythonExecutor._execute runs it with the PATCHED step methods, except that a step may take
the output that the pinned run recorded (build_memo.py) instead of running, when
  - its definition (type, fields, expressions with everything they reach, the steps that it
    depends on, in a canonical order of the plan) is byte-for-byte the pinned one, as a
    chained fingerprint of the plan up to this step;
  - every step that it depends on gave the pinned output (taken from the record, or run and
    hashed: equal content, equal sharing of tables and row lists);
  - none of the sqlglot lines that the step ran on the pinned commit (its dynamic extent,
    in a cold and in a warm run) is in a function that the commit changes.
Every other step runs; when a step runs whose output is needed, its inputs are real
objects with the pinned content. The root's table is what the task returns: the test then
compares its rows with the pinned rows (StageCheck.on_result) as before.

Anything unusual (no record, a plan with a step that two steps depend on, a tie in the
canonical order, a table that does not look like a Table, an error) raises Fallback, and the
check treats the task as before (it runs to its end)."""
from __future__ import annotations

import ast
import difflib
import hashlib
import io
import marshal
import os
import pickle
import re
import zlib

import stages

MEMO_VERSION = 3
STEP_METHODS = ("scan", "aggregate", "join", "sort", "set_operation")
# functions that the copy of _execute stands for, or that set up the executor's state: a
# commit that changes one of them gets no memo
FRAME = {"sqlglot/executor/python.py": ("PythonExecutor.__init__", "PythonExecutor.execute",
                                        "PythonExecutor._execute", "PythonExecutor.context")}


HOOK = ("sqlglot/executor/__init__.py", "execute")


class Fallback(Exception):
    pass


_DBG = os.environ.get("BR_MEMO_DBG") == "1"
_DBG_COUNTS = {}


# --- the plan -------------------------------------------------------------------------

class _KeyWriter(stages._Writer):
    """stages._Writer with every string written by its value: the plan from an optimized
    expression loaded from the record (memo.load_expr) holds the same strings as the pinned one,
    but not the same string objects (the arg keys of a loaded expression are not the interned
    names of sqlglot's code), so a key must not depend on string identity."""

    def persistent_id(self, obj):
        if type(obj) is str:
            return obj.encode("utf-8", "surrogatepass")
        return None


class _TieWriter(_KeyWriter):
    """_KeyWriter without the parent pointer of sqlglot objects: the standalone fingerprint of a
    step that breaks ties in canon_order (writing every parent would write the whole query once
    per step). Only canon_order uses it; the chained keys still write everything."""

    def reducer_override(self, obj):
        r = super().reducer_override(obj)
        if r is NotImplemented or type(r) is not tuple or len(r) != 3:
            return r
        info = self.info.get(type(obj))
        if info is None or info[0] != 0 or "parent" not in info[2]:
            return r
        state = list(r[2])
        state[info[2].index("parent")] = None
        return (r[0], r[1], tuple(state))


def _key_fp(obj, writer=_KeyWriter):
    buf = io.BytesIO()
    w = writer(buf, None)
    try:
        w.dump(obj)
    except Exception:  # noqa: BLE001
        return None
    if w.opaque:
        return None
    return hashlib.blake2b(buf.getvalue(), digest_size=16).digest()


def _merkle(plan):
    """{id(step): 16 bytes}: a hash of the step's own definition (standalone fingerprint of its
    state) and of the hashes of its dependencies (sorted). Only breaks ties of canon_order: the
    chained keys still check every definition."""
    out = {}
    stack = [(plan.root, False)]
    while stack:
        s, done = stack.pop()
        if id(s) in out:
            continue
        if not done:
            stack.append((s, True))
            for d in s.dependencies:
                if id(d) not in out:
                    stack.append((d, False))
            continue
        state = {k: v for k, v in vars(s).items() if k not in ("dependencies", "dependents")}
        fp = _key_fp(("%s.%s" % (type(s).__module__, type(s).__qualname__), state), _TieWriter)
        if fp is None:
            return None
        h = hashlib.blake2b(fp, digest_size=16)
        for x in sorted(out[id(d)] for d in s.dependencies):
            h.update(x)
        out[id(s)] = h.digest()
    return out


def canon_order(plan):
    """(steps in post order from the root, deps by (type, name), ties broken by the merkle hash
    of the dependency's subtree; {id(step): index}) or None for a plan that this module does not
    handle (two dependencies with equal type, name and subtree hash)."""
    order, idx = [], {}
    merkle = [None]
    stack = [(plan.root, False)]
    seen = set()
    while stack:
        s, done = stack.pop()
        if done:
            idx[id(s)] = len(order)
            order.append(s)
            continue
        if id(s) in seen:
            continue  # a step that two steps depend on (a CTE): its first place counts
        seen.add(id(s))
        deps = list(s.dependencies)
        names = [(type(d).__name__, str(d.name)) for d in deps]
        if len(set(names)) != len(names):
            if merkle[0] is None:
                merkle[0] = _merkle(plan)
                if merkle[0] is None:
                    return None
            names = [n + (merkle[0][id(d)],) for n, d in zip(names, deps)]
            if len(set(names)) != len(names):
                return None
        deps = [d for _, d in sorted(zip(names, deps), key=lambda x: x[0])]
        stack.append((s, True))
        for d in reversed(deps):
            stack.append((d, False))
    return order, idx


def plan_keys(order, idx):
    """Chained fingerprints: key i covers steps 0..i of the canonical order (16 bytes each)."""
    buf = io.BytesIO()
    w = _KeyWriter(buf, None)
    h = hashlib.blake2b(digest_size=16)
    keys = []
    try:
        for s in order:
            state = {k: v for k, v in vars(s).items() if k not in ("dependencies", "dependents")}
            w.dump(("%s.%s" % (type(s).__module__, type(s).__qualname__), sorted(idx[id(d)] for d in s.dependencies), state))
            if w.opaque:
                return None
            h.update(buf.getbuffer())
            buf.seek(0)
            buf.truncate()
            keys.append(h.copy().digest())
    except Exception:  # noqa: BLE001
        return None
    return keys


# --- step outputs ---------------------------------------------------------------------

_TABLE_KEYS = {"columns", "column_range", "reader", "rows", "range_reader"}


def out_state(ctx, T, RR, GR):
    """(names, tables, row lists) of a step's output context, or None."""
    names, tabs, rows_lists, tid, rid = [], [], [], {}, {}
    tables = getattr(ctx, "tables", None)
    if type(tables) is not dict:
        return None
    for name, t in tables.items():
        i = tid.get(id(t))
        if i is None:
            if type(t) is not T or set(t.__dict__) != _TABLE_KEYS:
                return None
            rd, rr = t.reader, t.range_reader
            if type(rd) is not RR or set(rd.__dict__) != {"columns", "row"} or type(rd.columns) is not dict:
                return None
            if type(rr) is not GR or set(rr.__dict__) != {"table", "range"} or rr.table is not t or type(rr.range) is not range:
                return None
            if type(t.rows) is not list or type(t.columns) is not tuple or type(rd.row) is not tuple:
                return None
            if t.column_range is not None and type(t.column_range) is not range:
                return None
            r = rid.get(id(t.rows))
            if r is None:
                r = rid[id(t.rows)] = len(rows_lists)
                rows_lists.append(t.rows)
            i = tid[id(t)] = len(tabs)
            tabs.append((t.columns, t.column_range, list(rd.columns.items()), rd.row, rr.range, r))
        names.append((name, i))
    return names, tabs, rows_lists


def state_hash(state):
    """Content hash without object identity (fast mode: no memo), plus the explicit sharing."""
    buf = io.BytesIO()
    p = pickle.Pickler(buf, 4)
    p.fast = True
    try:
        p.dump(state)
    except Exception:  # noqa: BLE001
        return None
    return hashlib.blake2b(buf.getbuffer(), digest_size=16).digest()


def content_hash(state):
    """state_hash without the readers' current row and range (what a reader leaves behind)."""
    names, tabs, rows_lists = state
    return state_hash((names, [t[:3] + (None, None) + t[5:] for t in tabs], rows_lists))


def out_matches(st, pinned):
    """The output state st is the pinned output: equal state hash, or, for a step whose readers
    were left at another row in the two build runs (b"c" + content hash), equal content."""
    if len(pinned) == 17 and pinned[:1] == b"c":
        return content_hash(st) == pinned[1:]
    return state_hash(st) == pinned


# --- the pinned optimized expression ----------------------------------------------------

def _new(cls):
    return object.__new__(cls)


def _set(obj, state):
    slots, d = state
    for n, v in slots:
        object.__setattr__(obj, n, v)
    if d is not None:
        object.__getattribute__(obj, "__dict__").update(d)


class _ExprPickler(pickle.Pickler):
    """Writes sqlglot objects from their slots and __dict__ (never through sqlglot's own
    __reduce__, which is code that a commit may change); loading runs no sqlglot code but
    the class lookups. Objects of sqlglot classes that subclass a builtin are refused."""

    def __init__(self, f):
        super().__init__(f, 4)
        self.info = {}

    def reducer_override(self, obj):
        t = type(obj)
        if not (t.__module__ or "").startswith("sqlglot") or isinstance(obj, type):
            return NotImplemented
        info = self.info.get(t)
        if info is None:
            d = stages._describe(t)
            if d[0] == 2:
                info = False  # an enum: pickle writes it by value
            elif d[0] != 0 or d[4] is not None:
                raise TypeError("cannot write %r" % (t,))
            else:
                info = (d[2], d[3])
            self.info[t] = info
        if info is False:
            return NotImplemented
        slots = [(n, getattr(obj, n)) for n in info[0] if hasattr(obj, n)]
        return (_new, (t,), (slots, dict(obj.__dict__) if info[1] else None), None, None, _set)


def dump_expr(e):
    buf = io.BytesIO()
    _ExprPickler(buf).dump(e)
    return zlib.compress(buf.getvalue(), 6)


def load_expr(blob):
    return pickle.loads(zlib.decompress(blob))


class _NameCheckedUnpickler(pickle.Unpickler):
    """Loads a dump_expr blob and refuses a global whose object does not carry the name that the
    blob gives it (a module attribute that was replaced at run time): the loaded graph then has
    exactly the classes, by module and qualified name, that the recorded graph has."""

    def find_class(self, module, name):
        obj = super().find_class(module, name)
        if isinstance(obj, type) or callable(obj):
            if getattr(obj, "__module__", None) != module or getattr(obj, "__qualname__", None) != name:
                raise pickle.UnpicklingError("%s.%s is another object" % (module, name))
        return obj


def load_expr_checked(blob):
    return _NameCheckedUnpickler(io.BytesIO(zlib.decompress(blob))).load()


class Out:
    """Stands for a step's output context: the copy of _execute reads only .tables."""
    __slots__ = ("tables",)

    def __init__(self, tables):
        self.tables = tables


def restore(state, T, RR, GR):
    names, tabs, rows_lists = state
    made = []
    for columns, column_range, rcols, row, rng, r in tabs:
        t = T.__new__(T)
        rd = RR.__new__(RR)
        rd.columns = dict(rcols)
        rd.row = row
        gr = GR.__new__(GR)
        t.columns = columns
        t.column_range = column_range
        t.reader = rd
        t.rows = rows_lists[r]
        gr.table = t
        gr.range = rng
        t.range_reader = gr
        made.append(t)
    return Out({name: made[i] for name, i in names})


# --- changed functions ------------------------------------------------------------------

def _functions(tree):
    """[(start, end, qualname)] of every def and lambda (start: the first decorator line)."""
    out = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                q = prefix + child.name
                out.append((start, child.end_lineno, q))
                walk(child, q + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
            elif isinstance(child, ast.Lambda):
                q = prefix + "<lambda:%d>" % child.lineno
                out.append((child.lineno, child.end_lineno, q))
                walk(child, prefix)
            else:
                walk(child, prefix)
    walk(tree, "")
    return out


def _innermost(funcs, lo, hi):
    best = None
    for s, e, q in funcs:
        if s <= lo and hi <= e and (best is None or e - s < best[1] - best[0]):
            best = (s, e, q)
    return best


_IDENTITY = re.compile(r"\bis\b(?!\s+(?:not\s+)?(?:None|True|False)\b)|\bid\s*\(|\bgetrefcount\b|\bweakref\b")


def _blanked(tree):
    """ast.dump of the module with every function and lambda body blanked (decorators,
    arguments with their defaults, annotations and everything outside a body stay)."""
    for node in list(ast.walk(tree)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = [ast.Pass()]
        elif isinstance(node, ast.Lambda):
            node.body = ast.Constant(None)
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def changed_functions(old_text, new_text, firsts=None):
    """{qualname: (start, end) in the old text} of the functions that hold the edits, or None
    when an edit is outside a function (the two modules differ once every function and lambda
    body is blanked: module-level code, a decorator, a default, a dict key next to a lambda on
    the same line), the functions differ by name, or an added line compares by identity."""
    try:
        old_tree, new_tree = ast.parse(old_text), ast.parse(new_text)
        old_funcs = _functions(old_tree)
        new_funcs = _functions(new_tree)
        if _blanked(old_tree) != _blanked(new_tree):
            return None
    except (SyntaxError, ValueError, RecursionError):
        return None
    a, b = old_text.split("\n"), new_text.split("\n")
    out = {}
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        for line in b[j1:j2]:
            code = line.split("#", 1)[0]
            if _IDENTITY.search(code):
                return None
        # 1-based spans; an insertion or deletion counts its neighbour lines too
        olo, ohi = (i1 + 1, i2) if i2 > i1 else (i1, i1 + 1)
        nlo, nhi = (j1 + 1, j2) if j2 > j1 else (j1, j1 + 1)
        fo, fn = _innermost(old_funcs, olo, ohi), _innermost(new_funcs, nlo, nhi)
        if fo is None or fn is None or fo[2] != fn[2]:
            return None
        out[fo[2]] = (fo[0], fo[1])
        if firsts is not None:
            firsts[fo[2]] = min(firsts.get(fo[2], olo), olo)
    return out


def _stmt_start(text, qualname, line):
    """The first line of the statement of the body of module-level function qualname that holds
    line (a statement of the body runs once and after every statement before it, so the lines
    before it run as on the pinned commit), or None."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == qualname:
            for st in node.body:
                if st.lineno <= line <= st.end_lineno:
                    return st.lineno
            return None
    return None


def _call_stmt(text, qualname, callee):
    """The last line of the one statement of the body of module-level function qualname that
    calls the plain name callee; None when no statement or several statements call it."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, TypeError):
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == qualname:
            found = [st for st in node.body
                     if any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == callee for c in ast.walk(st))]
            if len(found) != 1:
                return None
            return found[0].end_lineno
    return None


# --- the record -------------------------------------------------------------------------

_INDEX = {}  # maps dir -> the loaded index (the maps are read-only in a run; a forked child inherits it)


class Store:
    def __init__(self, maps_dir):
        index = _INDEX.get(maps_dir)
        if index is None:
            with open(os.path.join(maps_dir, "memo.bin"), "rb") as f:
                index = _INDEX[maps_dir] = marshal.loads(zlib.decompress(f.read()))
        self.index = index
        self.path = os.path.join(maps_dir, "memo.dat")
        self.fd = None

    def ok(self):
        return self.index.get("probe") == stages.hash_probe() and self.index.get("version") == MEMO_VERSION

    def raw(self, span):
        if self.fd is None:
            self.fd = open(self.path, "rb")
        self.fd.seek(span[0])
        return self.fd.read(span[1])

    def blob(self, span):
        return pickle.loads(zlib.decompress(self.raw(span)))


def prepare(store, files, old_texts, record=None):
    """{"d": dirty step ids ("q:i"), "full": tasks that run to their end without the memo,
    "noopt": tasks whose optimize() may give the pinned optimized expression}, or None (no
    memo). files: {repo path: new text}; old_texts: {repo path: pinned text}; record: the stage
    map's record of the test (its "late" lines per task).
    The lines of every changed function (its whole span in the pinned text) count as changed:
      - a line in the extent of step i of task q (build_memo.py): step q:i is dirty;
      - a line in the frame of task q (executor construction, the executor's work outside the
        steps: PythonExecutor.__init__, Python().generator, Generator.__init__, ...): task q
        runs to its end;
      - a line that the stage map puts in the later work of task q but that no bucket of q
        holds in the memo's record (steps, frame, plan, post; or the early part before the plan,
        which runs live and whose output compare() checks): task q runs to its end;
      - a line that runs before or inside optimize() for task q: q keeps its optimize()."""
    changed = {}
    hook_from = None
    for path, new in files.items():
        old = old_texts.get(path)
        if new is None or old is None or not path.endswith(".py"):
            return None
        firsts = {}
        fns = changed_functions(old, new, firsts)
        if not fns:
            return None  # (no edit found in a file that the commit changes: not this module's case)
        for q in fns:
            if q in FRAME.get(path, ()):
                return None
        changed[path] = list(fns.items())
        if path == HOOK[0] and HOOK[1] in fns:
            hook_from = _stmt_start(old, HOOK[1], firsts[HOOK[1]])
    idx = store.index
    if idx.get("tsets") is None or record is None:
        return None
    first = idx["first"]  # {path: set of lines} that run once per process
    cover = idx["cover"]  # {path: {line: set id}}
    sets = idx["sets"]
    tcov, tsets = idx["tcov"], idx["tsets"]  # {path: {line: set id}}; ("F"|"A"|"E", task) tokens
    late, lsets = record.get("late", {}), record.get("sets", [])
    dirty, full, early, replan = set(), set(), set(), set()
    textd, midd = set(), set()  # tasks whose DuckDB text / logger text may differ from the record's
    for path, spans in changed.items():
        f1 = first.get(path, ())
        by_line = cover.get(path, {})
        t_line = tcov.get(path, {})
        l_line = late.get(path, {})
        for name, (s, e) in spans:
            # execute() itself (the function that holds the stubs: the stage map counts all its
            # lines as later work of every task) runs live around the memo window; what it hands
            # to it is checked by compare(), what it returns by on_result()
            hook = (path, name) == HOOK
            for ln in range(s, e + 1):
                if ln in f1:
                    return None
                sid = by_line.get(ln)
                if sid is not None:
                    dirty.update(sets[sid])
                tid = t_line.get(ln)
                toks = tsets[tid] if tid is not None else ()
                seen = set()
                for kind, q in toks:
                    if kind == "T":
                        textd.add(q)  # (the stage check's transpile(), not a part of the task)
                        continue
                    if kind == "M" and not (hook and hook_from is not None and ln < hook_from):
                        midd.add(q)  # (a line of execute() before its first edited statement runs as pinned)
                    if kind == "F":
                        full.add(q)
                    elif kind == "P":
                        replan.add(q)
                    elif kind == "E" and not (hook and hook_from is not None and ln < hook_from):
                        # (a line of execute() before the statement with the first edit runs
                        # as on the pinned commit: it does not keep optimize())
                        early.add(q)
                    seen.add(q)  # (a line that the record places in any part of the task)
                lid = None if hook else l_line.get(ln)
                if lid is not None:
                    for q in lsets[lid]:
                        if q not in seen:
                            full.add(q)  # later work that the memo's record does not place
    noopt = {q for q, r in idx["queries"].items() if r.get("opt") is not None} - early
    # "k:q": a changed line runs while task q builds its plan: its steps are matched by their keys
    # (else the plan, built by the pinned planner code from the pinned expression, is the pinned
    # one: the keys are not computed)
    dirty.update("k:%d" % q for q in replan)
    # transpile(sql, write="duckdb") of the check (build_memo.text_lines): no changed function in
    # the lines that it runs on the pinned commit (cold and warm), the same SQL string: the pinned text
    notext = {q for q, r in idx["queries"].items() if r.get("text") is True and r.get("sqlh") is not None} - textd
    # execute()'s expression.sql(pretty=True) between optimize() and Plan(): the pinned text when
    # execute() itself is unchanged and no line of that call is in a changed function
    hook_changed = any(path == HOOK[0] and any(q == HOOK[1] for q, _ in spans) for path, spans in changed.items())
    hook_edit = hook_changed  # (execute() changed anywhere: StageCheck.schema_stub stays out)
    if hook_changed and hook_from is not None:
        # an edit of execute() after the statement that calls Plan(): the statements up to that
        # call run as pinned (the proxy of the stage check meets only pinned code until Plan())
        plan_at = _call_stmt(old_texts.get(HOOK[0]), HOOK[1], "Plan")
        if plan_at is not None and plan_at < hook_from:
            hook_changed = False
    # (with or without noopt: StageCheck.optimize_stub also wraps a computed result that is pinned)
    nomid = set() if hook_changed else ({q for q, r in idx["queries"].items() if r.get("mtext") is not None} - midd)
    return {"d": sorted(dirty), "full": sorted(full), "noopt": sorted(noopt), "notext": sorted(notext), "nomid": sorted(nomid),
            "hook": hook_changed, "hook_edit": hook_edit}


# --- the copy of PythonExecutor._execute --------------------------------------------------

def _objs(out):
    """ids of the tables and row lists of an output context."""
    s = set()
    for t in out.tables.values():
        s.add(id(t))
        s.add(id(getattr(t, "rows", None)))
    return s


def execute(executor, plan, query, rec, dirty, store, spend, guard, mods):
    """Runs the plan as PythonExecutor.execute does; returns the root table. rec: the pinned
    record of the query; dirty: step ids whose pinned lines hold a change; spend(cpu): budget
    (False: stop); guard: ids of the base tables and their row lists; mods: (planner, Table,
    RowReader, RangeReader).

    In-place changes and aliasing: a step may return an input table object (a Scan without
    projections and condition, a Join without joins) and may change its inputs in place (an
    Aggregate sorts its input rows and writes operand columns into them). So
      - a step that the pinned run recorded as returning an input object ("alias"), and a step
        whose two pinned runs left its readers at different rows (content-only match), is never
        restored from the record: when its output is needed it runs on its materialized inputs,
        so the live object structure is rebuilt;
      - every output that is live here carries the set of shared steps (several dependents)
        whose live objects it reaches through such aliases; after every step that runs, each of
        those shared outputs that it could reach must still have its recorded content, else
        Fallback;
      - a step that runs and returns an object that reaches a shared output counts as not
        pinned: the steps that depend on it run too (and are checked as above)."""
    planner, T, RR, GR = mods
    co = canon_order(plan)
    if co is None:
        raise Fallback("plan shape")
    order, idx = co
    pkeys, pouts, pcpu, pstore = rec["keys"], rec["outs"], rec["cpu"], rec["store"]
    if ("k:%d" % query) in dirty or rec.get("planl") is not True:
        keys = plan_keys(order, idx)
        if keys is None:
            raise Fallback("plan key")
    else:
        if [type(s).__name__ for s in order] != rec["order"]:
            raise Fallback("plan order")
        keys = pkeys
    alias_steps = rec.get("alias")
    if alias_steps is None or rec.get("chash") is None:
        raise Fallback("no alias record")
    alias_steps = set(alias_steps)
    n_pin = len(pkeys)
    same_def = [i < n_pin and keys[i] == pkeys[i] for i in range(len(order))]
    info = {}  # id(step) -> [pinned (bool), Out or None]
    shared = {}  # id(step) -> content hash of a step that several steps depend on
    reach = {}  # id(step) -> ids of the shared steps whose live objects its live output reaches
    stats = rec.setdefault("_stats", [0, 0, 0])

    def run_step(node, context):
        if isinstance(node, planner.Scan):
            out = executor.scan(node, context)
        elif isinstance(node, planner.Aggregate):
            out = executor.aggregate(node, context)
        elif isinstance(node, planner.Join):
            out = executor.join(node, context)
        elif isinstance(node, planner.Sort):
            out = executor.sort(node, context)
        elif isinstance(node, planner.SetOperation):
            out = executor.set_operation(node, context)
        else:
            raise Fallback("step type")
        if node.offset:
            table = out.tables[node.name]
            table.rows = table.rows[node.offset:]
        for t in out.tables.values():
            if id(t) in guard or id(getattr(t, "rows", None)) in guard:
                raise Fallback("base table in an output")
        return out

    def run_tracked(node, outs):
        """Runs node on the materialized outputs of its dependencies (outs, in order); checks
        the shared outputs that the run could change; returns (out, reaches a shared output)."""
        context = executor.context({name: table for o in outs for name, table in o.tables.items()})
        out = run_step(node, context)
        touched = set()
        for d in node.dependencies:
            touched |= reach.get(id(d), set())
        for sid in touched:
            slot = info.get(sid)
            o = slot[1] if slot is not None else None
            if o is None:
                continue
            st = out_state(o, T, RR, GR)
            if st is None or content_hash(st) != shared[sid]:
                raise Fallback("a step changed a shared output")
        mine = _objs(out)
        r = set()
        for d, o in zip(node.dependencies, outs):
            if mine & _objs(o):
                r |= reach.get(id(d), set())
        aliased = bool(r)
        if len(node.dependents) > 1:
            r.add(id(node))
        reach[id(node)] = r
        return out, aliased

    def materialize(node):
        slot = info[id(node)]
        if slot[1] is not None:
            return slot[1]
        i = idx[id(node)]
        span = pstore.get(i)
        content_only = len(pouts[i]) == 17 and pouts[i][:1] == b"c"
        if span is not None and not content_only and i not in alias_steps:
            out = restore(store.blob(span), T, RR, GR)
            reach[id(node)] = {id(node)} if len(node.dependents) > 1 else set()
            stats[2] += 1
        else:
            # a pinned step without a stored output, or one that must not be restored: it runs
            # (its code and input are pinned)
            if not spend(pcpu[i]):
                raise Fallback("budget")
            outs = [materialize(dep) for dep in node.dependencies]
            out, _ = run_tracked(node, outs)
            if _DBG:
                why = "alias" if i in alias_steps else ("content" if content_only else "nostore")
                _DBG_COUNTS[("mat", type(node).__name__, why)] = _DBG_COUNTS.get(("mat", type(node).__name__, why), 0) + 1
                _DBG_COUNTS[("matcpu", type(node).__name__, why)] = round(_DBG_COUNTS.get(("matcpu", type(node).__name__, why), 0) + pcpu[i], 4)
            st = out_state(out, T, RR, GR)
            # (content: an alias step's readers are where its other users left them)
            if st is None or content_hash(st) != rec["chash"][i]:
                raise Fallback("pinned step %d gave another output" % i)
            stats[1] += 1
        if len(node.dependents) > 1 and id(node) not in shared:
            raise Fallback("shared step without a content hash")
        slot[1] = out
        return out

    ctes, scope = executor._ctes, executor._outer_scope
    executor._ctes, executor._outer_scope = plan.ctes, None
    try:
        finished = set()
        queue = set(plan.leaves)
        while queue:
            node = queue.pop()
            i = idx.get(id(node))
            if i is None:
                raise Fallback("step outside the order")
            deps_pinned = all(info[id(d)][0] for d in node.dependencies)
            if same_def[i] and deps_pinned and ("%d:%d" % (query, i)) not in dirty:
                info[id(node)] = [True, None]
                stats[0] += 1
                if len(node.dependents) > 1:
                    c = rec["cont"].get(i)
                    if c is None:
                        raise Fallback("shared step without a content hash")
                    shared[id(node)] = c
            else:
                if not spend(pcpu[i] if i < n_pin else rec["late"]):
                    raise Fallback("budget")
                outs = [materialize(dep) for dep in node.dependencies]
                out, aliased = run_tracked(node, outs)
                stats[1] += 1
                pinned = False
                st = None
                if same_def[i] and deps_pinned and not aliased:
                    st = out_state(out, T, RR, GR)
                    pinned = st is not None and out_matches(st, pouts[i])
                    if _DBG:
                        _DBG_COUNTS[("runcpu", type(node).__name__, bool(pinned))] = round(_DBG_COUNTS.get(("runcpu", type(node).__name__, bool(pinned)), 0) + pcpu[i], 4)
                    if _DBG and not pinned and st is not None:
                        _DBG_COUNTS[("cmatch" if content_hash(st) == rec["chash"][i] else "differ", type(node).__name__)] = _DBG_COUNTS.get(("cmatch" if content_hash(st) == rec["chash"][i] else "differ", type(node).__name__), 0) + 1
                elif _DBG:
                    _DBG_COUNTS[("notcandcpu", type(node).__name__)] = round(_DBG_COUNTS.get(("notcandcpu", type(node).__name__), 0) + (pcpu[i] if i < n_pin else 0), 4)
                    _DBG_COUNTS[("notcand", type(node).__name__, bool(same_def[i]), bool(deps_pinned), bool(aliased))] = _DBG_COUNTS.get(("notcand", type(node).__name__, bool(same_def[i]), bool(deps_pinned), bool(aliased)), 0) + 1
                if len(node.dependents) > 1:
                    if st is None:
                        st = out_state(out, T, RR, GR)
                    if st is None:
                        raise Fallback("shared output state")
                    shared[id(node)] = content_hash(st)
                info[id(node)] = [pinned, out]
            finished.add(node)
            for dep in node.dependents:
                if all(id(d) in info for d in dep.dependencies):
                    queue.add(dep)
            for dep in node.dependencies:
                if all(d in finished for d in dep.dependents):
                    info[id(dep)][1] = None  # (as _execute pops it: the memory)
        root = plan.root
        if len(info) != len(order):
            raise Fallback("steps left")
        return materialize(root).tables[root.name]
    finally:
        executor._ctes, executor._outer_scope = ctes, scope
