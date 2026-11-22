"""Runs a test with only some of its independent parts.

A sqlglot test is mostly a list of independent checks: top-level statements of the test
method (self.validate_identity(...), ...), the entries of a fixture file that a loader
yields one by one, and the tasks that a test hands to a process pool. A patch can change
the result of such a part only when the part runs the patched code. The maps record, for
the pinned commit, which lines each part runs; the selector then runs each affected test
with only its affected parts. The rest of the test (setup, statements that later
statements depend on) always runs.

Part keys: ("s", statement index), ("f", fixture file name, entry index), ("t", task index).
"""
from __future__ import annotations

import ast
import sys
import types


class State:
    def __init__(self):
        self.keep_files = None  # None = every entry; else {fixture name: set of entry indices}
        self.keep_tasks = None  # None = every task; else set of task indices
        self.current = None  # the part that runs now
        self.task_counter = 0
        self.on_enter = None  # tracer hook
        self.on_submit = None  # tracer hook
        self.files_seen = {}  # fixture name -> number of entries (tracer)
        self.tasks_seen = 0
        # entries of a patched fixture file that differ from the pinned file:
        # {("p" or "x", fixture name): set of entry indices}; only these can change a result
        self.keep_changed = None
        self.changed_only = False  # True: nothing but data files changed in this commit
        self.in_test = False
        self.task_hook = None  # stages.py: called in place of a pool task (index, fn, args, kwargs)

    def reset(self, keep_files=None, keep_tasks=None):
        self.keep_files = keep_files
        self.keep_tasks = keep_tasks
        self.current = None
        self.task_counter = 0
        self.files_seen = {}
        self.tasks_seen = 0

    def enter(self, tag):
        self.current = tag
        hook = self.on_enter
        if hook is not None:
            hook(tag)


STATE = State()


# --- fixture loaders -------------------------------------------------------
# These two functions replace the code of tests.helpers.load_sql_fixture_pairs and
# load_sql_fixtures. They run with the globals of tests.helpers, where install() puts
# _br_state and the original functions.


def _pairs_wrapper(filename):
    st = _br_state  # noqa: F821
    if not st.in_test:  # class and module fixtures always load every entry
        yield from _br_orig_pairs(filename)  # noqa: F821
        return
    keep = st.keep_files
    wanted = None if keep is None else keep.get(filename, ())
    changed = st.keep_changed
    if changed is not None and ("p", filename) in changed:
        only = changed[("p", filename)]
        if wanted is not None:
            wanted = set(wanted) | only
        elif st.changed_only:
            wanted = only
    i = -1
    for item in _br_orig_pairs(filename):  # noqa: F821
        i += 1
        if wanted is None or i in wanted:
            st.enter(("f", filename, i))
            yield item
    st.files_seen[filename] = i + 1
    st.enter(None)


def _fixtures_wrapper(filename):
    st = _br_state  # noqa: F821
    if not st.in_test:
        yield from _br_orig_fixtures(filename)  # noqa: F821
        return
    keep = st.keep_files
    wanted = None if keep is None else keep.get(filename, ())
    changed = st.keep_changed
    if changed is not None and ("x", filename) in changed:
        only = changed[("x", filename)]
        if wanted is not None:
            wanted = set(wanted) | only
        elif st.changed_only:
            wanted = only
    i = -1
    for item in _br_orig_fixtures(filename):  # noqa: F821
        i += 1
        if wanted is None or i in wanted:
            st.enter(("f", filename, i))
            yield item
    st.files_seen[filename] = i + 1
    st.enter(None)


def install_loaders() -> bool:
    """Returns False when tests.helpers does not look like the pinned one."""
    helpers = sys.modules.get("tests.helpers")
    if helpers is None:
        import importlib

        helpers = importlib.import_module("tests.helpers")
    if getattr(helpers, "_br_state", None) is not None:
        return True
    pairs = getattr(helpers, "load_sql_fixture_pairs", None)
    fixtures = getattr(helpers, "load_sql_fixtures", None)
    if not isinstance(pairs, types.FunctionType) or not isinstance(fixtures, types.FunctionType):
        return False
    if pairs.__code__.co_freevars or fixtures.__code__.co_freevars:
        return False
    helpers._br_state = STATE
    helpers._br_orig_pairs = types.FunctionType(pairs.__code__, helpers.__dict__, "load_sql_fixture_pairs")
    helpers._br_orig_fixtures = types.FunctionType(fixtures.__code__, helpers.__dict__, "load_sql_fixtures")
    pairs.__code__ = _pairs_wrapper.__code__
    fixtures.__code__ = _fixtures_wrapper.__code__
    return True


# --- statements of a test method --------------------------------------------


def _names(node, ctx):
    return {x.id for x in ast.walk(node) if isinstance(x, ast.Name) and isinstance(x.ctx, ctx)}


def _binds(node):
    out = _names(node, ast.Store)
    for x in ast.walk(node):
        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(x.name)
        elif isinstance(x, (ast.Import, ast.ImportFrom)):
            for a in x.names:
                out.add(a.asname or a.name.partition(".")[0])
        elif isinstance(x, ast.ExceptHandler) and x.name:
            out.add(x.name)
        elif isinstance(x, (ast.MatchAs, ast.MatchStar)) and x.name:
            out.add(x.name)
    return out


def _escapes(node):
    """True when the statement can leave the function or changes scope rules."""
    for x in ast.walk(node):
        if isinstance(x, (ast.Return, ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal)):
            return True
    return False


def _stores_state(node):
    """True when the statement assigns to an attribute or an item (object state)."""
    for x in ast.walk(node):
        if isinstance(x, (ast.Attribute, ast.Subscript)) and isinstance(x.ctx, (ast.Store, ast.Del)):
            return True
    return False


def method_shape(func: ast.FunctionDef):
    """Returns (first line, last line, [(start, end, droppable)]) for a test method.

    A statement is droppable when no later statement reads a name that it binds, and it
    neither leaves the function nor assigns object state."""
    body = func.body
    first = min([func.lineno] + [d.lineno for d in func.decorator_list])
    used_later = set()
    flags = [False] * len(body)
    for i in range(len(body) - 1, -1, -1):
        s = body[i]
        ok = isinstance(s, (ast.Expr, ast.With, ast.For, ast.If, ast.Try, ast.While, ast.Assign, ast.AnnAssign, ast.AugAssign))
        if ok and isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            ok = False  # docstring
        if ok and (_escapes(s) or _stores_state(s)):
            ok = False
        if ok and (_binds(s) & used_later):
            ok = False
        if ok and isinstance(s, ast.AugAssign):
            ok = False
        flags[i] = ok
        used_later |= _names(s, ast.Load) | _names(s, ast.Del)
    return first, func.end_lineno, [(s.lineno, s.end_lineno, flags[i]) for i, s in enumerate(body)]


def file_shapes(src: str):
    """{(class name, method name): shape} for the test methods defined in a test file."""
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test"):
                    out[(node.name, item.name)] = method_shape(item)
    return out


def reduced_lines(file_lines, shape, keep):
    """The method's source lines with the dropped statements blanked. keep: set of indices."""
    first, last, stmts = shape
    lines = list(file_lines[first - 1:last])
    left = 0
    for i, (s, e, droppable) in enumerate(stmts):
        if droppable and i not in keep:
            for k in range(s, e + 1):
                lines[k - first] = ""
        else:
            left += 1
    if not left:
        s = stmts[0][0]
        src = file_lines[s - 1]
        lines[s - first] = src[: len(src) - len(src.lstrip())] + "pass"
    return lines


def find_code(module_code, name, first_line):
    """The code object of function 'name' that starts at first_line, inside compiled unit code."""
    stack = [module_code]
    while stack:
        c = stack.pop()
        for k in c.co_consts:
            if isinstance(k, types.CodeType):
                if k.co_name == name and k.co_firstlineno == first_line:
                    return k
                stack.append(k)
    return None


def _entry_key(item):
    if isinstance(item, tuple):
        return tuple(tuple(sorted(x.items())) if isinstance(x, dict) else x for x in item)
    return item


def changed_entries(name, read_old):
    """Entries of the patched fixture file 'name' that the pinned file does not have.

    read_old: context manager under which open() sees the pinned file. Returns
    {("p", name): set, ("x", name): set}, or {} when the file cannot be compared (then every
    entry runs). An entry that is equal to a pinned entry gives the pinned result, and all
    tests pass on the pinned commit, so only new entries can make a test fail."""
    helpers = sys.modules.get("tests.helpers")
    out = {}
    if helpers is None or getattr(helpers, "_br_state", None) is None:
        return out
    for kind, fn in (("p", helpers._br_orig_pairs), ("x", helpers._br_orig_fixtures)):
        try:
            new = [_entry_key(e) for e in fn(name)]
            with read_old():
                old = {_entry_key(e) for e in fn(name)}
            out[(kind, name)] = {i for i, e in enumerate(new) if e not in old}
        except Exception:  # noqa: BLE001  (a malformed file: the full test must run and fail)
            continue
    return out


# Small-scale variants of three tests that only check behaviour at a size: the same
# statements with smaller constants. Each entry: (text in the pinned test, replacement, count).
# A variant predicts the result of the real test; the real test is what gets selected.
SCALED = {
    "tests/test_parser.py::TestParser::test_max_nodes": [
        ("width = 10000", "width = 20", 1),
        # the pinned query has 100 * width + 1 counted nodes: one more than the first limit
        ("            parse_one(sql, max_nodes=1_000_000)", "            parse_one(sql, max_nodes=2_000)", 1),
        ("parse_one(sql, max_nodes=50_000)", "parse_one(sql, max_nodes=1_000)", 1),
        ('"(50000)"', '"(1000)"', 1),
    ],
    "tests/test_generator.py::TestGenerator::test_generator_memory": [
        ("for _ in range(100):", "for _ in range(2):", 1),
        ("for _ in range(1000):", "for _ in range(60):", 1),
    ],
    "tests/test_optimizer.py::TestOptimizer::test_scope": [
        ('["SELECT x FROM t"] * 10000))', '["SELECT x FROM t"] * 100))', 1),
        ('" UNION SELECT 1 INTERSECT SELECT 1" * 1000, read="hive")', '" UNION SELECT 1 INTERSECT SELECT 1" * 20, read="hive")', 1),
        ("traverse())), 4001)", "traverse())), 81)", 1),
    ],
}


def scale_lines(test_id, lines, reduced=False):
    """Applies the small-scale constants. Returns None when the text is not the pinned one.
    In a reduced run a statement with such a constant can be absent."""
    out = list(lines)
    for old, new, count in SCALED[test_id]:
        hits = [i for i, s in enumerate(out) if old in s]
        if len(hits) > count or (len(hits) < count and not reduced) or any(out[i].count(old) != 1 for i in hits):
            return None
        for i in hits:
            out[i] = out[i].replace(old, new)
    return out


# A child that serves several commits one after the other (selector.fixture_batch_child) puts
# a list here: apply_reduction then notes (function, the code it had) for each method that it
# replaces, and the child puts that code back after the commit's tests.
SWAPPED = None


def live_function(cls, method, old_code):
    """The function object behind cls.method (under decorators), when it has the pinned code."""
    f = cls.__dict__.get(method)
    seen = 0
    while f is not None and seen < 8:
        if isinstance(f, types.FunctionType) and f.__code__ == old_code:
            return f
        f = getattr(f, "__wrapped__", None)
        seen += 1
    return None


# Off unless BR_KEEP_SHARED=1: the rule costs CPU (it parses each reduced method, and methods
# whose statements all read one local, a schema for instance, are no longer reduced).
KEEP_SHARED = __import__("os").environ.get("BR_KEEP_SHARED") == "1"  # see keep_shared_state


def keep_shared_state(file_lines, shape, keep):
    """Adds to keep the droppable statements that share an object with a kept statement.

    A droppable statement binds no name that a later statement reads, but it can still act
    on an object that a kept statement uses: a call that mutates it (schema.add_table(...),
    hash(expr) filling a cache) or an assertion that reads its state afterwards
    (self.assertEqual(len(schema._find_cache), 1)). Such a statement runs no patched line
    itself, so the unit map does not choose it; dropping it can remove the state that leads
    the kept part to the patched code, or the only check that sees the patched behaviour.
    The test then passes falsely. Two statements share an object when both read the same
    local variable of the method (self excepted: every validate_* statement reads it; module
    names such as parse_one are not local). Returns a new set, or None when the method source
    cannot be parsed (then every statement stays)."""
    first, last, stmts = shape
    try:
        # the method keeps its indentation under a wrapper line (dedenting the text would
        # fail on a multi-line string that has text at column 0)
        fn = ast.parse("if 1:\n" + "\n".join(file_lines[first - 1:last]) + "\n").body[0].body[0]
    except (SyntaxError, IndexError, ValueError, AttributeError):
        return None
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or len(fn.body) != len(stmts):
        return None
    local = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    for s in fn.body:
        local |= _binds(s)
    local.discard("self")
    reads = [(_names(s, ast.Load) | _names(s, ast.Del)) & local for s in fn.body]
    out = set(keep)
    changed = True
    while changed:
        changed = False
        live = set()
        for i in out:
            if 0 <= i < len(reads):
                live |= reads[i]
        for j, (_, _, droppable) in enumerate(stmts):
            if droppable and j not in out and reads[j] & live:
                out.add(j)
                changed = True
    return out


def reduction_codes(test_id, full_path, shape, file_lines, flags, import_names, keep_stmts, scaled, pinned=None, iters=None):
    """The part of apply_reduction that only reads its arguments: (old code, new code) of the
    method, or None. The parent can do it before the fork (selector.prepare_keeps)."""
    import engine

    _, class_name, method = test_id.split("::")
    first = shape[0]
    old_lines = file_lines[first - 1:shape[1]]
    if keep_stmts is not None and KEEP_SHARED:
        keep_stmts = keep_shared_state(file_lines, shape, keep_stmts)
    new_lines = old_lines if keep_stmts is None else reduced_lines(file_lines, shape, keep_stmts)
    if scaled:
        new_lines = scale_lines(test_id, new_lines, keep_stmts is not None)
        if new_lines is None:
            return None
    if iters is not None:
        # loop-iteration parts (iterred.py): the kept loops iterate through _br_loop
        import iterred

        new_lines = iterred.rewrite_by_edits(new_lines, first, iters[2], iters[1])
        if new_lines is None:
            return None
    if pinned is not None:
        old = pinned
    else:
        old_mod = engine.compile_lines(old_lines, first, (class_name,), full_path, flags, import_names)
        if old_mod is None:
            return None
        old = find_code(old_mod, method, first)
    new_mod = engine.compile_lines(new_lines, first, (class_name,), full_path, flags, import_names)
    if new_mod is None:
        return None
    new = find_code(new_mod, method, first)
    if old is None or new is None:
        return None
    return old, new


def install_reduction(cls, method, old, new):
    """The part of apply_reduction that changes the live test class. Returns True on success."""
    func = live_function(cls, method, old)
    if func is None or len(old.co_freevars) != len(new.co_freevars):
        return False
    if SWAPPED is not None:
        SWAPPED.append((func, func.__code__))
    func.__code__ = new
    return True


def apply_reduction(cls, test_id, full_path, shape, file_lines, flags, import_names, keep_stmts, scaled, pinned=None, iters=None):
    """Replaces the code of a test method by its reduced and/or small-scale form.

    keep_stmts: set of statement indices to keep, or None for all. Returns True on success.
    pinned: the pinned code object of the method, when its compile alone is known to give
    an equal code object (the caller checks that): then it is not compiled again."""
    got = reduction_codes(test_id, full_path, shape, file_lines, flags, import_names, keep_stmts, scaled, pinned, iters)
    if got is None:
        return False
    return install_reduction(cls, test_id.split("::")[2], got[0], got[1])
