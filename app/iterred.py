"""Loop-iteration parts of a test (an extension of reduce.py).

A test statement that is a for loop (or holds one) runs many independent checks, one per
iteration. When only some iterations run the changed code, the others cannot change the
result (same rule as for statements: deterministic code, no state carried from one
iteration to the next). The test method's text is rewritten so that each eligible loop
iterates through _br_loop(iterable, site): the iterable is consumed as before (same items,
same order), and only the kept iterations enter the body.

An iteration is named by its path: ((site, index), ...) from the outermost eligible loop.
"""
from __future__ import annotations

import ast
import builtins
import os

NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.ListComp, ast.SetComp,
          ast.DictComp, ast.GeneratorExp)


def walk_local(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        for c in ast.iter_child_nodes(n):
            if not isinstance(c, NESTED):
                stack.append(c)


def _target_names(t):
    return {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}


def _stores(nodes):
    out = set()
    for st in nodes:
        for x in walk_local(st):
            if isinstance(x, ast.Name) and isinstance(x.ctx, (ast.Store, ast.Del)):
                out.add(x.id)
            elif isinstance(x, (ast.Import, ast.ImportFrom)):
                for a in x.names:
                    out.add(a.asname or a.name.partition(".")[0])
            elif isinstance(x, ast.ExceptHandler) and x.name:
                out.add(x.name)
            elif isinstance(x, ast.NamedExpr):
                out.add(x.target.id)
    return out


def _loads(node, names, assigned, bad):
    for x in walk_local(node):
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id in names and x.id not in assigned:
            bad.add(x.id)


def carried(loop):
    """Names that an iteration may read from an earlier iteration."""
    cand = _stores(loop.body) - _target_names(loop.target)
    bad = set()
    if not cand:
        return bad

    def seq(stmts, assigned):
        assigned = set(assigned)
        for s in stmts:
            if isinstance(s, ast.Assign) or (isinstance(s, ast.AnnAssign) and s.value is not None):
                _loads(s.value, cand, assigned, bad)
                targets = s.targets if isinstance(s, ast.Assign) else [s.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        assigned.add(t.id)
                    elif isinstance(t, (ast.Tuple, ast.List)) and all(isinstance(e, ast.Name) for e in t.elts):
                        assigned |= {e.id for e in t.elts}
                    else:
                        bad.add("<target>")
            elif isinstance(s, ast.With):
                for it in s.items:
                    _loads(it.context_expr, cand, assigned, bad)
                inner = set(assigned)
                for it in s.items:
                    if it.optional_vars is not None:
                        if not isinstance(it.optional_vars, ast.Name):
                            bad.add("<target>")
                        inner |= _target_names(it.optional_vars)
                # a with body may be left early by an exception that the manager swallows
                # (subTest): names assigned in it are not definitely assigned after it
                seq(s.body, inner)
            elif isinstance(s, ast.For):
                _loads(s.iter, cand, assigned, bad)
                seq(s.body, assigned | _target_names(s.target))
                if s.orelse:
                    bad.add("<else>")
            elif isinstance(s, ast.Expr):
                _loads(s, cand, assigned, bad)
            else:
                # if / try / while / other: loads checked; their stores are not definite
                _loads(s, cand, assigned, bad)
                if _stores([s]) & cand:
                    bad.add("<branch store>")
        return assigned

    seq(loop.body, set())
    return bad


def loop_problems(loop):
    why = []
    if loop.orelse:
        why.append("else")
    for st in loop.body:
        for x in walk_local(st):
            if isinstance(x, (ast.Return, ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal, ast.Break)):
                why.append(type(x).__name__)
            elif isinstance(x, (ast.Attribute, ast.Subscript)) and isinstance(x.ctx, (ast.Store, ast.Del)):
                why.append("state store")
    for x in walk_local(loop.iter):
        if isinstance(x, ast.Name) and "fixture" in x.id.lower():
            why.append("fixture loader")
        if isinstance(x, ast.Attribute) and "fixture" in x.attr.lower():
            why.append("fixture loader")
    c = carried(loop)
    if c:
        why.append("carried:" + ",".join(sorted(c)))
    return why


def method_sites(func: ast.FunctionDef, droppable):
    """Eligible loops of a test method: [(site, stmt index, loop node)], in source order.
    droppable: list of flags per top-level statement (reduce.method_shape)."""
    out = []
    for i, s in enumerate(func.body):
        if not droppable[i]:
            continue
        loops = [x for x in walk_local(s) if isinstance(x, ast.For)]
        loops.sort(key=lambda x: (x.lineno, x.col_offset))
        for lp in loops:
            if not loop_problems(lp):
                out.append((i, lp))
    return [(k, i, lp) for k, (i, lp) in enumerate(out)]


def rewrite_lines(lines, first, sites):
    """lines: the method's source lines (first = line number of lines[0]). Wraps the iterable of
    each site in _br_loop(..., site). Returns new lines (same count)."""
    new = [ln.encode("utf-8") for ln in lines]
    edits = []
    for k, _, lp in sites:
        it = lp.iter
        edits.append((it.end_lineno, it.end_col_offset, b", %d)" % k))
        edits.append((it.lineno, it.col_offset, b"_br_loop("))
    edits.sort(key=lambda e: (e[0], e[1]), reverse=True)
    for ln, col, text in edits:
        b = new[ln - first]
        new[ln - first] = b[:col] + text + b[col:]
    return [b.decode("utf-8") for b in new]


class Loops:
    def __init__(self):
        self.keep_all = None  # None: every iteration runs; else set of paths whose subtrees run
        self.keep_prefix = frozenset()  # paths that lead to kept paths
        self.path = ()
        self.hook = None  # tracer: called with the new path at each change
        self.bad_sites = set()  # tracer: sites whose iterable is not stable under reduction

    def reset(self, keep_all=None, keep_prefix=frozenset()):
        self.keep_all = keep_all
        self.keep_prefix = keep_prefix
        self.path = ()

    def wanted(self, p):
        ka = self.keep_all
        if ka is None or p in self.keep_prefix:
            return True
        for n in range(1, len(p) + 1):
            if p[:n] in ka:
                return True
        return False


LOOPS = Loops()


def _br_loop(iterable, site):
    L = LOOPS
    parent = L.path
    hook = L.hook
    i = -1
    for item in iterable:
        i += 1
        p = parent + ((site, i),)
        if L.keep_all is not None and not L.wanted(p):
            continue
        L.path = p
        if hook is not None:
            hook(p)
        yield item
        L.path = parent
        if hook is not None:
            hook(parent)
    L.path = parent


def keep_sets(paths_all):
    """keep_all paths -> (keep_all, keep_prefix)."""
    ka = frozenset(tuple(tuple(x) for x in p) for p in paths_all)
    pre = set()
    for p in ka:
        for n in range(1, len(p)):
            pre.add(p[:n])
    return ka, frozenset(pre)


def install():
    builtins._br_loop = _br_loop


# --- run time ---------------------------------------------------------------

_INDEX = None
_RECS = {}


def record(maps_dir, n):
    """The map record of test n, or None (maps_dir/i/<n>.bin, see build_iters.py)."""
    global _INDEX
    import marshal
    import os

    if _INDEX is None:
        try:
            with open(os.path.join(maps_dir, "i", "index.bin"), "rb") as f:
                _INDEX = marshal.loads(f.read())
        except (OSError, ValueError, EOFError):
            _INDEX = frozenset()
    if n not in _INDEX:
        return None
    rec = _RECS.get(n)
    if rec is None:
        import zlib

        try:
            with open(os.path.join(maps_dir, "i", "%d.bin" % n), "rb") as f:
                rec = marshal.loads(zlib.decompress(f.read()))
        except (OSError, ValueError, EOFError, zlib.error):
            rec = False
        _RECS[n] = rec
    return rec or None


# A statement is rewritten only when at least this share of its iterations would be skipped.
MIN_SKIP = float(os.environ.get("BR_ITERS_MIN_SKIP", "0.25"))


def choose(srec, lines):
    """srec: one statement of the map; lines: {file: changed lines}. Returns the iteration
    paths to keep, or None when the statement must run whole."""
    _, own, paths, cov = srec
    mask = 0
    for rel, lns in lines.items():
        o = own.get(rel)
        c = cov.get(rel)
        for ln in lns:
            if o is not None and ln in o:
                return None
            if c is not None:
                m = c.get(ln)
                if m:
                    mask |= m
    if not mask:
        return None  # no traced part of the statement runs the lines: no claim, whole run
    out = []
    i = 0
    while mask:
        if mask & 1:
            out.append(paths[i])
        mask >>= 1
        i += 1
    # every iteration runs anyway: no rewrite. The same when only a few would be skipped: the
    # rewritten loop then saves less than the rewrite and its compile cost.
    kept = set(out)
    inner = {p[:-1] for p in paths}
    leaves = [p for p in paths if p not in inner]
    run = sum(1 for p in leaves if any(p[:k] in kept for k in range(1, len(p) + 1)))
    if (len(leaves) - run) < MIN_SKIP * len(leaves) or run == len(leaves):
        return None
    return out


def rewrite_by_edits(lines, first, edits, sites):
    """Like rewrite_lines, from the map's recorded positions."""
    new = [ln.encode("utf-8") for ln in lines]
    todo = []
    for k in sites:
        l1, c1, l2, c2 = edits[k]
        todo.append((l2, c2, b", %d)" % k))
        todo.append((l1, c1, b"_br_loop("))
    todo.sort(key=lambda e: (e[0], e[1]), reverse=True)
    for ln, col, text in todo:
        i = ln - first
        if i < 0 or i >= len(new) or len(new[i]) < col:
            return None
        b = new[i]
        new[i] = b[:col] + text + b[col:]
    return [b.decode("utf-8") for b in new]
