"""Instruments a copy of the pinned sqlglot tree for the operand outcome map (phase 1 + 2).

usage (inside the pinned python image): python instrument.py REPO [--group REGEX] [--no-cmp] [--no-phase2] [--count] [--skip FILE]
--no-phase2: phase 1 only (the instrumentation that maps/infect.bin was built with).
--group: only files whose path (relative to REPO) matches REGEX get the phase 2 probes (the
suite is run once per group, which keeps each run short; site numbers are the same in every
group). --no-cmp: leave the phase 1 comparisons alone (only constant comparisons of the group
are probed, for their value classes). REPO/_infect_inst.json lists the probed site numbers.
REPO is a copy of the pinned checkout (copied with preserved modification times). The .py
sources are not changed: for every sqlglot/**/*.py the script writes an instrumented .pyc into
__pycache__ (valid for the unchanged source, so line numbers, inspect.getsource and tracebacks
stay as they are).

Phase 1 (comparisons): every comparison with exactly one operator ("a < b", not "a < b < c") is
compiled as a call __br_<Op>__(site, a, b) of the probe in _brprobe.py, which returns the
original result. When one operand is a literal constant (an int, bool, None, str or float, or
a negated number), the call is __br_K<Op>__(site, a, b, side) instead: it also records the value
class of the other operand, so that a changed constant can be judged later.

Phase 2 (boolean operators): every maximal and/or tree (a BoolOp whose parent is not a BoolOp,
with 2 to 8 leaf operands) becomes
    __br_E__(site, <the same tree, each leaf L_i replaced by __br_L__(site, i, __br_P__(site, i) or L_i)>)
__br_P__ returns None, so "__br_P__(...) or L_i" is L_i, evaluated in the same frame, in the same
order and with the same short-circuiting as before; __br_L__ returns its last argument. The
probe sees where each leaf is about to run (P), its value (L), and the end of the evaluation (E).
Leaves whose syntax is side-effect free (see PURE below) are also stored as source text, so that
the probe can evaluate a leaf that the original run skipped, with a checked evaluator that
never runs user code.

Each module body also counts how many sqlglot module bodies are executing, so that import-time
evaluations can be told apart. Comparisons and and/or inside annotations are left alone.
Writes REPO/_infect_sites.json (compare sites: [file, lineno, col, end_lineno, end_col, op]) and
REPO/_infect_bsites.json (and/or sites: [file, lineno, col, end_lineno, end_col, tree, context,
leaf sources]); tree is ["O"|"A", child, ...] with leaf indices as ints; context "T" when the
result is only tested for truth (test of if / while / assert / conditional expression /
comprehension filter, operand of not), else "V".
"""
import ast
import importlib._bootstrap_external as be
import importlib.util
import json
import os
import sys

SKIP_FIELDS = {"annotation", "returns", "type_params", "type_comment"}
COUNT = False  # --count: only count the evaluations of the phase 2 sites (to find the hottest)
SKIP_K = set()  # --skip FILE: sites left unprobed (too hot for the time limit; unmapped)
SKIP_B = set()
MAX_LEAVES = 8
PURE_CALLS = {"isinstance", "len", "bool"}


def _const_value(node):
    if isinstance(node, ast.UnaryOp):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    return node.value


def const_operand(node):
    """True for a literal constant operand: an int, bool, None, str or float, or -/+ a number."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
        return isinstance(node, ast.Constant) and type(node.value) in (int, float)
    return isinstance(node, ast.Constant) and type(node.value) in (int, float, str, bool, type(None))


def pure_syntax(node):
    """Syntax that can be evaluated without side effects when the values allow it: names,
    constants, attribute loads, subscripts, not, comparisons, and/or, tuples, and calls of
    isinstance / len / bool and of .get on a value. The probe checks the values at run time
    (no properties, no user __eq__, exact builtin containers, ...)."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return isinstance(node.ctx, ast.Load)
    if isinstance(node, ast.Attribute):
        return isinstance(node.ctx, ast.Load) and pure_syntax(node.value)
    if isinstance(node, ast.Subscript):
        return isinstance(node.ctx, ast.Load) and not isinstance(node.slice, ast.Slice) and pure_syntax(node.value) and pure_syntax(node.slice)
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.Not, ast.USub)) and pure_syntax(node.operand)
    if isinstance(node, ast.Compare):
        return pure_syntax(node.left) and all(pure_syntax(c) for c in node.comparators)
    if isinstance(node, ast.BoolOp):
        return all(pure_syntax(v) for v in node.values)
    if isinstance(node, ast.Tuple):
        return isinstance(node.ctx, ast.Load) and all(pure_syntax(e) and not isinstance(e, ast.Starred) for e in node.elts)
    if isinstance(node, ast.Call):
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            return False
        f = node.func
        if isinstance(f, ast.Name) and f.id in PURE_CALLS:
            return all(pure_syntax(a) for a in node.args)
        if isinstance(f, ast.Attribute) and f.attr == "get" and 1 <= len(node.args) <= 2:
            return pure_syntax(f.value) and all(pure_syntax(a) for a in node.args)
        return False
    return False


def tree_desc(node, leaves):
    out = ["O" if isinstance(node.op, ast.Or) else "A"]
    for v in node.values:
        if isinstance(v, ast.BoolOp):
            out.append(tree_desc(v, leaves))
        else:
            out.append(len(leaves))
            leaves.append(v)
    return out


def count_leaves(node):
    return sum(count_leaves(v) if isinstance(v, ast.BoolOp) else 1 for v in node.values)


def truth_context(parent, field):
    if isinstance(parent, (ast.If, ast.While, ast.Assert, ast.IfExp)) and field == "test":
        return True
    if isinstance(parent, ast.comprehension) and field == "ifs":
        return True
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
        return True
    return False


def _call(name, args, loc):
    c = ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=[])
    ast.copy_location(c, loc)
    c.func.lineno = loc.lineno
    return c


class Instrument(ast.NodeTransformer):
    def __init__(self, rel, sites, bsites, phase2=True, cmp=True, inst=None):
        self.rel = rel
        self.sites = sites
        self.bsites = bsites
        self.phase2 = phase2  # probe the and/or trees and constant comparisons of this file
        self.cmp = cmp  # probe every comparison (phase 1)
        self.inst = inst if inst is not None else {"K": [], "B": []}
        self.parent = None
        self.field = None
        self.func_depth = 0  # inside a def or lambda: names resolve through the frame

    def generic_visit(self, node):
        saved = (self.parent, self.field)
        is_func = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        is_class = isinstance(node, ast.ClassDef)
        saved_depth = self.func_depth
        for field, old in ast.iter_fields(node):
            if field in SKIP_FIELDS:
                continue
            # decorators, defaults and base classes run in the enclosing scope
            if is_func and field == "body":
                self.func_depth = saved_depth + 1
            elif is_class and field == "body":
                self.func_depth = 0
            else:
                self.func_depth = saved_depth
            self.parent, self.field = node, field
            if isinstance(old, list):
                new = []
                for v in old:
                    if isinstance(v, ast.AST):
                        self.parent, self.field = node, field
                        v = self.visit(v)
                        if v is None:
                            continue
                        if not isinstance(v, ast.AST):
                            new.extend(v)
                            continue
                    new.append(v)
                old[:] = new
            elif isinstance(old, ast.AST):
                nv = self.visit(old)
                if nv is None:
                    delattr(node, field)
                else:
                    setattr(node, field, nv)
        self.func_depth = saved_depth
        self.parent, self.field = saved
        return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if len(node.ops) != 1:
            return node  # chained comparison: not a mapped site
        op = type(node.ops[0]).__name__
        sid = len(self.sites)
        self.sites.append([self.rel, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, op])
        side = None
        if op not in ("In", "NotIn"):
            # the probe gets the original operand nodes' values; constants are recognised on the
            # pinned source (the transformed children of a constant are the constant itself)
            if const_operand(node.comparators[0]) and not const_operand(node.left):
                side = 1
            elif const_operand(node.left) and not const_operand(node.comparators[0]):
                side = 0
        if side is not None and type(_const_value(node.left if side == 0 else node.comparators[0])) is type(None):
            side = None  # x is None / x == None: no mutation kind changes a None literal
        if side is not None and self.phase2 and COUNT:
            call = ast.Call(func=ast.Name(id="__br_CK__", ctx=ast.Load()), args=[ast.Constant(sid), node], keywords=[])
            ast.copy_location(call, node)
            return call
        if side is not None and self.phase2 and sid not in SKIP_K:
            self.inst["K"].append(sid)
            name = "__br_K%s__" if self.cmp else "__br_k%s__"
            call = ast.Call(func=ast.Name(id=name % op, ctx=ast.Load()),
                            args=[ast.Constant(sid), node.left, node.comparators[0], ast.Constant(side)], keywords=[])
        elif not self.cmp:
            return node
        else:
            call = ast.Call(func=ast.Name(id="__br_%s__" % op, ctx=ast.Load()),
                            args=[ast.Constant(sid), node.left, node.comparators[0]], keywords=[])
        ast.copy_location(call, node)
        return call

    def visit_BoolOp(self, node):
        parent, field = self.parent, self.field
        root = not isinstance(parent, ast.BoolOp)
        if not root:
            return self.generic_visit(node)
        n = count_leaves(node)
        if not (2 <= n <= MAX_LEAVES):
            return self.generic_visit(node)
        leaves = []
        desc = tree_desc(node, leaves)
        ctx = "T" if truth_context(parent, field) else "V"
        func = self.func_depth > 0
        # leaf 0 runs first in every arrangement of the operators: it never needs the checked evaluation
        srcs = [ast.unparse(l) if func and i > 0 and pure_syntax(l) else None for i, l in enumerate(leaves)]
        sid = len(self.bsites)
        self.bsites.append([self.rel, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, desc, ctx, srcs])
        if not self.phase2 or sid in SKIP_B:
            return self.generic_visit(node)
        if COUNT:
            node = self.generic_visit(node)
            return _call("__br_CB__", [ast.Constant(sid), node], node)
        self.inst["B"].append(sid)
        # pre-hooks: leaf 0 (starts the record), pure leaves, and every leaf after a pure one
        # (skipped pure leaves before it are evaluated there, in the state the mutant would see)
        pre = []
        seen_pure = False
        for i, s in enumerate(srcs):
            pre.append(i == 0 or s is not None or seen_pure)
            seen_pure = seen_pure or s is not None
        # transform the leaves (their own comparisons and nested trees) first
        new_leaves = []
        for l in leaves:
            self.parent, self.field = node, "values"
            new_leaves.append(self.visit(l))
        self.parent, self.field = parent, field
        it = iter(range(n))

        def rebuild(b):
            vals = []
            for v in b.values:
                if isinstance(v, ast.BoolOp):
                    vals.append(rebuild(v))
                else:
                    i = next(it)
                    lv = new_leaves[i]
                    if pre[i]:
                        lv = ast.BoolOp(op=ast.Or(), values=[_call("__br_P__", [ast.Constant(sid), ast.Constant(i)], v), lv])
                        ast.copy_location(lv, v)
                    vals.append(_call("__br_L__", [ast.Constant(sid), ast.Constant(i), lv], v))
            nb = ast.BoolOp(op=b.op, values=vals)
            ast.copy_location(nb, b)
            return nb

        return _call("__br_E__", [ast.Constant(sid), rebuild(node)], node)

    def visit_Lambda(self, node):
        return self.generic_visit(node)


def wrap_module(tree):
    body = tree.body
    head = []
    i = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        head.append(body[0])
        i = 1
    while i < len(body) and isinstance(body[i], ast.ImportFrom) and body[i].module == "__future__":
        head.append(body[i])
        i += 1
    rest = body[i:]
    anchor = rest[0] if rest else (head[-1] if head else None)

    def call(d):
        e = ast.Expr(ast.Call(ast.Name("__br_imp__", ast.Load()), [ast.Constant(d)], []))
        if anchor is not None:
            ast.copy_location(e, anchor)
            ast.copy_location(e.value, anchor)
        return e

    if not rest:
        rest = [ast.Pass()]
        if anchor is not None:
            ast.copy_location(rest[0], anchor)
    tr = ast.Try(body=rest, handlers=[], orelse=[], finalbody=[call(-1)])
    if anchor is not None:
        ast.copy_location(tr, anchor)
    tree.body = head + [call(1), tr]
    ast.fix_missing_locations(tree)


def main():
    import re

    args = sys.argv[1:]
    repo = os.path.abspath(args[0])
    group = None
    cmp = "--no-cmp" not in args
    if "--group" in args:
        group = re.compile(args[args.index("--group") + 1])
    global COUNT
    COUNT = "--count" in args
    if "--skip" in args:
        with open(args[args.index("--skip") + 1]) as f:
            sk = json.load(f)
        SKIP_K.update(sk["K"])
        SKIP_B.update(sk["B"])
    sites = []
    bsites = []
    inst = {"K": [], "B": [], "group": group.pattern if group else None, "cmp": cmp}
    files = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(repo, "sqlglot")):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                files.append(os.path.join(dirpath, fn))
    for path in files:
        rel = os.path.relpath(path, repo)
        with open(path, "rb") as f:
            data = f.read()
        st = os.stat(path)
        tree = ast.parse(data, filename=path)
        phase2 = "--no-phase2" not in args and (group is None or bool(group.search(rel)))
        tree = Instrument(rel, sites, bsites, phase2=phase2, cmp=cmp, inst=inst).visit(tree)
        wrap_module(tree)
        code = compile(tree, path, "exec", dont_inherit=True)
        pyc = importlib.util.cache_from_source(path)
        os.makedirs(os.path.dirname(pyc), exist_ok=True)
        with open(pyc, "wb") as f:
            f.write(be._code_to_timestamp_pyc(code, int(st.st_mtime), st.st_size))
        os.chmod(pyc, 0o644)
    with open(os.path.join(repo, "_infect_sites.json"), "w") as f:
        json.dump(sites, f)
    with open(os.path.join(repo, "_infect_bsites.json"), "w") as f:
        json.dump(bsites, f)
    with open(os.path.join(repo, "_infect_inst.json"), "w") as f:
        json.dump(inst, f)
    print("instrumented", len(files), "files,", len(sites), "compare sites,", len(bsites), "and/or sites;",
          "probed:", len(inst["K"]), "constant comparisons,", len(inst["B"]), "and/or trees", file=sys.stderr)


if __name__ == "__main__":
    main()
