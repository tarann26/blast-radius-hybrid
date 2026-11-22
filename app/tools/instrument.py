"""Instruments a copy of the pinned sqlglot tree for the operand outcome map.

usage (inside the pinned python image): python instrument.py REPO
REPO is a copy of the pinned checkout (copied with preserved modification times). The .py
sources are not changed: for every sqlglot/**/*.py the script writes an instrumented .pyc into
__pycache__ (valid for the unchanged source, so line numbers, inspect.getsource and tracebacks
stay as they are). Every comparison with exactly one operator ("a < b", not "a < b < c") is
compiled as a call __br_<Op>__(site, a, b) of the probe in _brprobe.py, which returns the
original result. Each module body also counts how many sqlglot module bodies are executing,
so that import-time evaluations can be told apart. Comparisons inside annotations are left
alone (not mapped). Writes REPO/_infect_sites.json: per site id [file, lineno, col_offset,
end_lineno, end_col_offset, operator] of the Compare node in the pinned source.
"""
import ast
import importlib._bootstrap_external as be
import importlib.util
import json
import os
import sys

SKIP_FIELDS = {"annotation", "returns", "type_params", "type_comment"}


class Instrument(ast.NodeTransformer):
    def __init__(self, rel, sites):
        self.rel = rel
        self.sites = sites

    def generic_visit(self, node):
        for field, old in ast.iter_fields(node):
            if field in SKIP_FIELDS:
                continue
            if isinstance(old, list):
                new = []
                for v in old:
                    if isinstance(v, ast.AST):
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
        return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if len(node.ops) != 1:
            return node  # chained comparison: not a mapped site
        op = type(node.ops[0]).__name__
        sid = len(self.sites)
        self.sites.append([self.rel, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, op])
        call = ast.Call(
            func=ast.Name(id="__br_%s__" % op, ctx=ast.Load()),
            args=[ast.Constant(sid), node.left, node.comparators[0]],
            keywords=[],
        )
        ast.copy_location(call, node)
        return call


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
    repo = os.path.abspath(sys.argv[1])
    sites = []
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
        tree = Instrument(rel, sites).visit(tree)
        wrap_module(tree)
        code = compile(tree, path, "exec", dont_inherit=True)
        pyc = importlib.util.cache_from_source(path)
        os.makedirs(os.path.dirname(pyc), exist_ok=True)
        with open(pyc, "wb") as f:
            f.write(be._code_to_timestamp_pyc(code, int(st.st_mtime), st.st_size))
        os.chmod(pyc, 0o644)
    with open(os.path.join(repo, "_infect_sites.json"), "w") as f:
        json.dump(sites, f)
    print("instrumented", len(files), "files,", len(sites), "sites", file=sys.stderr)


if __name__ == "__main__":
    main()
