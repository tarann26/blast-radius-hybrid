"""Instruments a copy of the pinned sqlglot tree with reach probes, for maps/infect_hidden.bin.

usage (inside the pinned python image): python instrument_reach.py REPO MAPS_DIR
REPO is a copy of the pinned checkout (copied with preserved modification times); MAPS_DIR
holds infect.bin and infect2.bin. The .py sources are not changed: for every sqlglot/**/*.py
the script writes an instrumented .pyc into __pycache__ (valid for the unchanged source, so
line numbers, inspect.getsource and tracebacks stay as they are).

A "mapped site" is a comparison or an and/or tree that has a row in infect.bin or infect2.bin
(sites that those maps left out, e.g. the ones that run more than 1e6 times, get no probe).
For every mapped site:

  * its REGION is the smallest statement that holds the site (for a site in the test of an
    if / while, the iterable of a for, a with item, a decorator or a default value: that
    compound statement, whose header runs first), or the lambda whose body holds the site.
    The statement "__br_R__(region)" is put in front of the region's statement in the same
    block ("elif x:" becomes "else: __br_R__(n); if x:", which runs the same); for a lambda
    the body B becomes "__br_R__(region) or B".
  * the site expression E itself becomes "__br_S__(site) or E".

__br_R__ and __br_S__ take one literal number and return None, so "probe or E" is E: evaluated
in the same frame, at the same time, with the same value. Nothing of the program is evaluated
by a probe.

Each module body also counts how many sqlglot module bodies are executing (as in the outcome
map build), so that import-time records can be told apart.
Writes REPO/_reach_sites.json: {"regions": [[file, kind, first line, last line]],
"sites": [[file, line, col, end line, end col, "C" | "B", region number or -1]]}.
"""
import ast
import importlib._bootstrap_external as be
import importlib.util
import json
import marshal
import os
import sys
import zlib


def mapped_sites(maps_dir):
    """{file: {(line, col, end line, end col): "C" | "B"}} of the rows of the two outcome maps."""
    out = {}
    with open(os.path.join(maps_dir, "infect.bin"), "rb") as f:
        m1 = marshal.loads(f.read())
    for rel, blob in m1["files"].items():
        for row in marshal.loads(zlib.decompress(blob)):
            out.setdefault(rel, {})[tuple(row[:4])] = "C"
    with open(os.path.join(maps_dir, "infect2.bin"), "rb") as f:
        m2 = marshal.loads(f.read())
    for rel, blob in m2["kfiles"].items():
        for row in marshal.loads(zlib.decompress(blob)):
            out.setdefault(rel, {})[tuple(row[:4])] = "C"
    for rel, blob in m2["bfiles"].items():
        for row in marshal.loads(zlib.decompress(blob)):
            out.setdefault(rel, {})[tuple(row[:4])] = "B"
    return out


def _call(name, number, loc):
    c = ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=[ast.Constant(number)], keywords=[])
    ast.copy_location(c, loc)
    ast.copy_location(c.func, loc)
    ast.copy_location(c.args[0], loc)
    return c


def _or(call, expr):
    b = ast.BoolOp(op=ast.Or(), values=[call, expr])
    ast.copy_location(b, expr)
    return b


def find_region(node, parent, field):
    """("L", lambda) when the site is in a lambda body, else ("S", smallest statement), or None."""
    child = node
    p = parent.get(id(child))
    while p is not None:
        if isinstance(p, ast.Lambda) and field.get(id(child)) == "body":
            return "L", p
        if isinstance(p, ast.stmt):
            return "S", p
        child = p
        p = parent.get(id(child))
    return None


class Wrap(ast.NodeTransformer):
    def __init__(self, site_of, lambda_region):
        self.site_of = site_of  # id(node) -> site number
        self.lambda_region = lambda_region  # id(lambda node) -> region number

    def visit(self, node):
        node = self.generic_visit(node)
        r = self.lambda_region.get(id(node))
        if r is not None:
            node.body = _or(_call("__br_R__", r, node.body), node.body)
        s = self.site_of.get(id(node))
        if s is not None:
            return _or(_call("__br_S__", s, node), node)
        return node


def instrument(tree, rel, keys, regions, sites):
    parent, field = {}, {}
    nodes = list(ast.walk(tree))
    for n in nodes:
        for f, v in ast.iter_fields(n):
            for c in (v if isinstance(v, list) else [v]):
                if isinstance(c, ast.AST):
                    parent[id(c)] = n
                    field[id(c)] = f
    site_of, lambda_region, stmt_region = {}, {}, {}
    for n in nodes:
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            kind = "C"
        elif isinstance(n, ast.BoolOp) and not isinstance(parent.get(id(n)), ast.BoolOp):
            kind = "B"
        else:
            continue
        key = (n.lineno, n.col_offset, n.end_lineno, n.end_col_offset)
        if keys.get(key) != kind:
            continue
        reg = find_region(n, parent, field)
        rid = -1
        if reg is not None:
            table = lambda_region if reg[0] == "L" else stmt_region
            rid = table.get(id(reg[1]))
            if rid is None:
                rid = table[id(reg[1])] = len(regions)
                regions.append([rel, reg[0], reg[1].lineno, reg[1].end_lineno])
        site_of[id(n)] = len(sites)
        sites.append([rel, key[0], key[1], key[2], key[3], kind, rid])
    found = len(site_of)
    tree = Wrap(site_of, lambda_region).visit(tree)
    # the probe statements, in front of each region statement in its own block
    for n in list(ast.walk(tree)):
        for f, v in ast.iter_fields(n):
            if isinstance(v, list) and any(id(c) in stmt_region for c in v):
                new = []
                for c in v:
                    r = stmt_region.get(id(c))
                    if r is not None:
                        e = ast.Expr(_call("__br_R__", r, c))
                        ast.copy_location(e, c)
                        new.append(e)
                    new.append(c)
                v[:] = new
    return tree, found


def wrap_module(tree):
    """The module body runs between __br_imp__(1) and __br_imp__(-1) (same as the outcome map build)."""
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
    mapped = mapped_sites(sys.argv[2])
    regions, sites = [], []
    files = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(repo, "sqlglot")):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                files.append(os.path.join(dirpath, fn))
    missing = 0
    for path in files:
        rel = os.path.relpath(path, repo)
        with open(path, "rb") as f:
            data = f.read()
        st = os.stat(path)
        tree = ast.parse(data, filename=path)
        keys = mapped.get(rel, {})
        tree, found = instrument(tree, rel, keys, regions, sites)
        missing += len(keys) - found
        wrap_module(tree)
        code = compile(tree, path, "exec", dont_inherit=True)
        pyc = importlib.util.cache_from_source(path)
        os.makedirs(os.path.dirname(pyc), exist_ok=True)
        with open(pyc, "wb") as f:
            f.write(be._code_to_timestamp_pyc(code, int(st.st_mtime), st.st_size))
        os.chmod(pyc, 0o644)
    with open(os.path.join(repo, "_reach_sites.json"), "w") as f:
        json.dump({"regions": regions, "sites": sites}, f)
    print("instrumented", len(files), "files:", len(sites), "mapped sites in", len(regions), "regions",
          "(%d lambda bodies);" % sum(1 for r in regions if r[1] == "L"),
          "sites without a region:", sum(1 for s in sites if s[6] < 0), "; map rows not found in the source:", missing, file=sys.stderr)
    if missing:
        raise SystemExit("a mapped site was not found in the source: wrong checkout or maps")


if __name__ == "__main__":
    main()
