"""Builds the maps that the selector reads at run time. Everything comes from the pinned commit:
per-test line coverage (trace/agg.pkl), the statement structure of each source file, and the
set of lines that already ran when the parent process finished its imports.

usage (inside the pinned image): python build_maps.py /repo /trace/agg.pkl /data/tests.json OUT_DIR
"""
from __future__ import annotations

import ast
import glob
import marshal
import os
import pickle
import zlib
import sys
from array import array

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preload
import stmtmap

CO_OPTIMIZED = 0x1


def parent_executed_lines(root):
    """Function lines that run while the parent imports sqlglot and every test module."""
    hits = {}
    mon = sys.monitoring
    tool = 3
    mon.use_tool_id(tool, "brmaps")

    def on_line(code, line):
        if code.co_flags & CO_OPTIMIZED and code.co_filename.startswith(root):
            hits.setdefault(code.co_filename[len(root) + 1:], set()).add(line)
        return mon.DISABLE

    mon.register_callback(tool, mon.events.LINE, on_line)
    mon.set_events(tool, mon.events.LINE)
    preload.install_environment(root)
    preload.import_sqlglot()
    mods = sorted(p[:-3].replace("/", ".") for p in glob.glob("tests/**/test_*.py", recursive=True))
    preload.import_tests(mods)
    preload.build_class_fixtures(set(preload.SHARED_FIXTURES))
    mon.set_events(tool, 0)
    mon.free_tool_id(tool)
    return hits


def line_spans(tree, n_lines):
    """For each line, the smallest statement, statement header or lambda that holds it."""
    best = [None] * (n_lines + 2)

    def put(a, b):
        size = b - a
        for ln in range(a, b + 1):
            cur = best[ln]
            if cur is None or size < cur[1] - cur[0]:
                best[ln] = (a, b)

    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            put(node.lineno, node.end_lineno)
        elif isinstance(node, ast.stmt):
            start = node.lineno
            decos = getattr(node, "decorator_list", None)
            if decos:
                start = min(start, min(d.lineno for d in decos))
            put(start, node.end_lineno)
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.AST):
                first = body[0]
                fstart = first.lineno
                fd = getattr(first, "decorator_list", None)
                if fd:
                    fstart = min(fstart, min(d.lineno for d in fd))
                if fstart - 1 >= start:
                    put(start, fstart - 1)
        elif isinstance(node, ast.ExceptHandler):
            put(node.lineno, node.end_lineno)
            if node.body and node.body[0].lineno - 1 >= node.lineno:
                put(node.lineno, node.body[0].lineno - 1)
    return best


def main():
    root, agg_path, tests_path, out = sys.argv[1:5]
    root = os.path.abspath(root)
    os.makedirs(os.path.join(out, "f"), exist_ok=True)
    agg = pickle.load(open(agg_path, "rb"))
    parent = parent_executed_lines(root)

    sets = {(): 0}
    set_list = [()]

    def sid(items):
        key = tuple(sorted(set(items)))
        i = sets.get(key)
        if i is None:
            i = sets[key] = len(set_list)
            set_list.append(key)
        return i

    # candidate tests for lines that run at import time (see tools/build_defmap.py)
    defmap = {}
    if len(sys.argv) > 5 and os.path.exists(sys.argv[5]):
        defmap = pickle.load(open(sys.argv[5], "rb"))["lines"]
    call = agg["call"]
    # coverage measured with the direct runner (whole tests and their parts): the first
    # tracer could lose lines that ran in pool worker processes
    import gzip
    import json as _json

    extra_dirs = sys.argv[6:]
    merged = {}  # file -> line -> set of tests
    for d in extra_dirs:
        for p in glob.glob(os.path.join(d, "*.json.gz")):
            r = _json.load(gzip.open(p))
            number = r["number"]
            lines_of = {}
            covs = [r.get("cov") or {}, r.get("base") or {}]
            covs += [u.get("cov") or {} for u in r.get("units", [])]
            covs += [c for _, c in r.get("warm", [])]
            for cov in covs:
                for f, lines in cov.items():
                    lines_of.setdefault(f, set()).update(lines)
            for f, lines in lines_of.items():
                fd = merged.setdefault(f, {})
                for ln in lines:
                    fd.setdefault(ln, set()).add(number)
    added = 0
    for f, fd in merged.items():
        cd = call.setdefault(f, {})
        for ln, ts in fd.items():
            cur = cd.get(ln)
            if cur is None:
                cd[ln] = sorted(ts)
                added += len(ts)
            else:
                new = ts.difference(cur)
                if new:
                    cd[ln] = sorted(ts.union(cur))
                    added += len(new)
    print("coverage pairs added from direct-runner traces:", added)
    lazy_def = agg["lazy_def"]
    import_def_tests = agg["import_def_tests"]
    files = {}
    reads = {}  # attribute or variable name -> [(file, line, inside a function?)]

    def collect_reads(rel, tree):
        def visit(node, in_func):
            for child in ast.iter_child_nodes(node):
                inner = in_func or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                name = None
                if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Load):
                    name = child.attr
                elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    name = child.id
                elif isinstance(child, ast.Constant) and isinstance(child.value, str) and child.value.isidentifier():
                    name = child.value
                if name is not None:
                    reads.setdefault(name, []).append((rel, child.lineno, in_func))
                visit(child, inner)

        visit(tree, False)

    paths = sorted(
        p[len(root) + 1:]
        for pat in ("sqlglot/**/*.py", "tests/**/*.py")
        for p in glob.glob(os.path.join(root, pat), recursive=True)
    )
    trees = {}
    for rel in paths:
        trees[rel] = ast.parse(open(os.path.join(root, rel), encoding="utf-8").read())
        if rel.startswith("sqlglot/"):
            collect_reads(rel, trees[rel])
    fid_of = {rel: i for i, rel in enumerate(paths)}

    def class_constant_readers(tree, pex_by_file):
        """{line: ((file id, line), ...)} for the lines of class-level 'NAME = value' statements
        whose NAME is read only inside functions that do not run at import. A test part can
        see a change of such a value only when it runs one of those reader lines."""
        out = {}

        def walk(body):
            for node in body:
                if isinstance(node, ast.ClassDef):
                    for st in node.body:
                        target = None
                        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                            target = st.targets[0].id
                        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
                            target = st.target.id
                        if target is None or target.startswith("__"):
                            continue
                        rs = reads.get(target, ())
                        if not rs:
                            continue
                        ok = True
                        for rfile, rline, in_func in rs:
                            if not in_func or rline in pex_by_file.get(rfile, ()):
                                ok = False
                                break
                        if ok:
                            lines = tuple(sorted({(fid_of[rfile], rline) for rfile, rline, _ in rs}))
                            for ln in range(st.lineno, st.end_lineno + 1):
                                out[ln] = lines
                    walk(node.body)

        walk(tree.body)
        return out

    for fid, rel in enumerate(paths):
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        lines = src.split("\n")
        n = len(lines)
        tree = ast.parse(src)
        units, flags, inames = stmtmap.build_units(src)
        spans = line_spans(tree, n)
        cov = call.get(rel, {})
        pex = parent.get(rel, set())
        stmt_set = [0] * (n + 2)
        stmt_pex = bytearray(n + 2)
        span_a = [0] * (n + 2)
        span_b = [0] * (n + 2)
        cache = {}
        for ln in range(1, n + 1):
            sp = spans[ln]
            if sp is None:
                continue
            r = cache.get(sp)
            if r is None:
                acc = set()
                flag = 0
                for k in range(sp[0], sp[1] + 1):
                    t = cov.get(k)
                    if t:
                        acc.update(t)
                    if k in pex:
                        flag = 1
                r = cache[sp] = (sid(acc), flag)
            stmt_set[ln], stmt_pex[ln] = r
            span_a[ln], span_b[ln] = sp
        unit_set = []
        unit_pex = bytearray(len(units))
        for ui, u in enumerate(units):
            acc = set()
            for k in range(u[0], u[1] + 1):
                t = cov.get(k)
                if t:
                    acc.update(t)
                if k in pex:
                    unit_pex[ui] = 1
            unit_set.append(sid(acc))
        dm = defmap.get(rel, {})
        def_set = [0] * (n + 2)
        def_known = bytearray(n + 2)
        for ln, tset in dm.items():
            if 0 < ln <= n:
                def_set[ln] = sid(tset)
                def_known[ln] = 1
        importers = set(import_def_tests.get(rel, ()))
        for t in lazy_def.get(rel, {}).values():
            importers.update(t)
        file_set = set()
        for t in cov.values():
            file_set.update(t)
        rec = (
            [(u[0], u[1], tuple(u[2]), u[3]) for u in units],
            flags,
            tuple(inames),
            stmt_set,
            bytes(stmt_pex),
            unit_set,
            bytes(unit_pex),
            sid(importers),
            sid(file_set),
            span_a,
            span_b,
            def_set,
            bytes(def_known),
            class_constant_readers(tree, parent) if rel.startswith("sqlglot/") else {},
        )
        with open(os.path.join(out, "f", f"{fid}.bin"), "wb") as f:
            f.write(zlib.compress(marshal.dumps(rec), 9))
        files[rel] = fid

    import json

    tests = json.load(open(tests_path))["tests"]
    opened = {p: sid(t) for p, t in agg["opened"].items()}
    listed = {p: sid(t) for p, t in agg.get("listed", {}).items()}
    import reduce

    shapes = {}
    shape_cache = {}
    for t in tests:
        if t["id"] in reduce.SCALED:
            rel, cls_name, meth = t["id"].split("::")
            if rel not in shape_cache:
                shape_cache[rel] = reduce.file_shapes(open(os.path.join(root, rel), encoding="utf-8").read())
            sh = shape_cache[rel].get((cls_name, meth))
            if sh is not None:
                shapes[t["number"]] = (sh[0], sh[1], [tuple(x) for x in sh[2]])
    # tests that load each module (at import of the test file or later): they fail when the module breaks
    importers_of = {}
    for rel in paths:
        s = set(import_def_tests.get(rel, ()))
        for t in lazy_def.get(rel, {}).values():
            s.update(t)
        importers_of[rel] = sid(s)
    # tests that hand work to the class-level process pool of tests/test_optimizer.py
    pool_users = set()
    opt_src = open(os.path.join(root, "tests/test_optimizer.py"), encoding="utf-8").read().split("\n")
    for i, text in enumerate(opt_src, start=1):
        if "self.pool.submit(" in text:
            pool_users.update(call.get("tests/test_optimizer.py", {}).get(i, ()))
    meta = {
        "pool_users": sid(pool_users),
        "importers": importers_of,
        "shapes": shapes,
        "files": files,
        "opened": opened,
        "listed": listed,
        "test_ids": [t["id"] for t in tests],
    }
    with open(os.path.join(out, "meta.bin"), "wb") as f:
        marshal.dump(meta, f)
    # name -> the function lines that read it (attribute, variable, or the name as a string)
    readers = {}
    for name, rs in reads.items():
        lines = sorted({(fid_of[rfile], rline) for rfile, rline, in_func in rs if in_func})
        if lines:
            flat = array("I")
            for a, b in lines:
                flat.append(a)
                flat.append(b)
            readers[name] = flat.tobytes()
    with open(os.path.join(out, "readers.bin"), "wb") as f:
        f.write(zlib.compress(marshal.dumps(readers), 9))
    offsets = array("I", [0])
    blob = bytearray()
    for s in set_list:
        blob += array("H", s).tobytes()
        offsets.append(len(blob))
    with open(os.path.join(out, "sets.idx"), "wb") as f:
        f.write(offsets.tobytes())
    with open(os.path.join(out, "sets.bin"), "wb") as f:
        f.write(bytes(blob))
    print("files", len(files), "sets", len(set_list), "blob", len(blob), "parent files", len(parent),
          "parent lines", sum(len(v) for v in parent.values()))


if __name__ == "__main__":
    main()
