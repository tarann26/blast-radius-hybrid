"""Def-time candidate map for sqlglot @ eb11aec.

For every line of a module-level or class-level statement of sqlglot/**/*.py (and every function
line that runs while the parent imports sqlglot and the test modules, "kind F"), the set of tests
whose outcome can change when that line changes.

usage (inside br-label):
  docker run --rm --network none --cpuset-cpus 0-3 -e SKIP_INTEGRATION=0 -e PYTHONPATH=/repo \
    -v /root/project/work/sqlglot:/repo:ro -v /root/project/work/trace:/trace:ro \
    -v /root/project/work/rt:/rt:ro -v /root/project/work/deftime:/out \
    -v /root/project/work/units/raw:/units:ro -v /root/project/work/fullcov/raw:/fullcov:ro \
    -v /root/project/nap-time/sample/tests.json:/data/tests.json:ro -w /repo br-label \
    python /out/build_defmap.py [/out/defmap.pkl]
  (/units and /fullcov are optional: direct-runner coverage, merged like rt/build_maps.py does)

Model (see README section of the final report):
  * value-flow graph between def-time "targets" (class attribute K.NAME, module global M.NAME) and
    def-time statements that read them (class bodies, module statements, import statements);
  * function lines run at import ("parent executed") carry the def-time statement(s) on whose
    behalf they ran ("origins", from a PY_START monitor over the same import as build_maps);
  * a class attribute target K.NAME is consumed at call time by tests that use a class S <= K whose
    source-level MRO lookup of NAME lands in K (usage from agg["classes"], with dialect bundles)
    AND execute a function line that reads .NAME (or the string "NAME");
  * a module global M.NAME at call time: tests that execute a function line of M that reads NAME
    bare, or any line reading <module M>.NAME, or (name-based) any .NAME;
  * import-time reads propagate (fixpoint) to the consumers of the reading statement; reads done
    by import-time function code propagate to the consumers of that code's origins; when the
    origin is a class statement (metaclass / __init_subclass__), the consumers are the users of
    that class, its subclasses and its dialect bundle, but only if the class (or a bundle member)
    resolves NAME to K;
  * dict entry keyed by an Expression class exp.X: consumers of the dict restricted to tests that
    instantiate X or a subclass of X;
  * function header (decorators, def line, parameter defaults): tests that execute any line of the
    function (including nested closures) plus the consumers of its import-time origins;
  * everything else (control-flow headers at module level, class headers, docstrings, imports
    that cannot be resolved): all tests that import the module.
"""
from __future__ import annotations

import ast
import builtins
import glob
import json
import os
import pickle
import sys
import time
import types

sys.path.insert(0, "/rt")
T0 = time.time()
ROOT = "/repo"
CO_OPT = 0x1


def log(*a):
    print("[%.1fs]" % (time.time() - T0), *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------
# 1. import everything under a PY_START monitor: origins of import-time function code


ORIGINS = {}  # code -> set of (rel file, lineno) | None
CTX = {}  # (code, origin) -> set of classes of the first argument of the repo frames between them


def _on_start(code, off):
    fn = code.co_filename
    if not (code.co_flags & CO_OPT) or not fn.startswith("/repo/"):
        return sys.monitoring.DISABLE
    f = sys._getframe(1)
    o = None
    acls = []
    test_frame = None
    while f is not None:
        c = f.f_code
        if c.co_filename.startswith("/repo/"):
            if not (c.co_flags & CO_OPT):
                o = (c.co_filename[6:], f.f_lineno)
                break
            if c.co_filename.startswith("/repo/tests/"):
                test_frame = (c.co_filename[6:], c.co_firstlineno)  # e.g. a setUpClass run by the parent
            if c.co_argcount:
                try:
                    v = f.f_locals.get(c.co_varnames[0])
                    acls.append(v if isinstance(v, type) else type(v))
                except Exception:
                    pass
        f = f.f_back
    if o is None and test_frame is not None:
        o = test_frame
    s = ORIGINS.get(code)
    if s is None:
        s = ORIGINS[code] = set()
    s.add(o)
    k = (code, o)
    cs = CTX.get(k)
    if cs is None:
        cs = CTX[k] = set()
    cs.update(acls)


def run_imports():
    mon = sys.monitoring
    mon.use_tool_id(4, "deforig")
    mon.register_callback(4, mon.events.PY_START, _on_start)
    mon.set_events(4, mon.events.PY_START)
    import build_maps

    pex = build_maps.parent_executed_lines(ROOT)
    mon.set_events(4, 0)
    mon.free_tool_id(4)
    return build_maps, pex


# ---------------------------------------------------------------------------------------------
# helpers


class Bits:
    """test-number sets as python ints"""

    @staticmethod
    def of(it):
        b = 0
        for t in it:
            b |= 1 << t
        return b

    @staticmethod
    def to_list(b):
        out = []
        i = 0
        while b:
            if b & 1:
                out.append(i)
            b >>= 1
            i += 1
        return out


def bits_to_list(b, n):
    s = bin(b)[2:][::-1]
    return [i for i, ch in enumerate(s) if ch == "1"]


def is_ident(s):
    return isinstance(s, str) and s.isidentifier()


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/out/defmap.pkl"
    build_maps, pex = run_imports()
    log("imports done; origin codes", len(ORIGINS), "pex lines", sum(len(v) for v in pex.values()))

    agg = pickle.load(open("/trace/agg.pkl", "rb"))
    tests = json.load(open("/data/tests.json"))["tests"]
    NT = len(tests)
    ALL = (1 << NT) - 1
    call = agg["call"]
    # coverage of the direct-runner traces, merged as rt/build_maps.py does
    import gzip

    added = 0
    for d in ("/units", "/fullcov"):
        for p in glob.glob(os.path.join(d, "*.json.gz")):
            r = json.load(gzip.open(p))
            number = r["number"]
            covs = [r.get("cov") or {}, r.get("base") or {}]
            covs += [u.get("cov") or {} for u in r.get("units", [])]
            covs += [c for _, c in r.get("warm", [])]
            for cv in covs:
                for f, lines in cv.items():
                    fd = call.setdefault(f, {})
                    for ln in lines:
                        cur = fd.get(ln)
                        if cur is None:
                            fd[ln] = [number]
                            added += 1
                        elif number not in cur:
                            cur.append(number)
                            added += 1
    log("coverage pairs added from direct-runner traces:", added)
    cov_cache = {}

    def cov(rel, ln):
        k = (rel, ln)
        b = cov_cache.get(k)
        if b is None:
            t = call.get(rel, {}).get(ln)
            b = cov_cache[k] = Bits.of(t) if t else 0
        return b

    importers_cache = {}

    def importers(rel):
        b = importers_cache.get(rel)
        if b is None:
            s = set(agg["import_def_tests"].get(rel, ()))
            for t in agg["lazy_def"].get(rel, {}).values():
                s.update(t)
            b = importers_cache[rel] = Bits.of(s)
        return b

    # -----------------------------------------------------------------------------------------
    # 2. modules and classes

    mod_by_file = {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if f and f.startswith("/repo/") and f.endswith(".py"):
            mod_by_file[f[6:]] = m
    file_by_mod = {id(m): f for f, m in mod_by_file.items()}

    def in_repo_mod(modname):
        return isinstance(modname, str) and (modname.startswith("sqlglot") or modname.startswith("tests"))

    all_classes = set()
    stack = []
    for m in mod_by_file.values():
        for v in list(vars(m).values()):
            if isinstance(v, type) and in_repo_mod(getattr(v, "__module__", None)):
                stack.append(v)
    while stack:
        c = stack.pop()
        if c in all_classes:
            continue
        all_classes.add(c)
        for v in list(vars(c).values()):
            if isinstance(v, type) and in_repo_mod(getattr(v, "__module__", None)):
                stack.append(v)
        try:
            subs = type.__subclasses__(c)
        except TypeError:
            subs = []
        for s in subs:
            if in_repo_mod(getattr(s, "__module__", None)):
                stack.append(s)
    from sqlglot.dialects.dialect import Dialect
    from sqlglot.expressions import Expr

    log("classes", len(all_classes))

    def ckey(c):
        return "%s.%s" % (c.__module__, c.__qualname__)

    subclasses = {c: set() for c in all_classes}
    for c in all_classes:
        for b in c.__mro__:
            if b in subclasses:
                subclasses[b].add(c)

    # tests per class key
    cls_tests = {k: Bits.of(v) for k, v in agg["classes"].items()}
    # defgap: the first tracer ran the tests with real spawn pools, so the classes used inside pool
    # workers (tests/test_executor.py _mp_execute, tests/test_optimizer.py check_file) are missing
    # there. Merge the classes seen by the direct runner (in-process pool: worker work counts for
    # the submitting test): /classesfull = whole-test runs recorded like the first tracer
    # (defgap/rt_trace/trace_units.py BR_CLASSES_FULL), /classes = per-part __init__ trace.
    cls_added = 0
    for d in ("/classesfull", "/classes"):
        for p in glob.glob(os.path.join(d, "*.json.gz")):
            r = json.load(gzip.open(p))
            number = r["number"]
            names = set(r.get("classes") or ())
            names.update(r.get("base") or ())
            for u in r.get("units") or ():
                if isinstance(u, list):
                    names.update(u)
            bit = 1 << number
            for k in names:
                cur = cls_tests.get(k, 0)
                if not cur & bit:
                    cls_tests[k] = cur | bit
                    cls_added += 1
    log("class usage pairs added from direct-runner traces:", cls_added)
    known_keys = {ckey(c) for c in all_classes}
    unknown_usage = 0  # classes made at test time (local subclasses): they can inherit anything
    for k, b in cls_tests.items():
        if k not in known_keys:
            unknown_usage |= b
    log("unknown class keys", sum(1 for k in cls_tests if k not in known_keys), "tests", bin(unknown_usage).count("1"))

    bundles = {}  # dialect -> set of bundle classes
    member_of = {}
    for c in all_classes:
        if isinstance(c, type) and issubclass(c, Dialect):
            bs = set()
            for a in ("tokenizer_class", "jsonpath_tokenizer_class", "parser_class", "generator_class"):
                v = getattr(c, a, None)
                if isinstance(v, type):
                    bs.add(v)
            bundles[c] = bs
            for v in bs:
                member_of.setdefault(v, set()).add(c)

    key_count = {}
    for c in all_classes:
        key_count[ckey(c)] = key_count.get(ckey(c), 0) + 1
    plain_usage = {c: cls_tests.get(ckey(c), 0) for c in all_classes}
    ambiguous = {c for c in all_classes if key_count[ckey(c)] > 1}
    log("ambiguous class keys", len({ckey(c) for c in ambiguous}), "classes", len(ambiguous))
    for c in ambiguous:
        # e.g. the Tokenizer classes that _Dialect.__new__ makes with type(): one trace key for all;
        # a test that used dialect D's instance of it also made a D instance
        own = 0
        for d in member_of.get(c, ()):
            own |= cls_tests.get(ckey(d), 0)
        # no owner: a class of a dialect made and dropped at import time (kept alive only by CTX);
        # dialects made at test time are in unknown_usage
        plain_usage[c] &= own
    usage = {}
    for c in all_classes:
        b = plain_usage[c]
        for d in member_of.get(c, ()):  # a test that uses dialect d can use d's bundle classes
            b |= plain_usage.get(d, 0)
        if c in bundles:  # and the reverse, for members that belong to this dialect only
            for v in bundles[c]:
                if len(member_of.get(v, ())) == 1:
                    b |= plain_usage.get(v, 0)
        usage[c] = b

    if os.environ.get("DEFMAP_Q"):
        for c in all_classes:
            if ckey(c).endswith(os.environ["DEFMAP_Q"]):
                log("Q", ckey(c), "usage", bin(usage[c]).count("1"))
                for S in subclasses[c]:
                    log("   sub", ckey(S), id(S), "plain", bin(plain_usage[S]).count("1"), "usage", bin(usage[S]).count("1"),
                        "member_of", [ckey(d) for d in member_of.get(S, ())][:5], S in ambiguous)

    # -----------------------------------------------------------------------------------------
    # 3. AST of every file: def-time statements, reads, function-line reads

    paths = sorted(
        p[len(ROOT) + 1:]
        for pat in ("sqlglot/**/*.py", "tests/**/*.py")
        for p in glob.glob(os.path.join(ROOT, pat), recursive=True)
    )

    class Stmt:
        __slots__ = ("id", "file", "kind", "span", "cls", "targets", "func_span", "node", "scope_obj", "note")

    stmts = []
    line_stmt = {}  # (file, line) -> stmt id
    src_names = {}  # class obj -> set of names assigned in its body (source)
    class_obj_of = {}  # (file, qualname tuple) -> class obj
    class_node = {}  # class obj -> (file, ClassDef)

    def real_class(rel, path):
        m = mod_by_file.get(rel)
        if m is None:
            return None
        o = m
        for p in path:
            o = vars(o).get(p) if isinstance(o, (type, types.ModuleType)) else None
            if o is None:
                return None
        return o if isinstance(o, type) else None

    def body_start(stmts_):
        s0 = stmts_[0]
        ln = s0.lineno
        d = getattr(s0, "decorator_list", None)
        if d:
            ln = min(ln, min(x.lineno for x in d))
        return ln

    def class_body_names(node):
        names = set()

        def visit(body):
            for s in body:
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(s.name)
                elif isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    tg = s.targets if isinstance(s, ast.Assign) else [s.target]
                    for t in tg:
                        for n in ast.walk(t):
                            if isinstance(n, ast.Name):
                                names.add(n.id)
                elif isinstance(s, (ast.Import, ast.ImportFrom)):
                    for a in s.names:
                        names.add((a.asname or a.name).split(".")[0])
                for fld in ("body", "orelse", "finalbody", "handlers"):
                    sub = getattr(s, fld, None)
                    if sub and not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if fld == "handlers":
                            for h in sub:
                                visit(h.body)
                        else:
                            visit(sub)

        visit(node.body)
        return names

    def new_stmt(rel, kind, a, b, path, scope_obj, node=None, func_span=None):
        s = Stmt()
        s.id = len(stmts)
        s.file, s.kind, s.span, s.cls, s.node = rel, kind, (a, b), tuple(path), node
        s.scope_obj, s.func_span, s.targets, s.note = scope_obj, func_span, [], ""
        stmts.append(s)
        for ln in range(a, b + 1):
            k = (rel, ln)
            old = line_stmt.get(k)
            if old is None or (stmts[old].span[1] - stmts[old].span[0]) >= (b - a):
                line_stmt[k] = s.id
        return s

    trees = {}

    def walk_body(rel, body, path, scope_obj, parent_range=None):
        for s in body:
            start = s.lineno
            decos = getattr(s, "decorator_list", None)
            if decos:
                start = min(start, min(d.lineno for d in decos))
            if isinstance(s, ast.ClassDef):
                bs = body_start(s.body)
                hdr_end = max(s.lineno, bs - 1)
                obj = real_class(rel, list(path) + [s.name])
                new_stmt(rel, "classhdr", start, hdr_end, path, scope_obj, s)
                if obj is not None:
                    class_obj_of[(rel, tuple(path) + (s.name,))] = obj
                    class_node[obj] = (rel, s)
                    src_names[obj] = class_body_names(s)
                walk_body(rel, s.body, list(path) + [s.name], obj if obj is not None else False)
            elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bs = body_start(s.body)
                hdr_end = max(s.lineno, bs - 1)
                new_stmt(rel, "funchdr", start, hdr_end, path, scope_obj, s, (start, s.end_lineno))
            elif isinstance(s, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)) or (
                hasattr(ast, "TryStar") and isinstance(s, ast.TryStar)
            ):
                subs = []
                for fld in ("body", "orelse", "finalbody"):
                    sub = getattr(s, fld, None)
                    if sub:
                        subs.append(sub)
                for h in getattr(s, "handlers", []) or []:
                    subs.append(h.body)
                # header lines: everything in the compound statement not covered by a sub statement
                ctl = new_stmt(rel, "control", s.lineno, s.end_lineno, path, scope_obj, s)
                for sub in subs:
                    walk_body(rel, sub, path, scope_obj)
            else:
                new_stmt(rel, "simple", start, s.end_lineno, path, scope_obj, s)

    for rel in paths:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        trees[rel] = tree
        walk_body(rel, tree.body, [], None)
    log("stmts", len(stmts))

    # source-level attribute resolution
    def resolve_src(c, name):
        for b in c.__mro__:
            sn = src_names.get(b)
            if sn is not None:
                if name in sn:
                    return b
            elif not in_repo_mod(getattr(b, "__module__", None)):
                if name in vars(b):
                    return None  # builtin / third-party base: not a source target
            # repo class without a source body (made by type()): nothing defined there
        return None

    # name evaluation in a scope
    def eval_chain(node, rel, scope_obj):
        """value of a Name/Attribute chain, or raises"""
        if isinstance(node, ast.Name):
            if scope_obj:
                c = scope_obj
                if node.id in vars(c):
                    return vars(c)[node.id]
            m = mod_by_file.get(rel)
            if m is not None and node.id in vars(m):
                return vars(m)[node.id]
            if hasattr(builtins, node.id):
                return getattr(builtins, node.id)
            raise LookupError(node.id)
        if isinstance(node, ast.Attribute):
            return getattr(eval_chain(node.value, rel, scope_obj), node.attr)
        raise LookupError("expr")

    def classify_attr(base_obj, attr):
        """-> list of read keys for reading base_obj.attr"""
        if isinstance(base_obj, types.ModuleType):
            f = file_by_mod.get(id(base_obj))
            if f:
                return [("glob", f, attr)]
            return []
        if isinstance(base_obj, type):
            if in_repo_mod(getattr(base_obj, "__module__", None)):
                r = resolve_src(base_obj, attr)
                if r is not None:
                    return [("cls", r, attr)]
                return [("any", attr)]
            return []
        return []

    # targets registry, reads registry
    readers_of = {}  # read key -> set of stmt ids (def-time reads)
    any_readers = {}  # attr name -> stmt ids reading .attr on something unresolved
    star_readers = {}  # file -> stmt ids importing * from file

    def add_read(sid, key):
        if key[0] == "any":
            any_readers.setdefault(key[1], set()).add(sid)
        else:
            readers_of.setdefault(key, set()).add(sid)

    def collect_reads(sid, node, rel, scope_obj, cls_names):
        """def-time reads of an expression (lambda / nested function bodies excluded)"""
        todo = [node]
        while todo:
            n = todo.pop()
            if isinstance(n, ast.Lambda):
                for d in n.args.defaults + [x for x in n.args.kw_defaults if x is not None]:
                    todo.append(d)
                continue
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
                try:
                    base = eval_chain(n.value, rel, scope_obj)
                    keys = classify_attr(base, n.attr)
                except Exception:
                    keys = [("any", n.attr)]
                for k in keys:
                    add_read(sid, k)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if scope_obj and n.id in cls_names:
                    add_read(sid, ("cls", scope_obj, n.id))
                add_read(sid, ("glob", rel, n.id))
            elif isinstance(n, ast.Constant) and is_ident(n.value):
                any_readers.setdefault(n.value, set()).add(sid)
            todo.extend(ast.iter_child_nodes(n))

    def target_of(node, rel, scope_obj, cls_names):
        if isinstance(node, ast.Name):
            if scope_obj:
                return [("cls", scope_obj, node.id)]
            if scope_obj is False:
                return []
            return [("glob", rel, node.id)]
        if isinstance(node, (ast.Tuple, ast.List)):
            r = []
            for e in node.elts:
                r += target_of(e, rel, scope_obj, cls_names)
            return r
        if isinstance(node, ast.Starred):
            return target_of(node.value, rel, scope_obj, cls_names)
        if isinstance(node, ast.Subscript):
            return target_of(node.value, rel, scope_obj, cls_names) if isinstance(
                node.value, (ast.Name, ast.Attribute, ast.Subscript)) else []
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and scope_obj and node.value.id in cls_names:
                pass
            try:
                base = eval_chain(node.value, rel, scope_obj)
            except Exception:
                return []
            if isinstance(base, types.ModuleType):
                f = file_by_mod.get(id(base))
                return [("glob", f, node.attr)] if f else []
            if isinstance(base, type):
                r = resolve_src(base, node.attr)
                return [("cls", r or base, node.attr)]
            # attribute of a value object (dict.update ...): the object is what the chain reads
            return target_of(node.value, rel, scope_obj, cls_names)
        return []

    def module_name_of(rel):
        p = rel[:-3]
        if p.endswith("/__init__"):
            p = p[: -len("/__init__")]
        return p.replace("/", ".")

    def file_of_modname(name):
        m = sys.modules.get(name)
        if m is not None:
            return file_by_mod.get(id(m))
        return None

    for s in stmts:
        rel, node = s.file, s.node
        scope_obj = s.scope_obj
        cls_names = src_names.get(scope_obj, set()) if scope_obj else set()
        if scope_obj is False:
            s.note = "class not found"
            continue
        if s.kind == "simple":
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    s.targets += target_of(t, rel, scope_obj, cls_names)
                    if isinstance(t, (ast.Subscript, ast.Attribute)):
                        collect_reads(s.id, t, rel, scope_obj, cls_names)
                collect_reads(s.id, node.value, rel, scope_obj, cls_names)
            elif isinstance(node, ast.AnnAssign):
                if node.value is not None:
                    s.targets += target_of(node.target, rel, scope_obj, cls_names)
                    collect_reads(s.id, node.value, rel, scope_obj, cls_names)
                else:
                    s.note = "annotation only"
            elif isinstance(node, ast.AugAssign):
                s.targets += target_of(node.target, rel, scope_obj, cls_names)
                collect_reads(s.id, node.target, rel, scope_obj, cls_names)
                collect_reads(s.id, node.value, rel, scope_obj, cls_names)
            elif isinstance(node, ast.Expr):
                v = node.value
                if isinstance(v, ast.Constant):
                    s.note = "docstring"
                elif isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute):
                    s.targets += target_of(v.func.value, rel, scope_obj, cls_names)
                    collect_reads(s.id, v, rel, scope_obj, cls_names)
                    s.note = "method call"
                else:
                    collect_reads(s.id, v, rel, scope_obj, cls_names)
                    s.note = "expr"
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    pkg = module_name_of(rel)
                    if not rel.endswith("__init__.py"):
                        pkg = pkg.rsplit(".", 1)[0]
                    for _ in range(node.level - 1):
                        pkg = pkg.rsplit(".", 1)[0]
                    base = pkg + ("." + node.module if node.module else "")
                else:
                    base = node.module
                src_file = file_of_modname(base)
                for a in node.names:
                    if a.name == "*":
                        if src_file:
                            star_readers.setdefault(src_file, set()).add(s.id)
                        s.targets.append(("star", rel))
                        continue
                    nm = a.asname or a.name
                    s.targets += target_of(ast.Name(id=nm), rel, scope_obj, cls_names)
                    if src_file:
                        add_read(s.id, ("glob", src_file, a.name))
            elif isinstance(node, ast.Import):
                for a in node.names:
                    nm = (a.asname or a.name).split(".")[0]
                    s.targets += target_of(ast.Name(id=nm), rel, scope_obj, cls_names)
                s.note = "import"
            else:
                s.note = type(node).__name__
        elif s.kind == "funchdr":
            # the decorators and defaults are evaluated at def time
            for d in node.decorator_list:
                collect_reads(s.id, d, rel, scope_obj, cls_names)
            for d in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                collect_reads(s.id, d, rel, scope_obj, cls_names)
    log("reads", len(readers_of), "any", len(any_readers))

    # function-line reads (call time) and parent-executed line reads
    rd_attr = {}  # attr / string -> bits of tests executing a function line reading it
    rd_glob = {}  # (file, name) -> bits
    pex_lines_attr = {}  # attr -> set of (file, line) parent-executed lines reading .attr / "attr"
    pex_lines_glob = {}  # (file, name) -> set of (file, line)
    pex_set = {(f, l) for f, ls in pex.items() for l in ls}

    for rel, tree in trees.items():
        m = mod_by_file.get(rel)
        mg = vars(m) if m is not None else {}

        def scan_func(fn):
            for n in ast.walk(fn):
                ln = getattr(n, "lineno", None)
                if ln is None:
                    continue
                lines = range(n.lineno, (n.end_lineno or n.lineno) + 1)
                if isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Load, ast.Del)):
                    keys_attr = [n.attr]
                    keys_glob = []
                    if isinstance(n.value, ast.Name):
                        bo = mg.get(n.value.id)
                        if isinstance(bo, types.ModuleType):
                            f = file_by_mod.get(id(bo))
                            if f:
                                keys_glob.append((f, n.attr))
                elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    keys_attr = []
                    keys_glob = [(rel, n.id)]
                elif isinstance(n, ast.Constant) and is_ident(n.value):
                    keys_attr = [n.value]
                    keys_glob = []
                else:
                    continue
                b = 0
                for l in lines:
                    b |= cov(rel, l)
                    if (rel, l) in pex_set:
                        for k in keys_attr:
                            pex_lines_attr.setdefault(k, set()).add((rel, l))
                        for k in keys_glob:
                            pex_lines_glob.setdefault(k, set()).add((rel, l))
                if b:
                    for k in keys_attr:
                        rd_attr[k] = rd_attr.get(k, 0) | b
                    for k in keys_glob:
                        rd_glob[k] = rd_glob.get(k, 0) | b

        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                # only the outermost functions (ast.walk visits nested ones again; harmless but slow)
                pass
        seen = set()
        todo = [tree]
        while todo:
            n = todo.pop()
            for c in ast.iter_child_nodes(n):
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scan_func(ast.Module(body=c.body, type_ignores=[]))
                elif isinstance(c, ast.Lambda):
                    scan_func(c.body)
                else:
                    todo.append(c)
    log("function reads indexed", len(rd_attr), len(rd_glob))

    # origins per parent-executed line, and the classes active between line and origin
    line_origins = {}
    line_ctx = {}  # ((rel, line), origin) -> set of classes
    for code, os_ in ORIGINS.items():
        rel = code.co_filename[6:]
        ls = {l for _, _, l in code.co_lines() if l is not None}
        for l in ls:
            line_origins.setdefault((rel, l), set()).update(os_)
            for o in os_:
                line_ctx.setdefault(((rel, l), o), set()).update(CTX.get((code, o), ()))

    def ctx_relevant(acls, K, name):
        """can a read of .name, done with these classes active, read K's definition of name?"""
        repo = [c for c in acls if c in subclasses]
        if not repo:
            return True
        subK = subclasses.get(K, ())
        for S in repo:
            if S in subK and resolve_src(S, name) is K:
                return True
            for B in bundles.get(S, ()):
                if B in subK and resolve_src(B, name) is K:
                    return True
            for D in member_of.get(S, ()):
                if D in subK and resolve_src(D, name) is K:
                    return True
        return False

    # -----------------------------------------------------------------------------------------
    # 4. consumers: fixpoint over targets and statements

    def U_src(K, name):
        b = 0
        for S in subclasses.get(K, (K,)):
            if S is K or resolve_src(S, name) is K:
                b |= usage.get(S, 0)
        return b | unknown_usage

    def U_all(C):
        b = 0
        for S in subclasses.get(C, (C,)):
            b |= usage.get(S, 0)
            for v in bundles.get(S, ()):
                b |= usage.get(v, 0)
        return b | unknown_usage

    def origin_contrib(o, line_file):
        """-> (bits, stmt ids) for one origin of import-time function code"""
        if o is None:
            return importers(line_file), ()
        rel, ln = o
        if rel.startswith("tests/"):
            return importers(rel), ()
        sid = line_stmt.get((rel, ln))
        if sid is None:
            return importers(rel), ()
        st = stmts[sid]
        if st.kind == "classhdr":
            return None, sid  # caller decides
        return 0, (sid,)

    stmt_cls = {}

    def hdr_class(sid):
        st = stmts[sid]
        if sid not in stmt_cls:
            stmt_cls[sid] = class_obj_of.get((st.file, st.cls + (st.node.name,)))
        return stmt_cls[sid]

    # base bits and edges
    base = {}  # node -> bits ; node = ("t", key) | ("s", sid)
    edges = {}  # node -> set(nodes) whose consumers flow into node

    def add_edge(a, b):
        edges.setdefault(a, set()).add(b)

    def origins_into(node, lines, filt_K=None, name=None):
        """for parent-executed lines: add consumers of their origins to node"""
        bits = 0
        for (lf, ll) in lines:
            for o in line_origins.get((lf, ll), (None,)):
                if filt_K is not None and not ctx_relevant(line_ctx.get(((lf, ll), o), ()), filt_K, name):
                    continue
                b, sids = origin_contrib(o, lf)
                if b is None:
                    C = hdr_class(sids)
                    if C is None:
                        bits |= importers(stmts[sids].file)
                        continue
                    if filt_K is not None:
                        cands = [C] + list(bundles.get(C, ()))
                        ok = any(
                            isinstance(B, type) and B in subclasses.get(filt_K, ()) and resolve_src(B, name) is filt_K
                            for B in cands
                        )
                        if not ok:
                            continue
                    bits |= U_all(C)
                else:
                    bits |= b
                    for sid in sids:
                        add_edge(node, ("s", sid))
        return bits

    target_nodes = set()
    for s in stmts:
        for t in s.targets:
            target_nodes.add(t)
    log("targets", len(target_nodes))

    for t in target_nodes:
        node = ("t", t)
        if t[0] == "cls":
            _, K, name = t
            b = U_src(K, name) & rd_attr.get(name, 0)
            for sid in readers_of.get(t, ()):
                add_edge(node, ("s", sid))
            for sid in any_readers.get(name, ()):
                add_edge(node, ("s", sid))
            b |= origins_into(node, pex_lines_attr.get(name, ()), K, name)
        elif t[0] == "glob":
            _, f, name = t
            b = rd_glob.get((f, name), 0)
            b |= rd_attr.get(name, 0) & importers(f)  # name-based attribute reads (module.attr)
            for sid in readers_of.get(t, ()):
                add_edge(node, ("s", sid))
            for sid in star_readers.get(f, ()):
                add_edge(node, ("s", sid))
            lines = set(pex_lines_glob.get((f, name), ()))
            b |= origins_into(node, lines)
            # module-attribute access resolved by name only (from parent-executed code)
        elif t[0] == "star":
            b = importers(t[1])
        else:
            b = 0
        base[node] = b

    info_stmt = {}
    for s in stmts:
        node = ("s", s.id)
        if s.file.startswith("tests/"):
            base[node] = importers(s.file)
            info_stmt[s.id] = "test-module"
            continue
        if s.kind == "simple" and s.targets:
            base[node] = 0
            for t in s.targets:
                add_edge(node, ("t", t))
            info_stmt[s.id] = "targets:" + ",".join(
                (t[1].__qualname__ + "." + t[2]) if t[0] == "cls" else (t[2] if t[0] == "glob" else t[0]) for t in s.targets[:3]
            )
        elif s.kind == "funchdr":
            a, e = s.func_span
            b = 0
            lines = []
            for l in range(a, e + 1):
                b |= cov(s.file, l)
                if (s.file, l) in pex_set:
                    lines.append((s.file, l))
            b |= origins_into(node, lines)
            base[node] = b
            info_stmt[s.id] = "funchdr"
        else:
            base[node] = importers(s.file)
            info_stmt[s.id] = "fallback:" + (s.kind if s.kind != "simple" else (s.note or "no-target"))

    # debug dump: readable node names, base sizes, edges
    if os.environ.get("DEFMAP_DEBUG"):
        def nname(n):
            if n[0] == "t":
                t = n[1]
                if t[0] == "cls":
                    return "T:%s.%s" % (ckey(t[1]), t[2])
                return "T:%s" % (t[1:],)
            s = stmts[n[1]]
            return "S:%s:%d-%d:%s" % (s.file, s.span[0], s.span[1], s.kind)
        DBG = {"base": {}, "edges": {}}
    # fixpoint
    val = dict(base)
    order = list(val.keys())
    changed = True
    it = 0
    while changed:
        changed = False
        it += 1
        for n in order:
            v = val[n]
            for m in edges.get(n, ()):
                v |= val.get(m, 0)
            if v != val[n]:
                val[n] = v
                changed = True
    log("fixpoint iterations", it)
    if os.environ.get("DEFMAP_DEBUG"):
        for n in val:
            DBG["base"][nname(n)] = (bin(base.get(n, 0)).count("1"), bin(val[n]).count("1"))
        for n, ms in edges.items():
            DBG["edges"][nname(n)] = sorted(nname(m) for m in ms)
        DBG["rd_attr"] = {k: bin(v).count("1") for k, v in rd_attr.items()}
        DBG["rd_attr_bits"] = dict(rd_attr)
        DBG["rd_glob_bits"] = {"%s:%s" % k: v for k, v in rd_glob.items()}
        ub = {}
        for c, b in usage.items():
            ub[ckey(c)] = ub.get(ckey(c), 0) | b
        DBG["usage_bits"] = ub
        DBG["unknown_usage"] = unknown_usage
        DBG["pex_attr"] = {k: sorted(v) for k, v in pex_lines_attr.items()}
        DBG["origins"] = {"%s:%d" % k: sorted(map(str, v)) for k, v in line_origins.items()}
        pickle.dump(DBG, open("/out/debug.pkl", "wb"))

    # -----------------------------------------------------------------------------------------
    # 5. per-line output

    def expr_class_key(knode, rel, scope_obj):
        try:
            o = eval_chain(knode, rel, scope_obj)
        except Exception:
            return None
        if isinstance(o, type) and issubclass(o, Expr):
            return o
        return None

    def U_plain_sub(X):
        b = 0
        for S in subclasses.get(X, (X,)):
            b |= plain_usage.get(S, 0)
        return b | unknown_usage

    lines_out = {}
    info_out = {}
    intern = {}

    def put(rel, ln, bits, why):
        lst = intern.get(bits)
        if lst is None:
            lst = intern[bits] = bits_to_list(bits, NT)
        lines_out.setdefault(rel, {})[ln] = lst
        info_out.setdefault(rel, {})[ln] = why

    import functools

    def code_lines_bits(code):
        b = 0
        todo = [code]
        while todo:
            c = todo.pop()
            rel = c.co_filename[6:] if c.co_filename.startswith("/repo/") else None
            if rel is None:
                return None
            for _, _, l in c.co_lines():
                if l is not None:
                    b |= cov(rel, l)
            todo.extend(x for x in c.co_consts if isinstance(x, types.CodeType))
        return b

    def callable_bits(v):
        """tests that can call v (a dispatch-table value), or None when v is not a plain callable"""
        if isinstance(v, functools.partial):
            v = v.func
        if isinstance(v, types.MethodType):
            v = v.__func__
        if isinstance(v, (classmethod, staticmethod)):
            v = v.__func__
        if isinstance(v, types.FunctionType):
            return code_lines_bits(v.__code__)
        if isinstance(v, type) and v in subclasses:
            return U_plain_sub(v)
        return None

    def key_value(knode, rel, scope_obj):
        if isinstance(knode, ast.Constant):
            return True, knode.value
        try:
            return True, eval_chain(knode, rel, scope_obj)
        except Exception:
            return False, None

    def walk_dicts(node, path, out):
        if isinstance(node, ast.Dict):
            out.append((node, path))
            for k, x in zip(node.keys, node.values):
                if k is not None and isinstance(x, (ast.Dict, ast.BinOp)):
                    walk_dicts(x, path + [k], out)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            walk_dicts(node.left, path, out)
            walk_dicts(node.right, path, out)
        elif (
            isinstance(node, ast.DictComp)
            and len(node.generators) == 1
            and isinstance(node.key, ast.Name)
            and isinstance(node.value, ast.Name)
            and isinstance(node.generators[0].target, ast.Tuple)
            and [getattr(e, "id", None) for e in node.generators[0].target.elts] == [node.key.id, node.value.id]
            and isinstance(node.generators[0].iter, ast.Call)
            and isinstance(node.generators[0].iter.func, ast.Attribute)
            and node.generators[0].iter.func.attr == "items"
        ):
            # {k: v for k, v in {...}.items() if ...}: a filtered copy, same key -> same value
            walk_dicts(node.generators[0].iter.func.value, path, out)

    def entries_of(s):
        """[(first line, last line, refine bits, label)] for the dict entries of a def-time assignment"""
        node = s.node
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
        elif isinstance(node, ast.AugAssign):
            value = node.value
        else:
            return []
        if value is None or not s.targets:
            return []
        t = s.targets[0]
        top = None
        if len(s.targets) == 1:
            try:
                if t[0] == "cls":
                    top = vars(t[1]).get(t[2])
                elif t[0] == "glob":
                    top = vars(mod_by_file[t[1]]).get(t[2])
            except Exception:
                top = None
        dicts = []
        walk_dicts(value, [], dicts)
        out = []
        for d, path in dicts:
            obj = top
            for k in path:
                ok, kv = key_value(k, s.file, s.scope_obj or None)
                try:
                    obj = obj.get(kv) if ok and isinstance(obj, dict) else None
                except TypeError:
                    obj = None
            for k, x in zip(d.keys, d.values):
                if k is None:
                    continue
                bits = ALL
                label = []
                X = expr_class_key(k, s.file, s.scope_obj or None)
                ok, kv = key_value(k, s.file, s.scope_obj or None)
                cb = None
                if ok and isinstance(obj, dict) and not isinstance(x, (ast.Dict, ast.BinOp)):
                    try:
                        present = kv in obj
                    except TypeError:
                        present = False
                    if present:
                        cb = callable_bits(obj[kv])
                        if cb is not None:
                            bits &= cb
                            label.append("callable")
                # an Expression-class key restricts the entry to tests that make X instances only when
                # the entry is a handler (callable) or a per-node metadata dict: a flat flag such as
                # SET_OP_DISTINCT_BY_DEFAULT[exp.Union] is read before the node exists
                if X is not None and (cb is not None or isinstance(x, (ast.Dict, ast.BinOp))):
                    bits &= U_plain_sub(X)
                    label.append("key=" + X.__name__)
                if label:
                    out.append((k.lineno, x.end_lineno, bits, ",".join(label)))
        return out

    for s in stmts:
        if not s.file.startswith("sqlglot/"):
            continue
        v = val[("s", s.id)]
        why = info_stmt.get(s.id, "?")
        entry_spans = entries_of(s) if s.kind == "simple" else []
        a, b = s.span
        for ln in range(a, b + 1):
            if line_stmt.get((s.file, ln)) != s.id:
                continue
            bits, w = v, why
            if entry_spans:
                own = [e for e in entry_spans if e[0] <= ln <= e[1]]
                if own:
                    # the innermost refined entry; the line must not hold a part of another entry
                    e = min(own, key=lambda e: e[1] - e[0])
                    others = [o for o in entry_spans if o is not e and o[0] <= ln <= o[1] and not (o[0] <= e[0] and e[1] <= o[1])]
                    if not others:
                        bits = v & e[2]
                        w = why + "|entry:" + e[3]
            put(s.file, ln, bits, w)

    # kind F: parent-executed function lines
    spans_cache = {}
    nF = 0
    for rel, ls in pex.items():
        if not rel.startswith("sqlglot/"):
            continue
        tree = trees.get(rel)
        if tree is None:
            continue
        if rel not in spans_cache:
            n = len(open(os.path.join(ROOT, rel), encoding="utf-8").read().split("\n"))
            spans_cache[rel] = build_maps.line_spans(tree, n)
        sp = spans_cache[rel]
        for ln in ls:
            node = ("F", rel, ln)
            span = sp[ln] if ln < len(sp) and sp[ln] else (ln, ln)
            b = 0
            for l in range(span[0], span[1] + 1):
                b |= cov(rel, l)
            for l in range(span[0], span[1] + 1):
                if (rel, l) not in pex_set:
                    continue
                for o in line_origins.get((rel, l), (None,)):
                    bb, sids = origin_contrib(o, rel)
                    if bb is None:
                        C = hdr_class(sids)
                        b |= U_all(C) if C is not None else importers(stmts[sids].file)
                    else:
                        b |= bb
                        for sid in sids:
                            b |= val[("s", sid)]
            # every line of the statement that holds a parent-executed line
            for l in range(span[0], span[1] + 1):
                prev = lines_out.get(rel, {}).get(l)
                w = "F"
                if prev is not None:
                    b2 = b | Bits.of(prev)
                    if not info_out[rel][l].startswith("F"):
                        w = "F+" + info_out[rel][l]
                else:
                    b2 = b
                put(rel, l, b2, w)
            nF += 1
    log("F lines", nF)

    nlines = sum(len(v) for v in lines_out.values())
    log("lines", nlines, "distinct sets", len(intern))
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"lines": lines_out, "info": info_out}, f, protocol=4)
    os.replace(tmp, out_path)
    log("wrote", out_path)


if __name__ == "__main__":
    main()
