"""Builds maps/cone.json: for each sqlglot module X whose importer cone can be imported again on
its own (see cone.py), the cone in import order. Runs on the pinned tree, in the image.

The cone of X: X and the modules that import it at module level, directly or through others.
The import graph comes from the source (import statements outside functions and outside
'if TYPE_CHECKING:'), plus one edge for every name that a module imports from a package whose
value is defined in another module (a lazily loaded name: 'from sqlglot.optimizer import
optimize' loads sqlglot.optimizer.optimizer).

X is listed only when, on the pinned tree, all of these hold:
 1. cone.reimport() of the cone succeeds (the run-time checks pass);
 2. the full import-state fingerprint after it equals the pinned one, key by key and in order;
 3. the fingerprint of the cone modules alone (what the selector computes) equals the pinned
    one for those keys, in order;
 4. after a garbage collection, no class or function of the old cone modules is still alive
    (nothing outside the cone holds one that the new import did not replace);
 5. a fresh import of every module outside the cone with the cone modules made unimportable
    gives the same fingerprint for the keys outside the cone, apart from entries that hold an
    object of the cone (registries, package attributes, lazily cached names). So nothing
    outside the cone reads the cone at import time, and the cone writes nothing else into the
    state outside it.

usage: python build_cone.py OUT.json [MAX_CONE]   (cwd and sys.path as for the selector)"""
from __future__ import annotations

import ast
import copy
import gc
import importlib
import json
import os
import sys
import time
import types
import weakref

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.environ.get("BR_ROOT", "/repo")


def module_name(rel):
    m = rel[:-3].replace("/", ".")
    return m[: -len(".__init__")] if m.endswith(".__init__") else m


def source_modules(root):
    out = {}
    for d, _, files in os.walk(os.path.join(root, "sqlglot")):
        for f in files:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(d, f), root)
                out[module_name(rel)] = rel
    return out


def _type_checking(test):
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def static_imports(tree, mod, is_pkg, mods):
    """(modules imported at module level, [(package, name)] of 'from package import name')."""
    out, froms = set(), []
    pkg = mod if is_pkg else mod.rpartition(".")[0]

    def add(name):
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            if ".".join(parts[:i]) in mods:
                out.add(".".join(parts[:i]))

    def visit(body):
        for st in body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(st, ast.If) and _type_checking(st.test):
                visit(st.orelse)
                continue
            if isinstance(st, ast.Import):
                for a in st.names:
                    add(a.name)
            elif isinstance(st, ast.ImportFrom):
                if st.level:
                    base = pkg.split(".")
                    base = base[: len(base) - (st.level - 1)]
                    src = ".".join(base + ([st.module] if st.module else []))
                else:
                    src = st.module or ""
                add(src)
                for a in st.names:
                    if src + "." + a.name in mods:
                        out.add(src + "." + a.name)
                    elif src in mods:
                        froms.append((src, a.name))
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(st, field, None)
                if isinstance(sub, list):
                    visit(sub)
            for h in getattr(st, "handlers", None) or ():
                visit(h.body)
            if isinstance(st, ast.Match):
                for case in st.cases:
                    visit(case.body)

    visit(tree.body)
    out.discard(mod)
    return out, froms


def import_graph(root):
    mods = source_modules(root)
    deps, froms = {}, {}
    for m, rel in mods.items():
        with open(os.path.join(root, rel)) as f:
            tree = ast.parse(f.read())
        d, fr = static_imports(tree, m, rel.endswith("__init__.py"), mods)
        parts = m.split(".")
        for i in range(1, len(parts)):
            d.add(".".join(parts[:i]))  # importing a.b.c runs the packages a and a.b first
        deps[m], froms[m] = d, fr
    return mods, deps, froms


def in_child(fn, *args):
    """Runs fn(*args) in a forked child; returns its JSON result."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            out = fn(*args)
        except BaseException as e:  # noqa: BLE001
            import traceback

            out = {"error": repr(e), "trace": traceback.format_exc()[-1500:]}
        data = json.dumps(out).encode()
        while data:
            data = data[os.write(w, data):]
        os._exit(0)
    os.close(w)
    chunks = []
    while True:
        c = os.read(r, 1 << 16)
        if not c:
            break
        chunks.append(c)
    os.close(r)
    os.waitpid(pid, 0)
    return json.loads(b"".join(chunks) or b"null")


def value_edges(froms):
    """Module -> modules that define the names it imports from packages (in a child: a lookup
    can load and cache a lazy name)."""
    import cone

    out = {}
    for m, pairs in froms.items():
        for src, name in pairs:
            pkg = sys.modules.get(src)
            if pkg is None:
                continue
            try:
                v = getattr(pkg, name)
            except Exception:  # noqa: BLE001
                continue
            owner = cone.named(v) if not isinstance(v, types.ModuleType) else v.__name__
            if owner and owner != src and owner != m:
                out.setdefault(m, []).append(owner)
    return out


def cones(deps):
    rev = {}
    for m, d in deps.items():
        for x in d:
            rev.setdefault(x, set()).add(m)
    out = {}
    for m in deps:
        seen, stack = {m}, [m]
        while stack:
            for y in rev.get(stack.pop(), ()):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        out[m] = seen
    return out


def key_list(k):
    return [str(x) for x in k]


def check_cone(x, cone_list, base):
    """Checks 1-4 in this (forked) child of the pinned import."""
    import cone
    import statefp

    cset = set(cone_list)
    refs = []
    for m in cone_list:
        mod = sys.modules[m]
        refs.append(weakref.ref(mod))
        stack = list(vars(mod).values())
        seen = set()
        while stack:
            o = stack.pop()
            if id(o) in seen:
                continue
            seen.add(id(o))
            if (isinstance(o, type) or type(o) is types.FunctionType) and getattr(o, "__module__", None) in cset:
                refs.append(weakref.ref(o))
                if isinstance(o, type):
                    stack.extend(o.__dict__.values())
            elif isinstance(o, (staticmethod, classmethod)):
                stack.append(o.__func__)
            elif isinstance(o, dict):
                stack.extend(o.values())
            elif isinstance(o, (list, tuple, set, frozenset)):
                stack.extend(o)
    mod = stack = seen = o = None
    t0 = time.process_time()
    try:
        touched = cone.reimport(cone_list, lambda: None)
    except cone.Redo as e:
        return {"ok": False, "why": "reimport: " + str(e)}
    t1 = time.process_time()
    # what the selector's child computes (with the record of the parent still in memory)
    only = cset | touched
    part, _ = statefp.snapshot(None, None, only)
    t2 = time.process_time()
    want = [k for k in base["keys"] if k[1] in only]
    if list(part) != want or any(part[k] != base["fp"][k] for k in want):
        extra = [key_list(k) for k in part if k not in base["fp"]][:10]
        diff = [key_list(k) for k in want if part.get(k) != base["fp"][k]][:10]
        return {"ok": False, "why": "cone fingerprint differs", "extra": extra, "keys": diff}
    # The full fingerprint, without the record: it holds a reference to every object outside
    # the cone, and marshal (statefp's fast path) writes an object that has more than one
    # reference differently.
    cone.HEAP = None
    part = None
    gc.collect()
    new, _ = statefp.snapshot()
    keys = list(new)
    if keys != base["keys"] or [new[k] for k in keys] != [base["fp"][k] for k in keys]:
        d = statefp.diff(base["fp"], new)
        order = keys != base["keys"]
        return {"ok": False, "why": "full fingerprint differs" + (" (order)" if order else ""),
                "keys": [key_list(k) for k in d[:20]]}
    t3 = t2
    new = None
    gc.collect()
    alive = [repr(r())[:100] for r in refs if r() is not None]
    if alive:
        return {"ok": False, "why": "old objects alive", "alive": alive[:10], "n": len(alive)}
    return {"ok": True, "reimport": t1 - t0, "partial": t2 - t1}


def blocked_import(cone_list, order, base_noncone):
    """Check 5 (in a child that has not imported sqlglot): imports the modules outside the cone
    with the cone unimportable; returns {key: fingerprint by a fresh Fingerprinter} for the
    keys outside the cone that differ from the pinned ones, or an error."""
    import statefp

    cset = set(cone_list)

    class Block:
        def find_spec(self, name, path=None, target=None):
            if name in cset:
                raise ImportError("blocked: " + name)
            return None

    sys.meta_path.insert(0, Block())
    for m in order:
        if m in cset or m in sys.modules:
            continue
        try:
            importlib.import_module(m)
        except BaseException as e:  # noqa: BLE001
            return {"error": "import of %s fails: %r" % (m, e)}
    new, _ = statefp.snapshot()
    out = {}
    for k in set(new) | set(base_noncone):
        if k[1] in cset:
            continue
        if new.get(k) != base_noncone.get(k):
            v = locate(k)
            out[json.dumps(key_list(k))] = None if v is _GONE else statefp.Fingerprinter().fp(v)
    return {"diff": out}


_GONE = object()


def locate(k):
    """The value that a statefp key names in the current import, or _GONE."""
    m = sys.modules.get(k[1])
    if m is None:
        return _GONE
    if k[0] == "G":
        return vars(m).get(k[2], _GONE)
    if k[0] == "O" and len(k) == 4:
        holder = vars(m).get(k[2], _GONE)
        try:
            return _GONE if holder is _GONE else getattr(holder, k[3])
        except AttributeError:
            return _GONE
    obj = m
    path = k[2].split(".")
    for part in path:
        d = vars(obj) if isinstance(obj, types.ModuleType) else getattr(obj, "__dict__", {})
        obj = d.get(part, _GONE)
        if obj is _GONE:
            return _GONE
    if k[0] == "A":
        return obj.__dict__.get(k[3], _GONE) if isinstance(obj, type) else _GONE
    if k[0] == "O":
        holder = obj.__dict__.get(k[3], _GONE) if isinstance(obj, type) else _GONE
        if holder is _GONE:
            return _GONE
        try:
            return getattr(holder, k[4])
        except AttributeError:
            return _GONE
    return _GONE


def filtered_fp(keys, cone_list):
    """In the pinned import: for each key, the fingerprint of its value with the entries that
    hold an object of the cone left out (None when the value itself is such an object)."""
    import cone
    import statefp

    cset = set(cone_list)
    cone.HEAP = None
    gc.collect()
    out = {}
    for ks in keys:
        k = tuple(json.loads(ks))
        v = locate(k)
        if v is _GONE:
            out[ks] = "gone"
            continue
        if cone.named(v) in cset:
            out[ks] = None
            continue
        if isinstance(v, dict):
            c = copy.copy(v)
            for a in [a for a, b in v.items() if cone.named(b) in cset]:
                del c[a]
            v = c
        elif isinstance(v, (list, set)):
            v = type(v)(b for b in v if cone.named(b) not in cset)
        out[ks] = statefp.Fingerprinter().fp(v)
    return out


def lazy_entry(ks, blocked_value):
    """A name that a package with a module-level __getattr__ caches when it is first read (a
    lazily loaded dialect or optimizer rule): absent when the cone is not imported, because a
    cone module read it first. Its value is an object outside the cone, the same either way."""
    k = json.loads(ks)
    if k[0] != "G" or blocked_value is not None:
        return False
    m = sys.modules.get(k[1])
    return m is not None and "__getattr__" in vars(m) and k[2] in vars(m)


def main():
    out_path = sys.argv[1]
    max_cone = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    import preload

    preload.install_environment(ROOT)
    mods, deps, froms = import_graph(ROOT)
    # a process with the environment but without sqlglot, for check 5
    clean_r, clean_w = os.pipe()
    resp_r, resp_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(clean_w)
        os.close(resp_r)
        server(clean_r, resp_w)
        os._exit(0)
    os.close(clean_r)
    os.close(resp_w)
    clean = (clean_w, os.fdopen(resp_r, "rb"))

    preload.import_sqlglot()
    import cone
    import statefp

    extra = in_child(value_edges, froms)
    for m, owners in extra.items():
        deps[m].update(owners)
    reach = cones(deps)
    base_fp, f = statefp.snapshot()
    del f
    base = {"keys": list(base_fp), "fp": base_fp}
    order = [m for m in sys.modules if m == "sqlglot" or m.startswith("sqlglot.")]
    missing = set(mods) - set(order)
    walk = deps_first(order, deps)
    pre = in_child(test_imports, ROOT)
    cone.record(walk)
    result, failed = {}, {}
    targets = only or [m for m in order if m in reach]
    # A target whose state depends on the order of its import with other modules fails check 2
    # ("cone fingerprint differs": hive copies TypeAnnotator.COERCES_TO, which bigquery changes
    # in place later). The cones of these modules then join its cone, and the checks run again.
    with_mods = [m for m in os.environ.get(
        "BR_CONE_WITH", "sqlglot.dialects.bigquery,sqlglot.optimizer.annotate_types").split(",") if m]
    for x in targets:
        members = set(reach[x])
        joined = False
        why = None
        for attempt in range(6):
            cone_list = [m for m in order if m in members]
            why, grow = check_module(x, cone_list, members, base, base_fp, order, missing, max_cone, clean)
            if why is not None and not grow and not joined and why.get("why") == "cone fingerprint differs":
                joined = True
                for m in with_mods:
                    members |= reach.get(m, {m})
                continue
            if why is None or not grow:
                break
            # state of these modules depends on the cone: they (and their importers) join it
            for m in grow:
                members |= reach.get(m, {m})
        if why is not None:
            failed[x] = why
            continue
        result[x] = cone_list
        print(x, len(cone_list), "ok", file=sys.stderr, flush=True)
    for x, r in failed.items():
        print(x, "FAILED", json.dumps(r)[:300], file=sys.stderr, flush=True)
    os.close(clean[0])
    with open(out_path, "w") as fh:
        json.dump({"version": 1, "walk": walk, "test_imports": pre, "modules": result, "failed": {k: (v or {}).get("why") for k, v in failed.items()}},
                  fh, indent=0, sort_keys=True)


def deps_first(order, deps):
    """The modules with every module after the modules it imports (ties: import order)."""
    out, done = [], set()
    pos = {m: i for i, m in enumerate(order)}

    def visit(m, stack):
        if m in done or m in stack:
            return
        stack.add(m)
        for d in sorted(deps.get(m, ()), key=lambda x: pos.get(x, 1 << 30)):
            if d in pos:
                visit(d, stack)
        stack.discard(m)
        done.add(m)
        out.append(m)

    for m in order:
        visit(m, set())
    return out


def test_imports(root):
    """The modules outside sqlglot and tests that importing every test module loads, in order
    (the selector's parent imports them before the cone children, which import the tests)."""
    import unittest

    before = set(sys.modules)
    names = []
    for d, _, files in os.walk(os.path.join(root, "tests")):
        for f in sorted(files):
            if f.startswith("test_") and f.endswith(".py"):
                names.append(module_name(os.path.relpath(os.path.join(d, f), root)))
    for name in sorted(names):
        try:
            importlib.import_module(name)
        except unittest.SkipTest:
            pass
        except BaseException:  # noqa: BLE001
            pass
    return [m for m in sys.modules if m not in before and m.partition(".")[0] not in ("sqlglot", "tests")]


def check_module(x, cone_list, members, base, base_fp, order, missing, max_cone, clean):
    """(None, None) when checks 1-5 pass; else (reason, modules outside the cone whose state
    depends on it, to add to the cone)."""
    if len(cone_list) > max_cone:
        return {"why": "cone of %d modules" % len(cone_list)}, None
    if members & missing:
        return {"why": "cone has modules that the import does not load"}, None
    r = in_child(check_cone, x, cone_list, base)
    if not r or not r.get("ok"):
        return r, None
    noncone = {k: v for k, v in base_fp.items() if k[1] not in members}
    b = ask(clean, (cone_list, order, noncone))
    if "error" in b:
        return {"why": "blocked import: " + b["error"]}, None
    if b["diff"]:
        want = in_child(filtered_fp, list(b["diff"]), cone_list)
        bad = [k for k, v in b["diff"].items() if not (want.get(k) is None and v is None) and want.get(k) != v
               and not lazy_entry(k, v)]
        if bad:
            owners = {json.loads(k)[1] for k in bad}
            return {"why": "state outside the cone depends on it", "keys": bad[:10]}, owners
    return None, None


def ask(clean, args):
    import pickle

    w, r = clean
    data = pickle.dumps(args)
    os.write(w, len(data).to_bytes(8, "little"))
    while data:
        data = data[os.write(w, data):]
    n = int.from_bytes(r.read(8), "little")
    return json.loads(r.read(n))


def server(rfd, wfd):
    """Forks one child per request from a process that has not imported sqlglot."""
    import pickle

    rf = os.fdopen(rfd, "rb")
    while True:
        head = rf.read(8)
        if len(head) < 8:
            return
        args = pickle.loads(rf.read(int.from_bytes(head, "little")))
        res = json.dumps(in_child(blocked_import, *args)).encode()
        os.write(wfd, len(res).to_bytes(8, "little"))
        while res:
            res = res[os.write(wfd, res):]


if __name__ == "__main__":
    main()
