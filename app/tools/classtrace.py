"""Build-time tracer (pinned tree only): which class attributes does each test read?

It records, for every test, the pairs (receiver class, attribute name) that the test reads on
objects and on class objects of four class trees: the tokenizers (_TokenizerBase), the parsers
(Parser), the generators (Generator) and the dialects (Dialect). The receiver is the class of
the object (or the class object itself) through which the read happens, so a read reaches a
value that a class defines when that class is in the receiver's MRO. This is attribute-read
coverage of the pinned commit; nothing in it comes from patches or test outcomes.

How the reads are seen:
- objects: __getattribute__ on the root classes (inherited by every subclass);
- class objects: Tokenizer, Parser and Generator use the plain 'type' metaclass, whose slots
  cannot change, so a recording metaclass is put in at class creation (builtins.__build_class__)
  before sqlglot is imported; the dialects' own metaclass (_Dialect) gets the same method.
- every test runs cold, in a child forked from a parent that has imported sqlglot and only
  this test's module: caches that sqlglot fills on first use (generator._DISPATCH_CACHE) are
  empty again, as are the class fixtures, so setUpClass reads count for each test of the
  class. The sqlglot containers that the test module import changed are put back to their
  state after the sqlglot import.
- the reads during the import of a test file count for every test of that file.
- the process pools of the tests run in-process (pool.py), so the reads of their tasks count
  for the test that submits them.

usage (inside the pinned image br-label):
    python classtrace.py REPO TESTS_JSON OUT_DIR SHARD NSHARDS
"""
from __future__ import annotations

import builtins
import gzip
import importlib
import json
import os
import resource
import select
import signal
import sys
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REC = set()
_add = REC.add
_type_ga = type.__getattribute__
_object_ga = object.__getattribute__


class RecordingMeta(type):
    def __getattribute__(cls, name):
        if name[:2] != "__":
            _add((cls, name))
        return _type_ga(cls, name)


def _meta_ga(cls, name):
    if name[:2] != "__":
        _add((cls, name))
    return _type_ga(cls, name)


def _object_ga_rec(self, name):
    if name[:2] != "__":
        _add((type(self), name))
    return _object_ga(self, name)


ROOTS = {("sqlglot.tokens", "_TokenizerBase"), ("sqlglot.parser", "Parser"), ("sqlglot.generator", "Generator")}
_build_class = builtins.__build_class__


def _build_class_rec(func, name, *bases, **kw):
    if (func.__globals__.get("__name__"), name) in ROOTS and "metaclass" not in kw:
        kw["metaclass"] = RecordingMeta
    return _build_class(func, name, *bases, **kw)


def install_hooks():
    builtins.__build_class__ = _build_class_rec


def finish_hooks():
    import statefp
    from sqlglot.dialects.dialect import _Dialect

    builtins.__build_class__ = _build_class
    roots = statefp.root_classes()
    assert len(roots) == 4, roots
    for r in roots:
        r.__getattribute__ = _object_ga_rec
    _Dialect.__getattribute__ = _meta_ga
    for r in roots[:3]:
        assert type(r) is RecordingMeta, r
    return roots


def cpu_now():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def sqlglot_containers():
    """(container, copy) for every dict, list and set that a sqlglot module global or a class
    attribute of a sqlglot class holds (one level)."""
    out = []
    seen = set()

    def add(v):
        if type(v) in (dict, list, set) and id(v) not in seen:
            seen.add(id(v))
            out.append((v, v.copy()))

    for name, m in list(sys.modules.items()):
        if m is None or not (name == "sqlglot" or name.startswith("sqlglot.")):
            continue
        for v in list(vars(m).values()):
            add(v)
            if isinstance(v, type) and v.__module__.startswith("sqlglot"):
                for x in list(v.__dict__.values()):
                    add(x)
    return out


def restore(containers):
    """Puts back the containers whose content differs from the saved copy."""
    n = 0
    for v, saved in containers:
        if len(v) != len(saved) or (type(v) is dict and v.keys() != saved.keys()) or (type(v) is set and v != saved):
            v.clear()
            if type(v) is list:
                v.extend(saved)
            else:
                v.update(saved)
            n += 1
    return n


class Keys:
    """Stable names for the classes (see statefp.class_paths)."""

    def __init__(self, paths, roots):
        self.paths = {i: "%s:%s" % (mod, path) for i, (c, mod, path) in paths.items()}
        self.refs = [c for c, _, _ in paths.values()]  # keeps the ids alive
        self.roots = tuple(roots)
        self.memo = {}

    def key(self, c):
        k = self.memo.get(id(c))
        if k is None:
            k = self.paths.get(id(c)) or "%s:%s" % (c.__module__, c.__qualname__)
            self.memo[id(c)] = k
            self.refs.append(c)
        return k

    def mro(self, c):
        return [self.key(b) for b in c.__mro__ if b is not object and issubclass(b, self.roots)]


def pairs_of(keys, rec, mros):
    out = set()
    for c, name in rec:
        k = keys.key(c)
        out.add((k, name))
        if k not in mros:
            mros[k] = keys.mro(c)
    return out


SPAWNED = []


def _watch_subprocesses():
    """A test that starts another Python process reads attributes there, out of sight."""
    import subprocess

    real = subprocess.Popen.__init__

    def init(self, *a, **kw):
        SPAWNED.append(1)
        return real(self, *a, **kw)

    subprocess.Popen.__init__ = init
    for name in ("posix_spawn", "posix_spawnp", "execv", "execve", "system", "fork"):
        fn = getattr(os, name, None)
        if fn is not None:
            def wrapped(*a, _fn=fn, **kw):
                SPAWNED.append(1)
                return _fn(*a, **kw)

            setattr(os, name, wrapped)


def run_child(keys, containers, mod, test, w):
    from reduce import STATE

    out = {"ok": False}
    try:
        _watch_subprocesses()
        restore(containers)
        rel, cls_name, meth = test["id"].split("::")
        STATE.reset()
        result = unittest.TestResult()
        REC.clear()
        t0 = cpu_now()
        case = getattr(mod, cls_name)(meth)
        STATE.in_test = True
        unittest.TestSuite([case]).run(result)
        STATE.in_test = False
        cpu = cpu_now() - t0
        rec = list(REC)
        mros = {}
        pairs = pairs_of(keys, rec, mros)
        out = {"ok": not (result.failures or result.errors or result.unexpectedSuccesses), "cpu": round(cpu, 3),
               "pairs": sorted(pairs), "mro": mros, "skipped": len(result.skipped),
               "spawned": bool(SPAWNED)}
    except BaseException as e:  # noqa: BLE001
        out["err"] = repr(e)[:300]
    data = json.dumps(out).encode()
    while data:
        n = os.write(w, data)
        data = data[n:]


def fork_and_wait(fn, timeout):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            os.setpgid(0, 0)
        except OSError:
            pass
        try:
            fn(w)
        finally:
            os._exit(0)
    os.close(w)
    chunks = []
    deadline = time.monotonic() + timeout
    timed_out = False
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            timed_out = True
            break
        ready, _, _ = select.select([r], [], [], min(left, 5.0))
        if ready:
            chunk = os.read(r, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
    os.close(r)
    for kill in (lambda: os.killpg(pid, signal.SIGKILL), lambda: os.kill(pid, signal.SIGKILL)):
        try:
            kill()
        except (ProcessLookupError, PermissionError):
            pass
    os.waitpid(pid, 0)
    if timed_out:
        return {"ok": False, "timeout": True}
    try:
        return json.loads(b"".join(chunks))
    except ValueError:
        return {"ok": False, "err": "no record"}


def trace_file(keys, containers, rel, tests, out_path):
    import preload

    name = rel[:-3].replace("/", ".")
    REC.clear()
    rec = {"file": rel, "import_ok": True, "tests": {}}
    try:
        mod = importlib.import_module(name)
    except BaseException as e:  # noqa: BLE001
        rec["import_ok"] = False
        rec["import_err"] = repr(e)[:300]
        mod = None
    mros = {}
    rec["import"] = sorted(pairs_of(keys, list(REC), mros))
    preload._single_thread_duckdb()
    for t in tests:
        if mod is None:
            break
        t0 = time.time()
        m = fork_and_wait(lambda w: run_child(keys, containers, mod, t, w), 1800)
        m["wall"] = round(time.time() - t0, 2)
        mros.update(m.pop("mro", {}) or {})
        rec["tests"][t["number"]] = m
        print(t["number"], t["id"], "ok", m.get("ok"), "pairs", len(m.get("pairs", ())), "wall", m["wall"], m.get("err", ""), flush=True)
    rec["mro"] = mros
    with gzip.open(out_path + ".tmp", "wt") as f:
        json.dump(rec, f)
    os.replace(out_path + ".tmp", out_path)


def class_info(keys, paths):
    """For every pinned class of the hooked trees: its MRO (as keys) and the names in its own
    dict whose value is a function, a method wrapper or a property."""
    out = {}
    fn_types = (types.FunctionType, classmethod, staticmethod, property, types.BuiltinFunctionType)
    for i, (c, mod, path) in paths.items():
        k = keys.key(c)
        out[k] = {"mro": keys.mro(c),
                  "fn": sorted(n for n, v in c.__dict__.items() if isinstance(v, fn_types)),
                  "names": sorted(c.__dict__)}
    return out


def main():
    root, tests_path, out_dir, shard, nshards = sys.argv[1:6]
    shard, nshards = int(shard), int(nshards)
    tests = json.load(open(tests_path))["tests"]
    install_hooks()
    import preload

    preload.install_environment(os.path.abspath(root))
    preload.import_sqlglot()
    roots = finish_hooks()
    import statefp

    paths = statefp.class_paths()
    keys = Keys(paths, roots)
    containers = sqlglot_containers()
    os.makedirs(out_dir, exist_ok=True)
    if shard == 0:
        with open(os.path.join(out_dir, "classes.json"), "w") as f:
            json.dump(class_info(keys, paths), f)
    by_file = {}
    for t in tests:
        by_file.setdefault(t["file"], []).append(t)
    # files spread over the shards by recorded seconds (largest first)
    loads = [0.0] * nshards
    mine = []
    for rel in sorted(by_file, key=lambda r: (-sum(t["seconds"] + 0.05 for t in by_file[r]), r)):
        i = loads.index(min(loads))
        loads[i] += sum(t["seconds"] + 0.05 for t in by_file[rel])
        if i == shard:
            mine.append(rel)
    only = os.environ.get("BR_ONLY_FILES")
    if only:
        mine = [r for r in mine if r in only.split(",")]
    for rel in mine:
        out_path = os.path.join(out_dir, rel.replace("/", "__") + ".json.gz")
        if os.path.exists(out_path):
            continue
        t0 = time.time()
        res = fork_and_wait(lambda w: (trace_file(keys, containers, rel, by_file[rel], out_path), os.write(w, b"{}")), 7200)
        print("FILE", rel, len(by_file[rel]), "tests", round(time.time() - t0, 1), "s", res, flush=True)


if __name__ == "__main__":
    main()
