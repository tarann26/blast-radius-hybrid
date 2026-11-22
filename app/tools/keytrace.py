"""Build-time tracer (pinned tree only): which entries of the dict, set and frozenset
constants of sqlglot does each test look up?

After the pinned sqlglot is imported, every dict / set / frozenset that a sqlglot module global
or a class attribute holds (the keys of statefp.snapshot) is replaced by a recording copy:
- a dict becomes a dict subclass (TDict) whose lookups record the key. It overrides __iter__,
  so CPython's dict_merge (dict(d), d2.update(d), {**d}) takes its generic path through keys()
  and __getitem__ instead of reading the table directly;
- a set or frozenset becomes a proxy (TSet) that is not a set subclass at all: CPython's set
  operations read the table of any set subclass directly (PyAnySet_Check), so with a proxy
  they fall back to iteration or to the proxy's reflected operators, which record a whole read.
  isinstance(x, set) still holds (the proxy reports set or frozenset as its __class__);
- a lookup records its key (also a key that is not there); len() and truth record LEN, key
  iteration records KEYS, everything else that sees the whole table records WHOLE;
- a test that changes a table (setitem, update, add, ...) marks the table "changed by a test":
  the selector never uses the map for such a table.
Every reference to an original table is replaced (module dicts, class dicts, other dicts,
lists, cells, object attributes). A table that is also referenced from a place that cannot be
rewritten (a tuple, a dict key, a mappingproxy, C code) stays as it is and is left out of the
map; the reference counts of the originals are checked after the swap, so no reference is
missed.

Every test runs cold, in its own child forked from a parent that has imported sqlglot and only
the test's module (caches that sqlglot fills on first use are empty again, setUpClass runs for
each test). Reads during the import of a test file count for every test of that file. Process
pools run in-process (pool.py), so reads of pool tasks count for the test that submits them.
Tests that fail, time out, crash or start another process here are "unknown": never cut.

usage (inside the pinned image br-label):
    python keytrace.py REPO TESTS_JSON OUT_DIR SHARD NSHARDS
"""
from __future__ import annotations

import gc
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

import keymap  # noqa: E402

CHANGED = set()  # tables that a test changed (names)
SAVED = {}  # table name -> its content before the first change in this test (see restored())
_W = object()  # whole read
_L = object()  # len / truth
_K = object()  # keys in order

_dget = dict.__getitem__
_dgetm = dict.get
_dcont = dict.__contains__
_diter = dict.__iter__
_dlen = dict.__len__


class TDict(dict):
    __slots__ = ("_r", "_n")

    def __getitem__(self, k):
        try:
            self._r.add(k)
        except TypeError:
            self._r.add(_W)
        return _dget(self, k)

    def get(self, k, default=None):
        try:
            self._r.add(k)
        except TypeError:
            self._r.add(_W)
        return _dgetm(self, k, default)

    def __contains__(self, k):
        try:
            self._r.add(k)
        except TypeError:
            self._r.add(_W)
        return _dcont(self, k)

    def __iter__(self):
        self._r.add(_K)
        return _diter(self)

    def keys(self):
        self._r.add(_K)
        return dict.keys(self)

    def __reversed__(self):
        self._r.add(_K)
        return dict.__reversed__(self)

    def __len__(self):
        self._r.add(_L)
        return _dlen(self)

    def values(self):
        self._r.add(_W)
        return dict.values(self)

    def items(self):
        self._r.add(_W)
        return dict.items(self)

    def copy(self):
        self._r.add(_W)
        return dict(dict.items(self))

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        import copy

        self._r.add(_W)
        return copy.deepcopy(dict(dict.items(self)), memo)

    def __reduce_ex__(self, proto):
        self._r.add(_W)
        return (dict, (dict(dict.items(self)),))

    def __reduce__(self):
        return self.__reduce_ex__(2)

    def __repr__(self):
        self._r.add(_W)
        return dict.__repr__(self)

    __str__ = __repr__

    def __format__(self, spec):
        self._r.add(_W)
        return format(dict(dict.items(self)), spec)

    def __eq__(self, other):
        self._r.add(_W)
        return dict.__eq__(self, other)

    def __ne__(self, other):
        self._r.add(_W)
        return dict.__ne__(self, other)

    __hash__ = None

    def __or__(self, other):
        self._r.add(_W)
        return dict.__or__(dict(dict.items(self)), other)

    def __ror__(self, other):
        self._r.add(_W)
        if not isinstance(other, dict):
            return NotImplemented
        out = dict(other)
        dict.update(out, dict.items(self))
        return out

    def __sizeof__(self):
        return dict.__sizeof__(self)

    # changes: the table is then "changed by a test"
    def _changed(self):
        if self._n not in SAVED:
            SAVED[self._n] = (self, list(dict.items(self)))
        CHANGED.add(self._n)

    def __setitem__(self, k, v):
        self._changed()
        dict.__setitem__(self, k, v)

    def __delitem__(self, k):
        self._changed()
        dict.__delitem__(self, k)

    def update(self, *a, **kw):
        self._changed()
        dict.update(self, *a, **kw)

    def __ior__(self, other):
        self._changed()
        dict.update(self, other)
        return self

    def clear(self):
        self._changed()
        dict.clear(self)

    def pop(self, k, *default):
        try:
            self._r.add(k)
        except TypeError:
            self._r.add(_W)
        if _dcont(self, k):
            self._changed()
        return dict.pop(self, k, *default)

    def popitem(self):
        self._changed()
        self._r.add(_W)
        return dict.popitem(self)

    def setdefault(self, k, default=None):
        try:
            self._r.add(k)
        except TypeError:
            self._r.add(_W)
        if not _dcont(self, k):
            self._changed()
        return dict.setdefault(self, k, default)


def _plain(x):
    return x._s if type(x) is TSet else x


class TSet:
    """A recording stand-in for a set or frozenset (not a subclass: see the module doc)."""

    __slots__ = ("_s", "_r", "_n")

    @property
    def __class__(self):
        return type(self._s)

    def __contains__(self, x):
        try:
            self._r.add(x)
        except TypeError:
            self._r.add(_W)
        return x in self._s

    def __iter__(self):
        self._r.add(_W)
        return iter(self._s)

    def __len__(self):
        self._r.add(_L)
        return len(self._s)

    def __bool__(self):
        self._r.add(_L)
        return bool(self._s)

    def __hash__(self):
        self._r.add(_W)
        return hash(self._s)  # TypeError for a set, as for the real one

    def __repr__(self):
        self._r.add(_W)
        return repr(self._s)

    __str__ = __repr__

    def __format__(self, spec):
        self._r.add(_W)
        return format(self._s, spec)

    def __reduce_ex__(self, proto):
        self._r.add(_W)
        return (type(self._s), (list(self._s),))

    def __reduce__(self):
        return self.__reduce_ex__(2)

    def __copy__(self):
        self._r.add(_W)
        return self._s.copy()

    def __deepcopy__(self, memo):
        import copy

        self._r.add(_W)
        return copy.deepcopy(self._s, memo)

    def copy(self):
        self._r.add(_W)
        return self._s.copy()

    def _whole(self, name, *args):
        self._r.add(_W)
        return getattr(self._s, name)(*[_plain(a) for a in args])

    def union(self, *a):
        return self._whole("union", *a)

    def intersection(self, *a):
        return self._whole("intersection", *a)

    def difference(self, *a):
        return self._whole("difference", *a)

    def symmetric_difference(self, *a):
        return self._whole("symmetric_difference", *a)

    def issubset(self, *a):
        return self._whole("issubset", *a)

    def issuperset(self, *a):
        return self._whole("issuperset", *a)

    def isdisjoint(self, *a):
        return self._whole("isdisjoint", *a)

    def _op(self, other, op, reflected=False):
        self._r.add(_W)
        other = _plain(other)
        if not isinstance(other, (set, frozenset)):
            return NotImplemented
        return op(other, self._s) if reflected else op(self._s, other)

    def __or__(self, o):
        return self._op(o, lambda a, b: a | b)

    def __and__(self, o):
        return self._op(o, lambda a, b: a & b)

    def __sub__(self, o):
        return self._op(o, lambda a, b: a - b)

    def __xor__(self, o):
        return self._op(o, lambda a, b: a ^ b)

    def __ror__(self, o):
        return self._op(o, lambda a, b: a | b, True)

    def __rand__(self, o):
        return self._op(o, lambda a, b: a & b, True)

    def __rsub__(self, o):
        return self._op(o, lambda a, b: a - b, True)

    def __rxor__(self, o):
        return self._op(o, lambda a, b: a ^ b, True)

    def __eq__(self, o):
        self._r.add(_W)
        return self._s == _plain(o)

    def __ne__(self, o):
        self._r.add(_W)
        return self._s != _plain(o)

    def __lt__(self, o):
        return self._op(o, lambda a, b: a < b)

    def __le__(self, o):
        return self._op(o, lambda a, b: a <= b)

    def __gt__(self, o):
        return self._op(o, lambda a, b: a > b)

    def __ge__(self, o):
        return self._op(o, lambda a, b: a >= b)

    # changes (sets only; a frozenset has no such methods and raises AttributeError, as it should)
    def _mut(self, name, *args):
        if type(self._s) is frozenset:
            raise AttributeError("'frozenset' object has no attribute %r" % name)
        if self._n not in SAVED:
            SAVED[self._n] = (self, set(self._s))
        CHANGED.add(self._n)
        self._r.add(_W)
        return getattr(self._s, name)(*[_plain(a) for a in args])

    def add(self, x):
        return self._mut("add", x)

    def discard(self, x):
        return self._mut("discard", x)

    def remove(self, x):
        return self._mut("remove", x)

    def pop(self):
        return self._mut("pop")

    def clear(self):
        return self._mut("clear")

    def update(self, *a):
        return self._mut("update", *a)

    def intersection_update(self, *a):
        return self._mut("intersection_update", *a)

    def difference_update(self, *a):
        return self._mut("difference_update", *a)

    def symmetric_difference_update(self, *a):
        return self._mut("symmetric_difference_update", *a)

    def _imut(self, o, name):
        if type(self._s) is frozenset:
            return NotImplemented
        if not isinstance(_plain(o), (set, frozenset)):
            return NotImplemented
        self._mut(name, o)
        return self

    def __ior__(self, o):
        return self._imut(o, "update")

    def __iand__(self, o):
        return self._imut(o, "intersection_update")

    def __isub__(self, o):
        return self._imut(o, "difference_update")

    def __ixor__(self, o):
        return self._imut(o, "symmetric_difference_update")


try:
    import collections.abc as _abc

    _abc.MutableSet.register(TSet)
except Exception:  # noqa: BLE001
    pass


class DDict(dict):
    """The generator's dispatch table (generator._build_dispatch): a copy of cls.TRANSFORMS plus
    the class's *_sql methods for the expression classes that TRANSFORMS does not hold. Its entry
    for key k can differ from the pinned one only when TRANSFORMS[k] differs (or the method,
    which is code, not a table entry), so a lookup of k here counts as a lookup of k in the
    TRANSFORMS table that it was copied from, and the copy itself is not a whole read.
    Anything else than a lookup counts as a whole read of that table."""

    __slots__ = ("_src",)

    def _key(self, k):
        try:
            self._src._r.add(k)
        except TypeError:
            self._src._r.add(_W)

    def get(self, k, default=None):
        self._key(k)
        return _dgetm(self, k, default)

    def __getitem__(self, k):
        self._key(k)
        return _dget(self, k)

    def __contains__(self, k):
        self._key(k)
        return _dcont(self, k)


def _ddict_whole(name):
    real = getattr(dict, name)

    def method(self, *a, **kw):
        self._src._r.add(_W)
        if name in ("__setitem__", "__delitem__", "update", "clear", "pop", "popitem", "setdefault", "__ior__"):
            CHANGED.add(self._src._n)  # a changed dispatch table: leave TRANSFORMS out of the map
        return real(self, *a, **kw)

    method.__name__ = name
    return method


for _name in ("__iter__", "keys", "values", "items", "__len__", "__reversed__", "copy", "__repr__", "__eq__",
              "__ne__", "__or__", "__ror__", "__reduce_ex__", "__reduce__", "__setitem__", "__delitem__",
              "update", "clear", "pop", "popitem", "setdefault", "__ior__"):
    setattr(DDict, _name, _ddict_whole(_name))
DDict.__hash__ = None


def install_derived():
    """Lookups through the generator's dispatch table count as lookups in TRANSFORMS (DDict)."""
    gen = sys.modules["sqlglot.generator"]
    build = gen._build_dispatch

    def traced_build(cls):
        src = cls.TRANSFORMS
        if type(src) is not TDict:
            return build(cls)
        saved = src._r
        src._r = set()  # the copy that _build_dispatch makes is not itself a read (see DDict)
        try:
            d = build(cls)
        finally:
            src._r = saved
        out = DDict()
        dict.update(out, d)
        out._src = src
        return out

    gen._build_dispatch = traced_build


# ---------------------------------------------------------------- swapping the tables in

TRACKERS = {}  # table name -> tracker
INFO = {}  # table name -> {"holders": [...], "type": ..., "len": ..., "left_out": reason or None}


def _heap_referrers(targets, ignore):
    """{id(target): [referrer, ...]} over every object that the collector knows, and every
    object that those hold (untracked dicts and tuples are found through their holders)."""
    out = {i: [] for i in targets}
    seen = set()
    stack = gc.get_objects()
    for o in stack:
        seen.add(id(o))
    i = 0
    while i < len(stack):
        o = stack[i]
        i += 1
        if id(o) in ignore or isinstance(o, types.FrameType):
            continue
        for r in gc.get_referents(o):
            ir = id(r)
            if ir in out:
                out[ir].append(o)
            if ir not in seen and type(r) in (dict, tuple, list, set, frozenset):
                seen.add(ir)
                stack.append(r)
    return out


def install(objects):
    """objects: {statefp key: table} of the pinned import. Swaps recording copies in."""
    by_id = {}
    for key, v in sorted(objects.items(), key=lambda kv: keymap.holder_name(kv[0])):
        by_id.setdefault(id(v), (v, []))[1].append(key)
    names = {}
    for i, (v, keys) in by_id.items():
        names[i] = keymap.holder_name(keys[0])
        INFO[names[i]] = {"holders": [keymap.holder_name(k) for k in keys], "type": type(v).__name__,
                          "len": len(v), "left_out": None}
    origs = {i: v for i, (v, _) in by_id.items()}
    ignore = {id(objects), id(by_id), id(origs), id(names), id(sys._getframe())}
    ignore.update(id(x) for x in by_id.values())
    ignore.update(id(x[1]) for x in by_id.values())
    refs = _heap_referrers(set(origs), ignore)
    module_dicts = {id(vars(m)): m for m in list(sys.modules.values()) if m is not None}
    class_dicts = {}
    for o in gc.get_objects():
        if isinstance(o, type):
            try:
                for r in gc.get_referents(o.__dict__):
                    if type(r) is dict:
                        class_dicts[id(r)] = o
            except Exception:  # noqa: BLE001
                pass
    plan = {}  # id -> list of (kind, target, where)
    left = {}

    def leave(i, why):
        if i not in left:
            left[i] = why

    for i, rs in refs.items():
        acts = []
        for r in rs:
            ir = id(r)
            if ir in origs:
                if type(r) is dict:
                    if any(k is origs[i] for k in r):
                        leave(i, "dict key of a table")
                    acts.append(("nested", r, None))
                else:
                    leave(i, "element of a set table")
                continue
            if type(r) is dict:
                if any(k is origs[i] for k in r):
                    leave(i, "dict key")
                    continue
                where = [k for k, v in r.items() if v is origs[i]]
                if ir in class_dicts:
                    acts.append(("class", class_dicts[ir], where))
                else:
                    acts.append(("dict", r, where))
            elif type(r) is list:
                acts.append(("list", r, [j for j, v in enumerate(r) if v is origs[i]]))
            elif type(r) is types.CellType:
                acts.append(("cell", r, None))
            elif isinstance(r, type):
                leave(i, "class internals")  # held by the class object itself, not its dict
            elif type(r) in (tuple, set, frozenset, types.MappingProxyType, types.FunctionType, types.MethodType):
                leave(i, "held by a %s" % type(r).__name__)
            else:
                names_ = []
                d = getattr(r, "__dict__", None)
                if isinstance(d, dict):
                    names_ += [k for k, v in d.items() if v is origs[i]]
                for klass in type(r).__mro__:
                    slots = klass.__dict__.get("__slots__", ())
                    if isinstance(slots, str):
                        slots = (slots,)
                    for s in slots:
                        try:
                            if getattr(r, s) is origs[i]:
                                names_.append(s)
                        except Exception:  # noqa: BLE001
                            pass
                if names_:
                    acts.append(("attr", r, names_))
                else:
                    leave(i, "held by a %s.%s" % (type(r).__module__, type(r).__qualname__))
        plan[i] = acts
    # a table held by a table that stays as it is stays as it is too
    changed = True
    while changed:
        changed = False
        for i, acts in plan.items():
            if i in left:
                continue
            for kind, r, _ in acts:
                if kind == "nested" and id(r) in left:
                    leave(i, "held by a table that is left out")
                    changed = True
                    break
    # the recording copies
    trackers = {}
    for i, v in origs.items():
        if i in left:
            continue
        if type(v) is dict:
            t = TDict()
            t._r = set()
            t._n = names[i]
        else:
            t = TSet()
            t._r = set()
            t._n = names[i]
        trackers[i] = t

    def sub(x):
        t = trackers.get(id(x))
        return t if t is not None and x is origs.get(id(x)) else x

    for i, t in trackers.items():
        v = origs[i]
        if type(v) is dict:
            dict.update(t, {k: sub(x) for k, x in dict.items(v)})
        else:
            t._s = v  # the proxy holds the original set itself (its elements are hashable)
    # swap
    for i, t in trackers.items():
        v = origs[i]
        for kind, r, where in plan[i]:
            if kind == "class":
                for name in where:
                    type.__setattr__(r, name, t)
            elif kind == "dict":
                for k in where:
                    dict.__setitem__(r, k, t)
            elif kind == "list":
                for j in where:
                    r[j] = t
            elif kind == "cell":
                r.cell_contents = t
            elif kind == "attr":
                for name in where:
                    object.__setattr__(r, name, t)
    # check: no reference to a swapped original is left, other than the tables that held it
    # (originals of dicts; they are not used any more) and this function's own
    inside = {}
    for i, v in origs.items():
        if type(v) is dict and i in trackers:
            for x in dict.values(v):
                if id(x) in origs:
                    inside[id(x)] = inside.get(id(x), 0) + 1
    probe = {}
    base = sys.getrefcount(probe)  # a fresh object referenced by one local
    bad = []
    for i, t in trackers.items():
        v = origs[i]
        # the local 'v', origs, by_id and the caller's dict of tables (base counts one local)
        expect = base + 1 + len(by_id[i][1]) + inside.get(i, 0)
        if type(v) is not dict:
            expect += 1  # the proxy's own reference
        n = sys.getrefcount(v) - 1  # the local 'v'
        if n != expect:
            bad.append((names[i], n, expect))
    for i, why in left.items():
        INFO[names[i]]["left_out"] = why
    for i, t in trackers.items():
        TRACKERS[names[i]] = t
    return bad


def reset():
    for t in TRACKERS.values():
        t._r = set()
    SAVED.clear()


def restored():
    """The changed tables whose content is back to what it was before the test changed them
    (same keys in the same order, the very same values): a test that puts back what it set."""
    out = set()
    for name, (t, saved) in SAVED.items():
        if type(t) is TDict:
            now = list(dict.items(t))
            if len(now) == len(saved) and all(a[0] == b[0] and a[1] is b[1] for a, b in zip(now, saved)):
                out.add(name)
        elif t._s == saved:
            out.add(name)
    return out


def collect():
    """{table name: sorted stable keys} of the reads since reset()."""
    out = {}
    for name, t in TRACKERS.items():
        r = t._r
        if not r:
            continue
        keys = set()
        for k in r:
            if k is _W:
                keys.add(keymap.WHOLE)
            elif k is _L:
                keys.add(keymap.LEN)
            elif k is _K:
                keys.add(keymap.KEYS)
            else:
                keys.add(keymap.stable_key(k))
        if keymap.WHOLE in keys:
            keys = {keymap.WHOLE}
        out[name] = sorted(keys)
    return out


# ---------------------------------------------------------------- running the tests

def cpu_now():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


SPAWNED = []


def _watch_subprocesses():
    """A test that starts another Python process reads tables there, out of sight."""
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


REGISTRY_CLEAR_AT = [None]


def run_child(mod, test, w):
    from reduce import STATE

    out = {"ok": False}
    try:
        _watch_subprocesses()
        rel, cls_name, meth = test["id"].split("::")
        if REGISTRY_CLEAR_AT[0] is not None and test["number"] >= REGISTRY_CLEAR_AT[0]:
            # in the full run, tests/test_dialect_entry_points.py has emptied the dialect
            # registry before this test (see selector.registry_clear_at)
            reg = sys.modules["sqlglot.dialects.dialect"].Dialect._classes
            dict.clear(reg)
        STATE.reset()
        result = unittest.TestResult()
        CHANGED.clear()
        reset()
        t0 = cpu_now()
        case = getattr(mod, cls_name)(meth)
        STATE.in_test = True
        unittest.TestSuite([case]).run(result)
        STATE.in_test = False
        cpu = cpu_now() - t0
        reads = collect()
        back = restored()
        for name in back:
            reads[name] = [keymap.WHOLE]  # the test that changes a table reads all of it
        out = {"ok": not (result.failures or result.errors or result.unexpectedSuccesses), "cpu": round(cpu, 3),
               "reads": reads, "changed": sorted(CHANGED - back), "restored": sorted(back),
               "skipped": len(result.skipped),
               "spawned": bool(SPAWNED)}
        if not out["ok"]:
            out["fail"] = [str(x[1])[-600:] for x in (result.failures + result.errors)][:2]
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


def trace_file(rel, tests, out_path):
    import preload

    name = rel[:-3].replace("/", ".")
    CHANGED.clear()
    reset()
    rec = {"file": rel, "import_ok": True, "tests": {}}
    try:
        mod = importlib.import_module(name)
    except BaseException as e:  # noqa: BLE001
        rec["import_ok"] = False
        rec["import_err"] = repr(e)[:300]
        mod = None
    rec["import"] = collect()
    rec["import_changed"] = sorted(CHANGED)
    preload._single_thread_duckdb()
    for t in tests:
        if mod is None:
            break
        t0 = time.time()
        m = fork_and_wait(lambda w: run_child(mod, t, w), 1800)
        m["wall"] = round(time.time() - t0, 2)
        rec["tests"][t["number"]] = m
        print(t["number"], t["id"], "ok", m.get("ok"), "tables", len(m.get("reads", ())), "wall", m["wall"],
              m.get("err", ""), flush=True)
    with gzip.open(out_path + ".tmp", "wt") as f:
        json.dump(rec, f)
    os.replace(out_path + ".tmp", out_path)


def main():
    root, tests_path, out_dir, shard, nshards = sys.argv[1:6]
    shard, nshards = int(shard), int(nshards)
    tests = json.load(open(tests_path))["tests"]
    import preload
    import statefp

    preload.install_environment(os.path.abspath(root))
    preload.import_sqlglot()
    _, f = statefp.snapshot()
    objects = dict(f.objects)
    del f
    gc.collect()
    t0 = time.time()
    bad = install(objects)
    del objects
    install_derived()
    print("installed", len(TRACKERS), "trackers, left out", sum(1 for v in INFO.values() if v["left_out"]),
          "refcount mismatches", len(bad), "in", round(time.time() - t0, 1), "s", flush=True)
    for b in bad[:50]:
        print("REFCOUNT", b, flush=True)
    ids = [t["id"] for t in tests]
    last = None
    for i, tid in enumerate(ids):
        if tid.startswith("tests/test_dialect_entry_points.py::"):
            last = i
    REGISTRY_CLEAR_AT[0] = None if last is None else last + 1
    os.makedirs(out_dir, exist_ok=True)
    if shard == 0:
        with open(os.path.join(out_dir, "tables.json"), "w") as fo:
            json.dump({"info": INFO, "refcount_bad": bad}, fo)
    by_file = {}
    for t in tests:
        by_file.setdefault(t["file"], []).append(t)
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
    if bad:
        # a reference that the swap did not see: the trace would not be complete
        mine = []
        print("NOT TRACING: refcount mismatches", flush=True)
    for rel in mine:
        out_path = os.path.join(out_dir, rel.replace("/", "__") + ".json.gz")
        if os.path.exists(out_path):
            continue
        t0 = time.time()
        res = fork_and_wait(lambda w: (trace_file(rel, by_file[rel], out_path), os.write(w, b"{}")), 7200)
        print("FILE", rel, len(by_file[rel]), "tests", round(time.time() - t0, 1), "s", res, flush=True)


if __name__ == "__main__":
    main()
