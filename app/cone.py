"""Import-time commits: re-import only the importer cone of the patched modules.

A commit that changes import-time code makes the child drop every sqlglot module and import
the whole patched tree again (about 180 modules), then fingerprint the whole import state
(statefp.py). Most sqlglot modules are imported by few others: no module imports a
dialects/*.py module at module level, and a parsers/*.py or generators/*.py module is imported
by its dialect only. When the parent still holds the pinned import and nothing else has run in
it, the child can drop only the cone of the patched module X (X and every module that imports
it at module level, directly or through others) and import those again. A module outside the
cone never runs X's code at import time, so it is the same as after a fresh import.

That holds only when the modules outside the cone neither read nor hold anything of the cone at
import time, and the cone does not write into their state. build_cone.py checks this on the
pinned tree for every module (maps/cone.json lists the modules that pass, each with its cone in
import order). The child checks it again for the patched code (verify): every table, class and
namespace that the modules outside the cone reach must hold the same objects as before, an entry
that held an object of the cone (a registry such as Dialect._classes, a package attribute, a
lazily cached name) must now hold the new object of the same name, and every cone module must
import without an error. When a check fails the child reports REDO and the parent runs the full
import for that commit.
"""
from __future__ import annotations

import ast

from astwalk import walk as _ast_walk
import enum
import importlib
import marshal
import operator
import sys
import types

_IS = operator.is_
_LEAF = (str, int, float, bool, bytes, complex, type(None), types.ModuleType, types.CodeType,
         types.BuiltinFunctionType, types.MethodDescriptorType, types.WrapperDescriptorType)
_MISSING = object()


def _sqlglot(name):
    return name == "sqlglot" or name.startswith("sqlglot.")


_NAMED_MEMO = None  # id -> named(o), while the parent takes its record


def named(o):
    memo = _NAMED_MEMO
    if memo is not None:
        got = memo.get(id(o), _MISSING)
        if got is _MISSING:
            got = memo[id(o)] = _named(o)
        return got
    return _named(o)


def _named(o):
    """The module that defines o under its own name (a module, or a class or function that
    module.qualname leads to), or None. Such an object is replaced by a new one of the same
    name when its module is imported again."""
    if type(o) is types.ModuleType:
        return o.__name__ if _sqlglot(o.__name__) else None
    if not (isinstance(o, type) or type(o) is types.FunctionType):
        return None
    mod_name = getattr(o, "__module__", None)
    if not isinstance(mod_name, str) or not _sqlglot(mod_name):
        return None
    m = sys.modules.get(mod_name)
    qual = getattr(o, "__qualname__", "")
    if m is None or "<" in qual:
        return None
    obj = m
    for part in qual.split("."):
        d = vars(obj) if type(obj) is types.ModuleType else obj.__dict__
        obj = d.get(part, None)
        if isinstance(obj, (staticmethod, classmethod)):
            obj = obj.__func__
        if obj is None:
            return None
    return mod_name if obj is o else None


class Heap:
    """A record of every mutable container that the sqlglot modules reach, taken in the parent
    right after the pinned import: each container with a copy of its entries (references, or a
    deep copy for plain data), and for tables the entries that hold a named object (a module,
    or a class or function under its own name) with that object's module. The walk does not
    enter another module, or a named class or function of another module: those are walked
    from their own module."""

    def __init__(self, walk_order=()):
        self.order = [m for m, v in sys.modules.items() if _sqlglot(m) and v is not None]
        self.index = {m: i for i, m in enumerate(self.order)}
        self.modules = {m: sys.modules[m] for m in self.order}
        # (kind, container, saved entries, owner module, named entries ((position, module), ...))
        self.recs = []
        self._leaf = {}
        seen = set()
        # A container belongs to the first module whose walk reaches it. The modules are walked
        # with the imported modules before their importers (walk_order, from build_cone.py), so
        # that a table which an importer only refers to belongs to the module that made it.
        first = [m for m in walk_order if m in self.modules]
        rest = [m for m in self.order if m not in set(first)]
        for m in first + rest:
            self._walk(self.modules[m], seen)
        self.by_owner = {}
        for rec in self.recs:
            self.by_owner.setdefault(rec[3], []).append(rec)

    def _is_leaf_type(self, t):
        got = self._leaf.get(t)
        if got is None:
            if t in _LEAF or issubclass(t, enum.Enum):
                got = True
            elif t is types.FunctionType or t in (dict, list, set, tuple, frozenset, staticmethod, classmethod,
                                                  property, types.MethodType) or issubclass(t, (type, dict)):
                got = False
            else:
                tm = getattr(t, "__module__", "") or ""
                got = not (_sqlglot(tm) or tm == "functools" or tm == "collections")
            self._leaf[t] = got
        return got

    def _push(self, stack, values):
        leaf = self._leaf
        for v in values:
            t = type(v)
            got = leaf.get(t)
            if got is None:
                got = self._is_leaf_type(t)
            if not got:
                stack.append(v)

    def _scan(self, stack, vals, seen):
        """Pushes the values to walk; returns the named entries ((position, module), ...)."""
        leaf_get = self._leaf.get
        marks = []
        append = stack.append
        own = self._own
        module_type = types.ModuleType
        function_type = types.FunctionType
        named_ = named
        for pos, v in enumerate(vals):
            t = type(v)
            got = leaf_get(t)
            if got is None:
                got = self._is_leaf_type(t)
            if got:
                # a leaf is never walked; a module is a named entry
                if t is module_type:
                    owner = named_(v)
                    if owner is not None:
                        marks.append((pos, owner))
                continue
            if t is function_type or isinstance(v, type):
                owner = named_(v)
                if owner is not None:
                    marks.append((pos, owner))
                    if owner != own:
                        continue  # walked from its own module
            if id(v) not in seen:
                append(v)
        return marks

    def _walk(self, module, seen):
        self._own = module.__name__
        own = module.__name__
        stack = [module]
        push = self._push
        recs = self.recs
        module_type = types.ModuleType
        function_type = types.FunctionType
        method_type = types.MethodType
        is_leaf_type = self._is_leaf_type
        scan = self._scan
        seen_add = seen.add
        while stack:
            o = stack.pop()
            k = id(o)
            if k in seen:
                continue
            seen_add(k)
            t = type(o)
            kids = None
            name = None
            if t is module_type:
                if o is not module:
                    continue
                kind, kids, name = "n", vars(o), own
            elif isinstance(o, type):
                if o.__module__ != own and named(o) is not None:
                    seen.discard(k)  # walked from its own module
                    continue
                if not _sqlglot(getattr(o, "__module__", "") or ""):
                    continue
                kind, kids = "c", o.__dict__
            elif t is function_type:
                if not _sqlglot(o.__module__ or ""):
                    continue
                if o.__module__ != own and named(o) is not None:
                    seen.discard(k)
                    continue
                kind = "f"
            elif t is dict or t is list or t is set or t is tuple or t is frozenset:
                plain = _plain(o)
                if plain is not None:
                    # plain data (strings, numbers and containers of them, e.g. a keyword
                    # trie): one deep copy, compared with == in the child
                    if t is tuple or t is frozenset:
                        continue  # immutable all the way down
                    kind = "p"
                elif t is dict:
                    kind, kids = "d", o
                elif t is list:
                    kind = "l"
                elif t is set:
                    kind = "s"
                else:
                    push(stack, o)
                    continue
            elif isinstance(o, dict):
                kind, kids = "d", o
            elif t in (staticmethod, classmethod):
                stack.append(o.__func__)
                continue
            elif t is property:
                push(stack, (o.fget, o.fset, o.fdel))
                continue
            elif t is method_type:
                push(stack, (o.__func__, o.__self__))
                continue
            elif is_leaf_type(t):
                continue
            else:
                kind = "o"
            if kind == "p":
                saved, marks = plain, ()
            elif kids is not None:
                keys = tuple(kids)
                vals = tuple(kids.values())
                saved = (keys, vals, o.__bases__ if kind == "c" else None)
                marks = scan(stack, vals, seen)
            elif kind in ("l", "s"):
                saved = tuple(o)
                marks = scan(stack, saved, seen)
            elif kind == "f":
                cells = o.__closure__ or ()
                contents = tuple(_cell(c) for c in cells)
                saved = (o.__code__, o.__defaults__, o.__kwdefaults__, cells, contents)
                marks = ()
                push(stack, (o.__defaults__, o.__kwdefaults__) + contents)
                d = getattr(o, "__dict__", None)
                if d:
                    stack.append(d)
            else:
                saved = _object_state(o)
                if saved is None:
                    continue
                marks = ()
                push(stack, saved[1])
                if saved[0] is not None:
                    stack.append(saved[0])
            recs.append((kind, o if kids is None or kind != "n" else kids, saved, own, tuple(marks)))


def _plain(o):
    """A deep copy of o when it holds plain data only (what marshal can write), else None."""
    try:
        return marshal.loads(marshal.dumps(o, 2))
    except (ValueError, TypeError):
        return None


def _cell(c):
    try:
        return c.cell_contents
    except ValueError:
        return _MISSING


def _object_state(o):
    """(the object's __dict__ or None, its slot values) of an instance."""
    d = getattr(o, "__dict__", None)
    if not isinstance(d, dict):
        d = None
    slots = []
    for klass in type(o).__mro__:
        s = klass.__dict__.get("__slots__")
        if not s:
            continue
        for name in ((s,) if isinstance(s, str) else s):
            if name in ("__dict__", "__weakref__"):
                continue
            try:
                slots.append(getattr(o, name))
            except AttributeError:
                slots.append(_MISSING)
    if d is None and not slots:
        return None
    return (d, tuple(slots))


HEAP = None
_EXPECTED_CODE = {}  # id(function) -> the code object that the child gave it (expect_code)


def expect_code(swaps):
    """In the child, before reimport(): these functions outside the cone got a new code object
    ((function, code) pairs), on purpose. The record holds their pinned code."""
    for f, code in swaps:
        _EXPECTED_CODE[id(f)] = (f, code)


def record(walk_order=()):
    """In the parent, right after the pinned import (before anything else runs in it)."""
    global HEAP, _NAMED_MEMO
    _NAMED_MEMO = {}
    try:
        HEAP = Heap(walk_order)
    finally:
        _NAMED_MEMO = None
    return HEAP


def import_lines(text):
    """The import statements and dynamic import calls of a module, for comparison."""
    tree = ast.parse(text)
    out = []
    for node in _ast_walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append(ast.dump(node, include_attributes=False))
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if name in ("import_module", "__import__", "find_spec", "reload"):
                out.append(ast.dump(node, include_attributes=False))
    return sorted(out)


class Redo(Exception):
    pass


def reimport(cone, install):
    """In the child: drops the cone modules, imports them again (install() puts the patched
    sources in place first) and checks that the rest of the import state is as it was. Raises
    Redo when the result is not known to equal a fresh import of the whole patched tree.
    Returns the modules outside the cone whose fingerprint can still differ (see below)."""
    heap = HEAP
    cset = set(cone)
    for m in cone:
        if m not in heap.index:
            raise Redo("not in the pinned import: " + m)
    for m in heap.order:
        if sys.modules.get(m) is not heap.modules[m]:
            raise Redo("module changed since the record: " + m)
    # names outside the cone that hold a cone object: a package's attribute for its submodule,
    # a lazily cached name (dialects/__init__.py, optimizer/__init__.py). They are bound again
    # by the import, as in a fresh one; their place in the namespace is restored below.
    # the containers of the cone modules are left alone: only the cone reaches them
    for kind, o, saved, owner, marks in heap.recs:
        if kind == "n" and marks and owner not in cset:
            keys = saved[0]
            for pos, owner in marks:
                if owner in cset:
                    o.pop(keys[pos], None)
    for m in cone:
        del sys.modules[m]
    install()
    for m in cone:
        if m in sys.modules:
            continue  # imported by an earlier cone module, as in a fresh import
        try:
            importlib.import_module(m)
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            raise Redo("import fails: %s %r" % (m, e)) from None
    names = [m for m, v in sys.modules.items() if _sqlglot(m) and v is not None]
    if len(names) != len(heap.order) or set(names) != set(heap.order):
        raise Redo("other sqlglot modules after the import")
    for m in heap.order:
        new = sys.modules[m]
        if (new is heap.modules[m]) == (m in cset):
            raise Redo("module not imported again, or imported when it should not be: " + m)
    touched = set()
    for owner, recs in heap.by_owner.items():
        if owner in cset:
            continue
        for rec in recs:
            # A table that held a named object of the cone (a registry such as
            # Dialect._classes) must hold the new object now: there, an unchanged table is not
            # enough (the patched module may not register its class any more, and the table
            # would keep the pinned one). So the quick check is for the other tables only.
            marked = False
            for _, m in rec[4]:
                if m in cset:
                    marked = True
                    break
            if not marked and _fast_same(rec):
                continue
            if not _same(rec[0], rec[1], rec[2], rec[4], cset):
                raise Redo("state outside the cone changed: %s %s" % (rec[0], _describe(rec[1])))
            # An entry here now holds a new function of the cone (a lazily cached name): the
            # fingerprint of a function covers its code, so this module's keys can differ.
            vals = rec[2][1] if rec[0] in ("n", "c", "d") else rec[2]
            if any(type(vals[pos]) is types.FunctionType for pos, m in rec[4] if m in cset):
                touched.add(owner)
    return touched


def _describe(o):
    if isinstance(o, type) and HEAP is not None:
        for kind, c, saved, owner, marks in HEAP.recs:
            if c is o:
                now = dict(o.__dict__)
                old = dict(zip(saved[0], saved[1]))
                ch = [k for k in set(now) | set(old) if now.get(k, _MISSING) is not old.get(k, _MISSING)]
                return "class %s.%s %s" % (o.__module__, o.__qualname__, ch[:8])
    if type(o) is dict and "__name__" in o:
        return "namespace " + str(o.get("__name__"))
    if isinstance(o, type):
        return "class " + o.__module__ + "." + o.__qualname__
    return type(o).__name__ + " " + repr(o)[:80]


def _fast_same(rec):
    """Unchanged (entries equal, with identity first, in the same order)? False: look closer."""
    kind, o, saved = rec[0], rec[1], rec[2]
    if kind == "d" or kind == "n":
        return tuple(o.values()) == saved[1] and tuple(o) == saved[0]
    if kind == "p":
        return o == saved
    if kind == "c":
        d = o.__dict__
        return tuple(d.values()) == saved[1] and tuple(d) == saved[0] and o.__bases__ is saved[2]
    if kind == "f":
        return (o.__code__ is saved[0] and o.__defaults__ is saved[1] and o.__kwdefaults__ is saved[2]
                and (not saved[3] or tuple(map(_cell, saved[3])) == saved[4]) and (o.__closure__ or ()) == saved[3])
    if kind == "l" or kind == "s":
        return tuple(o) == saved
    return False


def _same(kind, o, saved, marks, cset):
    if kind == "p":
        return type(o) is type(saved) and o == saved
    cone_marks = [pm for pm in marks if pm[1] in cset]
    if kind in ("n", "c", "d"):
        keys, vals, bases = saved
        d = o.__dict__ if kind == "c" else o
        if kind == "c" and o.__bases__ is not bases:
            return False
        if len(d) != len(keys):
            return False
        if kind == "n" and cone_marks and not all(map(_IS, d, keys)):
            # the names bound again by the import sit at the end: back to the pinned order
            if set(d) != set(keys):
                return False
            items = [(k, d[k]) for k in keys]
            d.clear()
            d.update(items)
        if not all(map(_IS, d, keys)) and tuple(d) != keys:
            return False
        now = tuple(d.values())
        if not cone_marks:
            return all(map(_IS, now, vals))
        return _same_marked(now, vals, cone_marks)
    if kind in ("l", "s"):
        now = tuple(o)
        if len(now) != len(saved):
            return False
        if not cone_marks:
            if all(map(_IS, now, saved)):
                return True
            return kind == "s" and set(now) == set(saved) and len(set(map(id, now)) ^ set(map(id, saved))) == 0
        return kind == "l" and _same_marked(now, saved, cone_marks)
    if kind == "f":
        cells = o.__closure__ or ()
        code_ok = o.__code__ is saved[0]
        if not code_ok:
            exp = _EXPECTED_CODE.get(id(o))
            code_ok = exp is not None and exp[0] is o and exp[1] is o.__code__
        return (code_ok and o.__defaults__ is saved[1] and o.__kwdefaults__ is saved[2]
                and len(cells) == len(saved[3]) and all(map(_IS, cells, saved[3]))
                and all(map(_IS, map(_cell, cells), saved[4])))
    st = _object_state(o)
    return st is not None and st[0] is saved[0] and len(st[1]) == len(saved[1]) and all(map(_IS, st[1], saved[1]))


def _same_marked(now, vals, cone_marks):
    """Entries equal by identity, except that an entry that held a named object of the cone
    now holds the new object of the same name."""
    cone_pos = dict(cone_marks)
    for pos, (a, b) in enumerate(zip(now, vals)):
        if pos in cone_pos:
            if a is b or type(a) is not type(b):
                return False
            if type(a) is types.ModuleType:
                if a.__name__ != b.__name__ or sys.modules.get(a.__name__) is not a:
                    return False
            elif named(a) != cone_pos[pos] or a.__qualname__ != b.__qualname__:
                return False
        elif a is not b:
            return False
    return True
