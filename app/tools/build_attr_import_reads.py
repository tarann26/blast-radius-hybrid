"""Builds maps/attr_import_reads.json on the pinned tree: which class attributes of sqlglot classes
are read while the tree is imported, and by whom.

A class-level edit that changes the value of attribute A of class C changes the import state
beyond C.A only through code that reads C.A during the import: the creation of C itself (its
class body, __init_subclass__ of its bases, its metaclass), which the selector runs again for
the patched class (selector: class route), or anything else (the creation of a subclass, a
module body, a function that an import runs). This map lists the second kind, per (class,
attribute): [module whose import was running, class that was being created or null]. An
attribute without an entry is read by nobody but the creation of its own class.

How reads are seen (before sqlglot is imported):
- class objects: every sqlglot class whose metaclass would be 'type' gets a recording metaclass
  at creation (builtins.__build_class__); a metaclass that sqlglot defines gets the recording
  __getattribute__; abc.ABCMeta classes get a recording subclass of ABCMeta;
- objects: the first sqlglot class of each hierarchy gets a recording __getattribute__;
- a read of the class __dict__ counts as a read of every attribute of that class ("*").
Classes whose reads cannot be seen this way (enum classes, other metaclasses, a class with its
own __getattribute__) are not listed in "hooked": the selector never takes the class route for
them.

usage: python build_attr_import_reads.py OUT.json   (cwd and sys.path as for the selector)"""
from __future__ import annotations

import abc
import builtins
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.environ.get("BR_ROOT", "/repo")

_type_ga = type.__getattribute__
_object_ga = object.__getattribute__
_class_dict = type.__dict__["__dict__"].__get__
ON = [False]
READS = []  # (receiver class, name, instance read?, import module, creation entry or None)
_add = READS.append
MODS = []  # modules being imported (innermost last)
BUILDS = []  # classes being created: [class or None]
IGNORE = frozenset(("__name__", "__qualname__", "__module__", "__mro__", "__bases__", "__subclasses__", "__class__",
                    "__doc__", "__base__", "__subclasshook__", "__instancecheck__", "__subclasscheck__", "__flags__",
                    "__dictoffset__", "__basicsize__", "__weakrefoffset__", "__itemsize__", "__text_signature__",
                    "__hash__", "__eq__", "__ne__", "__call__", "__new__", "__init__", "__reduce_ex__", "__reduce__",
                    "__getstate__", "__sizeof__", "__dir__", "__repr__", "__str__", "__format__", "__init_subclass__",
                    "__set_name__", "__getattribute__", "__setattr__", "__delattr__", "__abstractmethods__",
                    "__type_params__", "__origin__", "__args__", "__or__", "__ror__", "__class_getitem__", "__get__",
                    "__set__", "__delete__", "__wrapped__", "__isabstractmethod__", "__func__", "__self__"))


def _rec(cls, name, inst):
    if ON[0] and name not in IGNORE:
        _add((cls, name, inst, MODS[-1] if MODS else None, BUILDS[-1] if BUILDS else None))


class RecordingMeta(type):
    def __getattribute__(cls, name):
        _rec(cls, name, False)
        return _type_ga(cls, name)


class RecordingABCMeta(RecordingMeta, abc.ABCMeta):
    pass


def _meta_ga(cls, name):
    _rec(cls, name, False)
    return _type_ga(cls, name)


def _make_object_ga(orig):
    def __getattribute__(self, name):
        _rec(type(self), name, True)
        return orig(self, name)

    return __getattribute__


HOOKED_META = set()
HOOKED_ROOT = {}  # root class -> its recording __getattribute__
_build_class = builtins.__build_class__


def _sqlglot(name):
    return isinstance(name, str) and (name == "sqlglot" or name.startswith("sqlglot."))


def _build_class_rec(func, name, *bases, **kw):
    mod = func.__globals__.get("__name__")
    if not _sqlglot(mod):
        return _build_class(func, name, *bases, **kw)
    if not all(isinstance(b, type) for b in bases):
        pass  # a base that is not a class (typing.TypedDict, Generic[T]): left as it is, not hooked
    elif "metaclass" not in kw:
        metas = {type(b) for b in bases}
        if metas <= {type, RecordingMeta}:
            kw["metaclass"] = RecordingMeta
        elif metas <= {type, RecordingMeta, abc.ABCMeta, RecordingABCMeta}:
            kw["metaclass"] = RecordingABCMeta
    elif kw["metaclass"] is abc.ABCMeta:
        kw["metaclass"] = RecordingABCMeta
    entry = [None]
    BUILDS.append(entry)
    try:
        cls = _build_class(func, name, *bases, **kw)
    finally:
        BUILDS.pop()
    entry[0] = cls
    if isinstance(cls, type):
        if issubclass(cls, type):
            # a metaclass that sqlglot defines: its classes record their reads
            if "__getattribute__" not in _class_dict(cls):
                type.__setattr__(cls, "__getattribute__", _meta_ga)
                HOOKED_META.add(cls)
        else:
            # object reads: the first sqlglot class of the hierarchy records
            definer = None
            for k in cls.__mro__[:-1]:
                if "__getattribute__" in _class_dict(k):
                    definer = k
                    break
            if definer is None:
                orig = None
                for k in cls.__mro__[1:]:
                    if "__getattribute__" in _class_dict(k):
                        orig = _class_dict(k)["__getattribute__"]
                        break
                try:
                    hook = _make_object_ga(orig)
                    type.__setattr__(cls, "__getattribute__", hook)
                    HOOKED_ROOT[cls] = hook
                except (TypeError, AttributeError):
                    pass
    return cls


def provider(cls, name):
    for k in cls.__mro__:
        if name in _class_dict(k):
            return k
    return None


def class_key(cls):
    import statefp

    mod = _type_ga(cls, "__module__")
    if not _sqlglot(mod) or not statefp._resolves(cls):
        return None
    return mod + ":" + _type_ga(cls, "__qualname__")


def is_hooked(cls):
    """Both kinds of reads of this class and its objects were recorded."""
    meta = type(cls)
    if not (meta is RecordingMeta or meta is RecordingABCMeta or any(m in HOOKED_META for m in meta.__mro__)):
        return False
    for k in cls.__mro__[:-1]:
        ga = _class_dict(k).get("__getattribute__")
        if ga is not None:
            return HOOKED_ROOT.get(k) is ga
    return False


def main():
    out_path = sys.argv[1]
    import preload

    preload.install_environment(ROOT)
    from importlib.machinery import SourceFileLoader

    orig_exec = SourceFileLoader.exec_module

    def exec_module(self, module):
        MODS.append(module.__name__)
        try:
            return orig_exec(self, module)
        finally:
            MODS.pop()

    SourceFileLoader.exec_module = exec_module
    builtins.__build_class__ = _build_class_rec
    ON[0] = True
    try:
        preload.import_sqlglot()
    finally:
        ON[0] = False
        builtins.__build_class__ = _build_class
        SourceFileLoader.exec_module = orig_exec
    reads = {}
    n_own = 0
    for cls, name, inst, mod, entry in READS:
        creating = entry[0] if entry is not None else None
        if name == "__dict__":
            if inst:
                continue  # the object's own dict
            p, attr = cls, "*"
        else:
            if inst:
                pass  # (an attribute of the object itself is found in no class dict below, or in a slot descriptor)
            p, attr = provider(cls, name), name
            if p is None:
                continue
        if creating is p:
            n_own += 1
            continue
        key = class_key(p)
        if key is None:
            continue
        ctx = [mod, class_key(creating) if isinstance(creating, type) else None]
        if isinstance(creating, type) and ctx[1] is None:
            ctx[1] = "?"
        lst = reads.setdefault(key, {}).setdefault(attr, [])
        if ctx not in lst:
            lst.append(ctx)
    hooked = []
    unhooked = []
    seen = set()
    for m in list(sys.modules.values()):
        if m is None or not _sqlglot(getattr(m, "__name__", None)):
            continue
        stack = [v for v in vars(m).values() if isinstance(v, type)]
        while stack:
            c = stack.pop()
            if id(c) in seen:
                continue
            seen.add(id(c))
            key = class_key(c)
            if key is None:
                continue
            (hooked if is_hooked(c) else unhooked).append(key)
            stack.extend(v for v in _class_dict(c).values() if isinstance(v, type))
    with open(out_path, "w") as f:
        json.dump({"version": 1, "hooked": sorted(hooked), "reads": reads}, f, separators=(",", ":"), sort_keys=True)
    print("attr_import_reads: %d read events, %d by the creation of the own class, %d classes with readers, %d hooked, %d not hooked"
          % (len(READS), n_own, len(reads), len(hooked), len(unhooked)), file=sys.stderr)
    print("not hooked:", sorted(unhooked)[:40], file=sys.stderr)


if __name__ == "__main__":
    main()
