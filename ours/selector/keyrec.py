"""Key-level records of string-keyed dicts held by sqlglot classes and modules.

An edit of one entry of a dict like `KEYWORDS`, `FUNCTIONS` or `TIME_MAPPING` can only
change what a test does where the test looks that key up, enumerates the dict, or uses
something derived from it at import time (a trie, an inverted mapping, a copy in a
subclass). Two pieces make that precise and keep it sound:

* build time (tools/trace_keys.py): every string-keyed dict held by a sqlglot class or
  module is replaced, after all sqlglot modules are imported, by a recording copy
  (`RecDict`); each test then runs alone in a child and records which keys of which
  dict it looked up (`k in d`, `d[k]`, `d.get(k)`, `setdefault`, `pop`, writes) and
  which dicts it enumerated or sized (iteration, `keys`/`items`/`values`, `len`,
  `copy`, comparisons: recorded as "*");
* decision time (`state_fingerprint`, run in a verification child for the pinned tree
  and for the patched tree): every attribute of every sqlglot class and module is
  fingerprinted (dicts per key), so the selector knows every dict entry and every value
  the patch changed, including those derived at import time. A changed entry reaches
  only tests that looked up that key or enumerated that dict; any other changed value,
  or anything that cannot be fingerprinted, falls back to the usual rules.
"""

from __future__ import annotations

import hashlib
import types

LOG: set = set()  # (id of RecDict, key or "*") seen in the current test
RECORDING = [True]


class RecDict(dict):
    """A dict that records key lookups and whole-dict operations in LOG."""

    __slots__ = ()

    def _note(self, key):
        if RECORDING[0]:
            LOG.add((id(self), key_id(key) or "*"))

    def __getitem__(self, key):
        self._note(key)
        return dict.__getitem__(self, key)

    def __contains__(self, key):
        self._note(key)
        return dict.__contains__(self, key)

    def get(self, key, default=None):
        self._note(key)
        return dict.get(self, key, default)

    def setdefault(self, key, default=None):
        self._note(key)
        return dict.setdefault(self, key, default)

    def pop(self, key, *default):
        self._note(key)
        return dict.pop(self, key, *default)

    def __setitem__(self, key, value):
        self._note(key)
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        self._note(key)
        dict.__delitem__(self, key)

    def _all(self):
        if RECORDING[0]:
            LOG.add((id(self), "*"))

    def __iter__(self):
        self._all()
        return dict.__iter__(self)

    def __len__(self):
        self._all()
        return dict.__len__(self)

    def __bool__(self):
        self._all()
        return dict.__len__(self) > 0

    def keys(self):
        self._all()
        return dict.keys(self)

    def values(self):
        self._all()
        return dict.values(self)

    def items(self):
        self._all()
        return dict.items(self)

    def copy(self):
        self._all()
        return dict(dict.items(self))

    def update(self, *args, **kwargs):
        self._all()
        dict.update(self, *args, **kwargs)

    def popitem(self):
        self._all()
        return dict.popitem(self)

    def clear(self):
        self._all()
        dict.clear(self)

    def __eq__(self, other):
        self._all()
        return dict.__eq__(self, other)

    def __ne__(self, other):
        self._all()
        return dict.__ne__(self, other)

    __hash__ = None

    def __or__(self, other):
        self._all()
        return dict.__or__(dict(dict.items(self)), other)

    def __ror__(self, other):
        self._all()
        return dict.__or__(other, dict(dict.items(self)))

    def __ior__(self, other):
        self._all()
        dict.update(self, other)
        return self

    def __repr__(self):
        self._all()
        return dict.__repr__(self)

    def __reversed__(self):
        self._all()
        return dict.__reversed__(self)

    def __reduce__(self):
        self._all()
        return (dict, (dict(dict.items(self)),))

    def __sizeof__(self):
        return dict.__sizeof__(self)


def key_id(key) -> str | None:
    """A process-independent name for a dict key: the string itself, an enum member
    or a class by its qualified name; None for anything else."""
    import enum

    if type(key) is str:
        return key
    if isinstance(key, enum.Enum):
        kind = type(key)
        return f"enum:{kind.__module__}.{kind.__qualname__}.{key.name}"
    if isinstance(key, type):
        return f"class:{key.__module__}:{key.__qualname__}"
    return None


def string_keyed(value) -> bool:
    """An exact dict whose keys all have a key_id (strings, enum members, classes)."""
    return type(value) is dict and all(key_id(k) is not None for k in value)


# -- fingerprints --------------------------------------------------------------
class Unknown(Exception):
    """A value whose equality we cannot decide from its fingerprint."""


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()[:20]


def _code_fp(code, depth):
    hit = _MEMO.get(("code", id(code)))
    if hit is not None and hit[0] is code:
        return hit[1]
    result = _code_fp_uncached(code, depth)
    _MEMO[("code", id(code))] = (code, result, False)
    return result


def _code_fp_uncached(code, depth):
    consts = []
    for c in code.co_consts:
        consts.append(_code_fp(c, depth + 1) if isinstance(c, types.CodeType) else fp(c, depth + 1))
    return _h(repr((code.co_code, tuple(consts), code.co_names, code.co_varnames, code.co_freevars,
                    code.co_cellvars, code.co_argcount, code.co_kwonlyargcount, code.co_flags)))


UNORDERED = [False]  # fingerprint lists/tuples as multisets (for values built from sets)
# Memo of fingerprints within one fingerprinting process: (id, UNORDERED) -> (value,
# fingerprint, whether a list/tuple of 2+ items was fingerprinted inside). The value is
# kept so that its id cannot be reused by another object. Sound because fingerprinting
# runs no sqlglot code, so nothing changes while it runs; the fingerprint of an object
# does not depend on where it is reached (depth only bounds recursion).
_MEMO: dict = {}
SEQ = [0]  # count of ordered sequences of 2+ items fingerprinted (memo hits included)


def _slots_of(kind) -> list:
    out = []
    for c in kind.__mro__:
        names = getattr(c, "__slots__", ())
        if isinstance(names, str):
            names = (names,)
        out.extend(n for n in names if isinstance(n, str) and n not in out)
    return out


# Read the position slots of expression nodes (arg_key, index: where the node sits in
# its parent). The state check after a test run turns this off: shared constant nodes
# (UnixToTime.MILLIS, duckdb's MAX_BIT_POSITION) are attached to trees at run time,
# which moves those slots on the pinned commit too, and the walk already records
# every child's key and position.
EXPR_POSITION = [True]


def _expr_fp(root, depth) -> str:
    """Fingerprint of a sqlglot expression tree: every node's class, args (in order),
    comments, type and meta, walked without recursion (trees can be deep). `parent` is
    the back link of `args` and `_hash` a cache, so neither is read."""
    parts = []
    stack = [("node", root)]
    while stack:
        tag, item = stack.pop()
        if tag == "text":
            parts.append(item)
            continue
        if tag == "leaf":
            parts.append(fp(item, depth + 1))
            continue
        node = item
        slots = _slots_of(type(node))
        if not ("args" in slots and "parent" in slots and isinstance(getattr(node, "args", None), dict)):
            parts.append(fp(node, depth + 1))
            continue
        parts.append(f"({type(node).__module__}.{type(node).__qualname__}")
        todo = []
        extra = getattr(node, "__dict__", None)
        if extra:
            todo.append(("text", "__dict__"))
            todo.append(("leaf", dict(extra)))
        for name in slots:
            if name in ("args", "parent", "_hash") or (name in ("arg_key", "index") and not EXPR_POSITION[0]):
                continue
            todo.append(("text", name))
            try:
                v = getattr(node, name)
            except AttributeError:
                todo.append(("text", "unset"))
                continue
            todo.append(("node" if v is not None and not isinstance(v, (str, int, float, bool, list, dict, tuple)) else "leaf", v))
        for k, v in node.args.items():
            todo.append(("text", repr(k)))
            if type(v) is list:
                todo.append(("text", "["))
                todo.extend(("node", x) for x in v)
                todo.append(("text", "]"))
            else:
                todo.append(("node" if v is not None and not isinstance(v, (str, int, float, bool, tuple, dict)) else "leaf", v))
        todo.append(("text", ")"))
        stack.extend(reversed(todo))
    return "expr:" + _h("\x1f".join(parts))


def fp(value, depth=0) -> str:
    """Deterministic fingerprint: equal fingerprints mean equal behaviour for the
    purpose of reading this value. Raises Unknown when it cannot tell."""
    if depth > 64:
        raise Unknown("too deep")
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return f"{type(value).__name__}:{value!r}"
    key = (id(value), UNORDERED[0])
    hit = _MEMO.get(key)
    if hit is not None and hit[0] is value:
        if hit[2]:
            SEQ[0] += 1
        return hit[1]
    before = SEQ[0]
    result = _fp(value, depth)
    _MEMO[key] = (value, result, SEQ[0] != before)
    return result


def _fp(value, depth) -> str:
    import enum
    import functools
    import re

    import datetime
    import typing

    kind = type(value)
    if isinstance(value, (datetime.date, datetime.time, datetime.timedelta, datetime.tzinfo)):
        return f"{kind.__name__}:{value!r}"
    if isinstance(value, (typing.TypeVar, types.GenericAlias)) or kind.__module__ == "typing":
        return f"typing:{value!r}"
    if kind.__name__ in ("member_descriptor", "getset_descriptor", "wrapper_descriptor", "method_descriptor",
                         "classmethod_descriptor", "_tuplegetter"):
        return f"{kind.__name__}:{getattr(value, '__qualname__', '')}:{getattr(value, '__name__', '')}:{getattr(value, '__doc__', '')!r}"
    import logging

    if isinstance(value, logging.Logger):
        return f"logger:{value.name}:{value.level}:{value.propagate}:{len(value.handlers)}:{value.disabled}"
    if isinstance(value, types.MappingProxyType):
        return "mappingproxy:" + fp(dict(value), depth + 1)
    if kind.__name__ in ("dict_values", "dict_keys", "dict_items"):
        return f"{kind.__name__}:" + _h(repr([fp(v, depth + 1) for v in value]))
    if isinstance(value, enum.Enum):
        # the member's value too: an edit of a member's value keeps its name
        return f"enum:{type(value).__module__}.{type(value).__qualname__}.{value.name}=" + fp(value._value_, depth + 1)
    if isinstance(value, type):
        return f"class:{value.__module__}:{value.__qualname__}"
    if isinstance(value, types.ModuleType):
        return f"module:{value.__name__}"
    if isinstance(value, (types.FunctionType, types.LambdaType)):
        parts = [value.__module__ or "", value.__qualname__, _code_fp(value.__code__, depth)]
        if value.__defaults__:
            parts.append(fp(value.__defaults__, depth + 1))
        if value.__kwdefaults__:
            parts.append(fp(value.__kwdefaults__, depth + 1))
        if value.__closure__:
            cells = []
            for cell in value.__closure__:
                try:
                    cells.append(fp(cell.cell_contents, depth + 1))
                except ValueError:
                    cells.append("empty-cell")
            parts.append(repr(cells))
        return "fn:" + _h(repr(parts))
    if isinstance(value, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        return f"builtin:{getattr(value, '__module__', '')}:{getattr(value, '__qualname__', repr(value))}"
    if isinstance(value, (staticmethod, classmethod)):
        return f"{type(value).__name__}:" + fp(value.__func__, depth + 1)
    if isinstance(value, property):
        return "property:" + _h(repr([fp(f, depth + 1) if f else "-" for f in (value.fget, value.fset, value.fdel)]))
    if isinstance(value, functools.partial):
        return "partial:" + _h(repr([fp(value.func, depth + 1), fp(value.args, depth + 1), fp(value.keywords, depth + 1)]))
    if isinstance(value, re.Pattern):
        return f"re:{value.pattern!r}:{value.flags}"
    if isinstance(value, (list, tuple)):
        if len(value) > 1:
            SEQ[0] += 1
        items = [fp(v, depth + 1) for v in value]
        return f"{type(value).__name__}:" + _h(repr(sorted(items) if UNORDERED[0] else items))
    if isinstance(value, (set, frozenset)):
        return f"{type(value).__name__}:" + _h(repr(sorted(fp(v, depth + 1) for v in value)))
    if isinstance(value, dict):
        # in order: code that iterates a dict sees its insertion order (the multiset
        # reading, for values whose order differs between hash seeds, sorts it)
        if len(value) > 1:
            SEQ[0] += 1
        items = [(fp(k, depth + 1), fp(v, depth + 1)) for k, v in value.items()]
        return f"{type(value).__name__}:" + _h(repr(sorted(items) if UNORDERED[0] else items))
    if isinstance(value, types.MethodType):
        return "method:" + _h(repr([fp(value.__func__, depth + 1), fp(value.__self__, depth + 1)]))
    kind = type(value)
    if kind.__module__ == "_thread" and kind.__name__ in ("lock", "RLock"):
        return f"lock:{kind.__name__}"  # carries no data
    wrapped = getattr(value, "__wrapped__", None)
    if wrapped is not None and callable(value):  # functools.lru_cache / wraps
        return f"wrapped:{kind.__qualname__}:" + fp(wrapped, depth + 1)
    state = getattr(value, "__dict__", None)
    slots = _slots_of(kind)
    if "args" in slots and "parent" in slots and isinstance(getattr(value, "args", None), dict):
        return _expr_fp(value, depth)
    if isinstance(state, dict) and slots:
        # Slotted classes below a class without __slots__ have both: read both (sqlglot
        # expressions keep their content in slots next to an empty __dict__).
        values = [("__dict__", fp(dict(state), depth + 1))]
        for name in slots:
            try:
                values.append((name, fp(getattr(value, name), depth + 1)))
            except AttributeError:
                values.append((name, "unset"))
        return f"slots:{kind.__module__}.{kind.__qualname__}:" + _h(repr(values))
    if isinstance(state, dict):
        return f"obj:{kind.__module__}.{kind.__qualname__}:" + fp(dict(state), depth + 1)
    if slots:
        values = []
        for name in slots:
            try:
                values.append((name, fp(getattr(value, name), depth + 1)))
            except AttributeError:
                values.append((name, "unset"))
        return f"slots:{kind.__module__}.{kind.__qualname__}:" + _h(repr(values))
    raise Unknown(kind.__qualname__)


def state_fingerprint(root_modules=("sqlglot",), loaded_roots=()) -> dict:
    """{holder\\tattr: {"d": {key: fp}} | {"v": fp} | {"u": reason}} for every attribute of
    every class defined in a sqlglot module and every global of every sqlglot module,
    after importing all of them. Holders: "module:qualname" for classes, "module" for
    modules. Modules under `loaded_roots` are visited too, but only those already imported."""
    import importlib
    import pkgutil
    import sys

    import builtins
    import importlib.util

    errors = {}
    _MEMO.clear()
    # Every import executed by sqlglot code while everything is imported (importer,
    # target, fromlist). Tests import only part of sqlglot, so a patch that changes
    # what gets imported (and registered) at import time could change what a test
    # sees without changing this all-modules state; such patches are not narrowed.
    edges = set()
    prefixes = tuple(r + "." for r in root_modules)
    real_import, real_import_module = builtins.__import__, importlib.import_module

    def importer_ok(name):
        return isinstance(name, str) and (name in root_modules or name.startswith(prefixes))

    def recording_import(name, globals=None, locals=None, fromlist=(), level=0):
        importer = (globals or {}).get("__name__")
        if importer_ok(importer):
            target = name
            if level:
                try:
                    package = globals.get("__package__") or importer.rpartition(".")[0]
                    target = importlib.util.resolve_name("." * level + name, package)
                except Exception:  # noqa: BLE001
                    target = f"?{level}:{name}"
            edges.add(f"{importer}\t{target}\t{','.join(sorted(fromlist or ()))}")
        return real_import(name, globals, locals, fromlist, level)

    def recording_import_module(name, package=None):
        importer = sys._getframe(1).f_globals.get("__name__")
        if importer_ok(importer):
            edges.add(f"{importer}\t{name}\t@{package or ''}")
        return real_import_module(name, package)

    builtins.__import__, importlib.import_module = recording_import, recording_import_module
    try:
        for root in root_modules:
            package = importlib.import_module(root)
            for item in pkgutil.walk_packages(package.__path__, root + "."):
                if item.name.endswith("__main__"):
                    continue
                try:
                    importlib.import_module(item.name)
                except Exception as error:  # noqa: BLE001
                    errors[item.name] = repr(error)
    finally:
        builtins.__import__, importlib.import_module = real_import, real_import_module
    out = {}
    seen_classes = set()

    def entry(value):
        try:
            if type(value) is dict or isinstance(value, RecDict):
                ids = [key_id(k) for k in value]
                if all(i is not None for i in ids) and len(set(ids)) == len(ids):
                    return {"d": {i: fp(v) for i, v in zip(ids, dict.values(value))}}
            SEQ[0] = 0
            out = {"v": fp(value)}
            if not SEQ[0]:  # no list/tuple of 2+ items inside: the multiset reading is the same
                return out
            UNORDERED[0] = True
            try:
                unordered = fp(value)
            finally:
                UNORDERED[0] = False
            if unordered != out["v"]:
                out["o"] = unordered
            return out
        except Unknown as error:
            return {"u": str(error)}
        except Exception as error:  # noqa: BLE001
            return {"u": repr(error)}

    def visit_class(cls):
        if id(cls) in seen_classes:
            return
        seen_classes.add(id(cls))
        key = f"{cls.__module__}:{cls.__qualname__}"
        for attr, value in list(vars(cls).items()):
            if attr in ("__dict__", "__weakref__", "__doc__", "__module__", "__qualname__", "__annotations__",
                        "__slots__", "__abstractmethods__", "_abc_impl", "__parameters__", "__orig_bases__"):
                continue
            out[f"{key}\t{attr}"] = entry(value)
            if isinstance(value, type) and value.__module__ == cls.__module__ and value.__qualname__.startswith(cls.__qualname__ + "."):
                visit_class(value)

    for name, module in sorted(sys.modules.items()):
        if not any(name == r or name.startswith(r + ".") for r in tuple(root_modules) + tuple(loaded_roots)) \
                or module is None:
            continue
        for attr, value in list(vars(module).items()):
            if attr.startswith("__") and attr.endswith("__"):
                continue
            if isinstance(value, type) and value.__module__ == name:
                visit_class(value)
                continue
            out[f"{name}\t{attr}"] = entry(value)
    return {"state": out, "import_errors": errors, "imports": sorted(edges)}


# -- decision time -------------------------------------------------------------
def changed_entries(pinned1: dict, pinned2: dict, patched: dict, changed_paths, loop_leftover) -> dict | None:
    """Compare state fingerprints. Returns {holder\tattr: set of changed keys} for
    string-keyed dicts whose entries changed, or None when something else changed (a
    non-dict value, a value that could not be fingerprinted, an import error): then no
    key-level narrowing. Only called for commits whose import-time edits are entries of
    string-keyed dict displays that keep the relative order of their other keys
    (Site.dict_order_same), so no deterministic order changes anywhere: dicts are
    compared as mappings, and a non-dict value whose fingerprint differs only in order
    ("o" equal: built from a set, whose order depends on the process) is unchanged. A
    variable left behind by a `for` loop in a class body holds whichever set element
    came last; it is ignored when the patch does not touch its module
    (loop_leftover(key) -> bool). pinned2 is kept for callers that compare runs."""
    if patched.get("import_errors") != pinned1.get("import_errors"):
        return None
    s1, p = pinned1["state"], patched["state"]
    out: dict = {}
    for key in set(s1) | set(p):
        a, b = s1.get(key), p.get(key)
        if a == b:
            continue
        if a is None or b is None:
            return None  # an attribute appeared or disappeared
        if "u" in a or "u" in b:
            return None
        if "d" in a and "d" in b:
            da, db = a["d"], b["d"]
            keys = {k for k in set(da) | set(db) if da.get(k) != db.get(k)}
            if keys:
                out[key] = keys
            continue
        if "d" in a or "d" in b:
            return None  # a dict became something else, or the reverse
        if a.get("o", a.get("v")) == b.get("o", b.get("v")):
            continue  # same elements; only a set-dependent order differs
        base = key.split("\t", 1)[0].split(":", 1)[0].replace(".", "/")
        if loop_leftover(key) and base + ".py" not in changed_paths and base + "/__init__.py" not in changed_paths:
            continue
        return None
    return out


def state_changes(pinned1: dict, patched: dict, changed_paths, loop_leftover, unstable=frozenset()):
    """Like changed_entries, but non-dict changes are returned instead of rejected:
    ({holder\tattr: changed keys} for dicts, {holder\tattr} for other values), or None
    for an import error, an attribute that appeared or disappeared, or a value that
    could not be fingerprinted. A non-dict value counts as unchanged when its
    fingerprint is equal, or, only for values whose order already differs between two
    hash seeds on the pinned commit (`unstable`: built from a set, so tests cannot rely
    on their order), when its order-insensitive fingerprint is equal. The patched tree
    is fingerprinted under the same hash seed as pinned1, so set orders agree unless
    the sets themselves changed."""
    if patched.get("import_errors") != pinned1.get("import_errors"):
        return None
    if patched.get("imports") != pinned1.get("imports"):
        return None  # other imports run at import time: tests may see other state
    s1, p = pinned1["state"], patched["state"]
    dicts: dict = {}
    others: set = set()
    for key in set(s1) | set(p):
        a, b = s1.get(key), p.get(key)
        if a == b and not ("d" in a and list(a["d"]) != list(b["d"])):  # == ignores dict order
            continue
        if a is None or b is None or "u" in a or "u" in b:
            return None
        if "d" in a and "d" in b:
            da, db = a["d"], b["d"]
            keys = {k for k in set(da) | set(db) if da.get(k) != db.get(k)}
            # The entries keep their dict's order (JSON objects keep insertion order):
            # if the keys present in both are in another order, every reader that
            # iterates the dict may see a difference (recorded as "*").
            reordered = key not in unstable and [k for k in da if k in db] != [k for k in db if k in da]
            if keys or reordered:
                dicts[key] = keys
            continue
        if "d" not in a and "d" not in b and key in unstable \
                and a.get("o", a.get("v")) == b.get("o", b.get("v")):
            continue
        base = key.split("\t", 1)[0].split(":", 1)[0].replace(".", "/")
        if "d" not in a and "d" not in b and loop_leftover(key) \
                and base + ".py" not in changed_paths and base + "/__init__.py" not in changed_paths:
            continue
        others.add(key)
    return dicts, others
