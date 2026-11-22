"""Import-time commits that change one class-level statement of one class, in a module that
everything imports (so that no small importer cone exists, see cone.py).

The class statement of the patched text is executed again, alone, in the names that its module
held when the class was first created. The new class is compared with the pinned one, attribute
by attribute (statefp). When the attributes that differ hold plain data, and the map of import-
time attribute reads (maps/attr_import_reads.json, build_attr_import_reads.py) shows that on the
pinned tree nothing but the creation of the class itself reads them during the import, the import
of the patched tree differs from the pinned one in exactly these attributes: the new values are
put into the pinned class object (which every other module refers to) and the new class is
dropped. Any doubt raises cone.Redo, and the whole tree is imported instead.
"""
from __future__ import annotations

import ast
import builtins
import gc
import marshal
import sys
import types
import weakref

import cone
import statefp

CO_OPTIMIZED = 0x1
MAX_LINES = 20000
_COMPILED = {}  # path -> (text, code) of the last module compiled (the build-time check compiles one module for many classes)


def class_slice(lines, name):
    """(first, last) 0-based line numbers of the top-level statement 'class name', or None."""
    heads = [i for i, line in enumerate(lines) if line.startswith("class " + name + "(") or line.startswith("class " + name + ":")]
    if len(heads) != 1:
        return None
    h = heads[0]
    if h and lines[h - 1].startswith("@"):
        return None
    last = h
    for i in range(h + 1, len(lines)):
        line = lines[i]
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if line[0] not in " \t":
            break
        last = i
    return h, last


def prepare(mod_name, name, text, full_path, reads, base_state, base_codes, alias):
    """Creates the patched class beside the pinned one. Returns (pinned class, new class, names of
    the attributes to set, their statefp keys in class order, the Fingerprinter); raises cone.Redo."""
    mod = sys.modules.get(mod_name)
    old = vars(mod).get(name) if mod is not None else None
    if not isinstance(old, type) or old.__module__ != mod_name or old.__qualname__ != name:
        raise cone.Redo("class route: no such class")
    if reads is None or "*" in reads:
        raise cone.Redo("class route: the class dict is read at import")
    lines = text.split("\n")
    span = class_slice(lines, name)
    if span is None or span[1] - span[0] > MAX_LINES:
        raise cone.Redo("class route: class statement not found")
    h, last = span
    try:
        tree = ast.parse("\n".join(lines[h:last + 1]))
    except (SyntaxError, ValueError):
        raise cone.Redo("class route: class statement does not parse") from None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
        raise cone.Redo("class route: not one class statement")
    node = tree.body[0]
    if node.name != name or node.decorator_list or node.keywords or getattr(node, "type_params", None):
        raise cone.Redo("class route: decorators or keywords")
    # the names that the module held when the class statement first ran
    g = {}
    found = False
    for k, v in vars(mod).items():
        if k == name:
            found = True
            break
        g[k] = v
    if not found:
        raise cone.Redo("class route: class not in its module")
    try:
        bases = tuple(eval(compile(ast.Expression(b), full_path, "eval"), g) for b in node.bases)  # noqa: S307
    except Exception:  # noqa: BLE001
        raise cone.Redo("class route: bases") from None
    if not bases:
        bases = (object,)
    if len(bases) != len(old.__bases__) or any(a is not b for a, b in zip(bases, old.__bases__)):
        raise cone.Redo("class route: other bases")
    code = _COMPILED.get(full_path)
    if code is None or code[0] is not text:
        try:
            code = compile(text.encode("utf-8", "surrogateescape"), full_path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError):
            raise cone.Redo("class route: module does not compile") from None
        code = (text, marshal.loads(marshal.dumps(code)))  # as engine._MemLoader does
        _COMPILED.clear()
        _COMPILED[full_path] = code
    code = code[1]
    bodies = [k for k in code.co_consts if isinstance(k, types.CodeType) and k.co_name == name and not k.co_flags & CO_OPTIMIZED]
    if len(bodies) != 1 or bodies[0].co_firstlineno != h + 1:
        raise cone.Redo("class route: class body code not found")
    fn = types.FunctionType(bodies[0], g, name)
    before = tuple(g.items())
    try:
        new = builtins.__build_class__(fn, name, *old.__bases__)
    except BaseException as e:  # noqa: BLE001
        if isinstance(e, KeyboardInterrupt):
            raise
        raise cone.Redo("class route: class body raises %r" % (e,)) from None
    # g is this function's own dict of the module's names: what the class body (or anything it
    # called) stored through globals() would be lost, so it must hold exactly what it held
    if len(g) != len(before) or any(k is not k2 or v is not v2 for (k, v), (k2, v2) in zip(g.items(), before)):
        raise cone.Redo("class route: the class body writes module names")
    if type(new) is not type(old) or new.__module__ != mod_name or new.__qualname__ != name:
        raise cone.Redo("class route: other kind of class")
    f = statefp.Fingerprinter(alias, base_codes)
    f.objects = {}
    f.function_keys = set()
    state = statefp.class_state(new, mod_name, name, f)
    prefix = name + "."
    base_keys = [k for k in base_state if k[1] == mod_name and (k[2] == name or (isinstance(k[2], str) and k[2].startswith(prefix)))
                 and k[0] in ("A", "O")]
    changed = [k for k, v in state.items() if base_state.get(k, cone._MISSING) != v]
    if any(k not in state for k in base_keys):
        raise cone.Redo("class route: an attribute is gone")
    names = []
    od, nd = old.__dict__, new.__dict__
    for k in changed:
        if len(k) != 4 or k[0] != "A" or k[2] != name or k[3] == "__bases__" or k not in base_state:
            raise cone.Redo("class route: change of another kind %r" % (k,))
        a = k[3]
        if a not in od or a not in nd or a in reads:
            raise cone.Redo("class route: %s is read at import" % a)
        for v in (od[a], nd[a]):
            if not isinstance(v, (bool, int, float, str, bytes, type(None))) and cone._plain(v) is None:
                raise cone.Redo("class route: %s does not hold plain data" % a)
        names.append(a)
    if tuple(od) != tuple(nd) and [a for a in od if a not in statefp.SKIP_ATTRS] != [a for a in nd if a not in statefp.SKIP_ATTRS]:
        raise cone.Redo("class route: other attribute order")
    # what the fingerprint leaves out (statefp.SKIP_ATTRS) stays as it is in the pinned class, so
    # it must not differ: a docstring, the annotations, the abstract methods, the generic bases
    for a in ("__doc__", "__annotations__", "__abstractmethods__", "__orig_bases__", "__parameters__", "__static_attributes__",
              "__slots__", "__match_args__"):
        x, y = od.get(a, cone._MISSING), nd.get(a, cone._MISSING)
        if x is not y:
            try:
                same = type(x) is type(y) and bool(x == y)
            except Exception:  # noqa: BLE001
                same = False
            if not same:
                raise cone.Redo("class route: %s differs" % a)
    return old, new, names, changed, state, f


def apply(old, new, names):
    """Puts the new values into the pinned class and drops the new class."""
    for a in names:
        setattr(old, a, new.__dict__[a])
    ref = weakref.ref(new)
    return ref


def gone(ref):
    gc.collect()
    return ref() is None
