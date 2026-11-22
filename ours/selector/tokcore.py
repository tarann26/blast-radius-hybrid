"""Where the tokenizer core can observe a tokenizer setting.

sqlglot's Tokenizer copies its settings (class attributes such as
NUMERIC_ESCAPES or DROP_UNKNOWN_ESCAPES) into a TokenizerCore when it is
created: Tokenizer._init_core passes `field=self.SETTING` (or
`field=bool(self.SETTING)`), and the core keeps them as plain fields that its
scanning methods read. A change to SETTING in tokenizer class C can therefore
only be seen where a scanning method of a core created by a C tokenizer (or a
subclass) reads that field, or a local copied from it.

`plumbing(tokens_source)` finds the `field=self.SETTING` keywords of
_init_core; `observers(core_source)` finds, per field, the lines that read it.
The build-time tracer (tools/trace_tokcore.py) records, per test, which of those
lines ran for cores of which tokenizer class; the selector uses that record
instead of "every test that created a C tokenizer".

Static rules (anything else falls back to the line that reads the field, which
runs in every call where a later use could happen, so it is a safe stand-in):
  * `x = self.field` with `x` bound nowhere else in the method and not used
    inside a nested function: the lines that use `x` observe the field;
  * an observation is "truth" when only the truth value is used (if/while/
    ternary test, `not`, or an operand of and/or in such a position), "keyed"
    for `in`, subscript and `.get(...)`, and "value" otherwise.
"""

from __future__ import annotations

import ast

CORE_FILE = "sqlglot/tokenizer_core.py"
TOKENS_FILE = "sqlglot/tokens.py"


def _settings_fields(tree: ast.Module) -> set[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TokenizerCore":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return {a.arg for a in item.args.args[1:] + item.args.kwonlyargs}
    return set()


def _parents(tree):
    out = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            out[child] = node
    return out


def _kind(node, parents) -> str:
    """How the value of `node` (an expression) is used: "member" (right operand
    of in / not in: only the keys matter), "keyed" (subscript, .get), "truth"
    (only the truth value matters) or "value" (anything else)."""
    parent = parents.get(node)
    if isinstance(parent, ast.Compare):
        if len(parent.ops) == 1 and parent.comparators[0] is node and isinstance(parent.ops[0], (ast.In, ast.NotIn)):
            return "member"
        return "value"
    if isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.ctx, ast.Load):
        return "keyed"
    if isinstance(parent, ast.Attribute) and parent.value is node and parent.attr == "get":
        call = parents.get(parent)
        if isinstance(call, ast.Call) and call.func is parent:
            return "keyed"
        return "value"
    if _truth_position(node, parents):
        return "truth"
    return "value"


def _truth_position(node, parents) -> bool:
    parent = parents.get(node)
    if isinstance(parent, (ast.If, ast.While, ast.IfExp, ast.Assert)) and parent.test is node:
        return True
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
        return True
    if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == "bool" \
            and len(parent.args) == 1 and parent.args[0] is node and not parent.keywords:
        return True
    if isinstance(parent, ast.BoolOp):
        # and/or return an operand: only its truth value is used when the whole
        # expression is itself in a truth position
        return _truth_position(parent, parents)
    return False


def observers(core_source: str) -> dict[str, list[tuple[int, str]]]:
    """field -> sorted [(line, kind)] of TokenizerCore methods (not __init__)."""
    tree = ast.parse(core_source)
    fields = _settings_fields(tree)
    parents = _parents(tree)
    out: dict[str, set] = {f: set() for f in fields}
    core = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "TokenizerCore"), None)
    if core is None:
        return {}
    for func in core.body:
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) or func.name == "__init__":
            continue
        # names bound in this method, and how often
        bound: dict[str, int] = {}
        nested_names: set[str] = set()
        for node in ast.walk(func):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bound[node.id] = bound.get(node.id, 0) + 1
            if node is not func and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Name):
                        nested_names.add(inner.id)
        for a in func.args.args + func.args.kwonlyargs + func.args.posonlyargs + \
                [x for x in (func.args.vararg, func.args.kwarg) if x is not None]:
            bound[a.arg] = bound.get(a.arg, 0) + 1
        for node in ast.walk(func):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                for name in node.names:
                    bound[name] = bound.get(name, 0) + 2  # not a local: no flow
        for node in ast.walk(func):
            if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"
                    and node.attr in fields and isinstance(node.ctx, ast.Load)):
                continue
            parent = parents.get(node)
            local = None
            if isinstance(parent, ast.Assign) and parent.value is node and len(parent.targets) == 1 \
                    and isinstance(parent.targets[0], ast.Name):
                name = parent.targets[0].id
                if bound.get(name) == 1 and name not in nested_names:
                    local = name
            if local is None:
                out[node.attr].add((node.lineno, _kind(node, parents)))
                continue
            for use in ast.walk(func):
                if isinstance(use, ast.Name) and use.id == local and isinstance(use.ctx, ast.Load):
                    out[node.attr].add((use.lineno, _kind(use, parents)))
    return {f: sorted(v) for f, v in out.items()}


def plumbing(tokens_source: str) -> dict[int, list[tuple[str, str, bool]]]:
    """Line of tokens.py -> [(SETTING, field, through_bool)] for the keywords of the
    TokenizerCore(...) call in Tokenizer._init_core that pass `self.SETTING`
    (through_bool: `bool(self.SETTING)`)."""
    tree = ast.parse(tokens_source)
    out: dict[int, list] = {}
    for cls in ast.walk(tree):
        if not (isinstance(cls, ast.ClassDef) and cls.name == "Tokenizer"):
            continue
        for func in cls.body:
            if not (isinstance(func, ast.FunctionDef) and func.name == "_init_core"):
                continue
            for call in ast.walk(func):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "TokenizerCore"):
                    continue
                for kw in call.keywords:
                    value, through_bool = kw.value, False
                    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "bool" \
                            and len(value.args) == 1 and not value.keywords:
                        value, through_bool = value.args[0], True
                    if kw.arg and isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) \
                            and value.value.id == "self" and kw.value.lineno == kw.value.end_lineno:
                        out.setdefault(kw.value.lineno, []).append((value.attr, kw.arg, through_bool))
    return out


def init_core_dump(tokens_source: str) -> str | None:
    """AST dump of Tokenizer._init_core (to tell whether a patch changed it)."""
    try:
        tree = ast.parse(tokens_source)
    except SyntaxError:
        return None
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == "Tokenizer":
            for func in cls.body:
                if isinstance(func, ast.FunctionDef) and func.name == "_init_core":
                    return ast.dump(func)
    return None


def parents_of(tree) -> dict:
    return _parents(tree)


def mentions_by_line(tree, names) -> dict:
    """line -> [Attribute `X.NAME` (by its last line) / Name `NAME` nodes] for `names`."""
    out: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in names:
            out.setdefault(node.end_lineno, []).append(node)
        elif isinstance(node, ast.Name) and node.id in names:
            out.setdefault(node.lineno, []).append(node)
    return out


def mention_index(tree) -> dict:
    """name -> line -> nodes, for every Attribute (by its last line) and Name."""
    out: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            out.setdefault(node.attr, {}).setdefault(node.end_lineno, []).append(node)
        elif isinstance(node, ast.Name):
            out.setdefault(node.id, {}).setdefault(node.lineno, []).append(node)
    return out


def mention_kinds(tree, parents, line: int, names, text: str, by_line=None) -> set | None:
    """How the reads of `names` on `line` are used, following a local they are
    assigned to inside the enclosing function (same rules as observers()). None
    when some mention on the line is not a plain read we can classify (a store,
    text the AST does not show, a local bound twice, ...)."""
    import re

    expected = sum(len(re.findall(r"\b" + re.escape(n) + r"\b", text)) for n in names)
    if by_line is None:
        by_line = mentions_by_line(tree, names)
    nodes = by_line.get(line, [])
    if not nodes or len(nodes) != expected:
        return None
    kinds = set()
    for node in nodes:
        if not isinstance(node.ctx, ast.Load):
            return None
        parent = parents.get(node)
        func = parent
        while func is not None and not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                                                          ast.ClassDef, ast.Module)):
            func = parents.get(func)
        local = None
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(parent, ast.Assign) \
                and parent.value is node and len(parent.targets) == 1 and isinstance(parent.targets[0], ast.Name):
            name = parent.targets[0].id
            bound = 0
            nested = False
            for inner in ast.walk(func):
                if isinstance(inner, ast.Name) and inner.id == name and isinstance(inner.ctx, (ast.Store, ast.Del)):
                    bound += 1
                if inner is not func and isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                                                            ast.ClassDef)):
                    if any(isinstance(x, ast.Name) and x.id == name for x in ast.walk(inner)):
                        nested = True
                if isinstance(inner, (ast.Global, ast.Nonlocal)) and name in inner.names:
                    nested = True  # not a local: other scopes see it
            args = func.args
            if any(a.arg == name for a in args.args + args.kwonlyargs + args.posonlyargs) \
                    or (args.vararg and args.vararg.arg == name) or (args.kwarg and args.kwarg.arg == name):
                bound += 1
            if bound == 1 and not nested:
                local = name
        if local is None:
            kinds.add(_kind(node, parents))
            continue
        for use in ast.walk(func):
            if isinstance(use, ast.Name) and use.id == local and isinstance(use.ctx, ast.Load):
                kinds.add(_kind(use, parents))
    return kinds
