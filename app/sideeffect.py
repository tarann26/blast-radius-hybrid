"""Import-time statements that change state outside their own module.

A statement that runs when its module is imported (module level, class body, and the blocks
inside them, not function bodies) can change state that the import-state diff (statefp) does
not see and that no reader map bounds: it stores into an attribute or an item of an object
that the module did not create (another module, a logger, sys.modules, builtins, a class of
another module), or it calls a function only for its effect (a bare call such as
logging.disable(), csv.field_size_limit(), gc.disable(), warnings.simplefilter()). Such a
change reaches every test that runs after the module is imported, whether or not that test
uses the module (lazily imported dialects load in the middle of a run). So a patch that adds,
changes or removes such a statement selects every test."""
from __future__ import annotations

import ast
import re


_OWN_CALLS = frozenset(("dict", "list", "set", "frozenset", "tuple", "defaultdict", "OrderedDict", "Counter", "deepcopy", "copy", "sorted", "bytearray"))
_OWN_METHODS = frozenset(("copy", "fromkeys"))
_OWN_VALUES = (ast.Dict, ast.List, ast.Set, ast.Tuple, ast.DictComp, ast.ListComp, ast.SetComp, ast.Constant, ast.JoinedStr, ast.BinOp, ast.Lambda)
# builtins that a bare call can use without an effect that outlives the call
_HARMLESS_BUILTINS = frozenset(("print", "len", "isinstance", "issubclass", "repr", "str", "int", "hash", "id", "type", "iter", "next", "sorted", "list", "dict", "set", "tuple", "frozenset", "any", "all", "min", "max", "sum", "range", "zip", "map", "filter", "enumerate", "callable", "getattr", "hasattr"))


def _own_value(v):
    if isinstance(v, _OWN_VALUES):
        return True
    if isinstance(v, ast.Call):
        f = v.func
        if isinstance(f, ast.Name) and f.id in _OWN_CALLS:
            return True
        if isinstance(f, ast.Attribute) and (f.attr in _OWN_METHODS or f.attr in _OWN_CALLS):
            return True
    return False


def _local_names(body):
    """Names that a scope (module or class body) binds to objects it creates itself:
    classes, functions, and names assigned from a display or a copy."""
    out, bound = set(), set()
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
            bound.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(t, ast.Name):
                    bound.add(t.id)
                    if node.value is not None and _own_value(node.value):
                        out.add(t.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
    return out, bound


def _root(node):
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _import_time_stmts(body, scope):
    """Statements run at import time, with the (own, bound) names of their scopes."""
    for node in body:
        yield node, scope
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.If) and _is_main_guard(node.test):
            continue  # runs only when the file is the program, never when a test imports it
        if isinstance(node, ast.ClassDef):
            own, bound = _local_names(node.body)
            yield from _import_time_stmts(node.body, (scope[0] | own, scope[1] | bound, True))
            continue
        for field in ("body", "orelse", "finalbody"):
            sub = getattr(node, field, None)
            if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                yield from _import_time_stmts(sub, scope)
        for h in getattr(node, "handlers", []) or []:
            yield from _import_time_stmts(h.body, scope)
        for c in getattr(node, "cases", []) or []:
            yield from _import_time_stmts(c.body, scope)


_SIMPLE = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _effects(stmt, scope):
    """Why this import-time statement can change outside state ('' = it cannot, by these rules)."""
    local, bound, in_class = scope
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return ""  # their headers (decorators, bases) are expressions: see the Expr rule below
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        f = stmt.value.func
        r = _root(f)
        if isinstance(f, ast.Name) and r not in bound and r in _HARMLESS_BUILTINS:
            return ""
        if isinstance(f, ast.Attribute) and r in local and not (isinstance(f.value, ast.Name) and r in local and _is_func(r, stmt)):
            return ""  # a method of an object that this scope made (KEYWORDS.pop(...))
        return "bare call %s" % (ast.unparse(f)[:60],)
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Await, ast.Yield, ast.YieldFrom)):
        return "bare call"
    targets = []
    if isinstance(stmt, ast.Assign):
        targets = stmt.targets
    elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
        targets = [stmt.target]
    elif isinstance(stmt, ast.Delete):
        targets = stmt.targets
    elif isinstance(stmt, (ast.For, ast.With)):
        targets = [stmt.target] if isinstance(stmt, ast.For) else [i.optional_vars for i in stmt.items if i.optional_vars is not None]
    for t in targets:
        for n in ast.walk(t):
            if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(getattr(n, "ctx", None), (ast.Store, ast.Del)):
                r = _root(n)
                if r is not None and r in local:
                    continue
                return "store into %s" % (ast.unparse(n)[:60],)
    # walrus or setattr-like calls inside expressions of the statement
    for n in ast.walk(stmt) if not isinstance(stmt, _SIMPLE) else []:
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("setattr", "delattr", "exec", "eval", "__import__", "globals", "vars"):
            return "call %s" % n.func.id
    return ""


def _is_main_guard(test):
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"
            and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value == "__main__")


def _is_func(name, stmt):
    return False


def import_time_effects(old_text, new_text, groups=None):
    """Reasons ('' none) why the patch's change of one Python file alters import-time state
    outside that module. old_text None = new file, new_text None = deleted file. groups: the
    edit groups of engine.apply_file_patch (old_line, n_removed, new_line, n_added, ...);
    None = every line of both texts counts as changed."""
    reasons = []
    old_ch, new_ch, inserts = set(), set(), set()
    if groups is None or old_text is None or new_text is None:
        old_ch = set(range(1, (old_text or "").count("\n") + 2))
        new_ch = set(range(1, (new_text or "").count("\n") + 2))
    else:
        for g in groups:
            old_ch.update(range(g[0], g[0] + g[1]))
            new_ch.update(range(g[2], g[2] + g[3]))
            if not g[1]:
                inserts.add(g[0] - 1)  # the insertion sits between old lines g[0]-1 and g[0]
    for text, changed, ins in ((old_text, old_ch, inserts), (new_text, new_ch, ())):
        if not text:
            continue
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return ""  # a file that does not compile is handled by the broken-module rules
        own, bound = _local_names(tree.body)
        for stmt, scope in _import_time_stmts(tree.body, (own, bound, False)):
            if isinstance(stmt, _SIMPLE):
                # only the header lines of a compound statement belong to it here
                end = stmt.body[0].lineno - 1 if getattr(stmt, "body", None) else stmt.lineno
                lo, hi = stmt.lineno, max(stmt.lineno, end)
            else:
                lo, hi = stmt.lineno, stmt.end_lineno or stmt.lineno
            if not any(lo <= x <= hi for x in changed) and not any(lo <= p and p + 1 <= hi for p in ins):
                continue
            why = _effects(stmt, scope)
            if why:
                reasons.append("line %d: %s" % (stmt.lineno, why))
    return "; ".join(sorted(set(reasons))[:3])


# ---- the same check on slices of the text (ast.parse of parser.py alone costs about 0.3 s)

_FALLBACK = object()
_NAME_LINE = re.compile(r"(\w+)\s*(?::[^=]*)?=(?!=)\s*(.*)")
_IMPORT_LINE = re.compile(r"(?:from\s+\S+\s+)?import\s+(.+)")
_DEF_LINE = re.compile(r"(?:async\s+def|def|class)\s+(\w+)")
_OWN_START = re.compile(r"(?:[\[{(\d'\"]|(?:dict|set|frozenset|list|tuple|defaultdict|OrderedDict|Counter|deepcopy|copy|sorted)\()")


def _scan_names(lines, lo, hi, ind):
    """(own, bound) of the statements at indentation ind in lines[lo:hi], by their first line."""
    own, bound = set(), set()
    k = len(ind)
    for i in range(lo, hi):
        line = lines[i]
        if not line.startswith(ind) or len(line) <= k or line[k] in " \t":
            continue
        s = line[k:]
        m = _DEF_LINE.match(s)
        if m:
            own.add(m.group(1)); bound.add(m.group(1))
            continue
        m = _IMPORT_LINE.match(s)
        if m:
            for part in m.group(1).replace("(", " ").replace(")", " ").split(","):
                w = part.split()
                if w:
                    bound.add((w[-1] if len(w) >= 3 and w[-2] == "as" else w[0]).split(".")[0])
            continue
        m = _NAME_LINE.match(s)
        if m:
            bound.add(m.group(1))
            v = m.group(2).rstrip()
            if _OWN_START.match(v) or v.endswith(".copy()"):
                own.add(m.group(1))
    return own, bound


def _starts(lines, lo, hi, ind):
    k = len(ind)
    out = []
    for i in range(lo, hi):
        line = lines[i]
        if line.startswith(ind) and len(line) > k and line[k] not in " \t)]}#":
            if out and lines[out[-1]][k:].startswith("@") and i == out[-1] + 1:
                continue  # a decorator and what it decorates are one statement
            out.append(i)
    return out


def _level(lines, lo, hi, ind, changed, ins, scope, reasons):
    starts = _starts(lines, lo, hi, ind)
    k = len(ind)
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else hi  # [s, e), 0-based
        if not any(s < x <= e for x in changed) and not any(s < p and p + 1 <= e for p in ins):
            continue
        head = s
        while lines[head][k:].startswith("@"):
            head += 1
        first = lines[head][k:]
        if first.startswith(("def ", "async def ")):
            continue
        if first.startswith("class "):
            body = None
            for i in range(head + 1, e):
                t = lines[i]
                if t.strip() and not t.lstrip().startswith("#"):
                    body = t[: len(t) - len(t.lstrip())]
                    break
            if body is None or len(body) <= k or not lines[head].rstrip().endswith(":"):
                return _FALLBACK  # a header over several lines, or a one-line class
            own, bound = _scan_names(lines, head + 1, e, body)
            got = _level(lines, head + 1, e, body, changed, ins, (scope[0] | own, scope[1] | bound, True), reasons)
            if got is _FALLBACK:
                return got
            continue
        text = "\n".join(t[k:] if t.startswith(ind) else t.lstrip() for t in lines[s:e])
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return _FALLBACK
        ast.increment_lineno(tree, s)
        for stmt, sc in _import_time_stmts(tree.body, scope):
            if isinstance(stmt, _SIMPLE):
                end = stmt.body[0].lineno - 1 if getattr(stmt, "body", None) else stmt.lineno
                a, b = stmt.lineno, max(stmt.lineno, end)
            else:
                a, b = stmt.lineno, stmt.end_lineno or stmt.lineno
            if not any(a <= x <= b for x in changed) and not any(a <= p and p + 1 <= b for p in ins):
                continue
            why = _effects(stmt, sc)
            if why:
                reasons.append("line %d: %s" % (stmt.lineno, why))
    return None


def import_time_effects_fast(old_text, new_text, groups=None):
    """import_time_effects on the top-level (and class-level) statements around the edit only;
    the whole text is parsed when a slice does not parse alone."""
    if groups is None or old_text is None or new_text is None:
        return import_time_effects(old_text, new_text, groups)
    old_ch, new_ch, inserts = set(), set(), set()
    for g in groups:
        old_ch.update(range(g[0], g[0] + g[1]))
        new_ch.update(range(g[2], g[2] + g[3]))
        if not g[1]:
            inserts.add(g[0] - 1)
    reasons = []
    for text, changed, ins in ((old_text, old_ch, inserts), (new_text, new_ch, ())):
        lines = text.split("\n")
        own, bound = _scan_names(lines, 0, len(lines), "")
        if _level(lines, 0, len(lines), "", changed, ins, (own, bound, False), reasons) is _FALLBACK:
            return import_time_effects(old_text, new_text, groups)
    return "; ".join(sorted(set(reasons))[:3])


# ---- call-time code (function bodies): a changed statement that writes into an imported
# object or calls an imported library for its effect changes state that outlives the test
# that runs it (sqlglot.pretty = True, csv.field_size_limit(8), logging...setLevel(...)).

_ANY_IMPORT = re.compile(r"^[ \t]*(?:from[ \t]+(\.*[\w.]*)[ \t]+import[ \t]+(.+)|import[ \t]+(.+))$", re.M)
_PURE_METHODS = frozenset(("cast", "assert_never", "reveal_type"))  # typing: no effect
_LOG_METHODS = frozenset(("debug", "info", "warning", "warn", "error", "exception", "critical", "log"))


def _imported_names(text):
    """{name: module} of every import statement of the text (any indentation)."""
    out = {}
    for m in _ANY_IMPORT.finditer(text):
        if m.group(3) is not None:
            for part in m.group(3).split(","):
                w = part.split()
                if w:
                    out[(w[-1] if len(w) >= 3 and w[-2] == "as" else w[0].split(".")[0])] = w[0]
        else:
            mod = m.group(1)
            for part in m.group(2).replace("(", " ").replace(")", " ").replace("\\", " ").split(","):
                w = part.split()
                if w and w[0] != "*":
                    out[w[-1] if len(w) >= 3 and w[-2] == "as" else w[0]] = mod
    return out


def _chain_root(node):
    """(root name, whether a call sits in the chain) of a.b[c].d / f(x).y."""
    called = False
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        if isinstance(node, ast.Call):
            called = True
            node = node.func
        else:
            node = node.value
    return (node.id if isinstance(node, ast.Name) else None), called


def _call_time_stmt_effect(stmt, ext_all, ext_lib):
    for node in ast.walk(stmt):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign) or (isinstance(node, ast.AnnAssign) and node.value is not None):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(getattr(n, "ctx", None), (ast.Store, ast.Del)):
                    r, called = _chain_root(n)
                    # through a call only into a library's object (decimal.getcontext().prec):
                    # what a sqlglot function returns is a new object
                    if r in ext_lib or (r in ext_all and not called):
                        return "store into %s" % ast.unparse(n)[:60]
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            f = node.value.func
            r = _root(f)
            if isinstance(f, ast.Attribute) and (f.attr in _LOG_METHODS or f.attr in _PURE_METHODS):
                continue
            if r in ext_lib:
                return "call %s" % ast.unparse(f)[:60]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("setattr", "delattr") and node.args:
            if _root(node.args[0]) in ext_all:
                return "call %s" % node.func.id
    return ""


_CONT_END = ("(", "[", "{", ",", "\\", "+", "-", "*", "/", "%", "|", "&", "=", ":", "and", "or", "not", "in", "is", "else", "if", "lambda")


def _complete_start(text_lines, line_no):
    """Whether the 1-based line can start a statement: the line before it (not blank, not a
    comment) does not leave an expression open. A line that cannot is read only inside its
    whole group."""
    i = line_no - 2
    while i >= 0 and (not text_lines[i].strip() or text_lines[i].lstrip().startswith("#")):
        i -= 1
    if i < 0:
        return True
    prev = text_lines[i].split("#")[0].rstrip() if '"' not in text_lines[i] and "'" not in text_lines[i] else text_lines[i].rstrip()
    if prev.endswith(":") and prev.lstrip().startswith(("if ", "elif ", "else", "for ", "while ", "with ", "try", "except", "finally", "def ", "async ", "class ", "match ", "case ")):
        return True
    return not prev.endswith(_CONT_END)


def _shape(tree):
    """Whether a parsed statement list holds a store into an attribute or item, a bare call
    or a setattr/delattr call: the shapes of _call_time_stmt_effect."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del)):
            return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("setattr", "delattr"):
            return True
    return False


def call_time_effects(old_text, new_text, groups):
    """Reasons ('' none) why a changed statement inside a function (indented lines of the
    edit groups) can change state outside the objects of the call: a store into an imported
    name's attributes or items, or a bare call on an imported library (not sqlglot) other
    than logging. The added and the removed lines are read one group, then one line, at a
    time; a line that continues the line before it in the file is read only within its group."""
    reasons = []
    for text, idx, start in ((old_text, 4, 0), (new_text, 5, 2)):
        if not text:
            continue
        cand = [g for g in groups if any(x[:1] in (" ", "\t") and x.strip() for x in g[idx])]
        if not cand:
            continue
        text_lines = None
        found = []
        for g in cand:
            block = g[idx]
            if text_lines is None:
                text_lines = text.split("\n")
            tries = []
            if _complete_start(text_lines, g[start]):
                tries.append(block)
            if len(block) > 1:
                for j, x in enumerate(block):
                    if j and _complete_start(text_lines, g[start] + j) or (not j and _complete_start(text_lines, g[start])):
                        tries.append([x])
            for lines in tries:
                body = [x for x in lines if x.strip() and not x.lstrip().startswith("#")]
                if not body:
                    continue
                k = min(len(x) - len(x.lstrip()) for x in body)
                if k == 0:
                    continue
                src = "\n".join(x[k:] if len(x) - len(x.lstrip()) >= k else x.lstrip() for x in lines)
                try:
                    tree = ast.parse(src)
                except (SyntaxError, ValueError):
                    continue
                if _shape(tree):
                    found.append((src, tree))
                if len(lines) > 1:
                    break  # the whole group parsed: its lines need no reading one by one
        if not found:
            continue
        imported = _imported_names(text)
        ext_all = set(imported)
        ext_lib = {n for n, m in imported.items() if not (m.startswith("sqlglot") or m.startswith("."))}
        for src, tree in found:
            own_imp = _imported_names(src)  # import csv; csv.field_size_limit(8)
            a = ext_all | set(own_imp)
            b = ext_lib | {n for n, m in own_imp.items() if not (m.startswith("sqlglot") or m.startswith("."))}
            for stmt in tree.body:
                why = _call_time_stmt_effect(stmt, a, b)
                if why:
                    reasons.append(why)
    return "; ".join(sorted(set(reasons))[:3])


def touches_top_level(groups):
    """Whether an edit group adds or removes a line that starts at column 0 (module level)."""
    for g in groups:
        for line in list(g[4]) + list(g[5]):
            s = line.lstrip()
            if s and s[0] != "#" and len(s) == len(line):
                return True
    return False
