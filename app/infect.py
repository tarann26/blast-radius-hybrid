"""Operand outcome map: run-time side (standard library only; the map loads on first use).

What the map is (maps/infect.bin, built by build_infect.py from runs of the pinned commit's
own test suite, no patches involved): for every comparison with one operator in sqlglot/
("a < b", "x is None", ...), the operand relations that occurred when the pinned suite ran,
and in which test. A relation "signature" says, for one evaluation, whether is, is not, ==,
!=, <, <=, >, >= gave True, False or "unknown" for the two operand values. This is condition
coverage with operand relations, a map built from the pinned commit like line coverage.

How it is used: when the edit of a file changes exactly one comparison operator at a mapped
site and nothing else (other changes in the file only comments or layout), a test whose
recorded evaluations of that site give the same result under the new operator runs exactly
as on the pinned commit: same values, same control flow. Such a test cannot fail because of
this edit. infected_tests() returns the tests with at least one evaluation whose result
differs (or is unknown), and whether such an evaluation happened in a phase that is not
attributed to single tests (module import, collection, between tests).
"""
from __future__ import annotations

import ast
import marshal
import re
import zlib

from rawz import unz as _unz
from array import array

OPS = ("Is", "IsNot", "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "In", "NotIn")
OP_INDEX = {name: i for i, name in enumerate(OPS)}
_CODING_RE = re.compile(r"coding[:=]\s*[-\w.]+")

_MAP = {}  # map path -> loaded header (or None)
_FILES = {}  # (map path, file) -> {site key: (operator index, records)}
_SETS = {}  # map path -> list of test-set bytes


class Result:
    __slots__ = ("tests", "global_phase", "site", "old_op", "new_op", "sure", "seen")

    def __init__(self, tests, global_phase, site, old_op, new_op, sure=frozenset(), seen=None):
        self.tests = tests  # frozenset of test numbers with an infected evaluation
        self.sure = sure  # the tests with an evaluation whose result is known to change
        # every test with at least one recorded evaluation of the site, infected or not
        self.seen = seen
        self.global_phase = global_phase  # an infected evaluation outside any single test
        self.site = site
        self.old_op = old_op
        self.new_op = new_op

    def __repr__(self):
        return "Result(%d tests, global=%s, site=%s, %s->%s)" % (
            len(self.tests), self.global_phase, self.site, self.old_op, self.new_op)


def _load(map_path):
    m = _MAP.get(map_path, 0)
    if m == 0:
        try:
            with open(map_path, "rb") as f:
                m = marshal.loads(f.read())
        except (OSError, ValueError, EOFError):
            m = None
        _MAP[map_path] = m
    return m


def _file_sites(map_path, m, rel):
    k = (map_path, rel)
    s = _FILES.get(k)
    if s is None:
        blob = m["files"].get(rel)
        if blob is None:
            return None
        s = {}
        for row in marshal.loads(_unz(blob)):
            s[tuple(row[:4])] = (row[4], row[5])
        _FILES[k] = s
    return s


def _set(map_path, m, sid):
    sets = _SETS.get(map_path)
    if sets is None:
        sets = _SETS[map_path] = marshal.loads(_unz(m["sets"]))
    a = array("H")
    a.frombytes(sets[sid])
    return a


def _seen(map_path, m, set_ids, tests):
    """The tests of the given record sets: every test with a recorded evaluation of a site.
    Only worked out when some evaluation changes (tests is not empty); None otherwise."""
    if not tests:
        return None
    seen = set()
    for sid in set_ids:
        seen.update(_set(map_path, m, sid))
    return frozenset(seen)


def is_infected(sig, old, new):
    """sig: 2 bits per operator (0 False, 1 True, 2 unknown); In / NotIn are never computed."""
    if old >= 8 or new >= 8:
        return True
    a = (sig >> (2 * old)) & 3
    b = (sig >> (2 * new)) & 3
    return a == 2 or b == 2 or a != b


def _blank_or_comment(lines):
    return all((not s.strip()) or s.lstrip().startswith("#") for s in lines)


def _same(a, b, out):
    """Structural equality of two ASTs (positions ignored). One-operator comparisons whose
    operator differs (operands equal) are collected in out; returns False on any other
    difference or when out gets more than one entry."""
    if type(a) is not type(b):
        return False
    if isinstance(a, ast.AST):
        if isinstance(a, ast.Compare) and len(a.ops) == 1 and len(b.ops) == 1 and type(a.ops[0]) is not type(b.ops[0]):
            sub = []  # the operands must be identical: no difference of any kind inside them
            if not (_same(a.left, b.left, sub) and _same(a.comparators[0], b.comparators[0], sub)) or sub:
                return False
            out.append((a, type(a.ops[0]).__name__, type(b.ops[0]).__name__))
            return len(out) <= 1
        for f in a._fields:
            if not _same(getattr(a, f, None), getattr(b, f, None), out):
                return False
        return True
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if not _same(x, y, out):
                return False
        return True
    if isinstance(a, (float, complex)):
        return repr(a) == repr(b)
    return a == b


def _parse_unit(lines, start, end, class_path):
    depth = len(class_path)
    pad = start - 1 - depth
    if pad < 0:
        return None
    head = "".join(" " * k + "class " + name + ":\n" for k, name in enumerate(class_path))
    src = "\n" * pad + head + "\n".join(lines[start - 1:end]) + "\n"
    try:
        return ast.parse(src)
    except (SyntaxError, ValueError):
        return None


def single_compare_edit(old_lines, new_lines, groups, units):
    """(site key, old operator, new operator) when the whole edit of the file is one operator
    change in a one-operator comparison; () when the edit changes no code; None otherwise.
    groups: (old line, n removed, new line, n added, removed lines, added lines) per change;
    units: [start, end, class path, kind] of the module-level and class-level statements of
    the pinned file (the selector's map)."""
    hit = {}
    for g in groups:
        o, nr, nl, na, removed, added = g
        hu = []
        for ui, u in enumerate(units):
            s, e = u[0], u[1]
            if nr > 0:
                if s <= o + nr - 1 and o <= e:
                    hu.append(ui)
            elif s < o <= e:
                hu.append(ui)
        if not hu:
            if not (_blank_or_comment(added) and _blank_or_comment(removed)):
                return None
            if o <= 2 or nl <= 2:
                if any(_CODING_RE.search(x) for x in added + removed):
                    return None
            continue
        for ui in hu:
            hit.setdefault(ui, []).append(g)
    found = []
    for ui in sorted(hit):
        s, e, class_path = units[ui][0], units[ui][1], units[ui][2]
        before = inside = 0
        for g in groups:
            if g[0] + g[1] - 1 < s and g[0] < s:
                before += g[3] - g[1]
            elif g[0] <= e:
                inside += g[3] - g[1]
        a = _parse_unit(old_lines, s, e, class_path)
        b = _parse_unit(new_lines, s + before, e + before + inside, class_path)
        if a is None or b is None:
            return None
        if not _same(a, b, found):
            return None
    if not found:
        return ()  # no code change at all (comments, layout)
    if len(found) != 1:
        return None
    node, old, new = found[0]
    return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset), old, new


def infected_tests(path, old_lines, new_lines, groups, units, map_path):
    """None: not a mapped one-operator edit (use the normal path). Otherwise a Result."""
    m = _load(map_path)
    if m is None or path not in m["files"]:
        return None
    edit = single_compare_edit(old_lines, new_lines, groups, units)
    if not edit:
        return None
    key, old, new = edit
    sites = _file_sites(map_path, m, path)
    site = sites.get(key) if sites else None
    if site is None:
        return None
    op, records = site
    oi, ni = OP_INDEX[old], OP_INDEX[new]
    if op != oi:
        return None
    tests = set()
    sure = set()
    seen = set()
    glob = False
    for i in range(0, len(records), 3):
        sig, sid, g = records[i], records[i + 1], records[i + 2]
        if sid:
            seen.add(sid)
        if is_infected(sig, oi, ni):
            if sid:
                found = _set(map_path, m, sid)
                tests.update(found)
                if oi < 8 and ni < 8:
                    a, b = (sig >> (2 * oi)) & 3, (sig >> (2 * ni)) & 3
                    if a != 2 and b != 2 and a != b:
                        sure.update(found)
            if g:
                glob = True
    return Result(frozenset(tests), glob, (path,) + key, old, new, frozenset(sure), _seen(map_path, m, seen, tests))



# === phase 2 (maps/infect2.bin): and/or trees, literal constants at comparisons ==============
# Built by build_infect.py all2 from its own instrumented runs of the pinned suite. For every
# and/or tree (a BoolOp with 2..8 leaf operands whose parent is not a BoolOp) the map holds the
# outcome signatures of its evaluations per test: for each leaf whether the original run
# evaluated it, its truth (when it can be known without running user code), whether it is
# side-effect free (a leaf the run skipped was then evaluated by the probe with a checked
# evaluator that refuses user code), and, where the result is used as a value, which leaves
# held the identical object. For every comparison with one literal operand (int, bool, str,
# float) it holds the value classes of the other operand per test. From these the effect of
# swapping "and"/"or", or of changing the literal, is computed exactly for every recorded
# evaluation; anything that cannot be decided counts as a changed evaluation.

B_RAISED = (1 << 63) - 1
_MAP2 = {}
_FILES2 = {}


def _load2(map_path):
    m = _MAP2.get(map_path, 0)
    if m == 0:
        try:
            with open(map_path, "rb") as f:
                m = marshal.loads(f.read())
            if m.get("version") != 2:
                m = None
        except (OSError, ValueError, EOFError):
            m = None
        _MAP2[map_path] = m
    return m


def _rows2(map_path, m, kind, rel):
    k = (map_path, kind, rel)
    s = _FILES2.get(k)
    if s is None:
        blob = m[kind].get(rel)
        if blob is None:
            return None
        s = {}
        for row in marshal.loads(_unz(blob)):
            s[tuple(row[:4])] = row[4:]
        _FILES2[k] = s
    return s


def _flatten(node, leaves):
    out = ["O" if isinstance(node.op, ast.Or) else "A"]
    for v in node.values:
        if isinstance(v, ast.BoolOp):
            out.append(_flatten(v, leaves))
        else:
            out.append(len(leaves))
            leaves.append(v)
    return tuple(out)


def _const_val(node):
    """(True, value) for a literal operand as the instrumentation saw it, else (False, None)."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        c = node.operand
        if isinstance(c, ast.Constant) and type(c.value) in (int, float):
            return True, (-c.value if isinstance(node.op, ast.USub) else c.value)
        return False, None
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, str, bool, type(None)):
        return True, node.value
    return False, None


def _same2(a, b, out):
    """Like _same, for the phase 2 edits: an and/or tree whose operators change over identical
    leaves ("bool"), or a literal operand of a one-operator comparison whose value changes
    ("const"). At most one difference; any other difference returns False."""
    if type(a) is not type(b):
        return False
    if isinstance(a, ast.AST):
        if isinstance(a, ast.BoolOp):
            la, lb = [], []
            da, db = _flatten(a, la), _flatten(b, lb)
            if len(la) != len(lb):
                return False
            if da == db:
                for x, y in zip(la, lb):
                    if not _same2(x, y, out):
                        return False
                return True
            for x, y in zip(la, lb):
                sub = []
                if not _same2(x, y, sub) or sub:
                    return False
            out.append(("bool", a, da, db))
            return len(out) <= 1
        if isinstance(a, ast.Compare) and len(a.ops) == 1 and len(b.ops) == 1 and type(a.ops[0]) is type(b.ops[0]):
            ops = (a.left, a.comparators[0]), (b.left, b.comparators[0])
            ka = [_const_val(x) for x in ops[0]]
            kb = [_const_val(x) for x in ops[1]]
            for side, other in ((1, 0), (0, 1)):
                # side: 1 = the literal is the right operand (the instrumentation's convention)
                lit = 1 if side == 1 else 0
                if ka[lit][0] and kb[lit][0] and not ka[other][0]:
                    va, vb = ka[lit][1], kb[lit][1]
                    if type(va) is type(vb) and (repr(va) == repr(vb) if type(va) is float else va == vb):
                        break  # the literal is unchanged: compare normally below
                    sub = []
                    if not _same2(ops[0][other], ops[1][other], sub) or sub:
                        return False
                    out.append(("const", a, side, va, vb))
                    return len(out) <= 1
        for f in a._fields:
            if not _same2(getattr(a, f, None), getattr(b, f, None), out):
                return False
        return True
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if not _same2(x, y, out):
                return False
        return True
    if isinstance(a, (float, complex)):
        return repr(a) == repr(b)
    return type(a) is type(b) and a == b


def single_edit2(old_lines, new_lines, groups, units):
    """The phase 2 edit of the file, or None: ("bool", key, old tree, new tree) or
    ("const", key, op, side, old literal, new literal); key is the position of the and/or
    tree or of the comparison in the pinned file."""
    hit = {}
    for g in groups:
        o, nr, nl, na, removed, added = g
        hu = []
        for ui, u in enumerate(units):
            s, e = u[0], u[1]
            if nr > 0:
                if s <= o + nr - 1 and o <= e:
                    hu.append(ui)
            elif s < o <= e:
                hu.append(ui)
        if not hu:
            if not (_blank_or_comment(added) and _blank_or_comment(removed)):
                return None
            if o <= 2 or nl <= 2:
                if any(_CODING_RE.search(x) for x in added + removed):
                    return None
            continue
        for ui in hu:
            hit.setdefault(ui, []).append(g)
    found = []
    for ui in sorted(hit):
        s, e, class_path = units[ui][0], units[ui][1], units[ui][2]
        before = inside = 0
        for g in groups:
            if g[0] + g[1] - 1 < s and g[0] < s:
                before += g[3] - g[1]
            elif g[0] <= e:
                inside += g[3] - g[1]
        a = _parse_unit(old_lines, s, e, class_path)
        b = _parse_unit(new_lines, s + before, e + before + inside, class_path)
        if a is None or b is None:
            return None
        if not _same2(a, b, found):
            return None
    if len(found) != 1:
        return None
    d = found[0]
    n = d[1]
    key = (n.lineno, n.col_offset, n.end_lineno, n.end_col_offset)
    if d[0] == "bool":
        return ("bool", key, d[2], d[3])
    return ("const", key, type(n.ops[0]).__name__, d[2], d[3], d[4])


class _Unknown(Exception):
    pass


def _bool_changed(sig, n, old, new, vctx):
    """True when the evaluation with signature sig can behave differently under the new tree."""
    if sig == B_RAISED:
        return True
    st = [(sig >> (4 * i)) & 15 for i in range(n)]
    cls = [(sig >> (32 + 3 * i)) & 7 for i in range(n)]
    ev_old = {i for i in range(n) if st[i] & 4}

    def run(node, ev, allowed):
        op = node[0]
        kids = node[1:]
        last = len(kids) - 1
        for k, ch in enumerate(kids):
            if isinstance(ch, tuple):
                r = run(ch, ev, allowed)
            else:
                if not allowed(ch):
                    raise _Unknown
                ev.append(ch)
                r = ch
            if k == last:
                return r
            t = st[r] & 3
            if t == 2:
                raise _Unknown
            if (op == "O") == (t == 1):
                return r
        raise _Unknown

    try:
        e_old = []
        r_old = run(old, e_old, lambda i: st[i] & 4)
        if set(e_old) != ev_old or (ev_old and r_old != max(ev_old)):
            return True  # the record does not fit the pinned tree: do not trust it
        e_new = []
        r_new = run(new, e_new, lambda i: st[i] & 12)
    except _Unknown:
        return True
    s_new = set(e_new)
    for i in ev_old ^ s_new:
        if not st[i] & 8:
            return True  # a leaf with possible side effects runs in only one of the two
    tested = (ev_old | s_new) if not vctx else ((ev_old - {r_old}) | (s_new - {r_new}))
    if not vctx:
        tested |= {r_old, r_new}
    for i in tested:
        if st[i] & 3 == 2:
            return True
    if vctx:
        return not (r_old == r_new or cls[r_old] == cls[r_new])
    return (st[r_old] & 3) != (st[r_new] & 3)


def _decode(code):
    c = code[0]
    if c == "b":
        return code == "b1"
    if c == "n":
        return None
    if c == "i":
        return int(code[1:])
    if c == "s":
        return bytes.fromhex(code[1:]).decode("utf-8", "surrogatepass")
    if c == "f":
        return float.fromhex(code[1:])
    raise _Unknown


_CMP = {
    "Eq": lambda a, b: a == b, "NotEq": lambda a, b: a != b, "Lt": lambda a, b: a < b,
    "LtE": lambda a, b: a <= b, "Gt": lambda a, b: a > b, "GtE": lambda a, b: a >= b,
}


def _cmp_value(op, x, lit, side):
    a, b = (x, lit) if side == 1 else (lit, x)
    if op in ("Is", "IsNot"):
        if type(a) is not type(b):
            r = False
        elif type(a) in (bool, type(None)):
            r = a == b
        else:
            raise _Unknown  # identity of equal ints / strings depends on the interpreter
        return r if op == "Is" else not r
    if type(a) is str and type(b) is not str or type(b) is str and type(a) is not str:
        if op in ("Eq", "NotEq"):
            return op == "NotEq"
        raise _Unknown
    try:
        return bool(_CMP[op](a, b))
    except Exception:
        raise _Unknown


def _const_changed(code, op, side, old, new):
    try:
        if code == "U":
            return True
        if code == "O":
            # == / != / is / is not of an object compared by identity (or an Expression) with
            # a literal: never equal, without user code; orderings are unknown
            if op in ("Is", "Eq"):
                return False
            if op in ("IsNot", "NotEq"):
                return False
            return True
        if code in ("l", "h"):
            # an int more than 4 below / above the pinned literal
            if type(old) is not int or type(new) is not int or op in ("Is", "IsNot"):
                return True
            if code == "l":
                if new < old - 4:
                    return True
                x = old - 5
            else:
                if new > old + 4:
                    return True
                x = old + 5
            return _cmp_value(op, x, old, side) != _cmp_value(op, x, new, side)
        x = _decode(code)
        return _cmp_value(op, x, old, side) != _cmp_value(op, x, new, side)
    except _Unknown:
        return True


def infected_tests2(path, old_lines, new_lines, groups, units, map_path):
    """None: not a mapped phase 2 edit (use the normal path). Otherwise a Result."""
    m = _load2(map_path)
    if m is None:
        return None
    edit = single_edit2(old_lines, new_lines, groups, units)
    if edit is None:
        return None
    kind, key = edit[0], edit[1]
    tests = set()
    seen = set()
    glob = False
    if kind == "bool":
        rows = _rows2(map_path, m, "bfiles", path)
        row = rows.get(key) if rows else None
        if row is None:
            return None
        desc, ctx, recs = row
        if desc != edit[2]:
            return None
        n = _count(desc)
        new = edit[3]
        vctx = ctx == "V"
        for i in range(0, len(recs), 3):
            sig, sid, g = recs[i], recs[i + 1], recs[i + 2]
            if sid:
                seen.add(sid)
            if _bool_changed(sig, n, desc, new, vctx):
                if sid:
                    tests.update(_set(map_path, m, sid))
                if g:
                    glob = True
        return Result(frozenset(tests), glob, (path,) + key, "and/or", "and/or", seen=_seen(map_path, m, seen, tests))
    op, side, old, new = edit[2], edit[3], edit[4], edit[5]
    if old is None or new is None or type(old) not in (int, float, str, bool):
        return None
    rows = _rows2(map_path, m, "kfiles", path)
    row = rows.get(key) if rows else None
    if row is None or OPS[row[0]] != op:
        return None
    recs = row[1]
    for i in range(0, len(recs), 3):
        code, sid, g = recs[i], recs[i + 1], recs[i + 2]
        if sid:
            seen.add(sid)
        if _const_changed(code, op, side, old, new):
            if sid:
                tests.update(_set(map_path, m, sid))
            if g:
                glob = True
    return Result(frozenset(tests), glob, (path,) + key, repr(old), repr(new), seen=_seen(map_path, m, seen, tests))


def _count(desc):
    return sum(_count(c) if isinstance(c, tuple) else 1 for c in desc[1:])


# === which tests ran a statement in the whole-suite runs ======================================


def _records_tests(map_path, m, recs, out):
    for i in range(1, len(recs), 3):
        if recs[i]:
            out.update(_set(map_path, m, recs[i]))


_DEFERRED = (ast.Lambda, ast.GeneratorExp, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _sites_in(node, out):
    """The positions of the comparisons and and/or trees that run as part of the statement
    `node` itself or of the statements nested in it (not the code of a lambda, a generator
    expression or a nested definition, which can run later without the statement)."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _DEFERRED):
            continue
        if isinstance(child, (ast.Compare, ast.BoolOp)):
            out.append((child.lineno, child.col_offset, child.end_lineno, child.end_col_offset))
        _sites_in(child, out)


def statement_runners(path, old_lines, units, key, map_path, map_path2):
    """Tests that are known to have run the statement holding the site `key` (line, column,
    end line, end column in the pinned file) when the maps were recorded, or None when the
    site is in a lambda (its code runs when the lambda is called, not with the statement).

    A test with a record at any mapped site that is part of the statement, of a statement
    nested in it, or of a later statement of the same block has run the statement: control
    reaches those places only through it. The set can be smaller than the truth (a test that
    ran the statement and left no record there is not in it), never larger."""
    unit = None
    for u in units:
        if u[0] <= key[0] and key[2] <= u[1] and (unit is None or u[1] - u[0] < unit[1] - unit[0]):
            unit = u
    if unit is None:
        return None
    tree = _parse_unit(old_lines, unit[0], unit[1], unit[2])
    if tree is None:
        return None
    # the chain of nodes from the tree down to the site
    chain = []

    def descend(node):
        chain.append(node)
        if isinstance(node, (ast.Compare, ast.BoolOp)) and (
            node.lineno, node.col_offset, node.end_lineno, node.end_col_offset
        ) == key:
            return True
        for child in ast.iter_child_nodes(node):
            if descend(child):
                return True
        chain.pop()
        return False

    if not descend(tree):
        return None
    at = max((i for i, n in enumerate(chain) if isinstance(n, ast.stmt)), default=None)
    if at is None or at == 0:
        return None
    stmt = chain[at]
    if isinstance(stmt, _DEFERRED) or any(isinstance(n, ast.Lambda) for n in chain[at:]):
        return None
    block = None  # the list of statements that holds stmt
    for name, value in ast.iter_fields(chain[at - 1]):
        if isinstance(value, list) and any(v is stmt for v in value):
            block = value
    if block is None:
        return None
    keys = []
    for later in block[next(i for i, v in enumerate(block) if v is stmt):]:
        if not isinstance(later, _DEFERRED):
            _sites_in(later, keys)
    out = set()
    m = _load(map_path)
    sites = _file_sites(map_path, m, path) if m is not None and path in m["files"] else None
    m2 = _load2(map_path2)
    brows = _rows2(map_path2, m2, "bfiles", path) if m2 is not None else None
    krows = _rows2(map_path2, m2, "kfiles", path) if m2 is not None else None
    for k in keys:
        if sites and k in sites:
            _records_tests(map_path, m, sites[k][1], out)
        if brows and k in brows:
            _records_tests(map_path2, m2, brows[k][2], out)
        if krows and k in krows:
            _records_tests(map_path2, m2, krows[k][1], out)
    return out


# === tests that a cache hides from the outcome maps ==========================================
# maps/infect_hidden.bin (built by build_hidden.py from runs of the pinned commit's own suite
# with reach probes; no patches involved). The outcome maps come from runs of the whole suite
# in one process. Code that runs once per process and whose result sqlglot keeps shows its
# evaluations only in the first test that needed it. For every mapped site the file lists the
# tests that depend on such a kept result: alone in a fresh process they get to the site's
# statement (or start the site's expression) more often than in their own phases of the
# whole-suite run.

_HIDDEN = {}  # map path -> {file: blob or decoded {site key: test-number bytes}} (or None)


def hidden_tests(map_path, rel, key):
    """The tests that a cache hides at the site `key` (line, column, end line, end column in
    the pinned file) of file `rel`: a frozenset, empty for most sites. None when the map has
    no reach record for the site (no map file, or the records differed between the runs)."""
    m = _HIDDEN.get(map_path, 0)
    if m == 0:
        try:
            with open(map_path, "rb") as f:
                m = marshal.loads(f.read())
            m = dict(m["files"]) if m.get("version") == 1 else None
        except (OSError, ValueError, EOFError, KeyError, TypeError, AttributeError):
            m = None
        _HIDDEN[map_path] = m
    if m is None:
        return None
    rows = m.get(rel)
    if rows is None:
        return None
    if isinstance(rows, bytes):
        rows = m[rel] = {tuple(row[:4]): row[4] for row in marshal.loads(zlib.decompress(rows))}
    raw = rows.get(tuple(key))
    if raw is None:
        return None
    a = array("H")
    a.frombytes(raw)
    return frozenset(a)
