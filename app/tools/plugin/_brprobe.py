"""Comparison probe for building the operand outcome map (condition coverage of the pinned commit).

The instrumented sqlglot code calls __br_<op>__(site, a, b) in place of "a <op> b" for every
comparison with one operator. The probe computes the original result first (same operands,
same order, same exceptions) and returns it unchanged. It then records, for the current test
phase, the "signature" of the operand pair: for each of the operators is, is not, ==, !=, <,
<=, >, >= whether it gives True, False, or "unknown" (raises, does not return a bool, or cannot
be computed without running user code that might change state). Only operand values of the
pinned run are recorded: nothing here knows about patches.

Records go to one file per process in $BR_PROBE_DIR. Line format:
  P <pid> <kind> <tag>   header (kind: main | worker | sub)
  T <tag>                the following plain keys belong to this test phase
  S <tag>                this test phase submitted a task to (or created) a process pool
  <hex>                  key = site << 16 | signature, seen in the current phase
  G<hex>                 seen while a sqlglot module body was executing (import time)
  X<hex>                 seen in a thread other than the main thread
"""
import builtins
import os
import sys
import _thread

_MAIN_THREAD = _thread.get_ident()
_get_ident = _thread.get_ident
DEPTH = 0  # number of sqlglot module bodies currently executing
_IS_WORKER = "--multiprocessing-fork" in sys.argv  # a spawn pool worker
# a worker's evaluations before its first task (start-up imports) get the tag "W"; a plain
# subprocess inherits the phase of the test that started it
_TAG = "W" if _IS_WORKER else os.environ.get("BR_PROBE_TAG", "?")
_CUR = set()
_SETS = {_TAG: _CUR}
_G = set()
_X = set()
_F = None
_KIND = None
_DIR = os.environ.get("BR_PROBE_DIR")


def _open():
    global _F, _KIND
    if _KIND is None:
        _KIND = "worker" if _IS_WORKER else "sub"
    d = _DIR or "/tmp"
    _F = open(os.path.join(d, "%d.txt" % os.getpid()), "a", buffering=1)
    _F.write("P %d %s %s\n" % (os.getpid(), _KIND, _TAG))


def _w(line):
    if _F is None:
        _open()
    _F.write(line)


def set_main():
    global _KIND
    _KIND = "main"


def settag(tag):
    """Called by the pytest plugin (main process) and by pool tasks (workers)."""
    global _TAG, _CUR
    if tag == _TAG:
        return
    _TAG = tag
    s = _SETS.get(tag)
    if s is None:
        s = _SETS[tag] = set()
    _CUR = s
    os.environ["BR_PROBE_TAG"] = tag  # subprocesses started now inherit the phase
    _w("T %s\n" % tag)


def pool_use():
    _w("S %s\n" % _TAG)


def tagname():
    return _TAG


class Tagged:
    """Wraps a pool task (or pool initializer) so that the worker knows the test phase."""

    __slots__ = ("tag", "fn")

    def __init__(self, tag, fn):
        self.tag = tag
        self.fn = fn

    def __call__(self, *args, **kwargs):
        settag(self.tag)
        return self.fn(*args, **kwargs)

    def __reduce__(self):
        return (_mk_tagged, (self.tag, self.fn))


def _mk_tagged(tag, fn):
    # runs in the worker while the task is unpickled: its arguments are unpickled after this
    settag(tag)
    return Tagged(tag, fn)


def _imp(d):
    global DEPTH
    DEPTH += d


# --- operand classification -------------------------------------------------------------
# kind bits: EQ = == and != are C slots of builtin types (no user code runs);
# ORD = <, <=, >, >= are C slots of builtin types; EXPR = sqlglot Expr equality; CONT = container
EQ, ORD, EXPR, CONT = 1, 2, 4, 8
_PLAIN_TYPES = (object, int, float, complex, str, bytes, bool, type(None))
_PEQ = set()
_PORD = set()
for _t in _PLAIN_TYPES:
    for _n in ("__eq__", "__ne__"):
        _PEQ.add(getattr(_t, _n))
    for _n in ("__lt__", "__le__", "__gt__", "__ge__"):
        _PORD.add(getattr(_t, _n))
_K = {}
_EXPR = None  # sqlglot Expression class (defines __eq__/__hash__), looked up when needed
_EXPR_BASE = None  # sqlglot Expr, the base class that Expression.__hash__ descends into
_NOEXPR = os.environ.get("BR_PROBE_NOEXPR") == "1"  # profiling switch: treat expressions as unknown
# most uncached nodes that one probed == may hash (0: use only hashes that are already cached;
# hashing more makes the suite run several times slower)
_BUDGET = int(os.environ.get("BR_PROBE_BUDGET", "0"))


def _kind(t):
    global _EXPR, _EXPR_BASE
    if t in (tuple, list):
        k = CONT | 16
    elif t in (frozenset, set, dict):
        k = CONT
    else:
        k = 0
        try:
            if t.__eq__ in _PEQ and t.__ne__ in _PEQ:
                k |= EQ
            if t.__lt__ in _PORD and t.__le__ in _PORD and t.__gt__ in _PORD and t.__ge__ in _PORD:
                k |= ORD
            if not k & EQ and not _NOEXPR:
                if _EXPR is None:
                    m = sys.modules.get("sqlglot.expressions.core")
                    if m is not None:
                        _EXPR_BASE = getattr(m, "Expr", None)
                        _EXPR = getattr(m, "Expression", None)
                        if _EXPR is None or _EXPR_BASE is None:
                            _EXPR = _EXPR_BASE = None
                if (
                    _EXPR is not None
                    and t.__eq__ is _EXPR.__eq__
                    and t.__ne__ is _EXPR.__ne__
                    and t.__hash__ is _EXPR.__hash__
                ):
                    k |= EXPR
        except Exception:
            k = 0
        if _EXPR is None and k == 0:
            return k  # do not cache: the Expr class may not be loaded yet
    _K[t] = k
    return k


def _flat(x, need):
    """Container whose elements are all plain (not containers) for the needed kind bits."""
    if len(x) > 16:
        return False
    if type(x) is dict:
        for kk, vv in x.items():
            for e in (kk, vv):
                ke = _K.get(type(e))
                if ke is None:
                    ke = _kind(type(e))
                if ke & CONT or not (ke & need) == need:
                    return False
        return True
    for e in x:
        ke = _K.get(type(e))
        if ke is None:
            ke = _kind(type(e))
        if ke & CONT or not (ke & need) == need:
            return False
    return True


def _code(v):
    if v is True:
        return 1
    if v is False:
        return 0
    return 2


def _expr_eq(a, b):
    """a == b for two sqlglot expressions of the same type, without leaving cached hashes
    behind (Expr.__hash__ caches the hash on every node it visits; those caches are reset)."""
    ha = a._hash
    hb = b._hash
    if ha is not None and hb is not None:
        return 1 if ha == hb else 0  # both hashes cached: Expression.__eq__ compares them
    E = _EXPR_BASE
    nodes = []
    for root in (a, b):
        if root._hash is None:
            stack = [root]
            while stack:
                n = stack.pop()
                nodes.append(n)
                if len(nodes) > _BUDGET:
                    return 2
                for v in n.args.values():
                    if isinstance(v, E):
                        if v._hash is None:
                            stack.append(v)
                    elif type(v) is list:
                        for x in v:
                            if isinstance(x, E) and x._hash is None:
                                stack.append(x)
    try:
        r = _code(a == b)
    except BaseException:
        r = 2
    for n in nodes:
        n._hash = None
    return r


def _sig(a, b, oi, r):
    ta = type(a)
    tb = type(b)
    ka = _K.get(ta)
    if ka is None:
        ka = _kind(ta)
    kb = _K.get(tb)
    if kb is None:
        kb = _kind(tb)
    s = 1 if a is b else 4  # is: bits 0-1, is not: bits 2-3
    # == and !=
    if ka & CONT and not _flat(a, EQ):
        ka2 = 0
    else:
        ka2 = ka | (EQ if ka & CONT else 0)
    if kb & CONT and not _flat(b, EQ):
        kb2 = 0
    else:
        kb2 = kb | (EQ if kb & CONT else 0)
    if ka2 & EQ and kb2 & EQ:
        try:
            e = _code(a == b)
        except BaseException:
            e = 2
        try:
            n = _code(a != b)
        except BaseException:
            n = 2
    elif ka & CONT and ta is tb and (a is b or len(a) != len(b)):
        # same builtin container type: identical -> equal (items compare by identity first);
        # different lengths -> unequal; no item is compared in either case
        e = 1 if a is b else 0
        n = 1 - e
    elif (ka2 | kb2) & EXPR and (ka2 & (EQ | EXPR)) and (kb2 & (EQ | EXPR)):
        # Expr.__eq__: identical -> True; different types -> False; same type -> hash compare
        if a is b:
            e = 1
        elif ta is not tb:
            e = 0
        elif oi == 2:
            e = _code(r)
        elif oi < 2:
            # only the alternatives of an "is" / "is not" site need == of two expressions
            e = _expr_eq(a, b)
        else:
            e = 2
        n = 2 if e == 2 else 1 - e
    else:
        e = n = 2
    s |= e << 4 | n << 6
    # <, <=, >, >=
    oka = ka & ORD or (ka & 16 and _flat(a, ORD))
    okb = kb & ORD or (kb & 16 and _flat(b, ORD))
    if oka and okb:
        try:
            c = _code(a < b)
        except BaseException:
            c = 2
        s |= c << 8
        try:
            c = _code(a <= b)
        except BaseException:
            c = 2
        s |= c << 10
        try:
            c = _code(a > b)
        except BaseException:
            c = 2
        s |= c << 12
        try:
            c = _code(a >= b)
        except BaseException:
            c = 2
        s |= c << 14
    else:
        s |= 0xAA00
    # the operator of the site itself: its real result
    sh = 2 * oi
    s = (s & ~(3 << sh)) | (_code(r) << sh)
    return s


def _rec(i, s):
    k = (i << 16) | s
    if DEPTH:
        if k not in _G:
            _G.add(k)
            _w("G%x\n" % k)
    elif _get_ident() != _MAIN_THREAD:
        if k not in _X:
            _X.add(k)
            _w("X%x\n" % k)
    elif k not in _CUR:
        _CUR.add(k)
        _w("%x\n" % k)


_RAISED = object()

_TEMPLATE = '''
def __br_%(name)s__(i, a, b):
    try:
        r = a %(sym)s b
    except BaseException:
        _rec(i, _sig(a, b, %(oi)d, _RAISED))
        raise
    _rec(i, _sig(a, b, %(oi)d, r))
    return r
'''
_TEMPLATE_IN = '''
def __br_%(name)s__(i, a, b):
    try:
        r = a %(sym)s b
    except BaseException:
        _rec(i, 0xFFFF)
        raise
    _rec(i, 0xFFFF)
    return r
'''
OPS = [("Is", "is"), ("IsNot", "is not"), ("Eq", "=="), ("NotEq", "!="), ("Lt", "<"), ("LtE", "<="),
       ("Gt", ">"), ("GtE", ">="), ("In", "in"), ("NotIn", "not in")]
for _oi, (_name, _sym) in enumerate(OPS):
    _src = (_TEMPLATE if _oi < 8 else _TEMPLATE_IN) % {"name": _name, "sym": _sym, "oi": _oi}
    exec(_src, globals())
    setattr(builtins, "__br_%s__" % _name, globals()["__br_%s__" % _name])
builtins.__br_imp__ = _imp
