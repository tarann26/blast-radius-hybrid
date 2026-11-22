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
  K<site>:<code>         phase 2: value class of the non-constant operand at a compare site with a
                         literal constant operand (KG / KX: import time / other thread)
  B<site>:<sig>          phase 2: outcome signature of one evaluation of an and/or tree (BG / BX)
See the phase 2 section at the end of this file.
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
    if _BO:
        _flush_open()
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


# === phase 2: constants at comparisons, and/or trees =====================================
import ast as _ast
import atexit as _atexit
import json as _json
import types as _types
import enum as _enum

_getframe = sys._getframe
_FAIL = object()
_NOTSET = object()
_MODULE = _types.ModuleType
_FUNC = _types.FunctionType
_MEMBER = _types.MemberDescriptorType
_GETSET = _types.GetSetDescriptorType
_OGA = object.__getattribute__
_TGA = type.__getattribute__
_OEQ = object.__eq__
_ONE = object.__ne__
_TIC = type.__instancecheck__
_ENUM_HASH = _enum.Enum.__hash__
_BUILTIN_DESC = (_types.WrapperDescriptorType, _types.MethodDescriptorType, _types.BuiltinFunctionType,
                 _types.ClassMethodDescriptorType, _FUNC, classmethod, staticmethod)
_SCALARS = (int, str, float, bool, type(None), bytes)
_STATS = {"pfail": 0, "mismatch": 0, "open_flushed": 0, "noopen": 0}
_DEBUG = os.environ.get("BR_PROBE_DEBUG") == "1"

# --- value class of the non-constant operand at a constant comparison -------------------
_KS = {}  # (site, code) seen in the current phase is in _KCUR
_KCUR = set()
_KSETS = {}
_KG = set()
_KX = set()


def _vcode(x):
    t = type(x)
    if t is bool:
        return "b1" if x else "b0"
    if t is int:
        return "i%d" % x if -10**15 < x < 10**15 else "U"
    if x is None:
        return "n"
    if t is str:
        return "s" + x.encode("utf-8", "surrogatepass").hex() if len(x) <= 64 else "U"
    if t is float:
        return "f" + x.hex()
    try:
        if t.__eq__ is _OEQ and t.__ne__ is _ONE and not isinstance(x, (int, str, float, bytes)):
            return "O"  # == and != compare identity: never equal to a literal
        ka = _K.get(t)
        if ka is None:
            ka = _kind(t)
        if ka & EXPR:
            return "O"  # Expression.__eq__: a different type is never equal
    except Exception:
        pass
    return "U"


_KCAP = 24  # distinct value classes per site and phase; more: "U" (treated as unknown)
_KCOUNT = {}


def _reck(i, x, lit):
    try:
        if x is _RAISED:
            c = "U"
        elif type(x) is int and type(lit) is int and not -4 <= x - lit <= 4:
            # far from the literal: only the side matters for literals changed by a few units
            c = "l" if x < lit else "h"
        else:
            c = _vcode(x)
    except BaseException:
        c = "U"
    k = (i, c)
    if DEPTH:
        if k not in _KG:
            _KG.add(k)
            _w("KG%x:%s\n" % k)
    elif _get_ident() != _MAIN_THREAD:
        if k not in _KX:
            _KX.add(k)
            _w("KX%x:%s\n" % k)
    else:
        s = _KSETS.get(_TAG)
        if s is None:
            s = _KSETS[_TAG] = set()
        if k not in s:
            ck = (_TAG, i)
            n = _KCOUNT.get(ck, 0)
            if n >= _KCAP:
                k = (i, "U")
                if k in s:
                    return
            _KCOUNT[ck] = n + 1
            s.add(k)
            _w("K%x:%s\n" % k)


_TEMPLATE_K = '''
def __br_K%(name)s__(i, a, b, side):
    try:
        r = a %(sym)s b
    except BaseException:
        _rec(i, _sig(a, b, %(oi)d, _RAISED))
        _reck(i, _RAISED, None)
        raise
    _rec(i, _sig(a, b, %(oi)d, r))
    if side:
        _reck(i, a, b)
    else:
        _reck(i, b, a)
    return r
'''
_TEMPLATE_KONLY = '''
def __br_k%(name)s__(i, a, b, side):
    try:
        r = a %(sym)s b
    except BaseException:
        _reck(i, _RAISED, None)
        raise
    if side:
        _reck(i, a, b)
    else:
        _reck(i, b, a)
    return r
'''
for _oi, (_name, _sym) in enumerate(OPS[:8]):
    exec(_TEMPLATE_K % {"name": _name, "sym": _sym, "oi": _oi}, globals())
    setattr(builtins, "__br_K%s__" % _name, globals()["__br_K%s__" % _name])
    exec(_TEMPLATE_KONLY % {"name": _name, "sym": _sym}, globals())
    setattr(builtins, "__br_k%s__" % _name, globals()["__br_k%s__" % _name])

# --- checked evaluation without user code ------------------------------------------------
_TRUTH = {}  # type -> 0 always true, 1 bool() runs only builtin code, 2 user code


def _mro_get(t, name):
    for k in t.__mro__:
        d = k.__dict__
        if name in d:
            return d[name]
    return _NOTSET


def _truth_kind(t):
    k = _TRUTH.get(t)
    if k is None:
        b = _mro_get(t, "__bool__")
        n = _mro_get(t, "__len__")
        if b is _NOTSET and n is _NOTSET:
            k = 0
        elif all(x is _NOTSET or type(x) in (_types.WrapperDescriptorType, _types.MethodDescriptorType) for x in (b, n)):
            k = 1
        else:
            k = 2
        _TRUTH[t] = k
    return k


def _ptruth(v):
    """0 / 1: the truth of v, computed without user code; 2: unknown."""
    if v is True:
        return 1
    if v is False or v is None:
        return 0
    t = type(v)
    if t in (int, str, list, tuple, dict, set, frozenset, float, bytes):
        return 1 if v else 0
    k = _truth_kind(t)
    if k == 0:
        return 1
    if k == 1:
        try:
            return 1 if v else 0
        except BaseException:
            return 2
    return 2


def _pname(name, f):
    code = f.f_code
    if code.co_flags & 1:  # optimized frame (function): locals are fast locals or cells
        if name in code.co_varnames or name in code.co_cellvars or name in code.co_freevars:
            try:
                return f.f_locals[name]
            except KeyError:
                return _FAIL
    else:
        return _FAIL
    g = f.f_globals
    if name in g:
        return g[name]
    b = f.f_builtins
    if name in b:
        return b[name]
    return _FAIL


def _pattr(o, name):
    t = type(o)
    if t is _MODULE:
        d = o.__dict__
        return d[name] if name in d else _FAIL
    if isinstance(o, type):
        if t.__getattribute__ is not _TGA:
            return _FAIL
        m = _mro_get(t, name)
        if m is not _NOTSET:
            mt = type(m)
            if hasattr(mt, "__set__") or hasattr(mt, "__delete__"):
                return _FAIL  # data descriptor of the metaclass
        try:
            mro = _TGA(o, "__mro__")
        except Exception:
            return _FAIL
        for k in mro:
            d = k.__dict__
            if name in d:
                v = d[name]
                vt = type(v)
                if vt is _FUNC:
                    return v
                if vt is staticmethod:
                    return v.__func__
                if vt is classmethod:
                    return _types.MethodType(v.__func__, o)
                if hasattr(vt, "__get__"):
                    return _FAIL
                return v
        return _FAIL
    if t.__getattribute__ is not _OGA:
        return _FAIL
    c = _mro_get(t, name)
    if c is not _NOTSET:
        ct = type(c)
        if hasattr(ct, "__set__") or hasattr(ct, "__delete__"):
            if ct is _MEMBER or ct is _GETSET:
                try:
                    return c.__get__(o, t)
                except AttributeError:
                    return _FAIL
            return _FAIL
    try:
        d = _OGA(o, "__dict__")
    except AttributeError:
        d = None
    if d is not None and type(d) is dict and name in d:
        return d[name]
    if c is not _NOTSET:
        ct = type(c)
        if hasattr(ct, "__get__"):
            if ct in _BUILTIN_DESC:
                try:
                    return c.__get__(o, t)
                except Exception:
                    return _FAIL
            return _FAIL
        return c
    return _FAIL


def _plain_hash(t):
    h = _mro_get(t, "__hash__")
    return h is _ENUM_HASH or type(h) is _types.WrapperDescriptorType


def _eq_kind(x):
    t = type(x)
    k = _K.get(t)
    if k is None:
        k = _kind(t)
    return k


def _peq(a, b):
    """a == b without user code: 0 / 1, or None."""
    if a is b and type(a) is not float:
        return 1
    ka = _eq_kind(a)
    kb = _eq_kind(b)
    if ka & CONT and not _flat(a, EQ):
        ka = 0
    elif ka & CONT:
        ka |= EQ
    if kb & CONT and not _flat(b, EQ):
        kb = 0
    elif kb & CONT:
        kb |= EQ
    if ka & EQ and kb & EQ:
        try:
            return 1 if a == b else 0
        except BaseException:
            return None
    if (ka | kb) & EXPR and ka & (EQ | EXPR) and kb & (EQ | EXPR):
        if type(a) is not type(b):
            return 0
        ha = a._hash
        hb = b._hash
        if ha is not None and hb is not None:
            return 1 if ha == hb else 0
    return None


def _pcontains(c, x):
    tc = type(c)
    if tc is str:
        if type(x) is str:
            return x in c
        return _FAIL
    if tc in (tuple, list):
        if len(c) > 16:
            return _FAIL
        for e in c:
            r = _peq(e, x)
            if r is None:
                return _FAIL
            if r:
                return True
        return False
    if tc in (set, frozenset, dict):
        if len(c) > 16 or not (_eq_kind(x) & EQ) or not _plain_hash(type(x)):
            return _FAIL
        for e in c:
            if not (_eq_kind(e) & EQ):
                return _FAIL
        try:
            return x in c
        except BaseException:
            return _FAIL
    return _FAIL


def _pcmp(op, a, b):
    if op is _ast.Is:
        return a is b
    if op is _ast.IsNot:
        return a is not b
    if op is _ast.Eq or op is _ast.NotEq:
        r = _peq(a, b)
        if r is None:
            return _FAIL
        return bool(r) if op is _ast.Eq else not r
    if op is _ast.In or op is _ast.NotIn:
        r = _pcontains(b, a)
        if r is _FAIL:
            return r
        return r if op is _ast.In else not r
    ka = _eq_kind(a)
    kb = _eq_kind(b)
    oka = ka & ORD or (ka & 16 and _flat(a, ORD))
    okb = kb & ORD or (kb & 16 and _flat(b, ORD))
    if not (oka and okb):
        return _FAIL
    try:
        if op is _ast.Lt:
            return a < b
        if op is _ast.LtE:
            return a <= b
        if op is _ast.Gt:
            return a > b
        if op is _ast.GtE:
            return a >= b
    except BaseException:
        return _FAIL
    return _FAIL


def _pe(node, f):
    t = type(node)
    if t is _ast.Constant:
        return node.value
    if t is _ast.Name:
        return _pname(node.id, f)
    if t is _ast.Attribute:
        o = _pe(node.value, f)
        if o is _FAIL:
            return o
        return _pattr(o, node.attr)
    if t is _ast.Compare:
        left = _pe(node.left, f)
        if left is _FAIL:
            return left
        for op, comp in zip(node.ops, node.comparators):
            right = _pe(comp, f)
            if right is _FAIL:
                return right
            r = _pcmp(type(op), left, right)
            if r is _FAIL or not r:
                return r
            left = right
        return True
    if t is _ast.UnaryOp:
        v = _pe(node.operand, f)
        if v is _FAIL:
            return v
        if type(node.op) is _ast.Not:
            tr = _ptruth(v)
            return _FAIL if tr == 2 else not tr
        if type(v) in (int, float):
            return -v
        return _FAIL
    if t is _ast.BoolOp:
        is_or = type(node.op) is _ast.Or
        last = len(node.values) - 1
        for i, sub in enumerate(node.values):
            x = _pe(sub, f)
            if x is _FAIL or i == last:
                return x
            tr = _ptruth(x)
            if tr == 2:
                return _FAIL
            if tr == is_or:
                return x
        return _FAIL
    if t is _ast.Tuple:
        out = []
        for e in node.elts:
            x = _pe(e, f)
            if x is _FAIL:
                return x
            out.append(x)
        return tuple(out)
    if t is _ast.Subscript:
        o = _pe(node.value, f)
        if o is _FAIL:
            return o
        k = _pe(node.slice, f)
        if k is _FAIL:
            return k
        to = type(o)
        if to in (tuple, list, str) and type(k) is int:
            try:
                return o[k]
            except IndexError:
                return _FAIL
        if to is dict and type(k) in (str, int, bool, type(None)):
            return o[k] if k in o else _FAIL
        return _FAIL
    if t is _ast.Call:
        fn = node.func
        if type(fn) is _ast.Name:
            g = _pname(fn.id, f)
            args = []
            for a in node.args:
                x = _pe(a, f)
                if x is _FAIL:
                    return x
                args.append(x)
            if g is isinstance and len(args) == 2:
                o, cls = args
                cl = cls if type(cls) is tuple else (cls,)
                for c in cl:
                    if not isinstance(c, type) or type(c).__instancecheck__ is not _TIC:
                        return _FAIL
                to = type(o)
                if to.__getattribute__ is not _OGA or _mro_get(to, "__class__") is not _mro_get(object, "__class__"):
                    return _FAIL
                return isinstance(o, cls)
            if g is len and len(args) == 1:
                if type(args[0]) in (list, tuple, dict, str, set, frozenset, bytes):
                    return len(args[0])
                return _FAIL
            if g is bool and len(args) == 1:
                tr = _ptruth(args[0])
                return _FAIL if tr == 2 else bool(tr)
            return _FAIL
        if type(fn) is _ast.Attribute and fn.attr == "get":
            o = _pe(fn.value, f)
            if o is _FAIL or type(o) is not dict:
                return _FAIL
            k = _pe(node.args[0], f)
            if k is _FAIL or type(k) not in (str, int, bool, type(None)):
                return _FAIL
            if len(node.args) == 2:
                dflt = _pe(node.args[1], f)
                if dflt is _FAIL:
                    return dflt
            else:
                dflt = None
            return o[k] if k in o else dflt
        return _FAIL
    return _FAIL


# --- and/or trees --------------------------------------------------------------------------
# per leaf, 4 bits: bits 0-1 truth (0 false, 1 true, 2 unknown), bit 2 evaluated by the
# original run, bit 3 side-effect free (checked evaluation succeeded; for an evaluated leaf
# it also gave the same value). Value contexts add 3 bits per leaf from bit 32: the index of
# the first leaf with the identical value (equal value for int / str / float / bytes / bool /
# None), its own index when none. RAISED: the evaluation ended by an exception or was not
# seen whole.
RAISED = (1 << 63) - 1
_BSITES = None
_BO = {}  # (frame id, site) -> open evaluation record


def _bsites():
    global _BSITES
    if _BSITES is None:
        p = os.environ.get("BR_BSITES")
        out = []
        if p:
            with open(p) as f:
                raw = _json.load(f)
            for row in raw:
                leaves = [None if s is None else _ast.parse(s, mode="eval").body for s in row[7]]
                out.append((row[6] == "V", leaves, len(leaves)))
        _BSITES = out
    return _BSITES


class _Ev:
    __slots__ = ("tag", "glob", "thr", "st", "val", "pre", "done", "n", "leaves", "vctx")


def _brec(sid, sig, tag, glob, thr):
    k = (sid, sig)
    if glob:
        if k not in _BG:
            _BG.add(k)
            _w("BG%x:%x\n" % k)
    elif thr:
        if k not in _BX:
            _BX.add(k)
            _w("BX%x:%x\n" % k)
    else:
        s = _BSETS.get(tag)
        if s is None:
            s = _BSETS[tag] = set()
        if k not in s:
            s.add(k)
            if tag != _TAG:
                _w("T %s\nB%x:%x\nT %s\n" % (tag, sid, sig, _TAG))
            else:
                _w("B%x:%x\n" % k)


_BG = set()
_BX = set()
_BSETS = {}


def _flush_open():
    items = list(_BO.items())
    _BO.clear()
    for (_fid, sid), ev in items:
        _STATS["open_flushed"] += 1
        _brec(sid, RAISED, ev.tag, ev.glob, ev.thr)


_atexit.register(lambda: _BO and _flush_open())


def _skip_to(ev, i, f):
    """Leaves before i that the original run skipped: evaluate the side-effect-free ones now."""
    j = ev.done
    while j < i:
        if ev.st[j] is None:
            leaf = ev.leaves[j]
            v = _FAIL
            if leaf is not None:
                try:
                    v = _pe(leaf, f)
                except BaseException:
                    v = _FAIL
            if v is _FAIL:
                ev.st[j] = 2
                if leaf is not None:
                    _STATS["pfail"] += 1
            else:
                ev.st[j] = 8 | _ptruth(v)
                if ev.vctx:
                    ev.val[j] = v
        j += 1
    ev.done = i


def __br_P__(sid, i):
    f = _getframe(1)
    key = (id(f), sid)
    if i == 0:
        old = _BO.pop(key, None)
        if old is not None:
            _STATS["open_flushed"] += 1
            _brec(sid, RAISED, old.tag, old.glob, old.thr)
        vctx, leaves, n = (_BSITES or _bsites())[sid]
        ev = _Ev()
        ev.tag = _TAG
        ev.glob = DEPTH > 0
        ev.thr = _get_ident() != _MAIN_THREAD
        ev.st = [None] * n
        ev.val = [_NOTSET] * n if vctx else None
        ev.pre = [_NOTSET] * n
        ev.done = 0
        ev.n = n
        ev.leaves = leaves
        ev.vctx = vctx
        _BO[key] = ev
    else:
        ev = _BO.get(key)
        if ev is None:
            return None
        if ev.done < i:
            _skip_to(ev, i, f)
    leaf = ev.leaves[i]
    if leaf is not None:
        try:
            ev.pre[i] = _pe(leaf, f)
        except BaseException:
            ev.pre[i] = _FAIL
    return None


def __br_L__(sid, i, v):
    ev = _BO.get((id(_getframe(1)), sid))
    if ev is None:
        _STATS["noopen"] += 1
        return v
    s = 4 | _ptruth(v)
    p = ev.pre[i]
    if p is not _NOTSET:
        if p is _FAIL:
            _STATS["pfail"] += 1
        elif p is v or (type(p) is type(v) and type(v) in _SCALARS and p == v):
            s |= 8
        else:
            _STATS["mismatch"] += 1
            if _DEBUG:
                try:
                    _w("M %d %d %s %r | %r\n" % (sid, i, type(v).__name__, p, v))
                except BaseException:
                    _w("M %d %d %s ?\n" % (sid, i, type(v).__name__))
        ev.pre[i] = _NOTSET
    st = ev.st
    j = ev.done
    while j < i:
        if st[j] is None:
            st[j] = 2  # skipped and not side-effect free (no pre-hook was needed for leaf i)
        j += 1
    st[i] = s
    if ev.vctx:
        ev.val[i] = v
    ev.done = i + 1
    return v


def _same_value(a, b):
    if a is b:
        return True
    t = type(a)
    return t is type(b) and t in (int, str, float, bytes) and a == b


def __br_E__(sid, r):
    f = _getframe(1)
    ev = _BO.pop((id(f), sid), None)
    if ev is None:
        _STATS["noopen"] += 1
        _brec(sid, RAISED, _TAG, DEPTH > 0, _get_ident() != _MAIN_THREAD)
        return r
    if ev.done < ev.n:
        _skip_to(ev, ev.n, f)
    sig = 0
    st = ev.st
    for i in range(ev.n):
        sig |= st[i] << (4 * i)
    if ev.vctx:
        val = ev.val
        last = -1
        for i in range(ev.n):
            if st[i] & 4:
                last = i
        if last < 0 or val[last] is not r:
            sig = RAISED
        else:
            for i in range(ev.n):
                c = i
                vi = val[i]
                if vi is not _NOTSET:
                    for j in range(i):
                        vj = val[j]
                        if vj is not _NOTSET and _same_value(vi, vj):
                            c = j
                            break
                sig |= c << (32 + 3 * i)
    glob = ev.glob or DEPTH > 0
    thr = ev.thr or _get_ident() != _MAIN_THREAD
    _brec(sid, sig, ev.tag, glob, thr)
    if ev.tag != _TAG:
        _brec(sid, sig, _TAG, glob, thr)
    return r


_CNT_K = {}
_CNT_B = {}


def __br_CK__(sid, r):
    _CNT_K[sid] = _CNT_K.get(sid, 0) + 1
    return r


def __br_CB__(sid, r):
    _CNT_B[sid] = _CNT_B.get(sid, 0) + 1
    return r


def _dump_counts():
    if _CNT_K or _CNT_B:
        _w("C %s\n" % _json.dumps({"K": _CNT_K, "B": _CNT_B}))


_atexit.register(_dump_counts)
builtins.__br_CK__ = __br_CK__
builtins.__br_CB__ = __br_CB__
builtins.__br_P__ = __br_P__
builtins.__br_L__ = __br_L__
builtins.__br_E__ = __br_E__


def _stats_line():
    if _F is not None or any(_STATS.values()):
        _w("Z %s\n" % _json.dumps(_STATS))


_atexit.register(_stats_line)
