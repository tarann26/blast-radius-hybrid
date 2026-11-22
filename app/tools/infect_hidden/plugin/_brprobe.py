"""Reach probe for building maps/infect_hidden.bin (the cache-hidden tests of the operand
outcome map). Same phase bookkeeping as the probe of the outcome maps (infect2/plugin/_brprobe.py:
tags set by the pytest plugin, pool tasks tagged with the submitting test, module bodies counted),
but it records only HOW OFTEN control arrived somewhere, never a value:

  __br_R__(region)   placed as a statement of its own in front of the smallest statement that
                     holds a mapped site (or at the start of a lambda body): the current test
                     phase reached that statement once more;
  __br_S__(site)     placed in front of the site's own expression ("__br_S__(n) or <site>"): the
                     current test phase started to evaluate the site once more.

Both take one integer literal, return None and evaluate nothing of the program. Only runs of
the pinned commit's own test suite are recorded: nothing here knows about patches.

Records go to one file per process in $BR_PROBE_DIR. Line format:
  P <pid> <kind> <tag>   header (kind: main | worker | sub)
  T <tag>                the following records belong to this test phase
  S <tag>                this test phase submitted a task to (or created) a process pool
  C <json>               counts of the phase so far, written when the phase ends or a pool task
                         returns: [regions, sites, regions in other threads, sites in other
                         threads], each {number: count}
  RG<hex> / sG<hex>      region reached / site started while a sqlglot module body was
                         executing (import time; not counted)
"""
import atexit
import builtins
import json
import os
import sys
import _thread

_MAIN_THREAD = _thread.get_ident()
_get_ident = _thread.get_ident
DEPTH = 0  # number of sqlglot module bodies currently executing
_IS_WORKER = "--multiprocessing-fork" in sys.argv  # a spawn pool worker
# a worker's records before its first task (start-up imports) get the tag "W"; a plain
# subprocess inherits the phase of the test that started it
_TAG = "W" if _IS_WORKER else os.environ.get("BR_PROBE_TAG", "?")
_RC = {}  # region -> times reached since the last "C" line (main thread, no module body running)
_SC = {}  # site -> times started
_RCX = {}  # the same in other threads
_SCX = {}
_RG, _SG = set(), set()
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


def _flush():
    """Writes the counts of the current phase and starts again from zero."""
    if _RC or _SC or _RCX or _SCX:
        _w("C %s\n" % json.dumps([_RC, _SC, _RCX, _SCX]))
        _RC.clear()
        _SC.clear()
        _RCX.clear()
        _SCX.clear()


atexit.register(_flush)


def settag(tag):
    """Called by the pytest plugin (main process) and by pool tasks (workers)."""
    global _TAG
    if tag == _TAG:
        return
    _flush()
    _TAG = tag
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
        try:
            return self.fn(*args, **kwargs)
        finally:
            _flush()  # a pool worker ends without running exit handlers

    def __reduce__(self):
        return (_mk_tagged, (self.tag, self.fn))


def _mk_tagged(tag, fn):
    # runs in the worker while the task is unpickled: its arguments are unpickled after this
    settag(tag)
    return Tagged(tag, fn)


def _imp(d):
    global DEPTH
    DEPTH += d


def __br_R__(i):
    if DEPTH:
        if i not in _RG:
            _RG.add(i)
            _w("RG%x\n" % i)
    elif _get_ident() != _MAIN_THREAD:
        _RCX[i] = _RCX.get(i, 0) + 1
    else:
        _RC[i] = _RC.get(i, 0) + 1
    return None


def __br_S__(i):
    if DEPTH:
        if i not in _SG:
            _SG.add(i)
            _w("sG%x\n" % i)
    elif _get_ident() != _MAIN_THREAD:
        _SCX[i] = _SCX.get(i, 0) + 1
    else:
        _SC[i] = _SC.get(i, 0) + 1
    return None


builtins.__br_R__ = __br_R__
builtins.__br_S__ = __br_S__
builtins.__br_imp__ = _imp
