"""Key-level reads of the dict, set and frozenset constants of sqlglot (shared by the
build-time trace keytrace.py, the map builder build_keymap.py and the selector).

A class or module constant such as Tokenizer.KEYWORDS or Generator.TRANSFORMS is a table;
most tests look up a few of its entries. The build-time trace of the pinned tree records,
for every test, which entries of which table it looked up (also lookups of keys that are not
in the table), and whether it used the table as a whole (iteration, copy, comparison, ...).
When a patch changes only some entries of such tables, a test that neither looked up one of
those entries nor used the table as a whole gets the same answers from the patched table.

Everything here works on the pinned commit (build time) or on the import state of the
patched tree in the selector's child (run time); nothing is derived from patches or labels.
"""
from __future__ import annotations

import array
import collections
import enum
import marshal
import numbers
import os
import types
import zlib

from rawz import unz as _unz

WHOLE = "*"  # the table was used as a whole: every change can matter
LEN = "#len"  # len() or truth value: matters when the number of entries changes
KEYS = "#keys"  # the keys in order (iteration over keys, keys()): matters when they change


def stable_key(k):
    """A text form of a key that is the same in every process for keys that compare equal.
    Different keys may share a form (that only adds tests); equal keys never differ."""
    try:
        if isinstance(k, str):
            return "s:" + str.__add__(k, "")  # the text itself, also for str enums
        if isinstance(k, numbers.Number):
            try:
                i = int(k)
                if i == k:
                    return "n:%d" % i
            except (TypeError, ValueError, OverflowError):
                pass
            return "n:" + repr(complex(k) if isinstance(k, complex) else float(k))
        if k is None:
            return "N"
        if isinstance(k, bytes):
            return "b:" + repr(bytes(k))
        if isinstance(k, tuple):
            return "(" + ",".join(stable_key(x) for x in k) + ")"
        if isinstance(k, (frozenset, set)):
            return "{" + ",".join(sorted(stable_key(x) for x in k)) + "}"
        if isinstance(k, type):
            return "C:%s.%s" % (k.__module__, k.__qualname__)
        if isinstance(k, enum.Enum):
            t = type(k)
            return "E:%s.%s.%s" % (t.__module__, t.__qualname__, k.name)
        if isinstance(k, types.FunctionType):
            return "F:%s.%s" % (k.__module__, k.__qualname__)
    except Exception:  # noqa: BLE001
        pass
    t = type(k)
    return "T:%s.%s" % (getattr(t, "__module__", "?"), getattr(t, "__qualname__", "?"))


def holder_name(key):
    """The name of a statefp key ("A", module, class path, attribute) or ("G", module, name)."""
    return "|".join(str(x) for x in key)


def changed_entries(old, new, fp_old, fp_new):
    """The stable keys of the entries that differ between two versions of one table, plus
    LEN when the number of entries differs and KEYS when the keys (in order) differ.
    fp_old / fp_new: value fingerprints (statefp.Fingerprinter.fp). None: cannot tell."""
    if type(old) is not type(new) or type(old) not in (dict, set, frozenset):
        return None
    out = set()
    if type(old) is dict:
        go, gn = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
        ko, kn = [], []
        for k, v in old.items():
            sk = stable_key(k)
            ko.append(sk)
            go[sk][fp_old(v)] += 1
        for k, v in new.items():
            sk = stable_key(k)
            kn.append(sk)
            gn[sk][fp_new(v)] += 1
        for sk in set(go) | set(gn):
            if go.get(sk) != gn.get(sk):
                out.add(sk)
        if ko != kn:
            out.add(KEYS)
    else:
        go, gn = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
        for x in old:
            go[stable_key(x)][fp_old(x)] += 1
        for x in new:
            gn[stable_key(x)][fp_new(x)] += 1
        for sk in set(go) | set(gn):
            if go.get(sk) != gn.get(sk):
                out.add(sk)
        if out:
            out.add(KEYS)
    if len(old) != len(new):
        out.add(LEN)
    return out


def diff_rows(changed, base_objects, base_state, new_objects, new_fpr, fingerprinter):
    """For each changed statefp key: the changed entries (sorted list) when the key holds a
    table on both sides whose pinned version is still as it was at the base snapshot, else
    None. Runs in the selector's child after the patched import."""
    out = {}
    f0 = None
    for key in changed:
        old = base_objects.get(key)
        new = new_objects.get(key)
        if old is None or new is None or type(old) is not type(new):
            continue
        if f0 is None:
            f0 = fingerprinter()
        try:
            # the parent may have changed the pinned table since the snapshot (it imports test
            # modules and builds fixtures): then its entries are not the pinned ones
            if f0.fp(old) != base_state.get(key):
                continue
            entries = changed_entries(old, new, f0.fp, new_fpr.fp)
        except Exception:  # noqa: BLE001
            continue
        if entries is not None:
            out[key] = sorted(entries)
    return out


class _LazyTables:
    """table id -> {stable key: set id}, each table read from its own marshal blob at its first
    use (the image stores keymap.bin so: rawmaps.py); a run reads a few of the 3,600 tables."""

    __slots__ = ("_blobs", "_done")

    def __init__(self, blobs):
        self._blobs = blobs
        self._done = {}

    def __getitem__(self, tid):
        t = self._done.get(tid)
        if t is None:
            t = self._done[tid] = marshal.loads(self._blobs[tid])
        return t

    def __len__(self):
        return len(self._blobs)

    def __iter__(self):
        return (self[i] for i in range(len(self._blobs)))


class KeyMap:
    """The key read map (keymap.bin, made by build_keymap.py)."""

    def __init__(self, path):
        with open(path, "rb") as f:
            d = marshal.loads(_unz(f.read()))
        self.holders = d["holders"]  # holder name -> table id (traced, never changed by a test)
        # table id -> {stable key: set id}
        self.tables = _LazyTables(d["tables"]) if d.get("lazy_tables") else d["tables"]
        self.sets = d["sets"]  # set id -> bytes of array('H') test numbers
        unknown = array.array("H")
        unknown.frombytes(d["unknown"])
        self.unknown = set(unknown)  # tests whose reads are not known: never cut

    def tests(self, sid):
        a = array.array("H")
        a.frombytes(self.sets[sid])
        return a

    def readers(self, rows):
        """rows: [(statefp key, changed entries or None)] for every changed key. The tests
        that can see the change, or None when the map cannot tell."""
        out = set(self.unknown)
        for key, entries in rows:
            if entries is None:
                return None
            tid = self.holders.get(holder_name(key))
            if tid is None:
                return None  # not a traced table (or one that some test changes)
            table = self.tables[tid]
            for sk in [WHOLE] + list(entries):
                sid = table.get(sk)
                if sid is not None:
                    out.update(self.tests(sid))
        return out


_MAP = [None]


def load(maps_dir):
    if _MAP[0] is None:
        try:
            _MAP[0] = KeyMap(os.path.join(maps_dir, "keymap.bin"))
        except (OSError, ValueError, KeyError, EOFError, zlib.error):
            _MAP[0] = False
    return _MAP[0] or None


def plan_ok(plan, root):
    """True when the commit's Python edits change nothing but import-time values that the
    import-state diff names: no function body that also runs at import (fpairs, stmt_lines),
    no edit that the child applies in memory (work), no removed name, no function or class
    header. Only then can the changed entries alone decide which tests see the commit."""
    if plan.fpairs or plan.stmt_lines or plan.gone or plan.work:
        return False
    for path, lines in plan.def_units.items():
        try:
            with open(os.path.join(root, path), encoding="utf-8") as f:
                src = f.read().split("\n")
        except OSError:
            return False
        for ln in lines:
            if 0 < ln <= len(src) and src[ln - 1].lstrip().startswith(("def ", "async def ", "@", "class ")):
                return False
    return True


def cut(plan, rows, maps_dir, root):
    """The tests that can see the commit's import-time change by the key read map, or None
    (no cut). rows: the child's diff rows [key, is_function, changed entries or None]."""
    if not rows or not plan_ok(plan, root):
        return None
    if any(len(r) < 3 or r[1] or r[2] is None for r in rows):
        return None  # a function, or a value that is not a table, or entries not known
    km = load(maps_dir)
    if km is None:
        return None
    return km.readers([(r[0], r[2]) for r in rows])
