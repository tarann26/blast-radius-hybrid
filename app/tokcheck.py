"""Token streams of the inputs that a few long tests tokenize (maps/tok_inputs.bin).

The map lists, for each of these tests, the (tokenizer, sql) inputs of Tokenizer.tokenize that
the test makes in every process of a pytest run (the test process and its spawn workers),
with a digest of the pinned tree's token stream for each input (build_tok_inputs.py). After a
patch to the tokenizer, changed(...) tokenizes those inputs again with the patched code: a
test with an input whose stream differs (or raises, or does not end in time) sees the patch,
so it is selected when it is too costly to run."""
from __future__ import annotations

import hashlib
import importlib
import json
import marshal
import os
import signal
import sys
import zlib

from rawz import unz as _unz

_MAP = {}


def load(maps_dir):
    m = _MAP.get(maps_dir, 0)
    if m == 0:
        try:
            with open(os.path.join(maps_dir, "tok_inputs.bin"), "rb") as f:
                m = marshal.loads(_unz(f.read()))
        except (OSError, ValueError, EOFError, zlib.error):
            m = None
        _MAP[maps_dir] = m
    return m


def tests_of(maps_dir):
    m = load(maps_dir)
    return frozenset(m["tests"]) if m else frozenset()


def build_tokenizer(key):
    """The tokenizer that the recorded key describes (see xt/tokrec.py), or None."""
    kind, mod, qual, version, ns, settings = key
    obj = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part)
    kwargs = json.loads(settings) if settings else {}
    if version and tuple(version) != (sys.maxsize, 0, 0):
        kwargs["version"] = ".".join(str(v) for v in version)
    if ns is not None and str(getattr(obj.NORMALIZATION_STRATEGY, "value", obj.NORMALIZATION_STRATEGY)) != ns:
        kwargs["normalization_strategy"] = ns
    dialect = obj(**kwargs)
    if kind == "tokenizer":
        cls = type(dialect).tokenizer_class
    elif kind == "jsonpath":
        cls = type(dialect).jsonpath_tokenizer_class
    else:
        _, tmod, tqual = kind.split(":", 2)
        cls = importlib.import_module(tmod)
        for part in tqual.split("."):
            cls = getattr(cls, part)
    return cls(dialect=dialect)


def digest(tokenizer, sql):
    """A digest of every field of every token, or of the exception that tokenize raised."""
    tokens_mod = sys.modules["sqlglot.tokens"]
    try:
        tokens = tokens_mod.Tokenizer.tokenize(tokenizer, sql)
        rows = [(int(t.token_type), t.text, t.line, t.col, t.start, t.end, tuple(t.comments)) for t in tokens]
    except _OutOfTime:
        raise
    except Exception as e:  # noqa: BLE001
        rows = ("raised", type(e).__name__, str(e))
    return hashlib.sha1(repr(rows).encode("utf-8", "surrogatepass")).digest()


class _OutOfTime(BaseException):
    pass


def _alarm(signum, frame):
    raise _OutOfTime()


def changed(maps_dir, wanted, budget):
    """The tests in wanted whose recorded inputs give another token stream in this process
    (patched code), or that could not all be checked within budget CPU seconds."""
    m = load(maps_dir)
    if not m:
        return set(wanted)
    inputs, digests, keys = m["inputs"], m["digests"], m["keys"]
    out = set()
    todo = {n: list(m["tests"].get(n, ())) for n in wanted}
    for n, ids in todo.items():
        if not ids or n not in m["tests"]:
            out.add(n)
    idx = sorted({i for n, ids in todo.items() if n not in out for i in ids})
    if not idx:
        return out
    bad = set()
    cache = {}
    old = signal.signal(signal.SIGVTALRM, _alarm)
    signal.setitimer(signal.ITIMER_VIRTUAL, max(0.05, budget))
    try:
        for i in idx:
            k, sql = inputs[i]
            try:
                tok = cache.get(k)
                if tok is None:
                    tok = cache[k] = build_tokenizer(keys[k])
                if digest(tok, sql) != digests[i]:
                    bad.add(i)
            except _OutOfTime:
                raise
            except Exception:  # noqa: BLE001  (the tokenizer cannot be built: count as changed)
                bad.add(i)
    except _OutOfTime:
        done = set(idx[: idx.index(i)]) if i in idx else set()
        bad.update(set(idx) - done)
    finally:
        signal.setitimer(signal.ITIMER_VIRTUAL, 0)
        signal.signal(signal.SIGVTALRM, old)
    for n, ids in todo.items():
        if n not in out and not bad.isdisjoint(ids):
            out.add(n)
    return out
