
"""Records the inputs of Tokenizer.tokenize (sqlglot/tokens.py) in every process of a test run.

A recorded input is (tokenizer key, sql). The key tells how to build the same tokenizer again:
the dialect (module, class, version, normalization strategy, settings) and whether the
tokenizer is that dialect's tokenizer_class, its jsonpath_tokenizer_class, or another class
(module, qualified name)."""
import os, sys, json, atexit

_seen = {}
_installed = [False]


def _key(tok):
    d = getattr(tok, "dialect", None)
    if d is None:
        return None
    dc = type(d)
    tc = type(tok)
    if tc is getattr(dc, "tokenizer_class", None):
        kind = "tokenizer"
    elif tc is getattr(dc, "jsonpath_tokenizer_class", None):
        kind = "jsonpath"
    else:
        kind = "class:%s:%s" % (tc.__module__, tc.__qualname__)
    ns = getattr(d, "normalization_strategy", None)
    try:
        settings = json.dumps(getattr(d, "settings", {}) or {}, sort_keys=True)
    except TypeError:
        settings = None
    return (kind, dc.__module__, dc.__qualname__, list(getattr(d, "version", ()) or ()),
            None if ns is None else str(getattr(ns, "value", ns)), settings)


def install():
    if _installed[0]:
        return
    mod = sys.modules.get("sqlglot.tokens")
    if mod is None:
        return
    _installed[0] = True
    cls = mod.Tokenizer
    orig = cls.tokenize

    def tokenize(self, sql):
        try:
            k = _key(self)
            if k is not None and isinstance(sql, str):
                _seen.setdefault(json.dumps(k), set()).add(sql)
        except Exception:  # noqa: BLE001
            pass
        return orig(self, sql)

    cls.tokenize = tokenize


def dump(path):
    out = [[k, sorted(v)] for k, v in _seen.items()]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)
