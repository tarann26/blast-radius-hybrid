"""Lazy stand-ins for numpy and pandas in the selector's parent.

Only two test files import them (tests/test_executor.py and tests/test_optimizer.py, at module
level), and only some of their tests call into them (DataFrame results of duckdb, pandas'
assert_frame_equal). Importing pandas costs about 0.3 s of CPU, and its about 350 extra memory
mappings make every later fork and exit dearer.

Each stand-in is the module object itself, made from the real spec, with the module attributes
that the import system reads (__spec__, __name__, __loader__, __file__) but without __path__:
the first read of any other attribute (including __path__, which every submodule import
reads) runs the real module code in place, as a normal import would, so the object stays
the one in sys.modules and in the globals of the test modules. (importlib.util.LazyLoader
does not work here: the import statement itself reads __spec__, which loads the module.)
"from pandas.testing import assert_frame_equal" needs a value at import time: pandas.testing
is a stand-in whose assert_frame_equal loads pandas on its first call; once pandas runs, the
real pandas.testing takes its place in sys.modules and in the globals that bound it."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types

NAMES = ("numpy", "pandas")
_STATE = {"installed": False, "assert": None}
_TRACE = bool(__import__("os").environ.get("BR_LAZY_TRACE"))  # debugging: say where a module really loads
_AFTER = {}  # module name -> function called with the module right after its real code ran
# duckreplay.frame_hook while the DuckDB answer replay is installed: it sees the arguments of
# every assert_frame_equal call first (the replay's results are stand-ins for frames)
FRAME_HOOK = None


class _LazyModule(types.ModuleType):
    def __getattr__(self, name):
        _run(self)
        return getattr(self, name)


def _run(module):
    if type(module) is not _LazyModule:
        return
    spec = module.__dict__["__spec__"]
    if _TRACE:
        import traceback

        sys.stderr.write("lazymods: %s loads here\n%s" % (spec.name, "".join(traceback.format_stack(limit=12))))
    module.__class__ = types.ModuleType
    if spec.submodule_search_locations is not None:
        module.__path__ = spec.submodule_search_locations
    if spec.name == "pandas" and isinstance(sys.modules.get("pandas.testing"), _TestingStandIn):
        del sys.modules["pandas.testing"]  # pandas imports the real one while it runs
    spec._initializing = True
    try:
        spec.loader.exec_module(module)
    finally:
        spec._initializing = False
    after = _AFTER.get(spec.name)
    if after is not None:
        after(module)
    if spec.name == "pandas":
        real = sys.modules.get("pandas.testing")
        if real is not None and not isinstance(real, _TestingStandIn):
            _STATE["assert"] = real.assert_frame_equal
            if FRAME_HOOK is not None:
                # the stand-in stays where the test modules bound it, and a test module that
                # is imported from now on binds it too: it must see the replay's results first
                real.assert_frame_equal = assert_frame_equal
                return
            for mod in list(sys.modules.values()):
                d = getattr(mod, "__dict__", None)
                if type(d) is dict and d.get("assert_frame_equal") is assert_frame_equal and mod is not _SELF:
                    d["assert_frame_equal"] = real.assert_frame_equal


def _real_testing():
    load()
    return sys.modules["pandas.testing"]


def assert_frame_equal(*args, **kwargs):
    hook = FRAME_HOOK
    if hook is not None:
        got = hook(args, kwargs)
        if got is None:
            return None  # known to pass (see duckreplay.py)
        args, kwargs = got
    real = _STATE["assert"]
    if real is None:
        load()
        real = _STATE["assert"] or sys.modules["pandas.testing"].assert_frame_equal
    return real(*args, **kwargs)


class _TestingStandIn(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            # the import system probes __path__ (a package?) and others: a plain module
            raise AttributeError(name)
        return getattr(_real_testing(), name)


def install() -> bool:
    """Puts the stand-ins into sys.modules. False (and nothing changed) when one of the modules
    is loaded already or cannot be found."""
    if _STATE["installed"]:
        return True
    if any(n in sys.modules for n in NAMES + ("pandas.testing",)):
        return False
    specs = {}
    for name in NAMES:
        spec = importlib.machinery.PathFinder.find_spec(name)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return False
        specs[name] = spec
    for name, spec in specs.items():
        module = importlib.util.module_from_spec(spec)
        module.__dict__.pop("__path__", None)
        module.__class__ = _LazyModule
        sys.modules[name] = module
    testing = _TestingStandIn("pandas.testing")
    testing.assert_frame_equal = assert_frame_equal
    sys.modules["pandas.testing"] = testing
    _STATE["installed"] = True
    return True


def install_one(name, attrs, after=None) -> bool:
    """A stand-in for one more module. attrs: attributes that it has from the start (reading
    them does not run the module; the real code replaces them when it runs). after: called
    with the module right after its real code ran. False when the module is loaded already or
    cannot be found."""
    if name in sys.modules:
        return False
    spec = importlib.machinery.PathFinder.find_spec(name)
    if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
        return False
    module = importlib.util.module_from_spec(spec)
    module.__dict__.pop("__path__", None)
    module.__dict__.update(attrs)
    module.__class__ = _LazyModule
    if after is not None:
        _AFTER[name] = after
    sys.modules[name] = module
    return True


def load_one(name) -> None:
    """Runs the real module now (no-op when it has run or has no stand-in)."""
    m = sys.modules.get(name)
    if type(m) is _LazyModule:
        _run(m)


def loaded(name) -> bool:
    """True when the real module has run (or no stand-in was installed)."""
    m = sys.modules.get(name)
    return m is not None and type(m) is not _LazyModule


def load() -> None:
    """Runs the real numpy and pandas now (no-op when they have run)."""
    for name in NAMES:
        m = sys.modules.get(name)
        if type(m) is _LazyModule:
            _run(m)


_SELF = sys.modules[__name__]
