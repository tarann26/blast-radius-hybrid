
"""pytest plugin for the exact traces: the lines of each phase of the target tests go to their
own bucket; everything else goes to 'other'."""
import os
try:
    import xmon
except ImportError:
    xmon = None
if os.environ.get("XT_OUT") is None:
    class _Nop:
        @staticmethod
        def switch(b):
            pass
    xmon = _Nop()

TARGET_CLASSES = tuple(os.environ.get("XT_CLASSES", "").split(",")) if os.environ.get("XT_CLASSES") else ()


def _bucket(item, phase):
    nid = item.nodeid
    if TARGET_CLASSES and nid.rsplit("::", 1)[0] in TARGET_CLASSES:
        return "%s|%s" % (nid, phase)
    return "other"


import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    if os.environ.get("XT_TOK"):
        import tokrec
        tokrec.install()
    xmon.switch(_bucket(item, "setup"))
    yield
    xmon.switch("other")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    xmon.switch(_bucket(item, "call"))
    yield
    xmon.switch("other")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    xmon.switch(_bucket(item, "teardown"))
    yield
    xmon.switch("other")
