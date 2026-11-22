"""pytest plugin for the map-building run: tells the comparison probe (_brprobe) which test
phase is running, and tags process-pool tasks with the phase that submitted them, so that
evaluations in spawn workers are attributed to the right test."""
import concurrent.futures.process as _cfp

import pytest

import _brprobe as P

_orig_submit = _cfp.ProcessPoolExecutor.submit
_orig_init = _cfp.ProcessPoolExecutor.__init__


def _submit(self, fn, /, *args, **kwargs):
    P.pool_use()
    return _orig_submit(self, P.Tagged(P.tagname(), fn), *args, **kwargs)


def _init(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kwargs):
    P.pool_use()
    if initializer is not None:
        initializer = P.Tagged(P.tagname(), initializer)
    return _orig_init(self, max_workers, mp_context, initializer, initargs, **kwargs)


def pytest_configure(config):
    P.set_main()
    P.settag("|collect")
    _cfp.ProcessPoolExecutor.submit = _submit
    _cfp.ProcessPoolExecutor.__init__ = _init


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    P.settag(item.nodeid + "|s")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    P.settag(item.nodeid + "|c")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    P.settag(item.nodeid + "|t")


@pytest.hookimpl(trylast=True)
def pytest_runtest_logfinish(nodeid, location):
    P.settag("|between")


def pytest_sessionfinish(session, exitstatus):
    P.settag("|end")
