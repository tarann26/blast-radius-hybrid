"""Runs a list of sqlglot tests in the current process and reports each verdict."""
from __future__ import annotations

import functools
import importlib
import os
import unittest
from unittest.suite import _ErrorHolder

from reduce import STATE

try:
    from iterred import LOOPS
except ImportError:  # pragma: no cover
    LOOPS = None


class Verdicts(unittest.TestResult):
    """Records pass or fail for each test. A failing subtest ends its test at once."""

    def __init__(self, emit, stop_after_fail=False, started=None, enders=()):
        super().__init__()
        self._enders = enders or {}
        self._undecided = set(enders.get("held", ())) if enders else set()
        self.skip_rest = False
        self._started = started
        self.failfast = True  # makes TestCase.subTest stop the test after a failed subtest
        self._emit = emit
        self._count = 0
        self._current = None
        self._stop_after_fail = stop_after_fail
        self.class_errors = []
        self.any_failed = False

    def stop(self):  # failfast must not end the whole run
        pass

    def _exc_info_to_string(self, err, test):
        # Only pass or fail counts here. Formatting each traceback costs about 10 ms of CPU,
        # which adds up to seconds when a commit breaks hundreds of tests.
        return ""

    def _bad(self):
        return len(self.failures) + len(self.errors) + len(self.unexpectedSuccesses)

    def startTest(self, test):
        super().startTest(test)
        self._count = self._bad()
        self._current = test
        if self._started is not None:
            self._started(test._br_number)
        keep = getattr(test, "_br_keep", None)
        if keep is None:
            STATE.reset()
        else:
            STATE.reset(keep[0], keep[1])
        if LOOPS is not None:
            if keep is not None and len(keep) > 2:
                LOOPS.reset(keep[2], keep[3])
            else:
                LOOPS.reset()
        STATE.in_test = True

    def stopTest(self, test):
        super().stopTest(test)
        STATE.in_test = False
        STATE.reset()
        failed = self._bad() > self._count
        self._current = None
        if getattr(test, "_br_skipped", False):
            self._emit(test._br_number, None)  # not run: no verdict
            return
        self._emit(test._br_number, failed)
        if failed:
            self.any_failed = True
            decides = self._enders.get(test._br_number)
            if decides:
                # see run_tests: once every long held test is selected, the runs that are
                # left only go on when they cost little
                self._undecided -= decides[0]
                if not self._undecided and decides[1]:
                    self.skip_rest = True
            if self._stop_after_fail:
                self.shouldStop = True

    def addError(self, test, err):
        super().addError(test, err)
        if isinstance(test, _ErrorHolder):  # setUpClass / setUpModule / tearDownClass failed
            self.class_errors.append(test.description)


_FAILED_IMPORTS = {}
ON_IMPORT = {}  # module name -> function called once with the module after its first load here


def load_case(test_id: str, number: int):
    path, cls_name, meth = test_id.split("::")
    name = path[:-3].replace("/", ".")
    if name in _FAILED_IMPORTS:
        # pytest imports a test file once: when that fails, all its tests fail. Trying again
        # for each test would import the whole package again each time (seconds per file).
        raise _FAILED_IMPORTS[name]
    try:
        mod = importlib.import_module(name)
    except BaseException as e:
        if not isinstance(e, (KeyboardInterrupt, SystemExit)):
            _FAILED_IMPORTS[name] = e
        raise
    if ON_IMPORT:
        hook = ON_IMPORT.pop(name, None)
        if hook is not None:
            hook(mod)
    case = getattr(mod, cls_name)(meth)
    case._br_number = number
    return case


def _skip_when_failed(case, result_box, after_any_failure):
    """Skips the method body of a test that is not worth a run any more: after any failure
    (after_any_failure), or after the failure of one of the "enders" (see run_tests)."""
    name = case._testMethodName
    real = getattr(case, name)

    @functools.wraps(real)  # keeps unittest's skip and expected-failure marks
    def method(*args, **kwargs):
        result = result_box[0]
        if result is not None and (result.skip_rest or (after_any_failure and result.any_failed)):
            case._br_skipped = True
            raise unittest.SkipTest("not needed after a failure")
        return real(*args, **kwargs)

    setattr(case, name, method)


def run_tests(items, emit, stop_after_fail=False, keeps=None, started=None, skip_after_fail=(), enders=()):
    """items: list of (number, test id), already in run order with each class contiguous.

    emit(number, failed) is called once per test. keeps: {number: (keep_files, keep_tasks)}
    for the tests that run in reduced form (see reduce.py).
    skip_after_fail: tests that do not run once any test has failed.
    enders: {"held": the long tests that are held, number: (the held tests that a failure of
    this test selects, whether the runs after this test cost more than they can save)}. When
    failures have selected every long held test, the later tests do not run (the caller
    selects them as they are)."""
    suite = unittest.TestSuite()
    by_class = {}
    result_box = [None]
    for number, test_id in items:
        try:
            case = load_case(test_id, number)
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            emit(number, True)  # the test file cannot be imported, or the test is gone
            continue
        if keeps and number in keeps:
            case._br_keep = keeps[number]
        if enders or number in skip_after_fail:
            _skip_when_failed(case, result_box, number in skip_after_fail)
        suite.addTest(case)
        by_class.setdefault(type(case), []).append(number)
    seen = set()

    def emit_once(number, failed):
        seen.add(number)
        emit(number, failed)

    result = Verdicts(emit_once, stop_after_fail, started, enders)
    result_box[0] = result
    suite.run(result)
    if result.class_errors and not result.shouldStop:
        # a class whose setUpClass failed never starts its tests: all of them fail
        for cls, numbers in by_class.items():
            for number in numbers:
                if number not in seen:
                    emit(number, True)
    return result
