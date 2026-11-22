"""Run sqlglot's unittest-style tests by pytest node id, without pytest.

Every test in tests.json is ``path::Class::method`` on a unittest.TestCase, so
unittest itself can run them with the same semantics pytest uses for such tests:
setUpClass/tearDownClass, subtests (one failing subtest fails the test), skips,
and an import error in the file failing every test of that file. A per-test
timer turns a hang into a failure, matching the labeler's timeout rule.
"""

from __future__ import annotations

import importlib
import signal
import sys
import time
import unittest


ON_START = None  # optional callback(node id) when a test starts


class TestTimeout(BaseException):
    """Raised inside a test that ran past its time limit."""


def _module_name(path: str) -> str:
    return path[:-3].replace("/", ".")


class RecordingResult(unittest.TestResult):
    def __init__(self, id_of, limit_of):
        super().__init__()
        self.id_of = id_of  # TestCase instance id() -> node id
        self.limit_of = limit_of
        self.failed: set[str] = set()
        self.durations: dict[str, float] = {}
        self.cpu: dict[str, float] = {}
        self._started = 0.0
        self._cpu_started = 0.0
        self._current: str | None = None

    def _node(self, test):
        return self.id_of.get(id(test))

    def startTest(self, test):
        super().startTest(test)
        self._current = self._node(test)
        if ON_START is not None:
            ON_START(self._current)
        self._started = time.monotonic()
        self._cpu_started = time.process_time()
        limit = self.limit_of(self._current)
        if limit:
            signal.setitimer(signal.ITIMER_REAL, limit)

    def stopTest(self, test):
        signal.setitimer(signal.ITIMER_REAL, 0)
        if self._current is not None:
            self.durations[self._current] = time.monotonic() - self._started
            self.cpu[self._current] = time.process_time() - self._cpu_started
        self._current = None
        super().stopTest(test)

    def _fail(self, test):
        node = self._node(test)
        if node is None:
            # Class/module fixture errors arrive as _ErrorHolder objects; the
            # caller maps them back to every test of that class or module.
            self.fixture_errors.append(str(test))
        else:
            self.failed.add(node)

    fixture_errors: list

    def addError(self, test, err):
        super().addError(test, err)
        self._fail(test)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._fail(test)

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._fail(test)

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._fail(test)


def _alarm(signum, frame):
    raise TestTimeout("test timed out")


def run_tests(test_ids, limit_of=lambda node: None, on_import_error=None):
    """Run the given node ids in order. Returns (failed ids, durations)."""
    signal.signal(signal.SIGALRM, _alarm)
    failed: set[str] = set()
    suite = unittest.TestSuite()
    id_of: dict[int, str] = {}
    by_class: dict[tuple[str, str], list[str]] = {}
    by_module: dict[str, list[str]] = {}
    modules: dict[str, object] = {}
    keep = []
    for node in test_ids:
        path, cls_name, method = node.split("::")
        by_class.setdefault((path, cls_name), []).append(node)
        by_module.setdefault(path, []).append(node)
        if path not in modules:
            try:
                modules[path] = importlib.import_module(_module_name(path))
            except BaseException as error:  # noqa: BLE001 - any import failure fails the file
                modules[path] = None
                if on_import_error:
                    on_import_error(path, error)
        module = modules[path]
        if module is None:
            failed.add(node)
            continue
        try:
            case = getattr(module, cls_name)(method)
        except BaseException:  # noqa: BLE001 - a missing class or method fails the test
            failed.add(node)
            continue
        keep.append(case)
        id_of[id(case)] = node
        suite.addTest(case)

    result = RecordingResult(id_of, limit_of)
    result.fixture_errors = []
    try:
        suite.run(result)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    failed |= result.failed
    for description in result.fixture_errors:
        # e.g. "setUpClass (tests.test_x.TestX)" or "setUpModule (tests.test_x)"
        inside = description[description.find("(") + 1 : description.rfind(")")]
        parts = inside.split(".")
        hit = False
        for (path, cls_name), nodes in by_class.items():
            if _module_name(path) + "." + cls_name == inside:
                failed.update(nodes)
                hit = True
        for path, nodes in by_module.items():
            if _module_name(path) == inside:
                failed.update(nodes)
                hit = True
        if not hit:
            # Unknown fixture error: be safe and fail everything we ran.
            failed.update(test_ids)
        del parts
    run_tests.last_cpu = result.cpu
    return failed, result.durations


if __name__ == "__main__":  # manual use: python -m selector.runner <node ids...>
    bad, times = run_tests(sys.argv[1:], limit_of=lambda node: 60)
    for node in sys.argv[1:]:
        print(("FAIL " if node in bad else "ok   ") + f"{times.get(node, 0):.3f} {node}")
