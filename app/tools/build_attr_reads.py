"""Builds maps/class_attr_reads.json from the pinned commit: for the test classes whose class
fixture the selector shares between runs (preload.SHARED_FIXTURES), the fixture attributes
that each test reads. The attributes are the lists that setUpClass filled from .sql fixture
files (preload._fixture_attributes). Usage (in the pinned image, repo at /repo):

    python build_attr_reads.py /repo /data/tests.json OUT.json
"""
import json
import os
import sys
import unittest

root, tests_json, out_path = sys.argv[1:4]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)
os.environ.setdefault("SKIP_INTEGRATION", "0")

import preload  # noqa: E402

preload.install_environment(root)
preload.import_sqlglot()
tests = json.load(open(tests_json))
tests = tests["tests"] if isinstance(tests, dict) else tests
preload.import_tests(sorted({mod for mod, _ in preload.SHARED_FIXTURES}))
preload.build_class_fixtures(set(preload.SHARED_FIXTURES))

current = [None]
reads = {}


class Recorder:
    """Stands in the class for a fixture attribute: hands out the value and notes the reader."""

    def __init__(self, name, value):
        self.name, self.value = name, value

    def __get__(self, obj, owner):
        if current[0] is not None:
            reads.setdefault(current[0], set()).add(self.name)
        return self.value


class Result(unittest.TestResult):
    def startTest(self, test):
        super().startTest(test)
        current[0] = test._br_id

    def stopTest(self, test):
        super().stopTest(test)
        current[0] = None


for mod_name, cls_name in preload.SHARED_FIXTURES:
    cls = getattr(sys.modules.get(mod_name), cls_name, None)
    attrs = getattr(cls, "_br_attrs", None)
    if not attrs:
        continue
    for attr in attrs:
        setattr(cls, attr, Recorder(attr, cls.__dict__[attr]))
    prefix = mod_name.replace(".", "/") + ".py::" + cls_name + "::"
    suite = unittest.TestSuite()
    for t in tests:
        if t["id"].startswith(prefix):
            case = cls(t["id"].rsplit("::", 1)[1])
            case._br_id = t["id"]
            suite.addTest(case)
            reads.setdefault(t["id"], set())
    result = Result()
    suite.run(result)
    sys.stderr.write("%s: %d tests, %d failures, %d errors\n" % (cls_name, result.testsRun, len(result.failures), len(result.errors)))
    if result.failures or result.errors:
        sys.exit("the pinned tests must pass under the recorder")

with open(out_path, "w") as f:
    json.dump({k: sorted(v) for k, v in sorted(reads.items())}, f, indent=0)
