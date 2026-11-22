"""Builds maps/registry_start.json from the pinned commit: the dialect names that are in the
dialect registry (Dialect._classes) when a full pytest run starts its first test, that is,
after pytest has imported every test file. The other dialects are registered later, by the
first test that looks them up (Dialect._try_load). Usage (pinned image, repo at /repo):

    python build_registry_start.py /repo /data/tests.json OUT.json
"""
import importlib
import json
import os
import sys

root, tests_json, out_path = sys.argv[1:4]
sys.path.insert(0, root)
os.chdir(root)
os.environ.setdefault("SKIP_INTEGRATION", "0")
tests = json.load(open(tests_json))
tests = tests["tests"] if isinstance(tests, dict) else tests
for name in sorted({t["id"].split("::")[0][:-3].replace("/", ".") for t in tests}):
    importlib.import_module(name)
from sqlglot.dialects.dialect import Dialect  # noqa: E402

with open(out_path, "w") as f:
    json.dump(sorted(str(k) for k in Dialect._classes), f)
