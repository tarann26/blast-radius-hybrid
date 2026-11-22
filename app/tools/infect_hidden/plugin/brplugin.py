"""pytest plugin: records every failing test id, collection errors, and call durations."""
import json, os, time

_failed = {}      # nodeid -> list of (when, kind)
_collect_err = {} # nodeid(file) -> repr
_dur = {}
_outcomes = {}

def pytest_collectreport(report):
    if report.failed:
        _collect_err[report.nodeid] = str(report.longrepr)[:2000]

def pytest_runtest_logreport(report):
    nid = report.nodeid
    if report.when == "call" and type(report).__name__ == "TestReport":
        _dur[nid] = _dur.get(nid, 0.0) + report.duration
    if report.failed:
        _failed.setdefault(nid, []).append([report.when, type(report).__name__, str(report.longrepr)[-1500:]])
    elif report.skipped and report.when in ("setup", "call"):
        _outcomes[nid] = "skipped"

def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("BR_OUT", "/repo/_result.json")
    with open(out, "w") as f:
        json.dump({"failed": _failed, "collect_errors": _collect_err, "durations": _dur,
                   "skipped": [k for k in _outcomes], "exitstatus": int(exitstatus)}, f)


def pytest_collection_modifyitems(config, items):
    only = os.environ.get("BR_ONLY")
    if not only or not os.path.exists(only):
        return
    keep = set(open(only).read().split("\n"))
    selected = [i for i in items if i.nodeid in keep]
    deselected = [i for i in items if i.nodeid not in keep]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
