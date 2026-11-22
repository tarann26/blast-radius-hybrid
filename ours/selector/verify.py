"""Run candidate tests against a patched tree, cheaply and in isolation.

The pinned checkout is read-only, so each commit gets an overlay directory in
/tmp: real directories only along the paths the patch touches, symlinks to the
pinned tree everywhere else (including __pycache__, so unchanged modules load
from their precompiled bytecode). A child forked from the main process (which
has third-party libraries loaded, but never sqlglot) imports sqlglot and the
tests from the overlay, runs the requested tests with per-test time limits and
reports which failed.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import signal
import sys
import time

FILE = object()
_COPY = object()


def build_overlay(dst: str, root: str, files: dict[str, bytes | None], copy_dirs=("tests",)) -> None:
    """files: relative path -> new content, or None for a deleted file.

    Python files under `copy_dirs` are copied (with their mtime, so the
    symlinked __pycache__ stays valid) rather than symlinked: a test that
    resolves its own path (Path(__file__).resolve()) must land in the overlay.
    """
    for top in copy_dirs:
        top_dir = os.path.join(root, top)
        for dirpath, dirnames, filenames in os.walk(top_dir):
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "fixtures")]
            for name in filenames:
                if name.endswith(".py"):
                    rel = os.path.relpath(os.path.join(dirpath, name), root)
                    files.setdefault(rel, _COPY)
    tree: dict = {}
    for rel, content in files.items():
        node = tree
        parts = [p for p in rel.split("/") if p]
        for part in parts[:-1]:
            child = node.get(part)
            if child is FILE or child is None:
                child = node[part] = {}
            node = child
        node[parts[-1]] = (FILE, rel)
    os.makedirs(dst, exist_ok=True)

    def fill(dst_dir: str, src_dir: str, node: dict):
        names = set(os.listdir(src_dir)) if os.path.isdir(src_dir) else set()
        # modules of this folder whose source changes: their compiled files must not be seen
        # (an unchecked-hash pyc is loaded without looking at the source at all)
        changed = {name[:-3] for name, entry in node.items()
                   if isinstance(entry, tuple) and name.endswith(".py") and files[entry[1]] is not _COPY}
        for name in sorted(names | set(node)):
            src = os.path.join(src_dir, name)
            target = os.path.join(dst_dir, name)
            entry = node.get(name)
            if entry is None and name == "__pycache__" and changed and os.path.isdir(src):
                os.mkdir(target)
                for compiled in os.listdir(src):
                    if compiled.split(".", 1)[0] not in changed:
                        os.symlink(os.path.join(src, compiled), os.path.join(target, compiled))
                continue
            if entry is None:
                os.symlink(src, target)
            elif isinstance(entry, tuple):
                content = files[entry[1]]
                if content is None:
                    continue
                if content is _COPY:
                    shutil.copy2(src, target)
                    continue
                with open(target, "wb") as handle:
                    handle.write(content)
            else:
                os.mkdir(target)
                fill(target, src, entry)

    fill(dst, root, tree)


def module_name(rel: str) -> str | None:
    if not rel.endswith(".py"):
        return None
    name = rel[:-3].replace("/", ".")
    if name.endswith(".__init__"):
        name = name[: -len(".__init__")]
    return name


def _rehome_modules(root: str, overlay: str, warm: bool):
    """Point preloaded sqlglot modules at the overlay (warm), or drop them (cold)."""
    import importlib

    for name in list(sys.modules):
        if not (name in ("sqlglot", "tests") or name.startswith("sqlglot.") or name.startswith("tests.")):
            continue
        module = sys.modules[name]
        if not warm or name == "tests" or name.startswith("tests."):
            del sys.modules[name]
            continue

        def move(path):
            return overlay + path[len(root):] if isinstance(path, str) and path.startswith(root) else path

        if getattr(module, "__file__", None):
            module.__file__ = move(module.__file__)
        path = getattr(module, "__path__", None)
        if path is not None:
            module.__path__ = [move(p) for p in path]
        spec = getattr(module, "__spec__", None)
        if spec is not None:
            if spec.origin:
                spec.origin = move(spec.origin)
            if spec.submodule_search_locations is not None:
                spec.submodule_search_locations[:] = [move(p) for p in spec.submodule_search_locations]
    importlib.invalidate_caches()


PROBE_STATS: dict = {}


def _install_probes(overlay: str, probes: dict):
    """Compile patched modules so each probed expression evaluates the old and
    the new version side by side and records whether they ever differ."""
    import ast
    import builtins
    import importlib.abc
    import importlib.util

    current = {"test": None}

    def same(old, new, truth):
        (ov, oe), (nv, ne) = old, new
        if (oe is None) != (ne is None):
            return False
        if oe is not None:
            return type(oe) is type(ne)
        if truth:
            try:
                return bool(ov) == bool(nv)
            except BaseException:  # noqa: BLE001
                return False
        if ov is nv:
            return True
        try:
            return type(ov) is type(nv) and bool(ov == nv)
        except BaseException:  # noqa: BLE001
            return False

    def probe(key, old_thunk, new_thunk, truth):
        try:
            new = (new_thunk(), None)
        except BaseException as error:  # noqa: BLE001
            new = (None, error)
        try:
            old = (old_thunk(), None)
        except BaseException as error:  # noqa: BLE001
            old = (None, error)
        stats = PROBE_STATS.setdefault(key, {"count": 0, "infected": 0, "tests": set()})
        stats["count"] += 1
        stats["tests"].add(current["test"])
        if not same(old, new, truth):
            stats["infected"] += 1
        if new[1] is not None:
            raise new[1]
        return new[0]

    import enum
    import operator

    ops = {"Is": operator.is_, "IsNot": operator.is_not, "Eq": operator.eq, "NotEq": operator.ne}
    plain = (type(None), bool, int, float, complex, str, bytes, tuple, list, dict, set, frozenset, type)

    def user_eq(value):
        kind = type(value)
        if isinstance(value, plain) and kind in plain:
            return False
        if isinstance(value, enum.Enum):
            eq = kind.__eq__
            return not (eq is object.__eq__ or eq is enum.Enum.__eq__ or getattr(eq, "__objclass__", None) in plain)
        return kind.__eq__ is not object.__eq__

    def probe_cmp(key, operands_thunk, old_op, new_op, truth):
        left, right = operands_thunk()
        stats = PROBE_STATS.setdefault(key, {"count": 0, "infected": 0, "tests": set()})
        stats["count"] += 1
        stats["tests"].add(current["test"])
        unsafe = (right is not None and user_eq(left)) or (left is not None and user_eq(right))
        try:
            new = (ops[new_op](left, right), None)
        except BaseException as error:  # noqa: BLE001
            new = (None, error)
        try:
            old = (ops[old_op](left, right), None)
        except BaseException as error:  # noqa: BLE001
            old = (None, error)
        if unsafe or not same(old, new, truth):
            stats["infected"] += 1
        if new[1] is not None:
            raise new[1]
        return new[0]

    builtins.__br_probe__ = probe
    builtins.__br_probe_cmp__ = probe_cmp
    builtins.__br_probe_current__ = current

    def transform(tree, rel, items):
        wanted = {}
        for k, item in enumerate(items):
            old_src, pos, truth = item[0], item[1], item[2]
            wanted[tuple(pos)] = (f"{rel}:{k}", old_src, truth)

        class Rewriter(ast.NodeTransformer):
            def generic_visit(self, node):
                super().generic_visit(node)
                if isinstance(node, ast.expr):
                    pos = (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
                    if pos in wanted:
                        key, old_src, truth = wanted.pop(pos)
                        old = ast.parse(old_src, mode="eval").body
                        no_args = ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
                        identity_swap = (
                            isinstance(old, ast.Compare) and isinstance(node, ast.Compare)
                            and len(old.ops) == len(node.ops) == 1
                            and {type(old.ops[0]), type(node.ops[0])} in ({ast.Is, ast.Eq}, {ast.IsNot, ast.NotEq})
                            and ast.dump(old.left) == ast.dump(node.left)
                            and ast.dump(old.comparators[0]) == ast.dump(node.comparators[0])
                        )
                        if identity_swap:
                            # Evaluate the operands once and flag comparisons whose
                            # == runs user code (e.g. structural equality of two
                            # expressions): their cost and effects are not "inert".
                            operands = ast.Tuple(elts=[node.left, node.comparators[0]], ctx=ast.Load())
                            call = ast.Call(
                                func=ast.Name(id="__br_probe_cmp__", ctx=ast.Load()),
                                args=[ast.Constant(key), ast.Lambda(args=no_args, body=operands),
                                      ast.Constant(type(old.ops[0]).__name__), ast.Constant(type(node.ops[0]).__name__),
                                      ast.Constant(truth)],
                                keywords=[],
                            )
                        else:
                            call = ast.Call(
                                func=ast.Name(id="__br_probe__", ctx=ast.Load()),
                                args=[ast.Constant(key), ast.Lambda(args=no_args, body=old), ast.Lambda(args=no_args, body=node), ast.Constant(truth)],
                                keywords=[],
                            )
                        for sub in ast.walk(call):
                            if sub is not node and not (hasattr(sub, "lineno") and sub in ast.walk(node)):
                                ast.copy_location(sub, node)
                        return call
                return node

        tree = Rewriter().visit(tree)
        ast.fix_missing_locations(tree)
        return tree, not wanted

    modules = {}
    for rel, items in probes.items():
        name = module_name(rel)
        if name:
            modules[name] = (rel, items)

    class ProbeLoader(importlib.abc.Loader):
        def __init__(self, rel, items, path):
            self.rel, self.items, self.path = rel, items, path

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            with open(self.path, "rb") as handle:
                source = handle.read()
            tree = ast.parse(source, self.path)
            tree, complete = transform(tree, self.rel, self.items)
            if not complete:
                PROBE_STATS.setdefault("__incomplete__", {"count": 0, "infected": 1, "tests": set()})
            exec(compile(tree, self.path, "exec"), module.__dict__)

    class ProbeFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in modules:
                return None
            rel, items = modules[name]
            file_path = os.path.join(overlay, rel)
            search = [os.path.dirname(file_path)] if rel.endswith("__init__.py") else None
            return importlib.util.spec_from_file_location(name, file_path, loader=ProbeLoader(rel, items, file_path),
                                                          submodule_search_locations=search)

    sys.meta_path.insert(0, ProbeFinder())
    return current


STAGE_LIGHT_FIXTURE = os.environ.get("BR_STAGE_LIGHT_FIXTURE", "1") == "1"


class _HeaderOnlyCsv:
    """Stand-in for the csv module seen by tests/test_executor.py: readers of files under
    the given benchmark folders yield the header row only (all files when `skip` is None)."""

    def __init__(self, real, skip=None):
        self._real = real
        self._skip = skip

    def reader(self, f, *args, **kwargs):
        rows = self._real.reader(f, *args, **kwargs)
        name = str(getattr(f, "name", "") or "")
        if self._skip is not None and not any(part in name for part in self._skip):
            return rows
        first = next(rows, None)
        return iter([] if first is None else [first])

    def __getattr__(self, name):
        return getattr(self._real, name)


# The executor tests' class fixture parses every TPC-H and TPC-DS table into Python rows
# (about 1.4 s); only test_execute_tpch reads the TPC-H rows and only test_execute_tpcds the
# TPC-DS rows (the other tests build their own tables or query duckdb views, which read
# the files themselves). A child that runs neither in full skips those rows.
_ROW_USERS = {"tpc-h": "tests/test_executor.py::TestExecutor::test_execute_tpch",
              "tpc-ds": "tests/test_executor.py::TestExecutor::test_execute_tpcds"}


def _light_executor_fixture(full_tests):
    """Install the header-only reader for the benchmarks no test in `full_tests` (tests run
    in full, not stopped after optimize()) needs; returns an undo function."""
    if not STAGE_LIGHT_FIXTURE:
        return lambda: None
    skip = [folder + "/" for folder, user in _ROW_USERS.items() if user not in full_tests]
    if not skip:
        return lambda: None
    import importlib

    try:
        module = importlib.import_module("tests.test_executor")
    except BaseException:  # noqa: BLE001 - run_tests reports it
        return lambda: None
    real = getattr(module, "csv", None)
    if real is None or isinstance(real, _HeaderOnlyCsv):
        return lambda: None
    module.csv = _HeaderOnlyCsv(real, skip)
    real_duckdb = getattr(module, "duckdb", None)
    if LAZY_VIEWS and real_duckdb is not None and not isinstance(real_duckdb, _LazyDuckdb):
        module.duckdb = _LazyDuckdb(real_duckdb)

    def undo():
        module.csv = real
        if real_duckdb is not None:
            module.duckdb = real_duckdb

    return undo


LAZY_VIEWS = os.environ.get("BR_LAZY_VIEWS", "1") == "1"


class _LazyConn:
    """A DuckDB connection whose CREATE VIEW statements (TestExecutor's class fixture makes one
    per TPC table: about 0.5 s) run on its first other use, in the same order; tests that never
    query DuckDB never pay for them."""

    def __init__(self, make):
        self._make = make
        self._conn = None
        self._pending = []

    def _real(self):
        if self._conn is None:
            conn = self._make()
            for sql in self._pending:
                conn.execute(sql)
            self._conn = conn
        return self._conn

    def execute(self, sql, *args, **kwargs):
        if (self._conn is None and not args and not kwargs and isinstance(sql, str)
                and sql.lstrip().upper().startswith("CREATE VIEW")):
            self._pending.append(sql)
            return None
        return self._real().execute(sql, *args, **kwargs)

    def close(self):
        if self._conn is not None:
            self._conn.close()

    def __getattr__(self, name):
        return getattr(self._real(), name)


class _LazyDuckdb:
    def __init__(self, real):
        self._real_module = real

    def connect(self, *args, **kwargs):
        real = self._real_module
        return _LazyConn(lambda: real.connect(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._real_module, name)


# Tests that check behaviour at a size: a copy of the pinned method with smaller constants
# runs instead, and its verdict stands for the real test's (the policy schedules this only
# when the patch leaves the test file alone). Constants keep each limit's relation (one node
# more than the first limit; the same union/intersect structure).
SCALED = {
    "tests/test_parser.py::TestParser::test_max_nodes": [
        ("width = 10000", "width = 20"),
        ("            parse_one(sql, max_nodes=1_000_000)", "            parse_one(sql, max_nodes=2_000)"),
        ("parse_one(sql, max_nodes=50_000)", "parse_one(sql, max_nodes=1_000)"),
        ('"(50000)"', '"(1000)"'),
    ],
}


def _install_scaled(tests):
    """Swap in the small-scale copies; a test whose source does not match keeps its real method."""
    import importlib
    import inspect
    import textwrap

    done = []
    for tid in tests:
        reps = SCALED.get(tid)
        if not reps:
            continue
        try:
            path, cls_name, meth = tid.split("::")
            mod = importlib.import_module(path[:-3].replace("/", "."))
            cls = getattr(mod, cls_name)
            orig = cls.__dict__[meth]
            lines, first = inspect.getsourcelines(orig)
            src = "".join(lines)
            for old, new in reps:
                if src.count(old) != 1:
                    raise ValueError(old)
                src = src.replace(old, new)
            ns = {}
            exec(compile("\n" * (first - 1) + textwrap.dedent(src), inspect.getsourcefile(orig), "exec"), mod.__dict__, ns)
            setattr(cls, meth, ns[meth])
            done.append(tid)
        except BaseException:  # noqa: BLE001 - the real test runs instead
            pass
    return done


def _child(overlay: str, root: str, files: dict, tests: list[str], limits: dict, check_modules: list[str], out_fd: int, memory_limit: int | None, warm: bool = False, probes: dict | None = None, groups: list | None = None, leak_check: bool = False, scaled: list | None = None):
    result = {"failed": [], "import_errors": {}, "durations": {}, "error": None}
    try:
        if memory_limit:
            try:
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            except (ValueError, OSError):
                pass
        build_overlay(overlay, root, dict(files))
        _rehome_modules(root, overlay, warm)
        os.chdir(overlay)
        sys.path[:] = [overlay] + [p for p in sys.path if os.path.realpath(p or ".") != os.path.realpath(root) and p not in ("", ".")]
        sys.dont_write_bytecode = True
        # Silence test output: the parent's stdout is the answer channel.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        import importlib

        from selector import runner
        from selector.runner import run_tests

        if probes:
            current = _install_probes(overlay, probes)
            runner.ON_START = lambda node: current.__setitem__("test", node)

        restricted = bool(groups) and any(group[0] for group in groups)
        if restricted:
            # Wrap the fixture loaders before any test module imports them. If the
            # overlay's helpers cannot be wrapped, every group simply runs all of
            # its cases (more work, same answer).
            from selector import cases

            try:
                cases.install(importlib.import_module("tests.helpers"))
            except BaseException:  # noqa: BLE001
                pass

        for name in check_modules:
            signal.signal(signal.SIGALRM, _raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, 20)
            try:
                importlib.import_module(name)
            except BaseException as error:  # noqa: BLE001
                result["import_errors"][name] = repr(error)[:300]
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        failed, durations = set(), {}
        result["stage"] = {}
        if scaled:
            result["scaled"] = _install_scaled([t for t in scaled if t in tests])
        before = None
        if leak_check:
            # State check (selector/leak.py): import the test modules first (pytest
            # imports every test module before the first test runs), then fingerprint
            # every sqlglot class and module and every loaded test module.
            from selector import keyrec

            keyrec.EXPR_POSITION[0] = False
            for path in sorted({node.split("::", 1)[0] for node in tests}):
                try:
                    importlib.import_module(path[:-3].replace("/", "."))
                except BaseException:  # noqa: BLE001 - run_tests reports it as failing
                    pass
            before = keyrec.state_fingerprint(loaded_roots=("tests",))["state"]
        for group in (groups or [({}, tests)]):
            allowed, group_tests = group[0], group[1]
            spec = group[2] if len(group) > 2 else None
            if restricted:
                cases.ALLOWED.clear()
                cases.ALLOWED.update({os.path.normpath(f): frozenset(v) for f, v in allowed.items()})
            if spec is None:
                undo = _light_executor_fixture(set(group_tests)) if any(
                    t.startswith("tests/test_executor.py::") for t in group_tests) else (lambda: None)
                try:
                    group_failed, group_durations = run_tests(group_tests, limit_of=lambda node: limits.get(node, 30.0))
                finally:
                    undo()
            else:
                # Stage check (selector/stage.py): stop each case after optimize()
                # and compare fingerprints; the test's own result is not used.
                from selector import stage

                executor = importlib.import_module("sqlglot.executor")
                stage.EXPECTED[:] = [tuple(c) for _, calls in spec["cases"] for c in calls]
                stage.OBSERVED.clear()
                by_input, ordinal = {}, {}
                for k, calls in spec["cases"]:
                    for c in calls:
                        by_input.setdefault(c[0], set()).add((c[1], c[2]))
                        ordinal[c[0]] = k
                stage.ABORT.update({"by_input": by_input, "ordinal": ordinal, "cost": spec["full"],
                                    "warm": spec["warm"], "cap": spec["cap"]})
                stage.install(executor)
                # Every case stops right after optimize(), which reads the schema, never the
                # table rows, so the class fixture's CSV rows (most of its cost) are not
                # needed: its csv reader yields the header row only for this run. A re-run
                # of differing cases is another run, whose fixture loads the real rows.
                undo = _light_executor_fixture(set())
                try:
                    _, group_durations = run_tests(group_tests, limit_of=lambda node: limits.get(node, 30.0))
                finally:
                    stage.uninstall(executor)
                    undo()
                differing = stage.differing_cases(spec["cases"], [tuple(o) for o in stage.OBSERVED])
                cost = spec["warm"] + sum(spec["full"].get(k, spec["cap"] + 1.0) for k in differing)
                rerun = bool(differing) and cost <= spec["cap"]
                for node in group_tests:
                    result["stage"][node] = {"differing": len(differing), "of": len(spec["cases"]),
                                             "calls": len(stage.OBSERVED), "rerun": rerun}
                if not differing:
                    group_failed = set()
                elif rerun:
                    # optimize() returned something else for these cases: run them in full
                    cases.ALLOWED[os.path.normpath(spec["fixture"])] = frozenset(differing)
                    group_failed, more = run_tests(group_tests, limit_of=lambda node: limits.get(node, 30.0))
                    group_durations.update(more)
                else:
                    group_failed = set(group_tests)
            failed |= set(group_failed)
            durations.update(group_durations)
        if before is not None:
            from selector import keyrec, leak

            after = keyrec.state_fingerprint(loaded_roots=("tests",))["state"]
            result["leak"] = leak.state_leaks(before, after)[:20]
        result["failed"] = sorted(failed)
        result["durations"] = {k: round(v, 4) for k, v in durations.items()}
        result["probes"] = {k: {"count": v["count"], "infected": v["infected"], "tests": len(v["tests"])}
                            for k, v in PROBE_STATS.items()}
    except BaseException as error:  # noqa: BLE001
        result["error"] = repr(error)[:500]
    data = json.dumps(result).encode()
    view = memoryview(data)
    while view:
        written = os.write(out_fd, view)
        view = view[written:]
    os.close(out_fd)


_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _cpu_seconds(pid: int):
    """User+system CPU seconds of a running child, or None if unknown."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            fields = handle.read().rsplit(b")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / _TICKS
    except (OSError, IndexError, ValueError):
        pass
    try:
        import psutil

        times = psutil.Process(pid).cpu_times()
        return times.user + times.system
    except Exception:  # noqa: BLE001
        return None


class _ImportTimeout(BaseException):
    pass


def _raise_timeout(signum, frame):
    raise _ImportTimeout()


class Job:
    def __init__(self, key, files, tests, limits, check_modules, budget, warm=False, probes=None, groups=None):
        # groups: [(fixture -> case ordinals to keep, [test ids])], run in order in
        # the same child; None runs `tests` with every case.
        self.groups = groups
        self.warm = warm
        self.probes = probes
        self.key = key
        self.files = files
        self.tests = tests
        self.limits = limits
        self.check_modules = check_modules
        self.budget = budget
        self.pid = None
        self.fd = None
        self.started = None
        self.buffer = b""
        self.result = None
        self.cpu = 0.0
        self.wall = 0.0


def run_jobs(jobs: list[Job], root: str, tmp_base: str, parallel: int, deadline: float, memory_limit: int | None = None, log=None):
    """Run jobs (fork per job), at most `parallel` at once. A job that overruns
    its budget or the global deadline is killed and gets result None."""
    import gc
    import select

    gc.collect()
    gc.freeze()  # children must not drag the parent's objects through their GC (copy-on-write)
    pending = list(jobs)
    running: dict[int, Job] = {}
    while pending or running:
        while pending and len(running) < parallel:
            job = pending.pop(0)
            if time.monotonic() > deadline:
                job.result = None
                continue
            read_fd, write_fd = os.pipe()
            overlay = os.path.join(tmp_base, f"ov_{job.key}")
            sys.stdout.flush()
            sys.stderr.flush()
            pid = os.fork()
            if pid == 0:
                os.close(read_fd)
                code = 0
                try:
                    _child(overlay, root, job.files, job.tests, job.limits, job.check_modules, write_fd, memory_limit, job.warm, job.probes, job.groups, getattr(job, "leak_check", False), getattr(job, "scaled", None))
                except BaseException:  # noqa: BLE001
                    code = 1
                os._exit(code)
            os.close(write_fd)
            job.pid, job.fd, job.started = pid, read_fd, time.monotonic()
            running[pid] = job
        if not running:
            break
        fds = {job.fd: job for job in running.values() if job.fd is not None}
        wait_for = 0.02
        if not fds:
            time.sleep(wait_for)  # every pipe is closed: children are exiting
        else:
            ready, _, _ = select.select(list(fds), [], [], wait_for)
            for fd in ready:
                job = fds[fd]
                chunk = os.read(fd, 1 << 16)
                if chunk:
                    job.buffer += chunk
                else:
                    os.close(fd)
                    job.fd = None
        now = time.monotonic()
        for pid, job in list(running.items()):
            done_pid, status, usage = os.wait4(pid, os.WNOHANG)
            if done_pid == pid:
                if job.fd is not None:
                    while True:
                        chunk = os.read(job.fd, 1 << 16)
                        if not chunk:
                            break
                        job.buffer += chunk
                    os.close(job.fd)
                    job.fd = None
                job.cpu = usage.ru_utime + usage.ru_stime
                job.wall = time.monotonic() - job.started
                try:
                    job.result = json.loads(job.buffer.decode()) if job.buffer else None
                except ValueError:
                    job.result = None
                del running[pid]
                shutil.rmtree(os.path.join(tmp_base, f"ov_{job.key}"), ignore_errors=True)
                if log:
                    rss = usage.ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)  # MB
                    log(f"verify {job.key}: {len(job.tests)} tests, cpu {job.cpu:.2f}s wall {job.wall:.2f}s rss {rss:.0f}MB -> "
                        f"{'none' if job.result is None else len(job.result.get('failed', []))} failed"
                        f"{' ERROR ' + str(job.result.get('error')) if job.result and job.result.get('error') else ''}"
                        f"{' IMPORT ' + str(list(job.result.get('import_errors', {}))) if job.result and job.result.get('import_errors') else ''}")
                continue
            # Budgets are CPU seconds of the child (a loaded machine stretches
            # wall time without making a run any more expensive); a child stuck
            # off-CPU (a test that sleeps or waits at import) is stopped after twice its
            # budget in wall time: it holds one of the two job slots meanwhile.
            elapsed = now - job.started
            used = _cpu_seconds(pid)
            over_budget = (used if used is not None else elapsed) > job.budget or elapsed > 2 * job.budget + 15
            if over_budget or now > deadline:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status, usage = os.wait4(pid, 0)
                if job.fd is not None:
                    os.close(job.fd)
                    job.fd = None
                job.cpu = usage.ru_utime + usage.ru_stime
                job.wall = time.monotonic() - job.started
                job.result = None
                del running[pid]
                shutil.rmtree(os.path.join(tmp_base, f"ov_{job.key}"), ignore_errors=True)
                if log:
                    log(f"verify {job.key}: killed after {job.wall:.1f}s")
    return jobs
