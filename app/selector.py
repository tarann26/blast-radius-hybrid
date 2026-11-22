"""Blast Radius selector.

For each commit: find the tests that can be affected by the patch (maps built from the pinned
commit), apply the patch in a forked child process, run those tests, and select the tests
that fail. Logs go to stderr; only the answers go to stdout.
"""
from __future__ import annotations

import collections
import gc
import json
import marshal
import os
import re
import zlib
import select
import signal
import sys
import time
import types
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# The code objects of the pinned sqlglot and test modules, by source path, as this process
# loads them from their pycs. A fresh child (which imports the patched tree again) then
# executes the same code objects again instead of reading and unmarshalling the pycs: the
# patched modules never come here (engine.OverlayFinder serves them from memory).
from importlib.machinery import SourceFileLoader as _SourceFileLoader  # noqa: E402

CODE_CACHE = {}
_CODE_ROOT = os.path.join(os.environ.get("BR_ROOT", "/repo"), "")
_pyc_get_code = _SourceFileLoader.get_code


def _cached_get_code(self, fullname):
    path = self.path
    code = CODE_CACHE.get(path)
    if code is None:
        code = _pyc_get_code(self, fullname)
        if path.startswith(_CODE_ROOT):
            CODE_CACHE[path] = code
    return code


if os.environ.get("BR_NO_CODE_CACHE") != "1":
    _SourceFileLoader.get_code = _cached_get_code
    engine_keep_code = True
else:
    engine_keep_code = False

import engine  # noqa: E402
import patchrules  # noqa: E402
import sideeffect  # noqa: E402
from rawz import unz as _unz  # noqa: E402  (map blobs: raw in the image, zlib in the repo)
from astwalk import walk as _walk  # noqa: E402

engine.KEEP_CODE = engine_keep_code

try:
    import keymap  # noqa: E402  (key read map of the table constants; optional)
except ImportError:
    keymap = None
try:
    import cone  # noqa: E402  (re-import of the importer cone only; optional)
except ImportError:
    cone = None

ROOT = os.environ.get("BR_ROOT", "/repo")
DATA = os.environ.get("BR_DATA", "/data")
MAPS = os.environ.get("BR_MAPS", os.path.join(HERE, "maps"))
DEBUG = os.environ.get("BR_DEBUG")
RUN_ALL = os.environ.get("BR_RUN_ALL")  # validation mode: run every candidate, no early stop
RUN_SLOW = os.environ.get("BR_RUN_SLOW")  # validation mode: also run the slow tests
NO_REDUCE = os.environ.get("BR_NO_REDUCE")  # validation mode: always run whole tests
ITERS_ON = os.environ.get("BR_ITERS", "1") != "0"  # loop-iteration parts (iterred.py)
# Loop-iteration parts also for tests that have no statement parts (the map then traces the
# whole test). Off: paired runs showed no CPU gain for such tests (their time is in set-up).
ITERS_WHOLE = os.environ.get("BR_ITERS_WHOLE") == "1"
MAPS_OBJ = None
ANSWER_FD = None
BASE_STATE = None
SMALL_PLAN = float(os.environ.get("BR_SMALL_PLAN", "0.2"))  # CPU s; see small_plan in run()
BASE_FUNCTION_KEYS = frozenset()
BASE_OBJECTS = {}  # statefp key -> the pinned table (dict, set, frozenset) that it held at the snapshot
BASE_CODES = None  # statefp code hashes of the snapshot of the pinned tree (statefp.Fingerprinter.base_codes)
PRE_PATH = None  # sys.path before the parent imported the test modules' libraries (preimport_test_libraries)
REGISTRY_CLEAR_AT = None  # first test after tests/test_dialect_entry_points.py (see registry_clear_at)
REGISTRY_START = None  # dialect names registered when a full run starts (build_registry_start.py)
PANDAS_USERS = frozenset()  # tests that load numpy or pandas (maps/pandas_tests.json), when they are lazy (lazymods.py)
NO_DUCK_REPLAY = os.environ.get("BR_NO_DUCK_REPLAY") == "1"  # validation: every DuckDB query of the tests runs
NO_DIFF = os.environ.get("BR_NO_DIFF")  # validation: no import-state diff
CLASS_FILTER = os.environ.get("BR_CLASS_FILTER") == "1"  # off: the instantiation trace does not see every reader
FRESH_REDUCE = os.environ.get("BR_FRESH_REDUCE", "1") == "1"
NO_CLASSMAP = os.environ.get("BR_NO_CLASSMAP") == "1"  # validation: no cut by the class attribute map
NO_KEYMAP = os.environ.get("BR_NO_KEYMAP") == "1" or keymap is None  # validation: no key map cut (keymap.py)
TIGHT = os.environ.get("BR_NO_TIGHT") != "1"  # exact line sets of the long tests (maps/tight_lines.bin)
TOKCHECK = os.environ.get("BR_NO_TOKCHECK") != "1"  # token streams of the long tests' inputs (tokcheck.py)
TOKCHECK_SECONDS = 2.0  # CPU limit of that check; inputs left unchecked count as changed
# files whose code runs inside Tokenizer.tokenize (with the trie of the keywords)
TOKENIZER_FILES = frozenset(("sqlglot/tokens.py", "sqlglot/tokenizer_core.py", "sqlglot/trie.py", "sqlglot/jsonpath.py"))
NO_CONE = os.environ.get("BR_NO_CONE") == "1" or cone is None  # validation: every import-time commit imports the whole tree

WORKER_COST = 2.0
HOLD_SECONDS = float(os.environ.get("BR_HOLD", "1.5"))  # test runs below this CPU cost always start
COMMIT_SECONDS = float(os.environ.get("BR_COMMIT", "8.0"))  # CPU limit of the test runs of one commit
FRESH_SECONDS = float(os.environ.get("BR_FRESH", "3.0"))  # the same for a commit that needs a new import
INERT_SECONDS = 1.0  # the same when the import state did not change at all
# CPU seconds of the class fixtures that a child must build again after a fresh import (the
# parent builds them only once, for the commits that run in the parent's import). Shared by
# the candidates of the class.
CLASS_SETUP_SECONDS = {"tests/test_executor.py::TestExecutor": 1.0} if os.environ.get("BR_CLASS_SETUP", "1") == "1" else {}
HOLD_FACTOR = float(os.environ.get("BR_HOLD_FACTOR", "0.7"))
QUIET_HOLD_SECONDS = float(os.environ.get("BR_QUIET_HOLD", "5.0"))
AFTER_FAIL_RATIO = float(os.environ.get("BR_AFTER_FAIL", "0.5"))
# Short reruns (short_rerun in run): after a first failure, the child leaves the tests of
# the 'skip' set unrun, and finish() selects them as they are. When those short tests cost at
# least SHORT_MIN packed seconds in the answer, a second child runs them anyway (the most
# recorded seconds per CPU second first, as far as the expected saving pays for the CPU): a
# failing one stays selected, a passing one leaves the answer. BR_SHORT_RERUN=0 turns it off.
SHORT_RERUN = os.environ.get("BR_SHORT_RERUN", "1") == "1"
SHORT_MIN = float(os.environ.get("BR_SHORT_MIN", "2.0"))  # packed seconds of those tests at least
SHORT_PASS = float(os.environ.get("BR_SHORT_PASS", "0.75"))  # share of them that pass (labeled mutants: 0.77)
# charged seconds per estimated CPU second (a run costs about 0.7 of its estimate in local CPU,
# and the scoring machine charges about 0.6 of that; the board measured about 0.59 per local CPU second)
SHORT_PRICE = float(os.environ.get("BR_SHORT_PRICE", "0.6"))
MANY_FAILURES = 8  # with this many failing tests, a long test on the same code is selected too
# A plain pytest run stops at collection when a test file cannot be imported (see finish()).
# BR_STOP_AT_COLLECTION=0: such a commit also gets the tests that fail when they run.
STOP_AT_COLLECTION = os.environ.get("BR_STOP_AT_COLLECTION", "1") == "1"
# BR_COLL_RUN=1: the labels ran on past collection errors (pytest --continue-on-collection-errors).
# The tests of the files that do not import fail and are selected without a run; the other
# candidates run (or are held at the price of their selection beside those), and the answer is
# those tests + the tests that fail at run time + the held tests that could fail.
COLL_RUN = os.environ.get("BR_COLL_RUN", "1") == "1"
if COLL_RUN:
    STOP_AT_COLLECTION = False
COLL_LOOSE = os.environ.get("BR_COLL_LOOSE", "1") == "1"  # any run may be held, not only the costly ones
COLL_BUDGET = float(os.environ.get("BR_COLL_BUDGET", "inf"))  # CPU limit of the runs of such a commit
# The stage check of the long executor test (stages.py), for a commit that would select that
# test without a run: "1" on, "0" off. For validation: "always" = whenever the test is held,
# "force" = the test is always held and checked.
STAGE = os.environ.get("BR_STAGE", "1")
MEMO = os.environ.get("BR_MEMO", "1") == "1"  # the step memo of the stage check (memo.py); BR_MEMO=0 turns it off
# CPU seconds (as measured when the stage map was built) that a check may cost for each
# second that leaving the test out of the answer saves. A check that proves a pass costs CPU
# for all its tasks; at 1.3 it still pays when a CPU second is charged at 0.6 score seconds.
STAGE_GAIN = float(os.environ.get("BR_STAGE_GAIN", "1.7"))
STAGE_STATE_SECONDS = 0.6  # the snapshots of the module state for a check that proves a pass
TEST_TIMEOUT = 4.0  # a test that runs longer than this (and 25 times its usual time) counts as hung
SETUP_TIMEOUT = 6.0  # a child whose imports take longer than this is stopped
SLOWDOWN = float(os.environ.get("BR_SLOWDOWN", "1.0"))  # local runs on a loaded machine
# Per-test CPU guard: a test whose run has cost more CPU than selecting it unrun can cost is
# stopped and selected (see prepare_plan and run_child). The limit of test n is
#   max(GUARD_MIN, GUARD_FACTOR * estimate + GUARD_ADD, (recorded seconds + 2) / GUARD_PRICE)
# CPU seconds: never inside the range of a normal run, and never before the run has cost what
# the selection of the test costs at most (its seconds and one more worker). The guard only
# ever shortens the time that a test may take (TEST_TIMEOUT and 25 times its usual time): a
# test with a small estimate keeps that limit as it is.
GUARD = os.environ.get("BR_NO_GUARD") != "1"
GUARD_MIN = float(os.environ.get("BR_GUARD_MIN", "4.0"))  # (= TEST_TIMEOUT)
GUARD_FACTOR = float(os.environ.get("BR_GUARD_FACTOR", "6.0"))
GUARD_ADD = float(os.environ.get("BR_GUARD_ADD", "1.5"))
GUARD_PRICE = float(os.environ.get("BR_GUARD_PRICE", "1.0"))  # charged seconds per CPU second of a run
GUARD_RESTARTS = 3  # new children per commit after stopped tests; then the rest is selected unrun
CPU_GUARD = {}  # in a child: test -> CPU seconds that its run may cost (the parent's plan.guard)
# CPU seconds of a new child before its first test, by kind of commit (measured on the pinned tree)
RESTART_SECONDS = {"hot": 0.05, "fresh_cone": 0.5, "fresh": 1.5, "fresh_purge": 1.5}
# The price of a run: charged seconds per estimated CPU second. The estimates (cost.bin) are in
# the unit of the recorded seconds of tests.json (build_costs.py scales them so), and the CPU
# seconds of a run on the scoring machine are about as long: the price is near 1. It is set a
# little lower, so that a test is held only when that also pays on a machine that charges less.
CPU_PRICE = float(os.environ.get("BR_CPU_PRICE", "0.8"))
# Hold rule by packed cost (marginal_holds in run): a test that would run is held when the
# CPU of the runs costs more than the selection of the tests costs inside the answer.
MARGINAL = os.environ.get("BR_MARGINAL") == "1"
# Names of interpreter-wide settings (the ones that _interpreter_settings compares after each
# test). A run is what shows that a test leaves such a setting changed for the later tests. So
# marginal_holds holds no test of a commit whose added lines name one, and no test whose own
# test file names one.
SETTINGS_WORDS = (
    "os.chdir", "os.environ", "os.putenv", "setrecursionlimit", "setswitchinterval", "sys.path", "getcontext",
    "localcontext", "setcontext", "gc.", "logging.", "getLogger", "sys.stdout", "sys.stderr", "excepthook",
    "settrace", "setprofile",
)
# cost.bin corrections from runs of the pinned commit in today's set-up (build_cost_fix.py)
COST_FIX = os.environ.get("BR_NO_COST_FIX") != "1"
# One child serves all the commits of the batch that change nothing but entries of .sql fixture
# files (see fixture_only in run() and fixture_batch_child); BR_NO_FIXTURE_BATCH=1 gives each of
# them its own child again (validation: the answers must be the same).
FIXTURE_BATCH = os.environ.get("BR_NO_FIXTURE_BATCH") != "1"
FIXTURE_BATCH_MIN = 2  # commits in one shared child at least
PRE_KEEPS = os.environ.get("BR_PRE_KEEPS", "1") == "1"  # reduced test code compiled in the parent (prepare_keeps)
# cone commits under a holder process (cone_in_holder): 0 off, 1 when it pays, 2 always
CONE_HOLDER = os.environ.get("BR_CONE_HOLDER", "1")


def _holder_pays():
    """The holder only keeps this process's 2 MB pages whole: it pays where the allocator's
    arena gets transparent huge pages (always, or madvise with mimalloc's large OS pages)."""
    if CONE_HOLDER == "2":
        return True
    if CONE_HOLDER != "1":
        return False
    try:
        with open("/sys/kernel/mm/transparent_hugepage/enabled") as f:
            mode = f.read()
    except OSError:
        return False
    if "[always]" in mode:
        return True
    return ("[madvise]" in mode and os.environ.get("PYTHONMALLOC") == "mimalloc"
            and os.environ.get("MIMALLOC_LARGE_OS_PAGES") == "1")


WALL_LIMIT = float(os.environ.get("BR_WALL", "150"))  # the batch must end well inside 3 minutes
# Batches of more than WALL_SMALL commits (the final: about 60, limit 4.5 minutes) get more time:
# WALL_PER_COMMIT seconds for each commit beyond WALL_SMALL, up to WALL_MAX (BR_WALL overrides).
WALL_SMALL = 45
WALL_PER_COMMIT = 4.0
WALL_MAX = 210.0
INFECT_KEEP = os.environ.get("BR_NO_INFECT_KEEP") != "1"
# Memory guard (parent side): a child whose resident memory passes RSS_LIMIT, or a container
# whose memory passes CG_LIMIT, is stopped like a test that the CPU guard stops (its test is
# selected). The container's limit is 4 GB; an OOM kill there would end the batch.
RSS_LIMIT = float(os.environ.get("BR_RSS_LIMIT_MB", "2560")) * 1048576
CG_LIMIT = float(os.environ.get("BR_CG_LIMIT_MB", "3400")) * 1048576
MEM_POLL = 0.05
MEM_GUARD = os.environ.get("BR_NO_MEM_GUARD") != "1"
SLOTS_RULE = os.environ.get("BR_NO_SLOTS") != "1"


def wall_limit():
    if "BR_WALL" in os.environ:
        return WALL_LIMIT
    try:
        n = sum(1 for p in os.listdir(os.path.join(DATA, "commits")) if p.endswith(".patch"))
    except OSError:
        return WALL_LIMIT
    if n <= WALL_SMALL:
        return WALL_LIMIT
    return min(WALL_MAX, WALL_LIMIT + WALL_PER_COMMIT * (n - WALL_SMALL))


_CG_FD = []


def memory_over(pid, statm_fd):
    """A reason string when the child (or the container) uses too much memory, else None."""
    try:
        rss = int(os.pread(statm_fd, 128, 0).split()[1]) * 4096
        if rss > RSS_LIMIT:
            return "child resident %d MB" % (rss >> 20)
    except (OSError, ValueError, IndexError):
        pass
    if not _CG_FD:
        try:
            _CG_FD.append(os.open("/sys/fs/cgroup/memory.current", os.O_RDONLY))
        except OSError:
            _CG_FD.append(None)
    fd = _CG_FD[0]
    if fd is not None:
        try:
            cur = int(os.pread(fd, 64, 0))
            if cur > CG_LIMIT:
                # memory.current counts the page cache too: the anonymous and shared memory
                # decide (memory.stat)
                with open("/sys/fs/cgroup/memory.stat") as f:
                    used = 0
                    for row in f:
                        k, _, v = row.partition(" ")
                        if k in ("anon", "shmem"):
                            used += int(v)
                if used > CG_LIMIT:
                    return "container memory %d MB" % (used >> 20)
        except (OSError, ValueError):
            pass
    return None


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Maps


class Maps:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "meta.bin"), "rb") as f:
            self.meta = marshal.loads(f.read())  # much faster than marshal.load(f)
        with open(os.path.join(path, "sets.idx"), "rb") as f:
            self.offsets = array("I")
            self.offsets.frombytes(f.read())
        self.sets_file = open(os.path.join(path, "sets.bin"), "rb")
        self.file_cache = {}
        self.set_cache = {}
        self.unit_cache = {}
        self.test_cache = {}
        self._paths = None
        self._readers = None
        self._class_users = None
        self._classmap = None
        self._faithful = None
        self._tight = None
        self._stages = None
        try:
            with open(os.path.join(path, "cost.bin"), "rb") as f:
                self.cost = marshal.loads(f.read())
        except OSError:
            self.cost = None
        if self.cost and COST_FIX:
            try:
                with open(os.path.join(path, "cost_fix.json")) as f:
                    for n, value in json.load(f)["cost"].items():
                        self.cost[int(n)] = float(value)
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                pass

    def file(self, rel):
        rec = self.file_cache.get(rel)
        if rec is None:
            fid = self.meta["files"].get(rel)
            if fid is None:
                return None
            with open(os.path.join(self.path, "f", f"{fid}.bin"), "rb") as f:
                rec = self.file_cache[rel] = marshal.loads(_unz(f.read()))
        return rec

    def unit_entries(self, rel, line):
        """{test: pattern id} for the tests whose parts are known to run this line."""
        fid = self.meta["files"].get(rel)
        if fid is None:
            return {}
        key = (fid, line // 256)
        chunk = self.unit_cache.get(key)
        if chunk is None:
            try:
                with open(os.path.join(self.path, "u", "%d_%d.bin" % key), "rb") as f:
                    chunk = marshal.loads(_unz(f.read()))
            except OSError:
                chunk = {}
            self.unit_cache[key] = chunk
        raw = chunk.get(line)
        if raw is None:
            return {}
        if isinstance(raw, bytes):
            arr = array("H")
            arr.frombytes(raw)
            raw = chunk[line] = dict(zip(arr[0::2], arr[1::2]))
        return raw

    def test_units(self, number):
        rec = self.test_cache.get(number, 0)
        if rec == 0:
            try:
                with open(os.path.join(self.path, "t", "%d.bin" % number), "rb") as f:
                    rec = marshal.loads(_unz(f.read()))
            except OSError:
                rec = None
            self.test_cache[number] = rec
        return rec

    def readers(self, name):
        """The function lines that read 'name': list of (path, line)."""
        if self._readers is None:
            try:
                with open(os.path.join(self.path, "readers.bin"), "rb") as f:
                    self._readers = marshal.loads(_unz(f.read()))
            except OSError:
                self._readers = {}
        raw = self._readers.get(name)
        if not raw:
            return ()
        arr = array("I")
        arr.frombytes(raw)
        paths = self.paths()
        return [(paths[arr[i]], arr[i + 1]) for i in range(0, len(arr), 2)]

    def class_users(self, name):
        """Ids of the classes whose instances can read an attribute defined by class 'name',
        or None when that cannot be told from instantiations."""
        if self._class_users is None:
            try:
                with open(os.path.join(self.path, "classes.bin"), "rb") as f:
                    self._class_users = marshal.loads(_unz(f.read()))
            except OSError:
                self._class_users = {}
        return self._class_users.get(name)

    def classmap(self):
        """The class attribute read map (build_classmap.py), or None when there is none."""
        if self._classmap is None:
            try:
                with open(os.path.join(self.path, "classmap.bin"), "rb") as f:
                    cm = marshal.loads(_unz(f.read()))
                hooked = array("I")
                hooked.frombytes(cm["hooked"])
                unknown = array("H")
                unknown.frombytes(cm["unknown"])
                cm["hooked"] = {cm["classes"][i] for i in hooked}
                cm["unknown"] = set(unknown)
                cm["fnames"] = set(cm["fnames"])
                self._classmap = cm
            except (OSError, ValueError, KeyError, EOFError, zlib.error):
                self._classmap = False
        return self._classmap or None

    def tight_lines(self):
        """{test: {path: set of lines}}: every line that each of a few long tests runs, from
        exact traces of the pinned tree (build_tight_lines.py), or {} without the map."""
        if self._tight is None:
            self._tight = {}
            if TIGHT:
                try:
                    with open(os.path.join(self.path, "tight_lines.bin"), "rb") as f:
                        raw = marshal.loads(_unz(f.read()))
                    for n, files in raw.items():
                        out = self._tight[n] = {}
                        for rel, blob in files.items():
                            arr = array("I")
                            arr.frombytes(blob)
                            out[rel] = frozenset(arr)
                except (OSError, ValueError, EOFError, zlib.error):
                    self._tight = {}
        return self._tight

    def stages(self):
        """{test id: record} of the stage map (build_stages.py), each with the build-time
        measurements under "cost"; empty without the two files."""
        if self._stages is None:
            import stages

            self._stages = {}
            try:
                with open(os.path.join(self.path, "stage_cost.bin"), "rb") as f:
                    costs = marshal.loads(f.read())
                for test_id, record in stages.load_map(os.path.join(self.path, "stages.bin")).items():
                    cost = costs.get(test_id)
                    if cost and set(cost["cpu"]) == set(record["tasks"]):
                        self._stages[test_id] = dict(record, cpu=cost["cpu"], cost=cost)
            except (OSError, ValueError, EOFError, TypeError, KeyError, AttributeError):
                self._stages = {}
        return self._stages

    def faithful(self):
        """The tests whose method, compiled alone as apply_reduction does, equals the pinned
        method (build_faithful.py); empty without the map."""
        if self._faithful is None:
            try:
                with open(os.path.join(self.path, "faithful.json")) as f:
                    self._faithful = frozenset(json.load(f))
            except (OSError, ValueError):
                self._faithful = frozenset()
        return self._faithful

    def paths(self):
        if self._paths is None:
            self._paths = {fid: rel for rel, fid in self.meta["files"].items()}
        return self._paths

    def tests(self, set_id):
        if set_id == 0:
            return ()
        s = self.set_cache.get(set_id)
        if s is None:
            a, b = self.offsets[set_id], self.offsets[set_id + 1]
            self.sets_file.seek(a)
            arr = array("H")
            arr.frombytes(self.sets_file.read(b - a))
            s = self.set_cache[set_id] = arr
        return s


# ---------------------------------------------------------------------------
# Planning


class Plan:
    __slots__ = ("commit", "mode", "cands", "work", "files", "note", "all_fail", "blind", "lines", "keeps", "broken", "times", "pool_died", "fpairs", "stmt_lines", "diff", "def_units", "data_cands", "skipped", "diffable", "class_filter", "plain_lines", "unaffected", "protect", "enders", "gone", "fixture_users", "global_leak", "must", "added", "infect_fresh", "infect_files", "watch", "watched", "leaks", "tokcheck", "sure_infected", "cone", "guard", "prep", "suspects", "body", "econ", "settings", "iters", "swap", "swap_work", "klass", "class_work", "pre_keeps", "stage", "stage_info", "class_broken", "coll_dead", "coll_names", "coll_extra", "vanished")

    def __init__(self, commit):
        self.commit = commit
        self.mode = "noop"  # noop | hot | fresh | all
        self.cands = set()  # tests that can be affected
        self.work = []  # (abs path, [(old code, new code)]) for in-memory replacement
        self.files = {}  # repo-relative path -> new text (None = deleted)
        self.note = []
        self.all_fail = False
        self.blind = set()  # tests selected without a run when the commit is not clean
        self.lines = {}  # repo-relative .py path -> lines of the changed statements (call-time code)
        self.keeps = {}  # test -> (statements, fixture entries, pool tasks) to run; absent = whole test
        self.iters = {}  # test -> ((kept paths, their prefixes), rewritten sites, edits): see iterred.py
        self.pre_keeps = None  # (the keeps and iters it was made from, reduced code per test): prepare_keeps
        self.broken = []  # modules that fail to import with the patch
        self.class_broken = []  # test modules whose setUpClass raises (they import): see finish()
        self.coll_dead = set()  # COLL_RUN: tests that fail at collection, selected without a run
        self.coll_names = set()  # (the modules that do not import, as choose() knew them)
        self.coll_extra = set()  # tests that run a changed unit that also runs at import time
        self.times = {}  # test -> CPU seconds of its run (for logs)
        self.pool_died = []  # tests during which a class-level process pool died
        self.fpairs = []  # (old code, new code) of changed function bodies that also run at import
        self.stmt_lines = {}  # path -> lines of changed function statements (import-time commits)
        self.infect_fresh = None  # tests with a changed evaluation, for a function that also runs at import
        self.infect_files = []  # (tests with a changed evaluation, route) of each file planned with the operand outcome map
        self.diff = None  # names whose import-time value the patch changes (import-state diff)
        self.def_units = {}  # path -> lines of functions whose header (defaults, decorators) changed
        self.data_cands = set()  # tests that read a changed file as data
        self.skipped = set()  # tests not run because another test had failed already
        self.diffable = True  # every changed .py file is a modified sqlglot/ module
        self.class_filter = None  # class ids: only parts that create one of these can read the change
        self.plain_lines = set()  # (path, line) of changed code itself: no class filter there
        self.unaffected = set()  # candidates that cannot read the changed value at all
        self.protect = set()  # tests whose verdicts decide about long held tests
        self.enders = {}  # tests whose failure can end the runs of the commit (see execute)
        self.fixture_users = set()  # tests that walk a class fixture list of which few entries changed
        self.global_leak = None  # first test after which interpreter-wide settings differ
        self.gone = set()  # module or class names that the patch removes (no reduced runs then)
        self.must = set()  # tests that read changed test data directly: selected when held
        self.tokcheck = []  # held long tests whose tokenize inputs the child tokenizes again (tokcheck.py)
        self.sure_infected = set()  # tests whose own evaluations of an edited comparison change (infect.py)
        self.added = []  # (test id, first later test) of test methods that the patch adds
        self.vanished = set()  # tests of classes that the patch only renames: their ids are not in the full run
        self.watch = []  # (module, descriptor) of shared containers to watch (shared_write_watch)
        self.watched = None  # the statefp keys of the containers that the child watched (None: not reported)
        self.leaks = []  # (test, statefp key, changed entries or None): a watched container changed
        self.cone = None  # import-time commit: the modules to import again (cone.py), or None
        # path -> (abs path, [(old code, new code)], modules whose import runs the changed function):
        # a file whose only change is the body of one function that also runs at import (swap_candidate)
        self.swap = {}
        self.swap_work = []  # cone child: (abs path, pairs) of the files that it handles by code replacement
        self.klass = {}  # path -> (module, class): every change of the file lies in the body of this top-level class
        self.class_work = []  # cone child: (module, class, patched text, abs path, import-time reads of the class)
        self.guard = {}  # test -> CPU seconds after which its run is stopped and the test selected
        self.prep = None  # (allowed, skip, cap, cost) of prepare_plan, for a second child
        self.suspects = set()  # tests without a verdict that count like failing ones (stopped runs)
        self.body = {}  # test -> CPU seconds of the test alone, without class set-up (for logs)
        self.econ = set()  # tests held because their runs cost more than their selection (marginal_holds)
        self.settings = False  # a line that the patch adds names an interpreter-wide setting (SETTINGS_WORDS)
        self.stage = None  # the stage check that the child may run for a held long test (stage_plan)
        self.stage_info = None  # what that check found: {"n": test, "verdict": True (fails) / False (passes) / None, ...}


CODING_RE = re.compile(r"coding[:=]\s*[-\w.]+")


def python_text(text):
    """The text that Python compiles from a source file whose bytes are
    text.encode("utf-8", "surrogateescape") (how the patch was read): the coding declaration
    of line 1 or 2 decides the decoding (PEP 263), and a UTF-8 byte order mark is dropped.
    None when the bytes do not decode: the import then raises a SyntaxError."""
    i = text.find("\n")
    j = text.find("\n", i + 1) if i >= 0 else -1
    if text.isascii() and "coding" not in (text if j < 0 else text[:j]):
        return text
    import io
    import tokenize

    data = text.encode("utf-8", "surrogateescape")
    try:
        enc, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        return data.decode(enc)
    except (SyntaxError, LookupError, UnicodeDecodeError, ValueError):
        return None


def _line_groups(old_text, new_text):
    """The changed line groups between two texts, in the form of engine.FilePatch.groups."""
    import difflib

    a, b = old_text.split("\n"), new_text.split("\n")
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append([i1 + 1, i2 - i1, j1 + 1, j2 - j1, a[i1:i2], b[j1:j2]])
    return out


def _blank_or_comment(lines):
    return all((not s.strip()) or s.lstrip().startswith("#") for s in lines)


def _indent(s):
    return len(s) - len(s.lstrip())


def _lead(s):
    return s[: len(s) - len(s.lstrip())]


def _indentation_changed(fp):
    """True when an edit can change how the file's blocks nest: a changed line whose leading
    whitespace differs from the line it replaces, or added lines (other than blank or
    comment lines) without a matching removed line."""
    for g in fp.groups:
        removed, added = g[4], g[5]
        if len(removed) != len(added):
            if not _blank_or_comment(added):
                return True
            continue
        for x, y in zip(removed, added):
            if y.strip() and not y.lstrip().startswith("#") and _lead(x) != _lead(y):
                return True
    return False


def _file_compiles(text, filename, flags):
    try:
        compile(text, filename, "exec", flags=flags, dont_inherit=True)
    except (SyntaxError, ValueError):
        return False
    return True


_KIND_FLAGS = 0x20 | 0x80 | 0x100 | 0x200  # generator, coroutine, iterable coroutine, async generator


def _kind_changed(x, y):
    """True when a function (or a function nested in it) turns into or out of a generator or
    a coroutine."""
    if (x.co_flags ^ y.co_flags) & _KIND_FLAGS:
        return True
    cx = [c for c in x.co_consts if hasattr(c, "co_code")]
    cy = [c for c in y.co_consts if hasattr(c, "co_code")]
    if len(cx) != len(cy):
        return True
    return any(_kind_changed(a, b) for a, b in zip(cx, cy))


def _type_checking_spans(text):
    """Line ranges of the bodies of "if TYPE_CHECKING:" / "if t.TYPE_CHECKING:" blocks at module
    or class level, with TYPE_CHECKING taken from the typing module of the same file. The flag
    is False whenever the code runs (only a static type checker reads it as True), so those
    lines never run. Returns None when the text does not compile."""
    import ast

    try:
        tree = ast.parse(text)
        compile(tree, "<check>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError):
        return None
    typing_names, flag_names = set(), set()
    for s in tree.body:
        if isinstance(s, ast.Import):
            typing_names.update(a.asname or a.name for a in s.names if a.name == "typing")
        elif isinstance(s, ast.ImportFrom) and s.module == "typing" and not s.level:
            flag_names.update(a.asname or a.name for a in s.names if a.name == "TYPE_CHECKING")
    if not typing_names and not flag_names:
        return []
    # The names must keep that meaning: no statement at module or class level binds them again,
    # no function declares them global, and nothing assigns an attribute TYPE_CHECKING.
    # (A name bound inside a function or a comprehension is local to it.)
    names = typing_names | flag_names
    for n in _walk(tree):
        if isinstance(n, ast.Global) and names & set(n.names):
            return []
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del)) and n.attr == "TYPE_CHECKING":
            return []
    stack = list(tree.body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)) and n.id in names:
            return []
        if isinstance(n, ast.Import) and any(a.name != "typing" and (a.asname or a.name.split(".")[0]) in names for a in n.names):
            return []
        if isinstance(n, ast.ImportFrom) and any(
            (a.asname or a.name) in names and not (n.module == "typing" and a.name == "TYPE_CHECKING") for a in n.names
        ):
            return []
        stack.extend(ast.iter_child_nodes(n))

    def is_flag(e):
        if isinstance(e, ast.Name):
            return e.id in flag_names
        return isinstance(e, ast.Attribute) and e.attr == "TYPE_CHECKING" and isinstance(e.value, ast.Name) and e.value.id in typing_names

    spans = []

    def scan(body):
        for s in body:
            if isinstance(s, ast.If) and is_flag(s.test):
                spans.append((s.body[0].lineno, s.body[-1].end_lineno))
                scan(s.orelse)  # the else branch does run
            elif isinstance(s, ast.ClassDef):
                scan(s.body)

    scan(tree.body)
    return spans


_TC_SPANS = None


def _old_type_checking_spans(rel, old_text):
    """_type_checking_spans of a pinned file, from the map that build_tc_spans.py makes from the
    pinned tree (maps/tc_spans.bin), when the text is the one of the map; else computed."""
    global _TC_SPANS
    if _TC_SPANS is None:
        _TC_SPANS = {}
        for d in (MAPS, HERE):
            try:
                with open(os.path.join(d, "tc_spans.bin"), "rb") as f:
                    _TC_SPANS = marshal.loads(f.read())
                break
            except (OSError, ValueError, EOFError, TypeError):
                pass
    rec = _TC_SPANS.get(rel)
    if rec is not None and rec[0] == len(old_text) and rec[1] == zlib.crc32(old_text.encode("utf-8", "surrogateescape")):
        return rec[2]
    return _type_checking_spans(old_text)


def type_checking_groups(fp, old_text):
    """The hunks of a patch to a .py file whose removed lines (in the old text) and added lines
    (in the new text) all lie inside the body of an "if TYPE_CHECKING:" block (see
    _type_checking_spans). Such a hunk changes nothing that runs. Tests that read the source
    file as data are candidates anyway (the "opened" map, see make_plan)."""
    if "TYPE_CHECKING" not in old_text or "TYPE_CHECKING" not in fp.new_text:
        return set()
    old_spans = _old_type_checking_spans(fp.old_path, old_text)
    if not old_spans:
        return set()

    def inside(a, n, spans):
        return all(any(s <= ln <= e for s, e in spans) for ln in range(a, a + n))

    # the old text decides first (from the map): the new text is parsed only for the hunks
    # whose removed lines all lie inside a block
    # (a hunk of comments or blank lines needs no answer here: plan_python_file finds that the
    # code of its statement is the same, without a parse of the whole new text)
    maybe = [g for g in enumerate(fp.groups) if g[1][1] + g[1][3] and inside(g[1][0], g[1][1], old_spans)
             and not (_blank_or_comment(g[1][4]) and _blank_or_comment(g[1][5]))]
    if not maybe:
        return set()
    new_spans = _type_checking_spans(fp.new_text)
    if not new_spans:
        return set()  # also when the new text does not compile: the import fails then
    return {gi for gi, (o, nr, nl, na, removed, added) in maybe if inside(nl, na, new_spans)}



_CHAIN_CACHE = {}


def _if_chain_heads(old_text, key):
    """{line of an if or elif header: sorted header lines of its whole if/elif chain} for the
    pinned text, and {line of a statement: line of the statement right before it in the same
    block}."""
    got = _CHAIN_CACHE.get(key)
    if got is not None:
        return got
    import ast

    chains, prev = {}, {}
    try:
        tree = ast.parse(old_text)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            for field in ("body", "orelse", "finalbody", "handlers"):
                block = getattr(node, field, None)
                if isinstance(block, list):
                    for a, b in zip(block, block[1:]):
                        if isinstance(a, ast.stmt) and isinstance(b, ast.stmt):
                            prev[b.lineno] = a
            if isinstance(node, ast.If) and node.lineno not in chains:
                heads = [node.lineno]
                cur = node
                while len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If) and cur.orelse[0].col_offset == node.col_offset:
                    cur = cur.orelse[0]
                    heads.append(cur.lineno)
                for h in heads:
                    chains[h] = heads
    got = _CHAIN_CACHE[key] = (chains, prev)
    return got


def _chain_lines(old_text, key, groups):
    """Lines to add to the changed lines when an edit turns an elif into an if or an if into
    an elif. Such an edit changes which branches run for every evaluation of the chain, also
    for the runs that took an earlier branch and so never ran the edited line (the pinned
    coverage of that line misses them): the header lines of the whole chain count."""
    import ast

    out = []
    for g in groups:
        o, nr, nl, na, removed, added = g
        for i, (x, y) in enumerate(zip(removed, added)):
            kx = x.lstrip().split(" ", 1)[0].rstrip(":")
            ky = y.lstrip().split(" ", 1)[0].rstrip(":")
            if kx == ky or not ({kx, ky} & {"elif", "else"}):
                continue
            chains, prev = _if_chain_heads(old_text, key)
            ln = o + i
            if ln in chains:
                out.extend(chains[ln])
            p = prev.get(ln)
            if isinstance(p, ast.If) and p.lineno in chains:
                out.extend(chains[p.lineno])
    return out



def _header_changed(old_lines, s, e, groups):
    """True when an edit touches the decorator or def lines of the unit at lines s..e."""
    end = None
    seen_def = False
    for ln in range(s, e + 1):
        text = old_lines[ln - 1].split("#", 1)[0].rstrip()
        if not seen_def and text.lstrip().startswith(("def ", "async def ")):
            seen_def = True
        if seen_def and text.endswith(":"):
            end = ln
            break
    if end is None:
        return False
    return any(s <= o + i <= end for o, nr, *_ in groups for i in range(max(nr, 1)))


def plan_python_file(plan: Plan, fp, maps: Maps, old_text: str):
    """Classifies the edit of one known .py file. Returns True when a fresh import is needed."""
    rel = fp.old_path
    rec = maps.file(rel)
    if rec is None:
        return True
    if _indentation_changed(fp) and not _file_compiles(fp.new_text, os.path.join(ROOT, rel), rec[1]):
        # The units below compile one at a time, each inside one-line class wrappers. An
        # edit that only moves the indentation of a line (a def line, for instance) can
        # compile there and look like a layout-only edit, while the whole file no longer
        # compiles (IndentationError, TabError). Then importing the module fails, so the
        # child must import the patched tree.
        plan.note.append("file does not compile")
        return True
    inert = type_checking_groups(fp, old_text)
    if inert:
        plan.note.append("type checking only: %d of %d hunks" % (len(inert), len(fp.groups)))
    units, flags, inames, stmt_set, stmt_pex, unit_set, unit_pex, importers, file_set, span_a, span_b = rec[:11]
    touched = plan.lines.setdefault(rel, set())
    old_lines = old_text.split("\n")
    new_lines = fp.new_text.split("\n")
    full = os.path.join(ROOT, rel)
    hit_units = {}
    for gi, g in enumerate(fp.groups):
        if gi in inert:
            continue  # inside "if TYPE_CHECKING:": never runs
        o, nr, nl, na, removed, added = g
        hit = []
        for ui, u in enumerate(units):
            s, e = u[0], u[1]
            if nr > 0:
                if s <= o + nr - 1 and o <= e:
                    hit.append(ui)
            elif s < o <= e:
                hit.append(ui)
        if not hit:
            if _blank_or_comment(added) and _blank_or_comment(removed):
                if nl > 2 or not any(CODING_RE.search(a) for a in added + removed):
                    continue
            return True
        for ui in hit:
            hit_units.setdefault(ui, []).append(g)
    for ui, groups in hit_units.items():
        s, e, class_path, kind = units[ui]
        if kind == 1:
            return True
        delta_before = 0
        delta_in = 0
        for g in fp.groups:
            if g[0] + g[1] - 1 < s and g[0] < s:
                delta_before += g[3] - g[1]
            elif g[0] <= e:
                delta_in += g[3] - g[1]
        try:
            a = engine.compile_unit(old_lines, s, e, class_path, full, flags, inames)
            b = engine.compile_unit(
                new_lines, s + delta_before, e + delta_before + delta_in, class_path, full, flags, inames
            )
        except (SyntaxError, ValueError):
            return True
        if a is None or b is None:
            return True
        pairs = []
        if not engine.deftime_equal(a, b, pairs):
            pairs = []
            if patchrules.tests_read_docs(maps) or not patchrules.equal_but_doc(a, b, pairs):
                return True
            plan.note.append("docstring only")
        if all(engine.func_code_equal(x, y) for x, y in pairs):
            continue  # comments or layout only
        if not class_path and any(line.startswith("def __getattr__(") for line in old_lines[s - 1:e]):
            # A module __getattr__ answers the lookups of names that the module does not hold
            # yet (lazy imports), and the result is stored. So the pinned run shows only the
            # first lookup of each name, but a changed __getattr__ can fail every lookup:
            # every test that loads the module is a candidate, and no reduced run applies.
            plan.cands.update(maps.tests(importers))
            plan.gone.add("__getattr__")
            plan.note.append("module __getattr__")
        simple = True
        for g in groups:
            o, nr, nl, na, removed, added = g
            if nr != na or any(_indent(x) != _indent(y) for x, y in zip(removed, added)):
                simple = False
            elif any(x.lstrip().startswith(("nonlocal ", "global ")) for x in removed + added):
                # a scope declaration has no line of its own in the coverage map, and it
                # changes what every use of the name in the function means
                simple = False
        if simple and rel.startswith("tests/") and _header_changed(old_lines, s, e, groups):
            # a test method's def line (its parameters): unittest calls the method with self
            # only, so the call itself fails; no line of the body records that, and the def
            # line runs only when the test module is imported: every test of the unit is a
            # candidate and runs whole
            simple = False
            plan.note.append("test def header changed")
        if simple and any(_kind_changed(x, y) for x, y in pairs):
            # generator <-> plain function: every call behaves differently, not only the
            # runs that reach the changed line, so the tests that run the unit are candidates
            simple = False
            plan.note.append("code kind changed")
        if simple:
            hit_lines = [ln for g in groups for ln in range(g[0], g[0] + g[1])]
            groups_lines = set(hit_lines)
            for ln in _chain_lines(old_text, rel, groups):
                # elif <-> if: the runs that took an earlier branch of the chain change too
                if ln not in hit_lines:
                    hit_lines.append(ln)
            for g in groups:
                for i in range(g[1]):
                    # an edit beside a lambda on its line: the lines of the statement around it
                    around = patchrules.statement_around_lambda(old_lines, s, e, g[0] + i, g[4][i], g[5][i])
                    if around:
                        hit_lines.extend(l2 for l2 in range(around[0], around[1] + 1) if l2 not in hit_lines)
            # The tests of the exact line map (a few long tests, traced in every process they
            # use) are candidates only when they run a line that holds a changed instruction.
            # A changed line inside a longer statement does not need to run: the statement's
            # first line can hold the instruction (an and/or jump, a comparison). When the
            # instructions do not pair up, the statement lines decide as for the other tests.
            # This holds for every line looked up below, also for the lines that the lambda
            # rule above adds: a test that runs none of the changed instructions (in the
            # function or in a lambda inside it) runs as it does on the pinned code.
            tight = maps.tight_lines()
            dl = False  # the lines that hold changed instructions; None: cannot be told
            for ln in hit_lines:
                if stmt_pex[ln]:
                    # the function also runs at import: its body change is known here
                    if COLL_RUN:
                        for l2 in hit_lines:
                            if stmt_set[l2]:
                                plan.coll_extra.update(maps.tests(stmt_set[l2]))
                    plan.fpairs.extend(pairs)
                    sl = plan.stmt_lines.setdefault(rel, set())
                    for l2 in hit_lines:
                        sl.update(range(span_a[l2], span_b[l2] + 1) if span_a[l2] else (l2,))
                        if l2 not in groups_lines and stmt_set[l2]:
                            # lines added by a rule (if/elif chain, lambda statement): the
                            # import-time candidates below look only at the changed lines
                            plan.cands.update(maps.tests(stmt_set[l2]))
                    return True
                found = maps.tests(stmt_set[ln])
                if tight and dl is not None and not tight.keys().isdisjoint(found):
                    if dl is False:
                        dl = set()
                        for x, y in pairs:
                            d = engine.changed_lines(x, y)
                            if d is None:
                                dl = None
                                break
                            dl |= d
                    if dl:
                        kept = [n for n in found if n not in tight or not tight[n].get(rel, frozenset()).isdisjoint(dl)]
                        if len(kept) != len(found):
                            plan.note.append("exact lines: %s not run" % sorted(set(found) - set(kept)))
                            found = kept
                plan.cands.update(found)
                touched.update(range(span_a[ln], span_b[ln] + 1) if span_a[ln] else (ln,))
        else:
            if unit_pex[ui]:
                if COLL_RUN:
                    # the import-time path decides; when the import breaks (choose()), the
                    # tests that call the changed unit at run time are candidates too
                    plan.coll_extra.update(maps.tests(unit_set[ui]))
                return True
            plan.cands.update(maps.tests(unit_set[ui]))
            touched.update(range(s, e + 1))
        plan.work.append((full, pairs))
    return False


_TEST_READERS = None


def test_data_candidates(plan: Plan, fp, maps: Maps):
    """An edit that lies inside a module-level data literal of a test helper module (for
    instance a column of TPCDS_SCHEMA in tests/helpers.py). Only the tests whose code refers
    to the name that the literal defines can see the change (maps/test_readers.json, built by
    build_test_readers.py from the pinned test sources). Those that refer to it from their own
    method read the data directly: they are selected when they are too costly to run (for
    instance the TPC-DS run), because no cheaper test reads the same values."""
    global _TEST_READERS
    if _TEST_READERS is None:
        try:
            with open(os.path.join(maps.path, "test_readers.json")) as f:
                _TEST_READERS = json.load(f)
        except (OSError, ValueError):
            _TEST_READERS = {}
    info = _TEST_READERS.get(fp.old_path)
    if not info or fp.status != "modify":
        return False
    names = set()
    for o, nr, nl, na, removed, added in fp.groups:
        if nr != na:
            return False  # lines added or removed: the statements may have moved
        for ln in range(o, o + nr):
            # strictly inside the literal: the first and last lines hold the name and brackets
            hit = [s for s in info["stmts"] if s[0] < ln < s[1] and s[3]]
            if not hit:
                return False
            names.update(hit[0][2])
    cands = set()
    for name in names:
        cands.update(info["readers"].get(name, ()))
    if not cands:
        return False
    plan.cands.update(cands)
    for name in names:
        plan.must.update(info["direct"].get(name, ()))
    plan.lines["?"] = None  # no reduced runs for this commit
    plan.note.append("test data readers of %s" % sorted(names))
    return True



_ENUM_LINES = {}


def _enum_body_lines(rel):
    """Lines of the bodies of Enum classes in a pinned module. Enum members are read through
    the class's own tables (__members__, value lookups, iteration), which the def-use map of
    names does not see, and import-time code (a metaclass, a registry) can consume them."""
    out = _ENUM_LINES.get(rel)
    if out is not None:
        return out
    import ast
    import enum

    out = set()
    try:
        tree = ast.parse(patchrules.source(rel))
    except SyntaxError:
        _ENUM_LINES[rel] = out
        return out
    mod = sys.modules.get(rel[:-3].replace("/", ".").removesuffix(".__init__"))

    def visit(body, prefix):
        for node in body:
            if isinstance(node, ast.ClassDef):
                qual = prefix + [node.name]
                is_enum = None
                if mod is not None:
                    obj = mod
                    for part in qual:
                        obj = getattr(obj, part, None)
                        if obj is None:
                            break
                    if obj is not None:
                        is_enum = isinstance(obj, enum.EnumMeta)
                if is_enum is None:
                    is_enum = any("Enum" in ast.unparse(b) or ast.unparse(b).endswith("AutoName") for b in node.bases)
                if is_enum:
                    out.update(range(node.lineno, node.end_lineno + 1))
                visit(node.body, qual)

    visit(tree.body, [])
    _ENUM_LINES[rel] = out
    return out


_SLOTS_LINES = {}


_SLOTS_RE = re.compile(r"^[ \t]+__slots__[ \t]*(?::[^=\n]*)?=(?!=)", re.M)


def _slots_lines(rel):
    """The lines of the class-level __slots__ statements of a pinned file. An edit there
    changes which attributes the instances of the class can hold: every user of the class
    (any test that loads or runs the file) is a candidate, with whole runs (refine_with_diff).
    (A text scan, not a parse: ast.parse of parser.py costs about 0.14 s of CPU.)"""
    out = _SLOTS_LINES.get(rel)
    if out is not None:
        return out
    out = set()
    src = patchrules.source(rel)
    if "__slots__" in src:
        for m in _SLOTS_RE.finditer(src):
            line = src.count("\n", 0, m.start()) + 1
            first = line
            depth = 0
            i, end = m.end(), len(src)
            while i < end:
                ch = src[i]
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                elif ch == "\n":
                    if depth <= 0 and src[i - 1] != "\\":
                        break
                    line += 1
                i += 1
            out.update(range(first, line + 1))
    _SLOTS_LINES[rel] = out
    return out


def _touches_slots(fp):
    if not SLOTS_RULE:
        return False
    rel = fp.old_path
    if not rel or not rel.endswith(".py"):
        return False
    sl = _slots_lines(rel)
    if sl:
        for o, nr, nl, na, removed, added in fp.groups:
            lines = range(o, o + nr) if nr else (o - 1, o)
            if any(ln in sl for ln in lines):
                return True
    # an added __slots__ statement (or one edited into existence)
    for o, nr, nl, na, removed, added in fp.groups:
        if any("__slots__" in x for x in list(added) + list(removed)):
            return True
    return False


def add_import_time_candidates(plan: Plan, fp, rec, maps: Maps):
    """Candidates for an edit that changes what the module does at import. The map gives,
    for a line of module-level or class-level code, the tests that use what the line defines.
    A line without an entry falls back to every test that loads or runs the file."""
    if test_data_candidates(plan, fp, maps):
        return
    stmt_set, def_set, def_known = rec[3], rec[11], rec[12]
    readers = rec[13] if len(rec) > 13 else {}
    paths = maps.paths()
    n = len(def_set) - 2
    complete = True
    reader_lines = {}
    units = rec[0]
    # a change in a function header (defaults, decorators) reaches the tests that run the function
    for o, nr, nl, na, removed, added in fp.groups:
        for s, e, _, kind in units:
            if kind == 0 and s <= o <= e and e > s:
                plan.def_units.setdefault(fp.old_path, set()).update(range(s, e + 1))
    enum_lines = _enum_body_lines(fp.old_path) if fp.old_path.endswith(".py") else ()
    if _touches_slots(fp):
        complete = False
        plan.note.append("__slots__ edit")
    for o, nr, nl, na, removed, added in fp.groups:
        lines = range(o, o + nr) if nr else (o - 1, o)
        simple = nr == na and all(_indent(x) == _indent(y) for x, y in zip(removed, added))
        if not simple and not (_blank_or_comment(added) and _blank_or_comment(removed)):
            complete = False
        elif enum_lines and not (_blank_or_comment(added) and _blank_or_comment(removed)) and any(ln in enum_lines for ln in lines):
            # an Enum member: its readers go through the class tables (see _enum_body_lines)
            complete = False
        for ln in lines:
            if not 0 < ln <= n:
                continue
            if def_known[ln]:
                plan.cands.update(maps.tests(def_set[ln]))
                rs = readers.get(ln) if simple else None
                if rs is None:
                    reader_lines = None
                elif reader_lines is not None:
                    for rfid, rline in rs:
                        reader_lines.setdefault(paths[rfid], set()).add(rline)
            elif stmt_set[ln]:
                reader_lines = None
                plan.cands.update(maps.tests(stmt_set[ln]))
            elif nr:
                complete = False
    if not complete:
        plan.cands.update(maps.tests(rec[7]))
        plan.cands.update(maps.tests(rec[8]))
        plan.note.append("import-time fallback")
        reader_lines = None
    if reader_lines:
        # a class constant that only functions read: a test part is affected only when it
        # runs one of the reading lines, so the reduced runs apply
        for p, lines in reader_lines.items():
            plan.lines.setdefault(p, set()).update(lines)
        plan.note.append("reader lines")
    else:
        plan.lines["?"] = None  # no reduced runs for this commit


INFECT = os.environ.get("BR_NO_INFECT") != "1"  # operand outcome map, see infect.py
INFECT2 = os.environ.get("BR_NO_INFECT2") != "1"  # its phase 2 (and/or trees, literal operands)
INFECT_FRESH = os.environ.get("BR_NO_INFECT_FRESH") != "1"  # the same for functions that also run at import
INFECT_UNSEEN = os.environ.get("BR_NO_INFECT_UNSEEN") != "1"  # tests without a record at a site that they run
INFECT_STMT = os.environ.get("BR_NO_INFECT_STMT") != "1"  # ... unless the maps show that they ran its statement
INFECT_HIDDEN = os.environ.get("BR_NO_INFECT_HIDDEN") != "1"  # ... replaced by the recorded cache-hidden tests of the site


def plan_python_file_infect(plan: Plan, fp, maps: Maps, old_text: str, used: list):
    """plan_python_file, narrowed by the operand outcome map (infect.py, maps/infect.bin: the
    operand relations that each test saw at each comparison when the pinned suite ran).
    When the whole edit of the file is one operator change at a mapped comparison: with no
    evaluation in any test or phase whose result changes, the file adds nothing; otherwise
    its candidates shrink to the tests with such an evaluation. Any other edit, or a changed
    evaluation outside single tests (import time, collection), takes the normal path."""
    rec = maps.file(fp.old_path)
    res = None
    if rec is not None and _indentation_changed(fp) and not _file_compiles(
        fp.new_text, os.path.join(ROOT, fp.old_path), rec[1]
    ):
        # the patched file does not compile as a whole: no map applies (see plan_python_file)
        return plan_python_file(plan, fp, maps, old_text)
    if rec is not None:
        import infect

        old_lines, new_lines = old_text.split("\n"), fp.new_text.split("\n")
        res = infect.infected_tests(
            fp.old_path, old_lines, new_lines, fp.groups, rec[0], os.path.join(maps.path, "infect.bin"),
        )
        if res is None and INFECT2:
            res = infect.infected_tests2(
                fp.old_path, old_lines, new_lines, fp.groups, rec[0], os.path.join(maps.path, "infect2.bin"),
            )
    if res is None or res.global_phase:
        if res is not None and res.sure:
            # The comparison also runs outside single tests (at import), so the map does not
            # narrow the candidates. A test whose own recorded evaluations are known to give
            # the other result still runs differently: held, it is selected (see finish).
            plan.sure_infected.update(res.sure)
        return plan_python_file(plan, fp, maps, old_text)
    # The operand map comes from runs of the whole suite in one process. Code that runs once
    # per process and whose result sqlglot then keeps in a cache (generator._build_dispatch
    # fills _DISPATCH_CACHE for the first user of each generator class) shows its evaluations
    # only in the first test that needed it, while the patched result reaches every later
    # reader of the cache. Such a reader is not covered by the operand map: it counts as a
    # test with a changed evaluation. Only when some recorded evaluation does change: the
    # work that the first test did is the work that each other test repeats in a process of
    # its own, so with no changed evaluation in any test or phase no test can differ.
    #
    # Which tests a cache hides at a site is recorded (maps/infect_hidden.bin, see
    # infect.hidden_tests and build_hidden.py): the tests that get to the site's statement
    # (or start the site's expression) more often when they run alone in a fresh process
    # than in their own phases of the whole-suite runs. Every other test does at the site
    # in the whole-suite runs all that it does alone, and each of those evaluations has a
    # record in the operand map.
    eff = res.tests
    hidden = None
    if res.tests and INFECT_UNSEEN and INFECT_HIDDEN:
        hidden = infect.hidden_tests(os.path.join(maps.path, "infect_hidden.bin"), fp.old_path, res.site[1:5])
        if hidden:
            eff = res.tests | hidden
            plan.note.append("infect: %d tests behind a cache at the site" % len(hidden - res.tests))
    if res.tests and INFECT_UNSEEN and hidden is None:
        # No reach record for this site: the wider rule. The line coverage map runs every
        # test in a process of its own; a test that runs a line of the site there but has no
        # recorded evaluation of the site counts as hidden.
        #
        # The line map is kept per statement, so a test also "runs the site" when its statement
        # stops before the comparison (an "and" whose first operand is false). A test that the
        # operand map shows running the statement in the whole-suite runs (it has a record at
        # another site of the statement or of a later statement of the block, see
        # infect.statement_runners) did not skip the code there: it ran the statement and did not
        # evaluate the comparison, so it is covered by the map.
        stmt_set = rec[3]
        unseen = set()
        for ln in range(res.site[1], res.site[3] + 1):
            if 0 <= ln < len(stmt_set) and stmt_set[ln]:
                unseen.update(maps.tests(stmt_set[ln]))
        if res.seen is not None:
            unseen.difference_update(res.seen)
        if unseen and INFECT_STMT:
            ran = infect.statement_runners(
                fp.old_path, old_lines, rec[0], tuple(res.site[1:5]),
                os.path.join(maps.path, "infect.bin"), os.path.join(maps.path, "infect2.bin"),
            )
            if ran:
                plan.note.append("infect: %d of %d tests without a record ran the statement" % (len(unseen & ran), len(unseen)))
                unseen.difference_update(ran)
        if unseen:
            eff = res.tests | unseen
            plan.note.append("infect: %d tests run the site without a record" % len(unseen))
    used.append(fp.old_path)
    if not eff:
        plan.note.append("infect: no evaluation changes")
        plan.infect_files.append((eff, "hot"))
        return False
    before = set(plan.cands)
    n_pairs = len(plan.fpairs)
    fresh = plan_python_file(plan, fp, maps, old_text)
    if not fresh:
        added = plan.cands - before
        plan.cands = before | (added & eff)
        plan.note.append("infect: %d of %d candidates" % (len(added & eff), len(added)))
        plan.infect_files.append((eff, "hot"))
    elif plan.cands == before and len(plan.fpairs) > n_pairs:
        # The function also runs at import, so the child imports the patched tree. No
        # evaluation at import time changes its result (global_phase is false). When the
        # child then finds the import state equal to the pinned one, only the tests with a
        # changed evaluation can differ, as above (see make_plan and refine_with_diff).
        plan.infect_files.append((eff, "import"))
    else:
        plan.infect_files.append((eff, "other"))
    return fresh


_MUTATORS = frozenset(("pop", "popitem", "clear", "update", "setdefault", "append", "extend", "insert",
                       "remove", "add", "discard", "__setitem__", "__delitem__"))


def _writes_shared_state(fn):
    """True when a function body assigns to, deletes or mutates an attribute or item of an
    object that it did not create itself (an imported module, a class, a global): that state
    outlives the test. Names bound by assignment, loops and with-statements in the body, and
    the arguments (self, cls), count as the function's own objects."""
    import ast

    local = set()
    for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        local.add(a.arg)
    nodes = _walk(fn)
    for n in nodes:
        targets = []
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign, ast.For, ast.NamedExpr)):
            targets = [n.target]
        elif isinstance(n, ast.With):
            targets = [i.optional_vars for i in n.items]
        for t in targets:
            if isinstance(t, ast.Name):
                local.add(t.id)

    def root(e):
        depth = 0
        while isinstance(e, (ast.Attribute, ast.Subscript)):
            e = e.value
            depth += 1
        return (e.id if isinstance(e, ast.Name) else None), depth

    for n in nodes:
        targets = []
        if isinstance(n, (ast.Assign, ast.Delete)):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            targets = [n.target]
        for t in targets:
            if isinstance(t, (ast.Attribute, ast.Subscript)):
                r, _ = root(t)
                if r is not None and r not in local:
                    return True
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id in ("setattr", "delattr"):
                return True
            if isinstance(f, ast.Attribute) and f.attr in _MUTATORS:
                r, depth = root(f.value)
                if r is not None and r not in local and depth >= 1:
                    return True
    return False


_ADD_METHODS = frozenset(("append", "extend", "insert", "add", "update", "setdefault", "__setitem__", "__ior__"))
_REMOVE_METHODS = frozenset(("pop", "popitem", "clear", "remove", "discard", "__delitem__"))


def _container_writes(fn):
    """The writes of a function body into objects that outlive the call: a container held by
    a class attribute reached through the first argument (self.X[k] = v, self.X.pop(k), cls.X
    = v), or by a module-level name (G[k] = v, G.add(x), exp.T.update(...), global G; G = v).
    Returns {descriptor: set of "add" / "remove" / "rebind"}; descriptor ("A", X) is the class
    attribute X of whatever class the object has, ("G", name, attr, ...) a module-level name
    and the attributes below it."""
    import ast

    args = fn.args.posonlyargs + fn.args.args
    first = args[0].arg if args else None
    declared = set()
    local = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    for a in (fn.args.vararg, fn.args.kwarg):
        if a is not None:
            local.add(a.arg)
    nodes = _walk(fn)
    for n in nodes:
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            declared.update(n.names)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            local.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not fn:
            local.add(n.name)
        elif isinstance(n, ast.arg):
            local.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            local.update((a.asname or a.name).split(".")[0] for a in n.names)
    local -= declared

    def desc(e):
        chain = []
        while isinstance(e, ast.Attribute):
            chain.append(e.attr)
            e = e.value
        if not isinstance(e, ast.Name):
            return None
        chain.reverse()
        if e.id == first and first is not None and first not in declared:
            return ("A", chain[0]) if len(chain) == 1 else None
        if e.id in local:
            return None
        return ("G", e.id) + tuple(chain)

    out = {}

    def note(d, kind):
        if d is not None:
            out.setdefault(d, set()).add(kind)

    for n in nodes:
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = n.targets if isinstance(n, (ast.Assign, ast.Delete)) else [n.target]
            for t in targets:
                if isinstance(t, (ast.Tuple, ast.List)):
                    continue
                if isinstance(t, ast.Subscript):
                    note(desc(t.value), "remove" if isinstance(n, ast.Delete) else "add")
                elif isinstance(t, ast.Attribute):
                    d = desc(t)
                    if d is not None and d[0] == "A" and isinstance(n, (ast.Assign, ast.AnnAssign)) and first != "cls":
                        continue  # self.X = v: the instance's own attribute
                    note(d, "add" if isinstance(n, ast.AugAssign) else "rebind")
                elif isinstance(t, ast.Name) and t.id in declared:
                    note(("G", t.id), "rebind")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr in _ADD_METHODS:
                note(desc(n.func.value), "add")
            elif n.func.attr in _REMOVE_METHODS:
                note(desc(n.func.value), "remove")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("setattr", "delattr") and n.args:
            note(desc(n.args[0]), "rebind")
    return out


def _unit_function(unit_lines, class_path):
    """The function node of a unit (a method or function, its lines as in the file), parsed
    inside one-line class wrappers as engine.compile_lines does; None when it does not parse."""
    import ast

    head = "".join(" " * k + "class " + name + ":\n" for k, name in enumerate(class_path))
    try:
        body = ast.parse(head + "\n".join(unit_lines) + "\n").body
    except (SyntaxError, ValueError):
        return None
    for _ in class_path:
        body = body[0].body if body and isinstance(body[0], ast.ClassDef) else []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def shared_write_watch(units, old_text, new_text, groups):
    """The shared containers that a commit's edit of one module can leave with other content
    than the pinned code leaves: those written (see _container_writes) by a function that
    holds a changed line, in the new text or in the pinned one. A later test that reads such
    a container can change without running the changed line. Containers that the pinned
    function fills for good (caches, registries) are left out: there the tests that read the
    container also run the function, and the coverage maps see that. Containers that it
    writes and restores within the call (a temporary entry), or does not write at all, keep
    the pinned content between tests: the child watches those (see child_main).
    units: the file's units from the map (start, end, class path, kind; kind 0 = a function).
    Only the functions that hold changed lines are parsed (a whole parser.py costs 0.2 s)."""
    old_lines = old_text.split("\n")
    new_lines = new_text.split("\n")
    hit = set()
    for o, nr, nl, na, removed, added in groups:
        for ui, u in enumerate(units):
            s, e, kind = u[0], u[1], u[3]
            if kind == 0 and ((s <= o + nr - 1 and o <= e) if nr > 0 else s < o <= e):
                hit.add(ui)
    watch = set()
    for ui in sorted(hit):
        s, e, class_path, _ = units[ui]
        delta_before = delta_in = 0
        for g in groups:
            if g[0] + g[1] - 1 < s and g[0] < s:
                delta_before += g[3] - g[1]
            elif g[0] <= e:
                delta_in += g[3] - g[1]
        old_fn = _unit_function(old_lines[s - 1:e], class_path)
        new_fn = _unit_function(new_lines[s - 1 + delta_before:e + delta_before + delta_in], class_path)
        pinned = _container_writes(old_fn) if old_fn is not None else {}
        for writes in (pinned, _container_writes(new_fn) if new_fn is not None else {}):
            for d in writes:
                old = pinned.get(d, set())
                if not old or ({"add", "remove"} <= old and "rebind" not in old):
                    watch.add(d)
    return sorted(watch)


_KNOWN_TESTS = None


# Methods that unittest runs around the tests of a class (or a module): an edit there acts on
# every test of the class and of its subclasses, and the per-test runs of the child do not run
# the class and module ones (setUpClass, tearDownClass, setUpModule, tearDownModule).
_INSTANCE_HOOKS = frozenset(("setUp", "tearDown", "asyncSetUp", "asyncTearDown", "__init__", "run", "__call__", "debug",
                             "doCleanups", "_callSetUp", "_callTearDown", "_callTestMethod", "_callCleanup"))
_CLASS_HOOKS = frozenset(("setUpClass", "tearDownClass", "doClassCleanups", "__init_subclass__"))
_MODULE_HOOKS = frozenset(("setUpModule", "tearDownModule", "load_tests"))


def _hook_effect(fn):
    """True when a hook body can leave state behind for later tests: a write into an object it
    did not create (_writes_shared_state), or a bare call on anything but self, cls, super()
    and its own names (logging.disable(...), sqlglot.something(...))."""
    import ast

    if _writes_shared_state(fn):
        return True
    local = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    for n in _walk(fn):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.With)):
            ts = n.targets if isinstance(n, ast.Assign) else ([n.target] if not isinstance(n, ast.With) else [i.optional_vars for i in n.items])
            for t in ts:
                if isinstance(t, ast.Name):
                    local.add(t.id)
    for n in _walk(fn):
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
            f = n.value.func
            while isinstance(f, (ast.Attribute, ast.Subscript, ast.Call)):
                f = f.func if isinstance(f, ast.Call) else f.value
            if isinstance(f, ast.Name) and (f.id in local or f.id == "super"):
                continue
            if isinstance(f, ast.Name) and f.id in ("print", "len", "isinstance", "hasattr", "getattr"):
                continue
            return True
    return False


RENAMED_CLASSES = os.environ.get("BR_NO_RENAMED_CLASSES") != "1"
# names in a class body that can make its behaviour depend on the class's own name or identity
_NAME_WORDS = re.compile(
    r"__name__|__qualname__|__class__|__module__|_testMethodName|\binspect\b|\bgetattr\b|\bglobals\(|\bvars\(|"
    r"\blocals\(|__dict__|__subclasses__|\bmro\b|__mro__|__init_subclass__|(?<![\w.])(type|id|repr|str)\(\s*(self|cls)\b|"
    r"\bsys\._getframe|\btraceback\b|\bunittest\.main|\bTestLoader\b|\bpytest\b|\brequest\b"
)


def _renamed_classes(old_text, new_text):
    """{new name: old name} for top-level classes that the patch only renames: the old class
    is gone, the new one has the same bases, keywords, decorators and body, no other
    top-level class has that body, and the body names nothing that reads the class's own name
    (_NAME_WORDS)."""
    import ast

    try:
        old_tree, new_tree = ast.parse(old_text or ""), ast.parse(new_text or "")
    except (SyntaxError, ValueError):
        return {}

    def shape(c):
        return ast.dump(ast.Module(body=c.body, type_ignores=[])) + "|" + "|".join(
            ast.dump(x) for x in c.bases + c.keywords + c.decorator_list)

    old_c = {c.name: c for c in old_tree.body if isinstance(c, ast.ClassDef)}
    new_c = {c.name: c for c in new_tree.body if isinstance(c, ast.ClassDef)}
    gone = {n: shape(c) for n, c in old_c.items() if n not in new_c}
    out = {}
    for n, c in new_c.items():
        if n in old_c:
            continue
        s = shape(c)
        match = [o for o, so in gone.items() if so == s]
        if len(match) != 1:
            continue
        body = ast.unparse(c)
        if _NAME_WORDS.search(body):
            continue
        # the rest of the module must not read the old or new name (a registry, a subclass)
        rest = ast.unparse(ast.Module(body=[x for x in new_tree.body if x is not c], type_ignores=[]))
        if re.search(r"\b(%s|%s)\b" % (re.escape(n), re.escape(match[0])), rest):
            continue
        if not (n.startswith("Test") and match[0].startswith("Test")) and not _is_testcase(new_tree, c, 0):
            # a plain class that pytest collects by its name: a new name can stop its runs
            continue
        out[n] = match[0]
    return out


DELETED_TESTS = os.environ.get("BR_NO_DELETED_TESTS") != "1"


def _deleted_tests(plan, rel, old_text, new_text, old, new, renamed, test_ids):
    """Test methods that the patch only deletes: the module is the pinned one without them (the
    same tree otherwise), the class stays, nothing in the module names them any more, and their
    bodies leave no state behind for later tests (_hook_effect). Their ids are not in the full
    run, so they cannot fail, and the other tests run as before: plan.vanished."""
    import ast

    classes = {c for c, _ in new}
    gone = [(c, n) for (c, n) in old if (c, n) not in new and c in classes and c not in renamed.values()]
    if not gone:
        return
    try:
        old_tree, new_tree = ast.parse(old_text), ast.parse(new_text)
    except (SyntaxError, ValueError):
        return
    drop = set(gone)
    for cdef in old_tree.body:
        if isinstance(cdef, ast.ClassDef):
            cdef.body = [f for f in cdef.body
                         if not (isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and (cdef.name, f.name) in drop)]
    if ast.dump(old_tree) != ast.dump(new_tree):
        return
    n_before = len(plan.vanished)
    for c, n in gone:
        if _hook_effect(old[(c, n)]) or re.search(r"\b%s\b" % re.escape(n), new_text):
            continue
        tid = "%s::%s::%s" % (rel, c, n)
        plan.vanished.update(k for k, t in enumerate(test_ids) if t == tid or t.startswith(tid + "["))
    if len(plan.vanished) > n_before:
        plan.note.append("deleted tests: %d" % (len(plan.vanished) - n_before))


def _is_testcase(tree, c, depth):
    """True when the class is a unittest.TestCase subclass by its bases (in its module or in
    the pinned test modules it imports): pytest collects those whatever their names are."""
    import ast

    if depth > 5:
        return False
    for b in c.bases:
        if isinstance(b, ast.Attribute) and b.attr in ("TestCase", "IsolatedAsyncioTestCase") and isinstance(b.value, ast.Name) and b.value.id == "unittest":
            return True
        if not isinstance(b, ast.Name):
            continue
        local = [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == b.id]
        if local:
            if _is_testcase(tree, local[-1], depth + 1):
                return True
            continue
        for imp in tree.body:
            if isinstance(imp, ast.ImportFrom) and not imp.level and imp.module:
                for a in imp.names:
                    if (a.asname or a.name) != b.id:
                        continue
                    if imp.module == "unittest" and a.name in ("TestCase", "IsolatedAsyncioTestCase"):
                        return True
                    if imp.module.startswith("tests."):
                        try:
                            with open(os.path.join(ROOT, imp.module.replace(".", "/") + ".py"), encoding="utf-8") as f:
                                t2 = ast.parse(f.read())
                        except (OSError, SyntaxError, ValueError):
                            continue
                        found = [x for x in t2.body if isinstance(x, ast.ClassDef) and x.name == a.name]
                        if found and _is_testcase(t2, found[-1], depth + 1):
                            return True
    return False


def _changed_hooks(plan: Plan, maps: Maps, rel, old_text, new_text, renamed=None):
    """Hooks (_INSTANCE_HOOKS, _CLASS_HOOKS, _MODULE_HOOKS) that the patch adds, removes or
    changes. One that can leave state behind (_hook_effect) selects every test from the first
    test of its class on (from 0 for a class that the pinned run does not list, such as a base
    class of other files' classes); a class or module hook selects the tests of its class or
    file, since the child's runs do not run it; an instance hook makes the tests of its class
    candidates (their runs run it)."""
    import ast

    def hooks(text):
        try:
            tree = ast.parse(text or "")
        except (SyntaxError, ValueError):
            return {}
        out = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _MODULE_HOOKS:
                out[(None, node.name)] = node
            elif isinstance(node, ast.ClassDef):
                for fn in node.body:
                    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and (fn.name in _INSTANCE_HOOKS or fn.name in _CLASS_HOOKS):
                        out[(node.name, fn.name)] = fn
        return out

    new, old = hooks(new_text), hooks(old_text) if old_text is not None else {}
    if renamed:
        # a hook of a class that the patch only renames runs as before
        new = {((renamed.get(c, c)), h): fn for (c, h), fn in new.items()}
    test_ids = maps.meta["test_ids"]
    n_tests = len(test_ids)
    for key in set(new) | set(old):
        a, b = old.get(key), new.get(key)
        if a is not None and b is not None and ast.dump(a) == ast.dump(b):
            continue
        cls_name, name = key
        if cls_name is None:
            mine = [n for n, tid in enumerate(test_ids) if tid.split("::")[0] == rel]
        else:
            mine = sorted((_KNOWN_TESTS.get((rel, cls_name)) or {}).values())
        if cls_name is None and not mine:
            continue  # a module hook of a file without tests (in the pinned list) runs for none
        first = mine[0] if mine else 0
        if any(fn is not None and _hook_effect(fn) for fn in (a, b)):
            plan.blind.update(range(first, n_tests))
            plan.note.append("hook %s.%s can leave state behind: tests from %d on" % (cls_name, name, first))
        elif name in _CLASS_HOOKS or name in _MODULE_HOOKS or not mine:
            plan.blind.update(mine if mine else range(n_tests))
            plan.note.append("class or module hook %s.%s: %d tests" % (cls_name, name, len(mine) if mine else n_tests))
        else:
            plan.cands.update(mine)
            plan.lines["?"] = None
            plan.note.append("hook %s.%s: %d tests of its class" % (cls_name, name, len(mine)))


def added_tests(plan: Plan, maps: Maps, rel, old_text, new_text):
    """Test methods that the patch adds to a test file (in the new text, not in the old one).
    The full run runs them too, at their place in the pytest order (a class's methods run in
    name order). Their own result does not count, but a state they leave behind (a module
    flag, a replaced method) reaches every later test. Such a method is listed in plan.added
    with the first test after it; the child runs it and compares the shared state before and
    after (see run_added_tests). A method whose body writes to an object it did not create
    selects every later test at once (plan.blind): the comparison does not see every kind of
    state (the items of a dict, for instance)."""
    import ast

    global _KNOWN_TESTS
    test_ids = maps.meta["test_ids"]
    if _KNOWN_TESTS is None:
        _KNOWN_TESTS = {}
        for n, tid in enumerate(test_ids):
            parts = tid.split("::")
            if len(parts) >= 3:
                _KNOWN_TESTS.setdefault((parts[0], parts[1]), {}).setdefault(parts[2].split("[")[0], n)

    def methods(text):
        try:
            tree = ast.parse(text or "")
        except (SyntaxError, ValueError):
            return None
        out = {}
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef):
                for fn in cls.body:
                    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("test"):
                        out[(cls.name, fn.name)] = fn
        return out

    new = methods(new_text)
    old = methods(old_text) if old_text is not None else {}
    if new is None or old is None:
        return  # a file that does not parse fails at import: the import check covers it
    renamed = _renamed_classes(old_text, new_text) if RENAMED_CLASSES else {}
    if renamed:
        plan.note.append("renamed classes %s" % sorted(renamed.items())[:3])
        try:
            back = ast.parse(new_text)
            for c in back.body:
                if isinstance(c, ast.ClassDef) and c.name in renamed:
                    c.name = renamed[c.name]
            same = ast.dump(back) == ast.dump(ast.parse(old_text))
        except (SyntaxError, ValueError):
            same = False
        if same:
            # the module is the pinned one but for the class names: it imports as before, the
            # renamed classes run under their new ids, and the old ids are not in the full run
            for old_name in renamed.values():
                prefix = "%s::%s::" % (rel, old_name)
                plan.vanished.update(n for n, tid in enumerate(test_ids) if tid.startswith(prefix))
    _changed_hooks(plan, maps, rel, old_text, new_text, renamed)
    if DELETED_TESTS:
        _deleted_tests(plan, rel, old_text, new_text, old, new, renamed, test_ids)
    for (cls_name, name), fn in new.items():
        if (cls_name, name) in old:
            continue
        if cls_name in renamed and (renamed[cls_name], name) in old:
            # the method of a class that the patch only renames: the full run runs it at the
            # same place as the pinned run does (pytest orders by the file, not by the name)
            continue
        names = _KNOWN_TESTS.get((rel, cls_name))
        if names:
            later = [n for m, n in names.items() if m > name]
            start = min(later) if later else max(names.values()) + 1
        else:
            start = 0  # a class that the pinned run does not know (new, or a base class): anywhere
        plan.added.append(("%s::%s::%s" % (rel, cls_name, name), start))
        if _writes_shared_state(fn):
            plan.blind.update(range(start, len(test_ids)))
            plan.note.append("added test %s writes shared state" % name)


PKG_REFS = os.environ.get("BR_NO_PKG_REFS") != "1"  # new modules of the tests package: only their importers (package_module_refs)


def make_plan(commit, text, maps: Maps, use_infect=INFECT) -> Plan:
    plan = Plan(commit)
    infect_used = []  # files planned with the operand outcome map
    marks = []  # what the plan held before each file
    fps = engine.parse_patch(text)
    plan.settings = any(
        line.startswith("+") and not line.startswith("+++") and any(word in line for word in SETTINGS_WORDS)
        for line in text.splitlines()
    )
    if not fps:
        if text.strip():
            # text that holds no file section this parser knows (another diff format): the
            # tree it makes is unknown, so every test is selected
            plan.note.append("no file section understood")
            plan.mode = "all"
        else:
            plan.note.append("no files in patch")
        return plan
    for fp in fps:
        if not fp.ok:
            plan.note.append("patch not understood: %s (%s)" % (fp.new_path or fp.old_path, fp.why))
            plan.mode = "all"
            return plan
    fresh = False
    n_fresh = 0  # files so far that need a fresh import (fresh: the current file does)
    for fp in fps:
        n_fresh += fresh
        fresh = False
        marks.append((len(plan.cands), len(plan.work), n_fresh, len(plan.blind), len(plan.data_cands), len(plan.gone), len(infect_used)))
        path = fp.new_path if fp.status != "delete" else fp.old_path
        old_text = None
        if fp.status in ("modify", "delete", "rename", "copy"):
            try:
                with open(os.path.join(ROOT, fp.old_path), encoding="utf-8", newline="") as f:
                    old_text = f.read()
            except (OSError, UnicodeDecodeError):
                fp.ok = False
        if fp.status in ("add", "rename", "copy") and os.path.lexists(os.path.join(ROOT, path)):
            fp.ok = False  # git apply does not create a file that exists already
        if fp.ok:
            engine.apply_file_patch(fp, old_text)
        if not fp.ok:
            plan.note.append(f"cannot apply {path}")
            plan.mode = "all"
            return plan
        if fp.status == "copy":
            # the source stays as it is; the target is a new file with the copied (and
            # patched) text, planned like any added file
            fp.status = "add"
            fp.old_path = path
            old_text = None
        if fp.status == "rename":
            plan.files[fp.old_path] = None
        plan.files[path] = fp.new_text
        is_py = path.endswith(".py")
        in_tree = path.startswith(("sqlglot/", "tests/"))
        recoded = False
        new_py = python_text(fp.new_text) if is_py and in_tree and fp.status != "delete" else fp.new_text
        if new_py is not None and new_py != fp.new_text and (old_text is None or python_text(old_text) == old_text):
            # Python decodes the file's bytes by its coding declaration (PEP 263) and drops a
            # byte order mark, so the code it runs is new_py, not the UTF-8 text of the patch:
            # string literals on any line can differ. The planning below compares the
            # pinned text with new_py, line by line; plan.files keeps the patched text, whose
            # bytes the fresh child compiles as Python does (engine._MemLoader).
            fp.groups = _line_groups(old_text or "", new_py)
            fp.new_text = new_py
            plan.note.append("source encoding of %s: planned on the decoded text" % path)
        elif new_py != fp.new_text or (is_py and in_tree and old_text is not None and python_text(old_text) != old_text):
            # The bytes do not decode (an unknown codec, bytes that the declared encoding does
            # not accept): the import raises a SyntaxError. Every test that loads or runs the
            # file is a candidate, with whole runs; the fresh child compiles the bytes and
            # reports the modules that fail to import.
            recoded = True
            fresh = True
            plan.lines["?"] = None
            rec = maps.file(fp.old_path) if fp.old_path else None
            if rec is not None:
                plan.cands.update(maps.tests(rec[7]))
                plan.cands.update(maps.tests(rec[8]))
            if path.startswith("tests/"):
                plan.cands.update(n for n, tid in enumerate(maps.meta["test_ids"]) if tid.split("::")[0] in (path, fp.old_path))
            plan.note.append("source encoding changes the text of %s" % path)
        if is_py and path.startswith("tests/") and fp.status != "delete":
            added_tests(plan, maps, path, old_text if fp.status != "add" else None, fp.new_text)
        scope = patchrules.pytest_scope(maps, fp, path, old_text)
        if scope is not None:
            # pytest itself reads this file (conftest.py, its options); the direct runs do not
            plan.blind.update(scope)
            plan.note.append("pytest file %s: %d tests" % (path, len(scope)))
            continue
        if fp.status in ("add", "rename") and patchrules.shadows_module(path):
            plan.blind.update(range(len(maps.meta["test_ids"])))
            plan.note.append("shadows a module")
        if is_py and in_tree:
            refs = None
            if fp.status == "add":
                # a new module is loaded only by its name: the tests that can reach that name
                base = os.path.basename(path)[:-3]
                word = os.path.basename(os.path.dirname(path)) if base == "__init__" else base
                refs = patchrules.package_module_refs(maps, path, text) if PKG_REFS else None
                if refs is not None:
                    plan.note.append("module %s of the tests package" % path)
                else:
                    refs = patchrules.referencing_tests(maps, word)
                if refs is not None:
                    plan.cands.update(refs)
                    plan.note.append("new module, %d referencing tests" % len(refs))
                    if refs:
                        fresh = True
                        plan.lines["?"] = None
            if recoded:
                pass
            elif fp.status != "modify" and refs is None:
                fresh = True
                plan.lines["?"] = None
                rec = maps.file(fp.old_path) if fp.old_path else None
                if rec is not None:
                    plan.cands.update(maps.tests(rec[7]))
                    plan.cands.update(maps.tests(rec[8]))
            elif fp.status == "modify" and (n_fp := len(plan.fpairs)) >= 0 and (
                plan_python_file_infect(plan, fp, maps, old_text, infect_used) if use_infect
                else plan_python_file(plan, fp, maps, old_text)
            ):
                fresh = True
                swap_candidate(plan, fp, maps, old_text, plan.fpairs[n_fp:])
                if fp.old_path not in plan.swap:
                    class_candidate(plan, fp, maps, old_text)
                plan.lines.pop(fp.old_path, None)
                rec = maps.file(fp.old_path)
                if rec is not None:
                    add_import_time_candidates(plan, fp, rec, maps)
                # a name that the patch removes or renames fails every line that uses it
                gone, extra = patchrules.gone_name_tests(maps, fp, old_text)
                if gone:
                    plan.gone.update(gone)
                    plan.cands.update(extra)
                    plan.lines["?"] = None
                    plan.note.append("names gone %s: %d tests" % (sorted(gone)[:3], len(extra)))
            if os.path.basename(path) == "__init__.py" and path.startswith("tests/"):
                # pytest imports the package of every test file below it; the coverage maps do
                # not list the package __init__ files as imported
                plan.cands.update(patchrules.tests_under(maps, os.path.dirname(path)))
                plan.lines["?"] = None
        if is_py and in_tree and not recoded and (
            (path.startswith("sqlglot/") and (fresh or fp.status != "modify"))
            or (path.startswith("tests/") and (fp.status != "modify" or sideeffect.touches_top_level(fp.groups)))
        ):
            # import-time code that changes state outside its own module (sideeffect.py):
            # every test that runs after the import (or, for a test file, after collection) can see it
            why = sideeffect.import_time_effects_fast(old_text, fp.new_text if fp.status != "delete" else None,
                                                 fp.groups if fp.status == "modify" else None)
            if why:
                plan.blind.update(range(len(maps.meta["test_ids"])))
                plan.note.append("import-time effect outside %s: %s" % (path, why))
        if is_py and path.startswith("sqlglot/") and fp.status == "modify" and not recoded:
            why = sideeffect.call_time_effects(old_text, fp.new_text, fp.groups)
            if why:
                # the state stays changed after the test that runs the line: every later test
                plan.blind.update(range(len(maps.meta["test_ids"])))
                plan.note.append("call-time effect outside the call in %s: %s" % (path, why))
        if is_py and path.startswith("sqlglot/") and fp.status == "modify" and not recoded:
            rec = maps.file(fp.old_path)
            watch = shared_write_watch(rec[0], old_text, fp.new_text, fp.groups) if rec is not None else []
            if watch:
                mod = path[:-3].replace("/", ".").removesuffix(".__init__")
                plan.watch.extend((mod, d) for d in watch)
                plan.note.append("watch shared containers %s" % [".".join(d[1:]) for d in watch][:4])
        # every test that reads the file as data can change
        for key in {fp.old_path, path}:
            if key:
                sid = maps.meta["opened"].get(key)
                if sid:
                    plan.cands.update(maps.tests(sid))
                    plan.data_cands.update(maps.tests(sid))
        if fp.status != "modify":
            d = os.path.dirname(path)
            sid = maps.meta["listed"].get(d)
            if sid:
                # The 'listed' map also holds the listings that the import system makes for
                # itself. Only tests whose file calls a listing function can see the file set.
                listers = patchrules.listing_tests(maps)
                for n in maps.tests(sid):
                    if listers is None or n in listers:
                        # os.listdir shows the patch in the child (install_listing_overlay);
                        # other ways of listing do not: those tests are selected unrun
                        rel = maps.meta["test_ids"][n].split("::")[0]
                        scans = listers is not None and patchrules.SCAN_RE.search(patchrules.source(rel))
                        (plan.blind if scans else plan.cands).add(n)
                        plan.data_cands.add(n)
        if is_py and not (path.startswith("sqlglot/") and fp.status == "modify"):
            plan.diffable = False
    n_fresh += fresh
    fresh = n_fresh > 0
    if infect_used:
        # The recorded operand relations hold only while nothing else in the patch can change
        # the values that reach the comparison: another file that brings candidates (code,
        # a fixture, a new file) or that needs a fresh import ends that, and the patch is
        # planned again without the map. (The marks count the files that need a fresh
        # import: a second such file may bring only candidates that the plan already has.)
        marks.append((len(plan.cands), len(plan.work), n_fresh, len(plan.blind), len(plan.data_cands), len(plan.gone), len(infect_used)))
        for a, b in zip(marks, marks[1:]):
            if a[:6] != b[:6] and a[6] == b[6]:
                kept = set().union(*(t for t, _ in plan.infect_files)) if plan.infect_files else set()
                again = make_plan(commit, text, maps, use_infect=False)
                if kept and INFECT_KEEP:
                    # A long held test with a changed evaluation at a mapped comparison of this
                    # patch stays selected (finish: sure_infected), as when that file is alone in
                    # its patch and the test is its only candidate. Without this, the cheap
                    # candidates of the other file pass and the evidence rule drops it.
                    again.sure_infected.update(kept)
                    again.note.append("infect sets kept: %d tests" % len(kept))
                return again
        # Every file that brings anything is now one mapped comparison edit. A test without
        # a changed evaluation at any of them runs as on the pinned code, so the cut after
        # the import (refine_with_diff) keeps the tests of all these files, not of the last.
        routes = {route for _, route in plan.infect_files}
        if "import" in routes and "other" not in routes:
            plan.infect_fresh = set().union(*(t for t, _ in plan.infect_files))
    add_registry_candidates(plan, fps, maps)
    if plan.vanished:
        plan.cands -= plan.vanished
        plan.blind -= plan.vanished
        plan.data_cands -= plan.vanished
        plan.note.append("%d tests leave the run (renamed classes, deleted tests)" % len(plan.vanished))
    if plan.added:
        fresh = True  # the added tests run in a child that imports the patched test files
    if fresh:
        plan.mode = "fresh"
    elif plan.work or plan.cands - plan.blind:
        plan.mode = "hot"
    elif plan.blind:
        plan.mode = "blind"
    return plan


_CLEAR_AT = {}


DEBUG_DKEYS = {}  # commit -> the "D" rows of its child (debug output only)
CONE_MAP = None  # module -> its importer cone in import order (maps/cone.json, build_cone.py)
CONE_WALK = []  # the modules, each after the modules it imports (cone.Heap)
CONE_FIXTURES = os.environ.get("BR_NO_CONE_FIXTURES") != "1"  # a changed .sql fixture beside the modules does not end the cone path
NO_SWAP_CONE = os.environ.get("BR_NO_SWAP_CONE") == "1"  # validation: no code replacement in cone children
_IMPORT_EXEC = None  # path -> {line of a function body: modules whose import runs it} (maps/import_exec.json)


def import_exec(maps):
    global _IMPORT_EXEC
    if _IMPORT_EXEC is None:
        out = {}
        try:
            with open(os.path.join(maps.path, "import_exec.json")) as f:
                data = json.load(f)
            names = data["modules"]
            for rel, rows in data["files"].items():
                out[rel] = {ln: [names[i] for i in ms] for ln, ms in rows}
            out[""] = True  # the map is there
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            out = {}
        _IMPORT_EXEC = out
    return _IMPORT_EXEC


NO_CLASS_ROUTE = os.environ.get("BR_NO_CLASS_ROUTE") == "1"  # validation: no class route (classroute.py)
_ASSIGN_RE = re.compile(r"\s+([A-Za-z_]\w*)\s*(?::[^=]*)?=(?!=)")
_ATTR_READS = None  # maps/attr_import_reads.json


def attr_reads(maps):
    global _ATTR_READS
    if _ATTR_READS is None:
        try:
            with open(os.path.join(maps.path, "class_route.json")) as f:
                _ATTR_READS = json.load(f)["classes"]
        except (OSError, ValueError, KeyError, TypeError):
            _ATTR_READS = {}
    return _ATTR_READS


def class_candidate(plan: Plan, fp, maps: Maps, old_text: str):
    """For a modified sqlglot file: notes in plan.klass the top-level class C when every hunk
    that is not blank or comment lines changes, line for line, statements directly in the body
    of C (not its header, not a class nested in it). The child then decides (classroute.py)."""
    rel = fp.old_path
    rec = maps.file(rel)
    if NO_CLASS_ROUTE or rec is None or not rel.startswith("sqlglot/") or CONE_MAP is None:
        return
    if rel[:-3].replace("/", ".").removesuffix(".__init__") in CONE_MAP:
        return  # the module has a small importer cone
    if _touches_slots(fp):
        return  # instance layout: not a class-body constant (see _slots_lines)
    inert = type_checking_groups(fp, old_text)
    units = rec[0]
    names = set()
    targets = set()  # the names that the changed statements assign, where the first line shows it
    old_lines = old_text.split("\n")
    for gi, g in enumerate(fp.groups):
        if gi in inert:
            continue
        o, nr, nl, na, removed, added = g
        if _blank_or_comment(added) and _blank_or_comment(removed) and (
                nl > 2 or not any(CODING_RE.search(a) for a in added + removed)):
            continue  # comment lines change nothing, wherever they are
        if nr > 0:
            hit = [u for u in units if u[0] <= o + nr - 1 and o <= u[1]]
        else:
            hit = [u for u in units if u[0] < o <= u[1]]
        if not hit:
            return
        for u in hit:
            if len(u[2]) != 1 or u[3] != 0:
                return
            if old_lines[u[0] - 1].lstrip().startswith(("def ", "async def ", "@", "class ")):
                return  # a method: its function object would differ
            names.add(u[2][0])
            m = _ASSIGN_RE.match(old_lines[u[0] - 1])
            if m:
                targets.add(m.group(1))
    if len(names) != 1:
        return
    name = names.pop()
    mod = rel[:-3].replace("/", ".").removesuffix(".__init__")
    read = attr_reads(maps).get(mod + ":" + name)
    if read is None:
        return  # not a class that the build-time checks passed (build_class_reexec.py)
    if "*" in read or not targets.isdisjoint(read):
        return  # read at import: the child would only find that out and import the whole tree
    plan.klass[rel] = (mod, name, os.path.join(ROOT, rel), frozenset(read))


def swap_candidate(plan: Plan, fp, maps: Maps, old_text: str, fpairs):
    """For a modified sqlglot file that needs a fresh import only because one function body
    that also runs at import changed (fpairs: what plan_python_file added to plan.fpairs for
    it): notes in plan.swap the code pairs and the modules whose import runs that function on
    the pinned tree (maps/import_exec.json). Nothing else of the file differs at import time
    (deftime_equal for every unit that the patch touches), so the patched tree differs from the
    pinned one only by that body, and the import can differ only where the body runs: the
    child replaces the code object, as for a function-body commit, and imports again only the
    cones of those modules (cone_plan). Any doubt: no entry, and the whole tree is imported."""
    rel = fp.old_path
    rec = maps.file(rel)
    if NO_SWAP_CONE or not fpairs or rec is None or not rel.startswith("sqlglot/") or CONE_MAP is None:
        return
    if rel[:-3].replace("/", ".").removesuffix(".__init__") in CONE_MAP:
        return  # the module has a small importer cone
    table = import_exec(maps)
    if not table:
        return
    inert = type_checking_groups(fp, old_text)
    units, flags, inames = rec[0], rec[1], rec[2]
    old_lines = old_text.split("\n")
    new_lines = fp.new_text.split("\n")
    full = os.path.join(ROOT, rel)
    hit_units = set()
    for gi, g in enumerate(fp.groups):
        if gi in inert:
            continue
        o, nr, nl, na, removed, added = g
        hit = []
        for ui, u in enumerate(units):
            s, e = u[0], u[1]
            if nr > 0:
                if s <= o + nr - 1 and o <= e:
                    hit.append(ui)
            elif s < o <= e:
                hit.append(ui)
        if not hit:
            if _blank_or_comment(added) and _blank_or_comment(removed):
                if nl > 2 or not any(CODING_RE.search(a) for a in added + removed):
                    continue
            return
        hit_units.update(hit)
    changed = []
    for ui in sorted(hit_units):
        s, e, class_path, kind = units[ui]
        if kind != 0:
            return
        delta_before = 0
        delta_in = 0
        for g in fp.groups:
            if g[0] + g[1] - 1 < s and g[0] < s:
                delta_before += g[3] - g[1]
            elif g[0] <= e:
                delta_in += g[3] - g[1]
        try:
            a = engine.compile_unit(old_lines, s, e, class_path, full, flags, inames)
            b = engine.compile_unit(new_lines, s + delta_before, e + delta_before + delta_in, class_path, full, flags, inames)
        except (SyntaxError, ValueError):
            return
        if a is None or b is None:
            return
        pairs = []
        if not engine.deftime_equal(a, b, pairs):
            return
        if all(engine.func_code_equal(x, y) for x, y in pairs):
            continue
        if not class_path and any(line.startswith("def __getattr__(") for line in old_lines[s - 1:e]):
            return
        changed.append((ui, pairs))
    if len(changed) != 1:
        return
    ui, pairs = changed[0]
    # the fingerprint of the import state counts exactly these bodies as unchanged (plan.fpairs)
    if len(pairs) != len(fpairs) or not all(
            engine.func_code_equal(x, x2) and engine.func_code_equal(y, y2) for (x, y), (x2, y2) in zip(pairs, fpairs)):
        return
    s, e = units[ui][0], units[ui][1]
    lines = table.get(rel, {})
    execs = set()
    for ln in range(s, e + 1):
        execs.update(lines.get(ln, ()))
    plan.swap[rel] = (full, pairs, sorted(execs))


def cone_plan(plan: Plan) -> bool:
    """Sets plan.cone for an import-time commit that can take the cone path. Each file is an
    existing sqlglot module that either maps/cone.json lists, with its import statements
    unchanged (so the import graph, and the order of a fresh import, stay as they were): its
    cone is imported again from the patched text; or whose only change is a function body
    (plan.swap): its code is replaced and the cones of the modules whose import runs that
    function are imported again."""
    if CONE_MAP is None or plan.mode != "fresh" or plan.added or not plan.files:
        return False
    mods = set()
    work = []
    class_work = []
    for rel, text in plan.files.items():
        if (CONE_FIXTURES and text is not None and rel.startswith("tests/fixtures/") and rel.endswith(".sql")
                and os.path.isfile(os.path.join(ROOT, rel))):
            # an existing .sql fixture: no import reads it, and the child shows the tests the
            # patched text as it does after an import of the whole tree
            continue
        if text is None or not rel.endswith(".py") or not rel.startswith("sqlglot/"):
            return False
        name = rel[:-3].replace("/", ".").removesuffix(".__init__")
        if python_text(text) != text:
            return False
        if name not in CONE_MAP:
            sw = plan.swap.get(rel)
            if sw is not None and all(m in CONE_MAP for m in sw[2]):
                for m in sw[2]:
                    mods.update(CONE_MAP[m])
                work.append((sw[0], sw[1]))
                continue
            kl = plan.klass.get(rel)
            if kl is None:
                return False
            class_work.append((kl[0], kl[1], text, kl[2], kl[3]))
            continue
        try:
            with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
                old = f.read()
            if cone.import_lines(old) != cone.import_lines(text):
                return False
        except (OSError, SyntaxError, ValueError):
            return False
        mods.update(CONE_MAP[name])
    order = [m for m in sys.modules if m in mods]
    if len(order) != len(mods):
        return False
    if class_work and (len(class_work) > 1 or any(m == kl[0] for kl in class_work for m in mods)):
        return False
    plan.cone = order
    plan.swap_work = work
    plan.class_work = class_work
    return True


def registry_clear_at(test_ids):
    """The first test after tests/test_dialect_entry_points.py. Its setUp and tearDown clear
    Dialect._classes, so in the full run every later test finds an empty dialect registry and
    registers each dialect again through _Dialect._try_load (dialect.py, the dir(module) scan)."""
    key = (id(test_ids), len(test_ids))
    hit = _CLEAR_AT.get(key)
    if hit is not None and hit[0] is test_ids:
        return hit[1]
    _CLEAR_AT[key] = (test_ids, _registry_clear_at(test_ids))
    return _CLEAR_AT[key][1]


def _registry_clear_at(test_ids):
    last = None
    for i, tid in enumerate(test_ids):
        if tid.startswith("tests/test_dialect_entry_points.py::"):
            last = i
    return None if last is None else last + 1


REGISTRY_METHODS = ("classes", "_try_load", "__getitem__", "get", "get_or_raise")
_REGISTRY_SPANS = {}


def _registry_spans(text):
    """The line spans of the registry lookup methods in the text of dialect.py (memoized: the
    pinned file is the same for every patch of the batch)."""
    import ast

    spans = _REGISTRY_SPANS.get(text)
    if spans is None:
        # the spans of the pinned file come from the map that build_registry_spans.py made
        try:
            with open(os.path.join(MAPS, "registry_spans.json")) as f:
                rec = json.load(f)
            if rec["len"] == len(text) and rec["crc"] == zlib.crc32(text.encode("utf-8", "surrogateescape")):
                spans = _REGISTRY_SPANS[text] = [tuple(s) for s in rec["spans"]]
                return spans
        except (OSError, ValueError, KeyError, TypeError):
            pass
        tree = ast.parse(text)
        spans = []
        for node in _walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name in REGISTRY_METHODS:
                        start = min([item.lineno] + [d.lineno for d in item.decorator_list])
                        spans.append((start, item.end_lineno))
        _REGISTRY_SPANS[text] = spans
    return spans


def add_registry_candidates(plan: Plan, fps, maps: Maps):
    """An edit of the registry lookup code reaches, through the registry state, every test
    after the clear that looks a dialect up by name, not only the tests that ran the line."""
    import ast

    clear_at = registry_clear_at(maps.meta["test_ids"])
    if clear_at is None:
        return
    for fp in fps:
        if fp.old_path != "sqlglot/dialects/dialect.py" or fp.status != "modify":
            continue
        try:
            with open(os.path.join(ROOT, fp.old_path), encoding="utf-8") as f:
                spans = _registry_spans(f.read())
        except (OSError, SyntaxError, ValueError):
            return
        hit = False
        for o, nr, nl, na, removed, added in fp.groups:
            a, b = o, o + max(nr, 1) - 1
            if any(s <= b and a <= e for s, e in spans):
                hit = True
        if not hit:
            continue
        rec = maps.file(fp.old_path)
        if rec is None:
            return
        extra = set()
        for s, e in spans:
            for ln in range(s, e + 1):
                if ln < len(rec[3]):
                    extra.update(n for n in maps.tests(rec[3][ln]) if n >= clear_at)
        plan.cands.update(extra)
        plan.note.append("registry: %d tests after the clear" % len(extra))


BASE_PATTERN = 0xFFFF
SCALED_IDS = frozenset((
    "tests/test_parser.py::TestParser::test_max_nodes",
    "tests/test_generator.py::TestGenerator::test_generator_memory",
    "tests/test_optimizer.py::TestOptimizer::test_scope",
))


def compute_keeps(plan: Plan, maps: Maps, tests):
    """For each candidate test whose independent parts are mapped: the parts that run the
    changed code or read a changed fixture entry. Other parts cannot change their result."""
    keeps = {}
    if plan.mode not in ("hot", "fresh", "fresh_purge", "fresh_cone"):
        return keeps
    # The small-scale variants (reduce.SCALED) run on every path: their whole runs cost far
    # more than any budget, and a long test that is held needs evidence to be selected.
    # reduce.scale_lines checks that the test's text is the pinned one (else the whole test
    # runs and its time limit decides).
    for n in plan.cands:
        if tests[n]["id"] in SCALED_IDS:
            keeps[n] = (None, None, None, True, None)
    if any(p.startswith("tests/") and p.endswith(".py") for p in plan.files):
        return keeps
    if plan.mode != "hot" and (None in plan.lines.values() or not plan.lines):
        return keeps
    if plan.watch:
        return keeps  # whole runs: a part of a test can leave out a write to a watched container
    code_lines = [(rel, ln) for rel, lines in plan.lines.items() for ln in lines]
    fixtures = {}
    for p in plan.files:
        if p.startswith("tests/fixtures/"):
            sid = maps.meta["opened"].get(p)
            fixtures[p[len("tests/fixtures/"):]] = set(maps.tests(sid)) if sid else set()
    for n in plan.cands:
        trec = maps.test_units(n)
        if trec is None:
            continue
        stmts, units, patterns = trec[2], trec[3], trec[4]
        unit_classes = trec[8] if len(trec) > 8 else None
        base_classes = trec[9] if len(trec) > 9 else None
        cfilter = plan.class_filter if unit_classes is not None else None
        chosen = set()
        if code_lines:
            seen = False
            whole = False
            for rel, ln in code_lines:
                pat = maps.unit_entries(rel, ln).get(n)
                if pat is None:
                    continue
                seen = True
                plain = cfilter is None or (rel, ln) in plan.plain_lines
                if pat == BASE_PATTERN:
                    # the always-run part runs the line: the whole test runs, unless that
                    # part creates no object that can read the changed value
                    if plain or base_classes is None or not cfilter.isdisjoint(base_classes):
                        whole = True
                        break
                    continue
                if plain:
                    chosen.update(patterns[pat])
                else:
                    for u in patterns[pat]:
                        uc = unit_classes[u]
                        if uc is None or not cfilter.isdisjoint(uc):
                            chosen.add(u)
            if whole:
                continue
            if not seen and (plan.work or plan.mode != "hot"):
                continue  # the coverage maps disagree: run the whole test
        keep_stmts = set()
        keep_files = {}
        keep_tasks = set() if any(u[0] == "t" for u in units) else None
        for ui in chosen:
            kind, name, i, parent = units[ui]
            if kind == "s":
                keep_stmts.add(i)
            elif kind == "f":
                keep_files.setdefault(name, set()).add(i)
            else:
                keep_tasks.add(i)
            if parent >= 0:
                keep_stmts.add(parent)
        ok = True
        for name, readers in fixtures.items():
            if n in readers:
                parents = {u[3] for u in units if u[0] == "f" and u[1] == name}
                if not parents:
                    ok = False  # the test reads the file in a way that the maps do not know
                    break
                keep_stmts.update(p for p in parents if p >= 0)
        if not ok:
            continue
        if code_lines and not chosen and not keep_stmts and not fixtures and plan.class_filter is not None:
            # no part of this test creates an object that can read the changed value
            plan.unaffected.add(n)
            continue
        keeps[n] = (keep_stmts, keep_files, keep_tasks, n in keeps, chosen)
    if plan.mode == "hot" and ITERS_ON and plan.lines and None not in plan.lines.values():
        compute_iters(plan, maps, keeps)
    return keeps


def compute_iters(plan: Plan, maps: Maps, keeps):
    """Loop iterations (iterred.py): in a kept statement that is a loop, only the iterations
    that run a changed line run, unless the statement runs the line outside its iterations."""
    import iterred

    lines = {rel: lns for rel, lns in plan.lines.items() if rel.endswith(".py")}
    for n in plan.cands:
        rec = iterred.record(maps.path, n)
        if rec is None:
            continue
        mode, edits, stmts = rec
        k = keeps.get(n)
        if mode == "stmt":
            if k is None or k[0] is None or k[3]:
                continue
            todo = [s for s in k[0] if s in stmts]
        else:
            if not ITERS_WHOLE or k is not None or maps.test_units(n) is not None:
                continue
            todo = list(stmts)
        kept = []
        sites = []
        for s in todo:
            got = iterred.choose(stmts[s], lines)
            if got is not None:
                kept.extend(got)
                sites.extend(stmts[s][0])
        if kept:
            plan.iters[n] = (iterred.keep_sets(kept), tuple(sorted(set(sites))), edits)


_PINNED_INDEX = {}


def pinned_code(path, name, first_line):
    """The code object of function 'name' at first_line in the pinned module code that this
    process loaded from the pyc of path (CODE_CACHE), or None. Same search as reduce.find_code."""
    index = _PINNED_INDEX.get(path)
    if index is None:
        index = _PINNED_INDEX[path] = {}
        module_code = CODE_CACHE.get(path)
        stack = [module_code] if module_code is not None else []
        while stack:
            c = stack.pop()
            for k in c.co_consts:
                if isinstance(k, types.CodeType):
                    index.setdefault((k.co_name, k.co_firstlineno), k)
                    stack.append(k)
    return index.get((name, first_line))


def _moved_shape(full, lines, shape):
    """The shape (first line, last line, ...) of a pinned test method in the patched lines of
    its file, when the method's lines occur there once, unchanged; else None. The pinned file
    is read with os.read (open() shows the patched text in the child)."""
    try:
        fd = os.open(full, os.O_RDONLY)
        try:
            chunks = []
            while True:
                b = os.read(fd, 1 << 20)
                if not b:
                    break
                chunks.append(b)
        finally:
            os.close(fd)
        pinned = b"".join(chunks).decode("utf-8").split("\n")
    except (OSError, UnicodeDecodeError):
        return None
    block = pinned[shape[0] - 1:shape[1]]
    if not block:
        return None
    hits = [k for k, s in enumerate(lines) if s == block[0] and lines[k:k + len(block)] == block]
    if len(hits) != 1:
        return None
    move = hits[0] - (shape[0] - 1)
    return (shape[0] + move, shape[1] + move) + tuple(shape[2:])


def _reduction_inputs(maps: Maps, tests, n, keep_stmts, scaled, it, lines_cache, faithful):
    """The part of apply_keeps for test n that reads only the pinned test file and the maps:
    (test id, file, class name, method, old code, new code), with old = new = None when the
    whole method stays, or None when no reduction applies (the whole test runs). The caller
    makes sure that the patch does not change the test's file."""
    import reduce

    test_id = tests[n]["id"]
    rel, cls_name, meth = test_id.split("::")
    trec = maps.test_units(n)
    shape = None
    if trec is not None:
        shape = (trec[0], trec[1], trec[2])
        if keep_stmts is not None and {i for i, st in enumerate(trec[2]) if st[2]} <= keep_stmts:
            keep_stmts = None  # every statement stays
    if keep_stmts is None and not scaled and it is None:
        return (test_id, rel, cls_name, meth, None, None)
    frec = maps.file(rel)
    if frec is None:
        return None
    if shape is None:
        shape = maps.meta["shapes"].get(n)
        if shape is None:
            return None
    lines = lines_cache.get(rel)
    if lines is None:
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            lines = lines_cache[rel] = f.read().split("\n")
    full = os.path.join(ROOT, rel)
    pinned = pinned_code(full, meth, shape[0]) if n in faithful else None
    got = reduce.reduction_codes(test_id, full, shape, lines, frec[1], frec[2], keep_stmts, scaled, pinned, it)
    if got is None:
        return None
    return (test_id, rel, cls_name, meth) + got


_PRE_LINES = {}  # pinned test file -> its lines (prepare_keeps, parent side)


def prepare_keeps(plan: Plan, maps: Maps, tests):
    """Parent side, before the fork of a function-body commit's child: the reduced code of
    its test methods (what apply_keeps would compile in the child). Same results; the work
    (file reads, map records, compiles) is done once here, so that the child does not pay
    for it in copy-on-write page faults, and file reads and map records are shared by all
    children. Only when the patch changes no file that the reduced tests are in; otherwise
    the child does it all, as before. The result names the keeps and iters that it was made
    from: apply_keeps uses it only for exactly these."""
    plan.pre_keeps = None
    if not plan.keeps or NO_REDUCE or plan.mode != "hot":
        return
    iters = plan.iters if ITERS_ON else {}
    items = list(plan.keeps.items()) + [(n, (None, None, None, False, None)) for n in iters if n not in plan.keeps]
    rels = {tests[n]["id"].split("::")[0] for n, _ in items}
    if rels & set(plan.files):
        return
    faithful = maps.faithful()
    pre = {}
    for n, (keep_stmts, keep_files, keep_tasks, scaled, _) in items:
        it = iters.get(n)
        try:
            got = _reduction_inputs(maps, tests, n, keep_stmts, scaled, it, _PRE_LINES, faithful)
        except Exception:  # noqa: BLE001  (the child does it, as before)
            return
        pre[n] = None if got is None else got + (keep_files, keep_tasks, it)
    plan.pre_keeps = (plan.keeps, plan.iters, pre)


def apply_keeps(plan: Plan, maps: Maps, tests):
    """Child side: replaces the code of the reduced test methods. Returns {test: (files, tasks)}."""
    import importlib

    import reduce

    out = {}
    iters = plan.iters if plan.mode == "hot" and ITERS_ON else {}
    if iters:
        import iterred

        iterred.install()
    pre = plan.pre_keeps
    if pre is not None and plan.mode == "hot" and pre[0] is plan.keeps and pre[1] is plan.iters:
        # the parent compiled the reduced methods (prepare_keeps): only the swaps are left
        for n, item in pre[2].items():
            if item is None:
                continue
            test_id, rel, cls_name, meth, old, new, keep_files, keep_tasks, it = item
            if old is not None:
                try:
                    cls = getattr(importlib.import_module(rel[:-3].replace("/", ".")), cls_name)
                except Exception:  # noqa: BLE001
                    continue
                if not reduce.install_reduction(cls, meth, old, new):
                    if DEBUG:
                        sys.stderr.write("reduction not applied: %s\n" % test_id)
                    continue
            out[n] = (keep_files, keep_tasks) if it is None else (keep_files, keep_tasks, it[0][0], it[0][1])
        return out
    lines_cache = {}
    faithful = maps.faithful()
    items = list(plan.keeps.items())
    if iters:
        items += [(n, (None, None, None, False, None)) for n in iters if n not in plan.keeps]
    for n, (keep_stmts, keep_files, keep_tasks, scaled, _) in items:
        it = iters.get(n)
        test_id = tests[n]["id"]
        rel, cls_name, meth = test_id.split("::")
        trec = maps.test_units(n)
        shape = None
        if trec is not None:
            shape = (trec[0], trec[1], trec[2])
            if keep_stmts is not None and {i for i, st in enumerate(trec[2]) if st[2]} <= keep_stmts:
                keep_stmts = None  # every statement stays
        if keep_stmts is not None or scaled or it is not None:
            frec = maps.file(rel)
            if frec is None:
                continue
            if shape is None:
                shape = maps.meta["shapes"].get(n)
                if shape is None:
                    continue
            lines = lines_cache.get(rel)
            if lines is None:
                with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
                    lines = lines_cache[rel] = f.read().split("\n")
            try:
                cls = getattr(importlib.import_module(rel[:-3].replace("/", ".")), cls_name)
            except Exception:  # noqa: BLE001
                continue
            full = os.path.join(ROOT, rel)
            if scaled and keep_stmts is None and plan.files.get(rel) is not None:
                # The patch changes the test's file: the small-scale variant applies when the
                # method's lines are the pinned ones; the map's line numbers move with the
                # lines that the patch adds or removes above it.
                shape = _moved_shape(full, lines, shape)
                if shape is None:
                    continue
            # The method as it compiles alone is the pinned method (maps/faithful.json, checked
            # at build time): the pinned code object stands in for that compile.
            pinned = pinned_code(full, meth, shape[0]) if n in faithful and rel not in plan.files else None
            if not reduce.apply_reduction(cls, test_id, full, shape, lines,
                                          frec[1], frec[2], keep_stmts, scaled, pinned, it):
                if DEBUG:
                    sys.stderr.write("reduction not applied: %s\n" % test_id)
                continue
        out[n] = (keep_files, keep_tasks) if it is None else (keep_files, keep_tasks, it[0][0], it[0][1])
    return out


# ---------------------------------------------------------------------------
# Children


def _decimal_settings(c):
    return (c.prec, c.rounding, c.Emin, c.Emax, c.capitals, c.clamp, tuple(sorted(k.__name__ for k, v in c.traps.items() if v)))


def _environ_copy():
    """os.environ as plain data. os.environ keeps its encoded items in the dict _data (every
    change through os.environ goes there); a copy of that dict compares at C speed."""
    data = getattr(os.environ, "_data", None)
    if type(data) is dict:
        return dict(data)
    return hash(frozenset(os.environ.items()))


def _interpreter_settings():
    """Interpreter-wide settings that a test run should leave as it found them."""
    import decimal
    import logging

    root = logging.getLogger()
    return (
        sys.getrecursionlimit(), sys.getswitchinterval(), os.getcwd(), len(sys.path), sys.path[:3],
        _environ_copy(), _decimal_settings(decimal.getcontext()), gc.get_threshold(),
        root.level, len(root.handlers), logging.root.manager.disable, sys.stdout is sys.__stdout__,
        sys.stderr is sys.__stderr__, sys.excepthook, sys.gettrace() is None,
    )


_PRIMITIVE = (bool, int, float, str, bytes, type(None))
_FUNCTION_TYPES = (types.FunctionType, staticmethod, classmethod, property, types.MethodType, types.BuiltinFunctionType)


def _shared_marks():
    """A cheap picture of the state that a test can leave behind for the later tests: the
    plain values (flags, numbers, strings) and the identity of the functions, classes and
    modules held by the module globals of sqlglot and of the test modules, and by the class
    attributes of sqlglot's classes. A replaced method, a new method or a changed module flag
    shows here; caches that fill up (dicts and lists) and objects do not, because ordinary
    tests change those all the time. Measured on 400 pinned tests run one by one: one
    change (a test that replaces the global sqlglot.schema), about 20 ms per picture."""
    out = []
    seen = set()

    def class_marks(mod, path, c):
        if id(c) in seen:
            return
        seen.add(id(c))
        for k, v in list(vars(c).items()):
            t = type(v)
            if t in _PRIMITIVE:
                out.append((mod, path, k, v))
            elif t in _FUNCTION_TYPES:
                out.append((mod, path, k, id(v)))
            elif isinstance(v, type) and v.__module__ == mod:
                class_marks(mod, path + "." + k, v)

    for name, m in sorted(sys.modules.items()):
        if m is None or not (name == "sqlglot" or name.startswith(("sqlglot.", "tests."))):
            continue
        for k, v in list(vars(m).items()):
            t = type(v)
            if t in _PRIMITIVE:
                out.append((name, k, v))
            elif t in _FUNCTION_TYPES or t is types.ModuleType or isinstance(v, type):
                out.append((name, k, id(v)))
                if isinstance(v, type) and v.__module__ == name and name.startswith("sqlglot"):
                    class_marks(name, k, v)
    return out


# The libraries that the pinned test modules import at module level (besides sqlglot and the
# tests themselves; grep of the import lines of tests/**/*.py). Every fresh child imports all
# test modules, and with them these libraries (setuptools alone costs about 65 ms).
TEST_LIBRARIES = (
    "unittest.mock", "sqlite3", "tracemalloc", "runpy", "doctest", "multiprocessing", "concurrent.futures",
    "setuptools", "setuptools.command.build_ext", "pytz", "duckdb", "numpy", "pandas", "pandas.testing",
    "resource", "argparse", "csv", "pickle", "weakref", "tempfile", "pathlib", "ast", "pkgutil", "json",
    "datetime", "decimal", "contextlib", "math", "time", "collections.abc", "functools",
)


def preimport_for_fresh(fresh, plans, tests):
    """Before the fresh children: the parent imports the libraries of the test modules and
    loads the code objects of every test module, once, instead of each child. Not when a
    commit changes a test module (its imports may differ). sys.path as it was before is kept
    in PRE_PATH: the child imports the patched sqlglot tree with that path (see child_main)."""
    global PRE_PATH
    import importlib

    if any(p.startswith("tests/") and p.endswith(".py") for c in fresh for p in plans[c].files):
        return
    before = list(sys.path)
    for name in TEST_LIBRARIES:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001  (the child imports it again and sees the same)
            pass
    if sys.path != before:
        PRE_PATH = before
    for path in sorted({os.path.join(ROOT, t["file"]) for t in tests}):
        if path not in CODE_CACHE and path.endswith(".py") and os.path.exists(path):
            name = os.path.relpath(path, ROOT)[:-3].replace("/", ".")
            try:
                _SourceFileLoader(name, path).get_code(name)
            except Exception:  # noqa: BLE001
                pass
    gc.freeze()


def run_added_tests(added, wfd):
    """Runs each test method that the patch adds (see added_tests) once, after the candidates,
    and reports "G n" when the shared state or the interpreter settings differ afterwards:
    in the full run every test from n on sees that state. "A n" / "A ok" bracket each run, so
    that the parent can tell a run that never ended (a hang or a crash, which also reaches
    the later tests)."""
    import unittest

    import runner

    before = (_shared_marks(), _interpreter_settings())
    for test_id, start in added:
        os.write(wfd, b"A %d\n" % start)
        try:
            case = runner.load_case(test_id, -1)
            unittest.TestSuite([case]).run(unittest.TestResult())
        except Exception:  # noqa: BLE001  (its own result does not count)
            pass
        after = (_shared_marks(), _interpreter_settings())
        if after != before:
            os.write(wfd, b"G %d\n" % start)
        before = after
        os.write(wfd, b"A ok\n")


def _copy_value(v):
    t = type(v)
    return t(v) if t in (dict, set, list) else v


def resolve_watch(watch):
    """The objects behind plan.watch in this process: [statefp key or None, getter, object,
    copy]. ("A", X): the attribute X of every sqlglot class that holds X itself as a dict, set
    or list (self.X reaches the class's own table, or one that it inherits)."""
    out = []
    classes = None
    for mod_name, d in watch:
        if d[0] == "A":
            if classes is None:
                classes, seen = [], set()

                def walk(c, path):
                    if id(c) in seen:
                        return
                    seen.add(id(c))
                    classes.append((c, path))
                    for k, v in list(vars(c).items()):
                        if isinstance(v, type) and v.__module__ == c.__module__:
                            walk(v, path + "." + k)

                for name, m in sorted(sys.modules.items()):
                    if m is None or not (name == "sqlglot" or name.startswith("sqlglot.")):
                        continue
                    for k, v in list(vars(m).items()):
                        if isinstance(v, type) and v.__module__ == name:
                            walk(v, v.__qualname__)
            for c, path in classes:
                v = vars(c).get(d[1])
                if type(v) in (dict, set, list):
                    out.append([("A", c.__module__, path, d[1]), (lambda c=c, a=d[1]: vars(c).get(a)), v, _copy_value(v)])
        else:
            m = sys.modules.get(mod_name)
            if m is None:
                continue
            if len(d) == 2:
                key, getter = ("G", mod_name, d[1]), (lambda m=m, a=d[1]: vars(m).get(a))
            else:
                owner = vars(m).get(d[1])
                for a in d[2:-1]:
                    owner = getattr(owner, a, None)
                if owner is None:
                    continue
                key = None
                if isinstance(owner, type):
                    key = ("A", owner.__module__, owner.__qualname__, d[-1])
                elif isinstance(owner, types.ModuleType):
                    key = ("G", owner.__name__, d[-1])
                getter = lambda o=owner, a=d[-1]: getattr(o, a, None)
            v = getter()
            out.append([key, getter, v, _copy_value(v)])
    return out


def check_watches(watches):
    """[(statefp key, changed entries or None)] of the watched objects that differ from their
    copies; the copies follow (a change is reported once)."""
    out = []
    for w in watches:
        cur = w[1]()
        if cur is w[2] and (type(cur) not in (dict, set, list) or cur == w[3]):
            continue
        entries = None
        if keymap is not None and type(cur) is type(w[3]) and type(cur) in (dict, set):
            try:
                got = keymap.changed_entries(w[3], cur, id, id)
                entries = sorted(got) if got is not None else None
            except Exception:  # noqa: BLE001
                entries = None
        out.append((w[0], entries))
        w[2], w[3] = cur, _copy_value(cur)
    return out


def run_tokcheck(numbers, wfd):
    """Child side, after the patched code is in place and before any test runs: tokenizes the
    recorded inputs of these held long tests again (tokcheck.py). Each test that gets another
    token stream is reported ("M n"), and the parent selects it."""
    if not numbers or not TOKCHECK:
        return ()
    os.write(wfd, b"M -1\n")  # the check takes time: the parent's clock for the setup restarts
    try:
        import tokcheck

        found = tokcheck.changed(MAPS, numbers, TOKCHECK_SECONDS * SLOWDOWN)
    except Exception:  # noqa: BLE001  (no answer: the tests count as changed)
        found = set(numbers)
    for n in sorted(found):
        os.write(wfd, b"M %d\n" % n)
    return found


def memo_dirty(plan, record=None):
    """Parent side: what memo.prepare finds for the commit's changed functions ({"d": dirty step
    ids, "full": tasks without the memo, "noopt": tasks whose optimize() may be answered from
    the record}), or None (no memo). The pinned texts are read here, in the parent: the child
    sees the patched tree."""
    try:
        import memo

        store = memo.Store(MAPS)
        if not store.ok():
            return None
        old = {}
        for p in plan.files:
            with open(os.path.join(ROOT, p), encoding="utf-8", newline="") as f:
                old[p] = f.read()
        return memo.prepare(store, plan.files, old, record)
    except Exception:  # noqa: BLE001
        return None


def memo_cost(m, cpu, active, later):
    """Build-time CPU seconds of the check's tasks with the memo (memo.prepare's answer m): a
    task's early part without optimize() where the record answers it; a later task with a
    record: its dirty steps, plus restoring the root's output; any other later task: to its end."""
    import memo

    store = memo.Store(MAPS)
    queries = store.index["queries"]
    dirty, full, noopt = set(m["d"]), set(m["full"]), set(m["noopt"])
    total = 0.0
    for i in active:
        early, late = cpu.get(i, (0.0, 0.0))
        rec = queries.get(i)
        if i in noopt and rec is not None and rec.get("opt") is not None and os.environ.get("BR_MEMO_OPT", "1") == "1":
            early = max(0.0, early - rec["opt"]["cpu"])
        total += early
        if i not in later:
            continue
        if rec is None or i in full:
            total += late
            continue
        steps = [k for k in range(len(rec["keys"])) if "%d:%d" % (i, k) in dirty]
        total += sum(rec["cpu"][k] for k in steps) + 0.01 + 0.002 * len(rec["keys"])
    return total


def run_stage_check(plan: Plan, tests, started, keeps, seen, broken, forced, settings, code_pairs, wfd):
    """Child side, after the commit's other runs: the stage check of the held long test
    (stages.py), when the parent would select that test without a run. It reports what it
    found in an "X" line; the parent drops the test only on the verdict "passes".
    seen: the verdicts that this child reported; broken: the modules that do not import here;
    forced: held tests that are selected whatever the verdicts are; code_pairs: (old code,
    new code) of the function bodies that this child replaced."""
    st = plan.stage
    n = st["n"]
    if (broken and not COLL_RUN) or settings[1] or settings[2]:
        # modules that do not import, changed interpreter settings, a dead class pool: the
        # parent selects the test for another reason than the verdicts
        if DEBUG:
            sys.stderr.write("stage check %s not run: broken %r settings %r\n" % (plan.commit, broken[:5], settings[1:]))
        return
    if STAGE not in ("always", "force"):
        # the rule of finish() for a long held test, with what this child knows
        failing = [k for k, bad in seen.items() if bad]
        verdicts = [k for k, bad in seen.items() if bad is not None]
        wanted = st["forced"] or n in forced
        if not wanted and not (st["inert"] and not failing):
            wanted = not verdicts or len(failing) + st["more"] >= MANY_FAILURES or any(k in st["deciders"] for k in failing)
        if not wanted:
            return
    # What the check can save: the seconds that the test adds to the answer of this commit,
    # next to the tests that are selected in any case (the failing tests, the tests that did
    # not run after a failure, the other held tests). The check may cost STAGE_GAIN CPU
    # seconds per second of that; its planned part (st["est"]) must fit, and the rest is the
    # budget of the tasks that run on because their input differs.
    seconds = [t["seconds"] for t in tests]
    others = set(st["others"])
    others.update(k for k, bad in seen.items() if bad is not False)
    others.discard(n)
    saving = packed_seconds(others | {n}, seconds) - packed_seconds(others, seconds)
    budget = STAGE_GAIN * saving - st["est"]
    if STAGE == "force":
        budget = float("inf")
    if budget < 0.0:
        os.write(wfd, b"X " + json.dumps({"n": n, "verdict": None, "skip": "saves %.1f s, costs %.1f" % (saving, st["est"])}).encode() + b"\n")
        return
    import reduce
    import runner
    import stages

    if GUARD and os.environ.get("BR_STAGE_AUDIT") != "1":
        CPU_GUARD[n] = max(GUARD_MIN, min(1.5 * (st["est"] + min(budget, st["differ_most"])) + 3.0, STAGE_GAIN * saving + 1.0))  # (started() sets this limit; never more CPU than the check can save)
    os.write(wfd, b"X start\n")
    info = {"n": n, "verdict": None, "saving": round(saving, 2)}
    t0 = time.process_time()
    check = holder = None
    _prof = None
    if os.environ.get("BR_STAGE_PROF"):
        import cProfile

        _prof = cProfile.Profile()
        _prof.enable()
    try:
        test_id = tests[n]["id"]
        path, cls_name, _ = test_id.split("::")
        record = MAPS_OBJ.stages()[test_id]
        case = runner.load_case(test_id, n)  # loads the test module as a run does (class fixture hook)
        module = sys.modules[path[:-3].replace("/", ".")]
        cls = getattr(module, cls_name)
        active = keeps[n][1] if keeps and n in keeps else None
        if os.environ.get("BR_STAGE_AUDIT") == "1":
            import stage_audit

            check = stage_audit.AuditCheck(record, set(st["later"]), st["to_end"] + budget, active)
        else:
            check = stages.StageCheck(record, set(st["later"]), st["to_end"] + budget, active)
            if st.get("memo") is not None:
                try:
                    import memo

                    store = memo.Store(MAPS)
                    m = st["memo"]
                    if store.ok():
                        check.memo = (store, set(m["d"]))
                        check.memo_full = set(m["full"])
                        check.noopt = set(m["noopt"])
                        check.notext = set(m.get("notext", ()))
                        check.nomid = set(m.get("nomid", ()))
                        check.hook_changed = m.get("hook_edit", True) is not False  # (schema_stub only)
                        rowh = store.index.get("rowh")
                        check.rowh = rowh if type(rowh) is dict and rowh else None
                        info["memo_dirty"] = [len(m["d"]), len(m["full"]), len(m["noopt"])]
                except Exception as e:  # noqa: BLE001  (no memo: the check runs as before)
                    info["memo_err"] = repr(e)[:200]
        pairs = code_pairs + list(plan.fpairs or ())  # function bodies that the commit changes
        alias = stages.code_alias(pairs) if pairs else None
        # as for a test: the "S" line, the dialect registry of a full run, the CPU limit
        started(n)
        holder = stages.hold_state(record["effect"], alias)  # the module state before the check
        before_settings = _interpreter_settings()
        if not check.install(module, cls):
            info["why"] = "execute() is not the pinned one"
            return
        results = []
        reduce.STATE.task_hook = check.run_task
        try:
            if check.probe(case):
                runner.run_tests([(n, test_id)], lambda number, failed: results.append(failed), False, keeps)
            else:
                results.append(False)  # (over: no verdict below)
        finally:
            reduce.STATE.task_hook = None
            check.remove()
            signal.setitimer(signal.ITIMER_PROF, 0)  # (started() set the CPU limit of the check)
        info["t_run"] = round(time.process_time() - t0, 3)
        info.update(same=len(check.same), ran=[[i, why] for i, why in check.ran[:8]], nran=len(check.ran),
                    proved=len(check.proved), seq="".join(check.seq), spent=round(check.spent, 3), why=check.why,
                    memo=getattr(check, "memo_stats", None), opt=getattr(check, "opt_stats", None), fast=getattr(check, "fast_stats", None), rows_cpu=round(getattr(check, "rows_cpu", 0.0), 4), rowh=getattr(check, "rowh", None) is not None,
                    run=None if not results else results[0])
        if not results or results[0] is None:
            verdict = None
        elif results[0]:
            verdict = True  # an error in a task, or in the test outside its tasks: the test fails
        elif not check.complete():
            verdict = None  # the check ended early (another SQL text, other rows, the budget)
        else:
            verdict = False
        if verdict is False:
            # The tasks that stopped early count as passed only when the patched code left the
            # interpreter settings alone and the module state as the pinned test leaves it:
            # their later work reads both.
            if _interpreter_settings() != before_settings:
                info["state"] = "settings"
                verdict = None
            elif not check.rows_ok():
                info["state"] = "fixture rows"
                verdict = None
            elif check.same or check.memo_used:
                before = stages.held_state(holder)
                ok, changed = stages.state_ok(record["effect"], before, alias) if before is not None else (False, [("no", "snapshot")])
                if not ok:
                    info["state"] = [list(map(str, k)) for k in changed[:5]]
                    verdict = None
        if hasattr(check, "report"):
            info["audit"] = check.report()
            info["would"] = verdict
            verdict = None  # validation run: the answer stays what it is without the check
        info["verdict"] = verdict
    except Exception as e:  # noqa: BLE001  (no verdict: the parent treats the test as before)
        info["error"] = repr(e)[:200]
        info["verdict"] = None
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        reduce.STATE.reset()
        if holder is not None:
            stages.release(holder)
        info["cpu"] = round(time.process_time() - t0, 3)
        if _prof is not None:
            _prof.disable()
            _prof.dump_stats(os.environ["BR_STAGE_PROF"])
        os.write(wfd, b"X " + json.dumps(info).encode() + b"\n")


def child_reporters(wfd):
    """The two functions through which a child reports its test runs to the parent, and the
    state that they keep. Returns (emit, started, settings, watches, begin):
    emit(number, failed) writes the verdict line of a test (and the "G", "W" and "P" lines);
    started(number) writes the "S" line and gives the first test the dialect registry of a full
    run; settings = [interpreter settings before the first test, a change was reported, a
    class's shared pool died]; watches = the shared containers to watch (plan.watch);
    begin() makes the next test count as the first test of a child again (a child that serves
    several commits calls it before each of them, see fixture_batch_child)."""
    settings = [None, False, False]
    watches = []  # shared containers that the patched code may leave changed (plan.watch)

    last = [time.process_time()]
    mark = [last[0], None]  # CPU time at the start of the test that runs now, and that test

    import pool as pool_module

    def emit(number, failed):
        if armed[0]:
            armed[0] = False
            signal.setitimer(signal.ITIMER_PROF, 0)  # the test ended inside its CPU limit
        now = time.process_time()
        code = 2 if failed is None else (1 if failed else 0)  # 2 = not run
        if settings[0] is not None and not settings[1] and _interpreter_settings() != settings[0]:
            # The patched code changed an interpreter-wide setting (recursion limit, working
            # directory, environment, ...). In a full run, every later test runs with it.
            settings[1] = True
            os.write(wfd, b"G %d\n" % number)
        if watches:
            # a watched container left with other content than before this test: in a full
            # run every later test that reads it sees that content (see finish)
            for key, entries in check_watches(watches):
                os.write(wfd, b"W %d %s\n" % (number, json.dumps([key, entries]).encode()))
        if DEBUG:
            alone = now - mark[0] if mark[1] == number else 0.0  # (a test that cannot be loaded never starts)
            os.write(wfd, b"%d %d %d %d\n" % (number, code, int((now - last[0]) * 1e6), int(alone * 1e6)))
        else:
            os.write(wfd, b"%d %d %d\n" % (number, code, int((now - last[0]) * 1e6)))
        last[0] = now
        if pool_module.InlinePool.broke_any:
            # the class's shared process pool died in this test: in a full run every later
            # test of the class that uses the pool fails
            pool_module.InlinePool.broke_any = False
            settings[2] = True
            os.write(wfd, b"P %d\n" % number)

    cleared = [False]
    started_once = []
    armed = [False]

    def begin():
        settings[0], settings[1], settings[2] = None, False, False
        cleared[0] = False
        del started_once[:]
        last[0] = time.process_time()

    def started(number):
        if REGISTRY_START is not None and not started_once:
            # The full run starts with the dialects that importing the test files registers
            # (maps/registry_start.json); the others are registered by the first test that
            # looks them up. This process registered all of them at start-up.
            started_once.append(True)
            try:
                dialect = sys.modules["sqlglot.dialects.dialect"]
                classes = dialect.Dialect._classes
                for key in [k for k in classes if k in dialect.DIALECT_MODULE_NAMES and k not in REGISTRY_START]:
                    del classes[key]
            except Exception:  # noqa: BLE001  (sqlglot does not import: the tests report that)
                pass
        if not cleared[0] and REGISTRY_CLEAR_AT is not None and number >= REGISTRY_CLEAR_AT:
            # the full run reaches this test with an empty dialect registry (see registry_clear_at)
            cleared[0] = True
            try:
                sys.modules["sqlglot.dialects.dialect"].Dialect._classes.clear()
            except Exception:  # noqa: BLE001  (sqlglot does not import: the tests report that)
                pass
        os.write(wfd, b"S %d\n" % number)
        mark[0], mark[1] = time.process_time(), number
        limit = CPU_GUARD.get(number)
        if limit is not None:
            # The CPU guard (see GUARD): the kernel counts the CPU time of this process, all
            # threads, from here. At the limit it sends SIGPROF, which ends the process (no
            # handler is installed). The parent reads the signal from the wait status and
            # selects the test.
            armed[0] = True
            signal.setitimer(signal.ITIMER_PROF, limit)

    return emit, started, settings, watches, begin


def child_main(plan: Plan, order, tests, wfd, index, skip_after_fail, plan_fd=None):
    import runner

    emit, started, settings, watches, _ = child_reporters(wfd)
    seen = {}  # what this child reported, for run_stage_check: the verdicts,
    broken_here = []  # the modules that do not import,
    forced = set()  # the held tests whose token streams changed,
    code_pairs = []  # and the function bodies that it replaced
    if STAGE != "0":
        report = emit

        def emit(number, failed):
            seen[number] = failed  # (None: not run after a failure, the parent selects it)
            report(number, failed)

    try:
        signal.signal(signal.SIGPROF, signal.SIG_DFL)  # the CPU guard ends this process with it
        CPU_GUARD.clear()
        CPU_GUARD.update(plan.guard)
        if ANSWER_FD is not None:
            os.close(ANSWER_FD)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        if not DEBUG:
            os.dup2(devnull, 2)
        if DEBUG:
            import faulthandler

            faulthandler.enable(file=sys.stderr)
        if os.environ.get("BR_FAULT"):
            import faulthandler

            faulthandler.dump_traceback_later(float(os.environ["BR_FAULT"]), file=sys.stderr)
        t_child = time.process_time()
        duckreplay = sys.modules.get("duckreplay")
        if duckreplay is not None and duckreplay.ACTIVE and not duckreplay.patch_ok(plan.files):
            # the patch changes test code or data: the recorded DuckDB answers are not valid
            # for it, so every query of this child runs on a real connection
            duckreplay.disable()
        py_files = {p: t for p, t in plan.files.items() if p.endswith(".py")}
        if any(not (p.startswith("sqlglot/") or (p.startswith("tests/fixtures/") and p.endswith(".sql")))
               for p in plan.files):
            # the commit changes test code or data: no on-demand table dicts (preload._LazyTables)
            import preload as _preload

            _preload.exec_tables_strict()
        if plan.mode in ("fresh", "fresh_purge", "fresh_cone"):
            post_path = None
            if PRE_PATH is not None:
                # the parent imported the libraries of the test modules (setuptools adds a path):
                # the patched tree is imported with the path that it had before
                post_path = list(sys.path)
                sys.path[:] = PRE_PATH
            import preload

            if plan.mode == "fresh_cone":
                # Only the cone of the patched modules is imported again (cone.py). The child
                # reports REDO (and the parent imports the whole tree) when the checks fail.
                # The objects of the parent are not collected here (as in the children of the
                # other commits, whose parent froze them): a collection would walk all of them.
                gc.freeze()
                try:
                    if plan.swap_work:
                        # files whose only change is a function body: the code objects are
                        # replaced, as for a function-body commit (the module is not in the cone)
                        swaps = []
                        for full, pairs in plan.swap_work:
                            got = engine.plan_swaps(pairs, index) if index is not None else None
                            if got is None:
                                raise cone.Redo("no safe code replacement")
                            swaps.extend(got)
                        for f, code in swaps:
                            f.__code__ = code
                        cone.expect_code(swaps)
                    class_done = None
                    if plan.class_work:
                        # one class-level statement of a class changed: the patched class is
                        # created beside the pinned one (classroute.py); the checks of
                        # reimport() below also cover what its creation did outside it
                        import classroute

                        kmod, kname, ktext, kfull, kreads = plan.class_work[0]
                        if BASE_STATE is None:
                            raise cone.Redo("class route: no base state")
                        class_done = classroute.prepare(kmod, kname, ktext, kfull, kreads, BASE_STATE, BASE_CODES, {})
                    cone_touched = cone.reimport(plan.cone, lambda: engine.install_overlay(ROOT, py_files))
                    if class_done is not None:
                        k_old, k_new, k_names, k_changed = class_done[:4]
                        k_ref = classroute.apply(k_old, k_new, k_names)
                        class_done = k_new = None
                        if not classroute.gone(k_ref):
                            raise cone.Redo("class route: the new class is still alive")
                        class_done = (k_old, kmod, kname, k_changed)
                except Exception as e:  # noqa: BLE001  (cone.Redo: a check failed; anything else too)
                    if DEBUG:
                        sys.stderr.write("child %s cone: %r\n" % (plan.commit, e))
                    os.write(wfd, b"REDO\n")
                    os._exit(0)
                test_mods = []
                bad = []
                if DEBUG:
                    sys.stderr.write("child %s cone %d modules: %.3f\n" % (plan.commit, len(plan.cone), time.process_time() - t_child))
            else:
                if plan.mode == "fresh_purge":
                    for name in [m for m in sys.modules if m.partition(".")[0] in ("sqlglot", "tests")]:
                        del sys.modules[name]
                engine.install_overlay(ROOT, py_files)
                # a module that cannot be imported any more fails every test that loads it

                # the patched test modules too: a test package or helper module that raises fails
                # every test file that imports it
                test_mods = [
                    p[:-3].replace("/", ".").removesuffix(".__init__") for p in py_files
                    if p.startswith("tests/") and os.path.basename(p) != "conftest.py"
                ]
                bad = preload.broken_modules(test_mods)
            broken_here.extend(bad)
            for name in bad:
                os.write(wfd, b"B %s\n" % name.encode())

            def import_all_tests():
                # Every test file: one that cannot be imported any more fails all its tests
                # (pytest imports every test file before the first test runs). This comes
                # after the import-state comparison, because test modules change that state.
                if post_path is not None:
                    sys.path[:] = post_path
                names = sorted({t["id"].split("::")[0][:-3].replace("/", ".") for t in tests} - set(test_mods))
                for name in preload.broken_test_modules(names):
                    broken_here.append(name)
                    os.write(wfd, b"B %s\n" % name.encode())

            if plan_fd is None:
                import_all_tests()
            if preload.exec_fresh_ok(plan.files):
                runner.ON_IMPORT[preload.EXEC_KEY[0]] = preload.exec_fresh_hook
            if plan_fd is not None:
                # What did the import leave behind, compared with the pinned tree? The parent
                # turns the changed names into the tests (and test parts) that read them.
                import statefp

                alias = {}
                if plan.fpairs:
                    f0 = statefp.Fingerprinter()
                    stack = list(plan.fpairs)
                    while stack:
                        old, new = stack.pop()
                        alias[f0.code(new)] = f0.code(old)
                        oc = [k for k in old.co_consts if isinstance(k, type(old))]
                        nc = [k for k in new.co_consts if isinstance(k, type(new))]
                        if len(oc) == len(nc):
                            stack.extend(zip(oc, nc))
                if plan.mode == "fresh_cone":
                    # the modules outside the cone hold the pinned objects (checked by
                    # cone.reimport): their keys are the pinned ones
                    only = set(plan.cone) | cone_touched
                    new_state, fpr = statefp.snapshot(alias, BASE_CODES, only)
                    changed = statefp.diff_only(BASE_STATE, new_state, only)
                    if class_done is not None:
                        # the keys of the class that now holds the patched values, placed
                        # where a snapshot of all modules lists them (modules by name)
                        k_old, kmod, kname, k_changed = class_done
                        k_state = statefp.class_state(k_old, kmod, kname, fpr)
                        k_now = [k for k, v in k_state.items() if BASE_STATE.get(k, _MISSING) != v]
                        if k_now != k_changed:
                            os.write(wfd, b"REDO\n")
                            os._exit(0)
                        at = len(changed)
                        for i, k in enumerate(changed):
                            if k in new_state and k[1] > kmod:
                                at = i
                                break
                            if k not in new_state:
                                at = i
                                break
                        changed[at:at] = k_now
                else:
                    new_state, fpr = statefp.snapshot(alias, BASE_CODES)
                    changed = statefp.diff(BASE_STATE, new_state)
                if len(changed) > 300:
                    os.write(wfd, b"D many\n")
                else:
                    entries = {}
                    if not NO_KEYMAP:
                        try:  # the changed entries of the changed tables (keymap.py)
                            entries = keymap.diff_rows(changed, BASE_OBJECTS, BASE_STATE, fpr.objects, fpr,
                                                       statefp.Fingerprinter)
                        except Exception:  # noqa: BLE001
                            entries = {}
                    rows = [[list(k), k in fpr.function_keys or k in BASE_FUNCTION_KEYS, entries.get(k)] for k in changed]
                    os.write(wfd, b"D " + json.dumps(rows).encode() + b"\n")
                t_diff = time.process_time()
                import_all_tests()
                if COLL_RUN:
                    os.write(wfd, b"T\n")  # every test file imported: the parent knows plan.broken
                if DEBUG:
                    sys.stderr.write("child %s import+diff %.3f tests %.3f\n" % (plan.commit, t_diff - t_child, time.process_time() - t_diff))
                chunks = []
                while True:
                    chunk = os.read(plan_fd, 1 << 16)
                    if not chunk:
                        break
                    chunks.append(chunk)
                os.close(plan_fd)
                todo = marshal.loads(b"".join(chunks)) if chunks else None
                if todo:
                    forced.update(run_tokcheck(todo.get("tokcheck"), wfd))
                if not todo or (not todo["order"] and not todo.get("stage")):
                    os.write(wfd, b"END\n")
                    os._exit(0)
                order = todo["order"]
                plan.stage = todo.get("stage")
                CPU_GUARD.update(todo.get("guard") or {})
                skip_after_fail = todo["skip"]
                plan.enders = todo["enders"]
                plan.keeps = todo["keeps"]
            if bad:
                # a test that loads a module that cannot be imported fails (the parent selects
                # it anyway): running it would only repeat the failing import
                dead = patchrules.importer_tests(MAPS_OBJ, bad)
                for n in order:
                    if n in dead:
                        emit(n, True)
                order = [n for n in order if n not in dead]
        else:
            for full, pairs in plan.work:
                swaps = engine.plan_swaps(pairs, index)
                if swaps is None:
                    # "REDOC": only the cells of a live closure do not fit the new code; the
                    # line map of the changed lines still gives the candidates
                    os.write(wfd, b"REDOC\n" if engine.SWAP_FAIL == "cells" else b"REDO\n")
                    os._exit(0)
                for f, code in swaps:
                    if plan.stage is not None:
                        code_pairs.append((f.__code__, code))
                    f.__code__ = code
        read_old = engine.install_open_overlay(ROOT, plan.files)
        patchrules.install_listing_overlay(ROOT, plan.files)
        if plan.mode == "hot":
            import preload

            # the parent built the shared class fixtures from the pinned files
            for mod_name, failed in preload.refresh_class_fixtures(ROOT, plan.files):
                if failed:  # setUpClass raises in the full run: every test of the class fails
                    broken_here.append(mod_name)
                    os.write(wfd, b"K %s\n" % mod_name.encode())
        import reduce

        keeps = None
        if plan.mode != "hot" and plan.keeps and not NO_REDUCE:
            fixture_files = [rel for rel in plan.files if rel.startswith("tests/fixtures/")]
            loaders = False
            if fixture_files or any(k[0] is not None or k[1] is not None for k in plan.keeps.values()):
                loaders = reduce.install_loaders()
            # A reduced run keeps the fixture entries that reach the changed code. When the
            # patch also changes a fixture file, the entries that it changes must run too.
            changed = {}
            comparable = True
            for rel in fixture_files:
                got = {}
                if loaders and plan.files[rel] is not None:
                    got = reduce.changed_entries(rel[len("tests/fixtures/"):], read_old)
                if not got:
                    comparable = False  # deleted, or not readable as entries: no reduced runs
                changed.update(got)
            if comparable:
                if changed:
                    reduce.STATE.keep_changed = changed
                keeps = apply_keeps(plan, MAPS_OBJ, tests)
        if plan.mode == "hot" and reduce.install_loaders():
            # an unchanged fixture entry that does not run changed code cannot change a result
            changed = {}
            for rel, text in plan.files.items():
                if rel.startswith("tests/fixtures/") and text is not None:
                    changed.update(reduce.changed_entries(rel[len("tests/fixtures/"):], read_old))
            if changed:
                reduce.STATE.keep_changed = changed
                reduce.STATE.changed_only = not plan.work
                if not plan.work:
                    import preload

                    preload.reduce_class_fixtures(changed)
            if plan.keeps and not NO_REDUCE:
                keeps = apply_keeps(plan, MAPS_OBJ, tests)
        if PANDAS_USERS and any(n in PANDAS_USERS for n in order):
            import lazymods

            lazymods.load()  # before the settings snapshot below: as in the parent
        if plan.watch:
            watches.extend(resolve_watch(plan.watch))
            os.write(wfd, b"H %s\n" % json.dumps([w[0] for w in watches]).encode())
        forced.update(run_tokcheck(plan.tokcheck, wfd))
        if plan.mode != "hot":
            # The objects of the new import stay for the life of this child. Frozen, they are
            # not walked again by every collection during the test runs (the children of the
            # function-body commits get the same from their parent, which froze its objects
            # before the fork).
            gc.freeze()
        t_setup = time.process_time()
        settings[0] = _interpreter_settings()
        runner.run_tests([(n, tests[n]["id"]) for n in order], emit, False, keeps, started, skip_after_fail, plan.enders)
        if plan.added and plan.mode in ("fresh", "fresh_purge"):
            run_added_tests(plan.added, wfd)
        if plan.stage is not None:
            run_stage_check(plan, tests, started, keeps, seen, broken_here, forced, settings, code_pairs, wfd)
        if DEBUG:
            sys.stderr.write("child %s setup %.3f run %.3f\n" % (plan.commit, t_setup - t_child, time.process_time() - t_setup))
            if "lazymods" in sys.modules:
                import lazymods

                sys.stderr.write("child %s libs numpy=%d pandas=%d duckdb=%d replay=%s\n" % (
                    plan.commit, lazymods.loaded("numpy"), lazymods.loaded("pandas"), lazymods.loaded("duckdb"),
                    json.dumps([duckreplay.ACTIVE, duckreplay.STATS]) if duckreplay is not None else "off"))
        os.write(wfd, b"END\n")
    except BaseException as e:  # noqa: BLE001
        try:
            os.write(wfd, ("ERR %r\n" % (e,)).encode()[:400])
        except OSError:
            pass
    finally:
        os._exit(0)


_MISSING = object()
# Class attributes that every run of tests changes: unittest's own bookkeeping (the class
# cleanups run when the class's tests end, and leave the list of their errors) and the copy
# that refresh_class_fixtures keeps.
_ROUTINE_CLASS_STATE = frozenset(("_classSetupFailed", "_class_cleanups", "tearDown_exceptions", "_br_parent_state"))
_ROUTINE_POOL_STATE = frozenset(("_started", "_shutdown"))  # pool.InlinePool: a task ran; the class cleanup shut it down


def _same_items(a, b):
    """True when two lists, dicts or sets hold the same objects."""
    if type(a) is not type(b) or len(a) != len(b):
        return False
    if type(a) is list:
        return all(x is y for x, y in zip(a, b))
    if type(a) is dict:
        return all(b.get(k, _MISSING) is v for k, v in a.items())
    return a == b


def _same_value(a, b):
    return a is b or (type(a) is type(b) and type(a) in _PRIMITIVE and a == b)


def shared_state_picture():
    """For a child that serves several commits: [(object, copy of its attribute dict)] for the
    loaded modules of sqlglot and of the tests, and for the classes that sqlglot's modules
    define (with the classes nested in them). The copies hold the same objects, so
    shared_state_change compares at the speed of a dict comparison (_shared_marks walks the
    same objects in Python, which costs about 30 ms a picture)."""
    out = []
    seen = set()

    def add_class(c):
        if id(c) in seen:
            return
        seen.add(id(c))
        d = dict(vars(c))
        out.append((c, d))
        mod = c.__module__
        for v in d.values():
            if isinstance(v, type) and v.__module__ == mod:
                add_class(v)

    for name, m in list(sys.modules.items()):
        if m is None or not (name == "sqlglot" or name.startswith(("sqlglot.", "tests."))):
            continue
        d = dict(vars(m))
        out.append((m, d))
        if name.startswith("sqlglot"):
            for v in d.values():
                if isinstance(v, type) and v.__module__ == name:
                    add_class(v)
    return out


def shared_state_change(picture):
    """None when every module and class of the picture holds what it held: the same names,
    each bound to the same object (or to an equal one). Else the first name that differs.
    Like _shared_marks, this does not look into the dicts, lists and objects that the names
    are bound to."""
    for obj, saved in picture:
        try:
            if vars(obj) == saved:
                continue
        except Exception:  # noqa: BLE001  (a value that cannot be compared is a new value)
            pass
        now = vars(obj)
        names = [k for k in now if k not in saved] + [k for k, v in saved.items() if now.get(k, _MISSING) is not v]
        if names == ["__slotnames__"]:
            # copyreg caches the slot names of a class there when an instance is first pickled
            # or copied: a value that follows from the class alone
            saved["__slotnames__"] = now["__slotnames__"]
            continue
        return "%s.%s" % (getattr(obj, "__qualname__", None) or getattr(obj, "__name__", "?"), names[0] if names else "?")
    return None


def picture_of_tests(test_ids):
    """For a child that serves several commits: what the classes of these tests hold now. For
    each test class and its bases below unittest.TestCase: every class attribute, the items of
    the lists, dicts and sets among them, and the flags of a process pool stand-in (pool.py).
    None when a class is not loaded."""
    import pool as pool_module

    classes = []
    seen = set()
    keys = sorted({(tid.split("::")[0][:-3].replace("/", "."), tid.split("::")[1]) for tid in test_ids})
    for mod_name, cls_name in keys:
        cls = getattr(sys.modules.get(mod_name), cls_name, None)
        if not isinstance(cls, type):
            return None
        for c in cls.__mro__:
            if c is object or c.__module__.partition(".")[0] == "unittest" or id(c) in seen:
                continue
            seen.add(id(c))
            attrs = dict(vars(c))
            items = {}
            for k, v in attrs.items():
                if type(v) in (list, dict, set):
                    items[k] = type(v)(v)
                elif isinstance(v, pool_module.InlinePool):
                    items[k] = dict(vars(v))
            classes.append((c, attrs, items))
    return classes


def restore_tests(picture, graveyard):
    """Puts the classes back as picture_of_tests saw them. Returns the names of what differed,
    without the class state that every run changes (see _ROUTINE_CLASS_STATE): an empty list
    means that the tests left these classes as a new child would find them. Objects that
    leave a class go to 'graveyard', so that no finalizer runs here (a database connection,
    for instance)."""
    odd = []
    for c, attrs, items in picture:
        now = vars(c)
        for k in [k for k in now if k not in attrs]:
            graveyard.append(now[k])
            delattr(c, k)
            if k not in _ROUTINE_CLASS_STATE:
                odd.append("%s.%s" % (c.__name__, k))
        for k, v in attrs.items():
            cur = now.get(k, _MISSING)
            if cur is not v:
                graveyard.append(cur)
                setattr(c, k, v)
                if k not in _ROUTINE_CLASS_STATE:
                    odd.append("%s.%s" % (c.__name__, k))
        for k, copy in items.items():
            v = attrs[k]
            if type(v) in (list, dict, set):
                if not _same_items(v, copy):
                    graveyard.append(type(v)(v))
                    v.clear()
                    (v.extend if type(v) is list else v.update)(copy)
                    if k not in _ROUTINE_CLASS_STATE:
                        odd.append("%s.%s items" % (c.__name__, k))
            else:  # the process pool stand-in of the class
                flags = vars(v)
                if not _same_items(flags, copy):
                    if any(not _same_value(flags.get(f, _MISSING), copy.get(f, _MISSING))
                           for f in set(flags) | set(copy) if f not in _ROUTINE_POOL_STATE):
                        odd.append("%s.%s pool" % (c.__name__, k))
                    flags.clear()
                    flags.update(copy)
    return odd


def fixture_batch_child(batch, tests, wfd):
    """Child side of run_fixture_batch_child. batch: [(plan, tests to run, skip_after_fail)] of
    commits that change nothing but entries of .sql fixture files: their tests run on the
    pinned code, each test reduced to the changed entries of the commit's file. This child
    serves them one after the other, each as child_main does for such a commit (same steps
    in the same order, same options of the test runs), between "C i" and "E i" lines.

    After the tests of a commit, everything that the steps or the tests changed is put back:
    open() and os.listdir, the reduce state, the code of the reduced test methods, the dialect
    registry (each commit starts with the registry that this process had at its fork, as its
    own child would), and the attributes of the test classes (class fixture lists, the class
    cleanups that unittest ran, the flags of the class's pool). Then the checks: the
    interpreter settings are the ones before the first commit, the test classes hold what they
    held (but for the state named above), no thread or module stand-in changed, no pool died,
    no class setup failed, and the modules of sqlglot and of the tests and the classes of
    sqlglot are bound to the objects that they were bound to before the first commit
    (shared_state_picture). When a check fails, or the commit cannot be compared entry by
    entry ("N i"), the commit gets no "E i": the parent gives it a child of its own. A failed
    check ("X i") also ends this child, so the commits after it get their own child too."""
    import builtins
    import io
    import threading
    import warnings

    import runner

    emit, started, settings, watches, begin = child_reporters(wfd)
    try:
        if ANSWER_FD is not None:
            os.close(ANSWER_FD)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        if not DEBUG:
            os.dup2(devnull, 2)
        if DEBUG:
            import faulthandler

            faulthandler.enable(file=sys.stderr)
        import preload
        import reduce

        t_child = time.process_time()
        picture = picture_of_tests({tests[n]["id"] for _, order, _ in batch for n in order})
        dialect = sys.modules.get("sqlglot.dialects.dialect")
        if picture is None or dialect is None or not reduce.install_loaders():
            raise RuntimeError("fixture batch: tests or loaders are not as expected")
        registry = dialect.Dialect._classes
        registry_start = dict(registry)
        shared = shared_state_picture()
        t_shared = time.process_time() - t_child
        io_names = (builtins.open, io.open, io.open_code, os.listdir)
        filters = list(warnings.filters)
        threads = threading.active_count()
        pinned = None  # the interpreter settings before the tests of the first commit
        graveyard = []
        for i, (plan, order, skip_after_fail) in enumerate(batch):
            os.write(wfd, b"C %d\n" % i)
            odd = []
            ran = False
            reduce.SWAPPED = swapped = []
            stand_ins = None
            try:
                read_old = engine.install_open_overlay(ROOT, plan.files)
                patchrules.install_listing_overlay(ROOT, plan.files)
                # the parent built the shared class fixtures from the pinned files
                comparable = not any(failed for _, failed in preload.refresh_class_fixtures(ROOT, plan.files))
                # an unchanged fixture entry that does not run changed code cannot change a result
                changed = {}
                for rel in plan.files:
                    got = reduce.changed_entries(rel[len("tests/fixtures/"):], read_old)
                    if not got:
                        comparable = False  # not readable as entries: whole tests, in its own child
                    changed.update(got)
                if comparable:
                    reduce.STATE.keep_changed = changed
                    reduce.STATE.changed_only = True
                    preload.reduce_class_fixtures(changed)
                    keeps = None
                    if plan.keeps and not NO_REDUCE:
                        keeps = apply_keeps(plan, MAPS_OBJ, tests)
                    if PANDAS_USERS and any(n in PANDAS_USERS for n in order):
                        import lazymods

                        lazymods.load()  # before the settings snapshot below: as in the parent
                    stand_ins = [type(sys.modules.get(name)) for name in ("numpy", "pandas", "pandas.testing")]
                    begin()
                    now = _interpreter_settings()
                    if pinned is None:
                        pinned = now
                    if now != pinned:
                        odd.append("settings before the tests")
                    else:
                        settings[0] = now
                        result = runner.run_tests([(n, tests[n]["id"]) for n in order], emit, False, keeps, started,
                                                  skip_after_fail, plan.enders)
                        ran = True
                        if settings[1]:
                            odd.append("settings changed in a test")
                        if settings[2]:
                            odd.append("pool died")
                        if result.class_errors:
                            odd.append("class setup failed")
                        if runner._FAILED_IMPORTS:
                            odd.append("test module not imported")
            finally:
                # back to the state in which a new child would start
                reduce.STATE.keep_changed = None
                reduce.STATE.changed_only = False
                reduce.STATE.in_test = False
                reduce.STATE.reset()
                for f, code in reversed(swapped):
                    f.__code__ = code
                reduce.SWAPPED = None
                builtins.open, io.open, io.open_code, os.listdir = io_names
            if dialect.Dialect._classes is not registry:
                odd.append("dialect registry replaced")
            registry.clear()
            registry.update(registry_start)
            odd.extend(restore_tests(picture, graveyard))
            if pinned is not None and _interpreter_settings() != pinned:
                odd.append("settings after the tests")
            if stand_ins is not None and stand_ins != [type(sys.modules.get(name)) for name in ("numpy", "pandas", "pandas.testing")]:
                odd.append("numpy or pandas loaded in a test")
            if threading.active_count() != threads:
                odd.append("threads")
            if len(warnings.filters) != len(filters) or any(a is not b for a, b in zip(warnings.filters, filters)):
                odd.append("warning filters")
            t_same = time.process_time()
            change = shared_state_change(shared)
            if change is not None:
                odd.append("shared state: " + change)
            t_shared += time.process_time() - t_same
            if odd:
                os.write(wfd, ("X %d %s\n" % (i, "; ".join(odd)[:300])).encode())
                break
            os.write(wfd, b"%s %d\n" % (b"E" if ran else b"N", i))
        if DEBUG:
            sys.stderr.write("fixture batch child: %d commits, cpu %.3f (of that, the picture of the shared state and its checks: %.3f)\n"
                             % (len(batch), time.process_time() - t_child, t_shared))
        os.write(wfd, b"END\n")
    except BaseException as e:  # noqa: BLE001
        try:
            os.write(wfd, ("ERR %r\n" % (e,)).encode()[:400])
        except OSError:
            pass
    finally:
        os._exit(0)


def run_fixture_batch_child(batch, tests, time_left):
    """Runs the tests of several fixture-only commits in one forked child (fixture_batch_child).
    batch: [(plan, tests to run, allowed seconds per test, skip_after_fail, cap)], as run_child
    takes them for one commit. Returns {position in batch: (verdicts, tests not run after a
    failure, CPU seconds per test)} for the commits whose runs ended with every check passed
    ("E i"). A test that hangs, runs past the limits of run_child or kills the process ends
    the child: that commit and the ones after it have no result here, and each runs in its
    own child, as without this function."""
    if PRE_KEEPS:
        for plan, *_ in batch:
            if plan.pre_keeps is None or plan.pre_keeps[0] is not plan.keeps:
                prepare_keeps(plan, MAPS_OBJ, tests)
    r, w = os.pipe()
    release_pin_if_slow()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            os.setpgid(0, 0)
        except OSError:
            pass
        fixture_batch_child([(plan, order, skip) for plan, order, _, skip, _ in batch], tests, w)
        os._exit(0)
    os.close(w)
    done = {}
    record = None  # (verdicts, skipped, times) of the commit whose tests run now
    status = "DIED"
    buf = b""
    t_end = time.monotonic() + time_left
    t_test = time.monotonic() + SETUP_TIMEOUT * SLOWDOWN  # time for the child's own setup
    allowed, cap, t_cap = {}, None, None
    try:
        while True:
            now = time.monotonic()
            if now >= t_end or now >= t_test or (t_cap is not None and now >= t_cap):
                status = "TIMEOUT"
                break
            ready, _, _ = select.select([r], [], [], max(0.01, min(t_end, t_test, t_cap or t_end) - now))
            if not ready:
                continue
            chunk = os.read(r, 65536)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if line == b"END":
                    status = "END"
                elif line.startswith(b"ERR"):
                    status = "ERR"
                    log("fixture batch", line.decode(errors="replace"))
                elif line.startswith(b"C "):
                    # the next commit: the limits of its own tests, as in run_child
                    cur = int(line[2:])
                    record = ({}, set(), {})
                    allowed, cap, t_cap = batch[cur][2], batch[cur][4], None
                    t_test = time.monotonic() + SETUP_TIMEOUT * SLOWDOWN
                elif line.startswith(b"E "):
                    if record is not None:
                        done[int(line[2:])] = record
                    record, t_cap = None, None
                    t_test = time.monotonic() + SETUP_TIMEOUT * SLOWDOWN
                elif line.startswith(b"N "):
                    record, t_cap = None, None
                    t_test = time.monotonic() + SETUP_TIMEOUT * SLOWDOWN
                elif line.startswith(b"X "):
                    record, t_cap = None, None
                    log("fixture batch ends:", batch[int(line.split()[1])][0].commit, line.decode(errors="replace"))
                    t_test = time.monotonic() + SETUP_TIMEOUT * SLOWDOWN
                elif line.startswith(b"S "):
                    if t_cap is None and cap is not None:
                        t_cap = time.monotonic() + cap
                    t_test = time.monotonic() + allowed.get(int(line[2:]), 30.0)
                elif line[:1].isalpha():
                    # "G", "P" (the child ends the batch itself) or a line that such a commit
                    # never writes: no result for this commit
                    record = None
                else:
                    parts = line.split()
                    if record is not None:
                        if parts[1] == b"2":
                            record[1].add(int(parts[0]))
                        else:
                            record[0][int(parts[0])] = parts[1] == b"1"
                        if len(parts) > 2:
                            record[2][int(parts[0])] = int(parts[2]) / 1e6
                    t_test = time.monotonic() + 30.0
    finally:
        # as in run_child: the child must not run on, and no zombie may stay
        os.close(r)
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)
        try:
            while os.waitpid(-1, os.WNOHANG)[0] > 0:
                pass
        except ChildProcessError:
            pass
    if status != "END":
        log("fixture batch: child ended with status %s after %d of %d commits" % (status, len(done), len(batch)))
    return done


def run_child(plan, order, tests, index, allowed, time_left, skip_after_fail=(), on_diff=None, cap=None):
    """Runs the tests in a forked child. Returns (verdicts, status, hung test or None).

    status: END, REDO, ERR, TIMEOUT, DIED, OVER, COST (the CPU guard stopped the returned
    test, see plan.guard). allowed[n] = seconds that test n may take.
    cap: seconds that all the tests together may take from the start of the first one.
    on_diff: for a commit that changes import-time code. The child first reports which
    import-time values differ from the pinned tree; on_diff(names) then decides what to run."""
    if plan.mode == "hot" and PRE_KEEPS and (plan.pre_keeps is None or plan.pre_keeps[0] is not plan.keeps):
        prepare_keeps(plan, MAPS_OBJ, tests)
    r, w = os.pipe()
    plan_r = plan_w = None
    if on_diff is not None:
        plan_r, plan_w = os.pipe()
    release_pin_if_slow()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        if plan_w is not None:
            os.close(plan_w)
        try:
            os.setpgid(0, 0)
        except OSError:
            pass
        child_main(plan, order, tests, w, index, skip_after_fail, plan_r)
        os._exit(0)
    os.close(w)
    if plan_r is not None:
        os.close(plan_r)
    verdicts = {}
    status = "DIED"
    hung = None
    broken = plan.broken
    buf = b""
    t_end = time.monotonic() + time_left
    current = None
    t_test = time.monotonic() + SETUP_TIMEOUT * SLOWDOWN  # time for the child's own setup (imports)
    t_cap = None
    added_open = None  # the first later test of an added test that runs now
    stage_open = False  # the child is inside its stage check (run_stage_check)
    pending_keys = None  # COLL_RUN: the import state, until the child has imported every test file
    statm_fd = None
    if MEM_GUARD:
        try:
            statm_fd = os.open("/proc/%d/statm" % pid, os.O_RDONLY)
        except OSError:
            pass
    t_mem = time.monotonic() + MEM_POLL
    try:
        while True:
            now = time.monotonic()
            if now >= t_end:
                status = "TIMEOUT"
                break
            if t_cap is not None and now >= t_cap:
                status = "OVER"  # the runs cost far more than planned: the rest is selected unrun
                break
            if now >= t_test:
                status = "TIMEOUT"
                hung = current
                break
            if statm_fd is not None and now >= t_mem:
                t_mem = now + MEM_POLL
                why = memory_over(pid, statm_fd)
                if why is not None:
                    log(plan.commit, "memory guard:", why, "test", current)
                    plan.note.append("memory guard: %s" % why)
                    status = "COST" if current is not None else "DIED"
                    hung = current
                    break
            wait = max(0.01, min(t_end, t_test, t_cap or t_end) - now)
            ready, _, _ = select.select([r], [], [], min(wait, MEM_POLL) if statm_fd is not None else wait)
            if not ready:
                continue
            chunk = os.read(r, 65536)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if line in (b"END", b"REDO", b"REDOC"):
                    status = line.decode()
                elif line.startswith(b"ERR"):
                    status = "ERR"
                    log(plan.commit, line.decode(errors="replace"))
                elif line.startswith(b"S "):
                    current = int(line[2:])
                    if t_cap is None and cap is not None:
                        t_cap = time.monotonic() + cap
                    t_test = time.monotonic() + allowed.get(current, 30.0)
                elif line.startswith(b"B "):
                    broken.append(line[2:].decode())
                elif line.startswith(b"K "):
                    broken.append(line[2:].decode())
                    plan.class_broken.append(line[2:].decode())
                elif line.startswith(b"P "):
                    plan.pool_died.append(int(line[2:]))
                elif line.startswith(b"W "):
                    n, _, rest = line[2:].partition(b" ")
                    key, entries = json.loads(rest)
                    plan.leaks.append((int(n), key, entries))
                elif line.startswith(b"H "):
                    plan.watched = json.loads(line[2:])
                elif line.startswith(b"M "):
                    n = int(line[2:])
                    if n >= 0:
                        plan.must.add(n)  # the token streams of this held test changed
                        plan.note.append("token streams of %d changed" % n)
                    t_test = time.monotonic() + SETUP_TIMEOUT * SLOWDOWN
                elif line.startswith(b"G "):
                    n = int(line[2:])
                    plan.global_leak = n if plan.global_leak is None else min(plan.global_leak, n)
                elif line == b"X start":
                    # the stage check of a held long test begins: every test of the commit has
                    # reported, so the cap on their runs is over; the check has its own limits
                    stage_open = True
                    t_cap = None
                    cap = None
                    t_test = time.monotonic() + SETUP_TIMEOUT * SLOWDOWN
                elif line.startswith(b"X "):
                    plan.stage_info = json.loads(line[2:])
                    stage_open = False
                    t_test = time.monotonic() + 30.0
                elif line == b"A ok":
                    added_open = None
                elif line.startswith(b"A "):
                    # an added test runs (see run_added_tests): no time cap from here, but a
                    # run that does not end counts as a change for every later test
                    added_open = int(line[2:])
                    t_cap = None
                    cap = None
                    t_test = time.monotonic() + 2 * TEST_TIMEOUT * SLOWDOWN
                elif line.startswith(b"D ") or line == b"T":
                    if line != b"T":
                        keys = None if line == b"D many" else json.loads(line[2:])
                        if DEBUG:
                            DEBUG_DKEYS[plan.commit] = keys
                        if COLL_RUN:
                            pending_keys = [keys]  # the plan waits for the test-file imports ("T")
                            t_test = time.monotonic() + SETUP_TIMEOUT * SLOWDOWN
                            continue
                    else:
                        if pending_keys is None:
                            continue
                        keys = pending_keys[0]
                        pending_keys = None
                    todo = on_diff(keys)
                    allowed = todo.pop("allowed")
                    cap = todo.pop("cap")
                    data = marshal.dumps(todo)
                    while data:
                        data = data[os.write(plan_w, data):]
                    os.close(plan_w)
                    plan_w = None
                    t_test = time.monotonic() + SETUP_TIMEOUT * SLOWDOWN
                else:
                    parts = line.split()
                    if parts[1] == b"2":
                        plan.skipped.add(int(parts[0]))
                    else:
                        verdicts[int(parts[0])] = parts[1] == b"1"
                    if len(parts) > 2:
                        plan.times[int(parts[0])] = int(parts[2]) / 1e6
                    if len(parts) > 3:
                        plan.body[int(parts[0])] = int(parts[3]) / 1e6
                    current = None
                    t_test = time.monotonic() + 30.0
    finally:
        # also when this process fails while it talks to the child: the child must not run on
        os.close(r)
        if statm_fd is not None:
            os.close(statm_fd)
        if plan_w is not None:
            os.close(plan_w)
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _, wstatus = os.waitpid(pid, 0)
        # In the container this process is PID 1: a process that a test started (a subprocess)
        # and that outlived the child is now this process's child. Its zombie would hold a
        # PID of the container's limit (256) until reaped.
        try:
            while os.waitpid(-1, os.WNOHANG)[0] > 0:
                pass
        except ChildProcessError:
            pass
    if added_open is not None:
        plan.global_leak = added_open if plan.global_leak is None else min(plan.global_leak, added_open)
        plan.note.append("added test did not end")
    if stage_open:
        # The child ended inside the stage check (its CPU limit, a run that does not end, a
        # crash). The check comes after the runs of the commit, so their verdicts are complete:
        # the commit ends as without the check, and the check gives no verdict.
        log(plan.commit, "stage check stopped:", status, wstatus)
        plan.stage_info = {"n": plan.stage["n"] if plan.stage else None, "verdict": None, "stopped": status}
        status, hung = "END", None
    elif status == "DIED" and current is not None and os.WIFSIGNALED(wstatus) and os.WTERMSIG(wstatus) == signal.SIGPROF \
            and current in plan.guard:
        # the CPU guard stopped this test (see child_reporters): it has no verdict, and the
        # caller selects it
        status = "COST"
        hung = current
        log(plan.commit, "test", current, "stopped after", round(plan.guard[current], 2), "CPU seconds")
    elif status == "DIED":
        log(plan.commit, "child died, wait status", wstatus, "after", len(verdicts), "tests")
        if current is not None:
            hung = current  # the test that killed the process fails in a full run too
    return verdicts, status, hung


def classmap_readers(keys, maps: Maps):
    """The tests that can read a changed value, by the class attribute map, or None when the
    map cannot tell.

    It can tell when every changed key is a class attribute that is not a function, of a class
    of the traced trees (tokenizers, parsers, generators, dialects), or an attribute of an
    object that such a class attribute holds. A test can then see the change only by reading
    that attribute through an object or class object whose class inherits from the changed
    class; the trace of the pinned tree lists those reads for every test. The keys are those
    of the import-state diff, so values that the metaclass or __init_subclass__ derive from
    the edited one are keys of their own."""
    cm = maps.classmap()
    if cm is None or not keys:
        return None
    classes = cm["classes"]
    out = set(cm["unknown"])  # tests whose reads are not known
    receivers = None
    for key, is_function in (row[:2] for row in keys):
        if is_function:
            return None
        if key[0] == "A" and len(key) == 4:
            owner, name = "%s:%s" % (key[1], key[2]), key[3]
        elif key[0] == "O" and len(key) == 5:
            owner, name = "%s:%s" % (key[1], key[2]), key[3]
        else:
            return None  # a module global, or deeper
        if not isinstance(name, str) or name[:2] == "__" or name in cm["fnames"]:
            return None
        if owner not in cm["hooked"]:
            return None
        if receivers is None:
            receivers = []
            for raw in cm["receivers"]:
                arr = array("I")
                arr.frombytes(raw)
                receivers.append({classes[i] for i in arr})
        raw = cm["names"].get(name)
        if not raw:
            continue  # no test reads this name on any class of the trees
        arr = array("I")
        arr.frombytes(raw)
        for i in range(0, len(arr), 2):
            if owner in receivers[arr[i]]:
                s = array("H")
                s.frombytes(cm["sets"][arr[i + 1]])
                out.update(s)
    return out


def classmap_cut(plan: Plan, keys, maps: Maps):
    """The tests that can see the commit's import-time change, or None (no cut). Only for a
    commit whose edits change nothing but values that the import-state diff names: no function
    body that runs at import (fpairs), no function header, no removed name, no edit that the
    parent applies in memory."""
    if (not keys or NO_CLASSMAP or plan.fpairs or plan.stmt_lines or plan.gone or plan.work
            or _changes_def_header(plan)):
        return None
    return classmap_readers(keys, maps)


def _changes_def_header(plan: Plan):
    """True when an edited multi-line statement at module or class level is a function or
    class definition (its header: defaults, decorators). Other multi-line statements (a
    constant written over several lines) run only at import."""
    for path, lines in plan.def_units.items():
        try:
            with open(os.path.join(ROOT, path), encoding="utf-8") as f:
                src = f.read().split("\n")
        except OSError:
            return True
        for ln in lines:
            if 0 < ln <= len(src) and src[ln - 1].lstrip().startswith(("def ", "async def ", "@", "class ")):
                return True
    return False


def _edits_own_definition(plan: Plan, key):
    """True when the pinned function that a changed key of the import-state diff names lies
    inside a statement that the patch edits and whose lines plan.def_units holds one by one.
    Then every test part that called the function under this name in the pinned runs ran
    lines that the reduced runs keep.

    key: ("G", module, name) or ("A", module, class path, name). The function is looked up
    in this process, which holds the pinned import: the plain function under classmethod,
    staticmethod or a bound method; for a wrapper whose own code lies elsewhere, the
    function that it keeps as __wrapped__; of a property, every accessor must lie in such
    a statement. Anything else gives False (whole runs)."""
    m = sys.modules.get(key[1]) if len(key) > 2 and isinstance(key[1], str) else None
    if m is None:
        return False
    d = vars(m)
    if key[0] == "A" and len(key) == 4 and isinstance(key[2], str):
        for part in key[2].split("."):
            obj = d.get(part)
            if not isinstance(obj, type):
                return False
            d = obj.__dict__
    elif not (key[0] == "G" and len(key) == 3):
        return False
    todo = [d.get(key[-1])]
    found = 0
    steps = 0
    while todo:
        steps += 1
        if steps > 16:
            return False
        v = todo.pop()
        if isinstance(v, (classmethod, staticmethod, types.MethodType)):
            todo.append(v.__func__)
        elif isinstance(v, property):
            todo.extend(f for f in (v.fget, v.fset, v.fdel) if f is not None)
        elif isinstance(v, types.FunctionType):
            code = v.__code__
            rel = os.path.relpath(code.co_filename, ROOT)
            if code.co_firstlineno in plan.def_units.get(rel, ()):
                found += 1  # every caller runs lines of this code
                continue
            # a wrapper made by a decorator elsewhere: the function that it wraps decides
            inner = getattr(v, "__wrapped__", None)
            if inner is None:
                return False
            todo.append(inner)
        else:
            return False
    return found > 0


_GETATTR_SPANS = {}


def _module_getattr_span(rel):
    """First and last line of the module-level "def __getattr__" of a pinned file, or None."""
    if rel not in _GETATTR_SPANS:
        span = None
        text = patchrules.source(rel)
        if "def __getattr__" in text:
            import ast

            try:
                for node in ast.parse(text).body:
                    if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                        span = (node.lineno, node.end_lineno)
            except (SyntaxError, ValueError):
                pass
        _GETATTR_SPANS[rel] = span
    return _GETATTR_SPANS[rel]


def _lazy_attr_users(maps: Maps, rel):
    """Tests that can hold a name bound through the module __getattr__ of 'rel': the tests of
    the test files whose source imports from that module ("from m import x", "from parent
    import m", "import m")."""
    import re

    mod = rel[:-3].replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    parent, _, last = mod.rpartition(".")
    pats = [r"^\s*from\s+%s\s+import\b" % re.escape(mod), r"^\s*import\s+%s\b" % re.escape(mod)]
    if parent:
        pats.append(r"^\s*from\s+%s\s+import\s+[^\n]*\b%s\b" % (re.escape(parent), re.escape(last)))
    pat = re.compile("|".join(pats), re.M)
    out = set()
    for trel, numbers in patchrules.test_files(maps).items():
        if pat.search(patchrules.source(trel)):
            out.update(numbers)
    # A sqlglot module that binds a lazy name at its own import (sqlglot/lineage.py, "from
    # sqlglot.optimizer import qualify") hands it to the tests that call it; those tests load
    # it from test files that import the same package in the cases seen so far. The importer
    # sets of sqlglot modules hold nearly every test (the parent imports everything), so they
    # are not used here: an open risk, see the report.
    return out


def refine_with_diff(plan: Plan, keys, maps: Maps, tests):
    """Uses the import-state diff of a commit that changes import-time code.

    keys: the module globals and class attributes whose value differs from the pinned tree
    (None = too many). Nothing differs and no function body changed: the Python edits do
    nothing. Otherwise a test part can see the change only when it runs a line that reads
    one of the changed names (or a changed function): the reduced runs apply to those lines."""
    if keys is None:
        plan.note.append("import state: many changes")
        plan.lines = {"?": None}
        return
    # A patch can hold, beside its import-time edit, an edit in call-time code of another
    # function (planned by plan_python_file: plan.work, and the lines of the changed
    # statements in plan.lines). The import-state diff says nothing about which tests run
    # those statements, so their lines stay among the lines of the reduced runs below.
    work_files = {full for full, _ in plan.work}
    call_lines = {rel: set(lns) for rel, lns in plan.lines.items()
                  if lns and os.path.join(ROOT, rel) in work_files}
    names = set()
    whole = bool(plan.gone)  # a removed name can fail any line that mentions it
    owners = set()
    for row in keys:
        key, is_function = row[0], row[1]
        owners.add("%s.%s" % (key[1], key[2]) if key[0] == "A" else None)
        name = key[-1]
        if name in ("__bases__", "__slots__") or not isinstance(name, str):
            whole = True
        elif is_function:
            # A function is also reached by computed names (dispatch tables). The reduced
            # runs are safe for it only when the patch changed that function's own header:
            # then every user runs its body, and the body lines are known. That is checked
            # for this very function: an edited definition elsewhere in the patch says
            # nothing about it. (So a function whose body the patch changed in call-time
            # code, beside an import-time edit, gives whole runs.)
            if not _edits_own_definition(plan, key):
                whole = True
        else:
            names.add(name)
    plan.diff = sorted(names)
    if not whole:
        # Only class constants changed: the candidates that cannot read any of them drop out.
        # Tests that read a changed file as data stay.
        readers = classmap_cut(plan, keys, maps)
        if readers is not None:
            before = len(plan.cands)
            plan.cands &= readers | plan.data_cands
            plan.note.append("class map: %d -> %d" % (before, len(plan.cands)))
    if not whole and not NO_KEYMAP:
        # Only entries of traced tables changed: a candidate that looked up none of them and
        # never used one of those tables as a whole gets the same answers (keymap.py).
        readers = keymap.cut(plan, keys, MAPS, ROOT)
        if readers is not None:
            before = len(plan.cands)
            plan.cands &= readers | plan.data_cands
            plan.note.append("key map: %d -> %d" % (before, len(plan.cands)))
    if not keys and not plan.fpairs and not plan.work:
        # Nothing that the import leaves behind differs. The edit very likely does nothing;
        # the cheapest candidates still run as a check, and nothing is selected unrun.
        plan.note.append("import state unchanged")
        plan.lines = {"?": None}
        plan.keeps = compute_keeps(plan, maps, tests)
        return
    lines = {}
    for name in names:
        rs = maps.readers(name)
        if not rs or len(rs) > 600:
            whole = True  # read in an unknown way, or read everywhere
            break
        for path, ln in rs:
            lines.setdefault(path, set()).add(ln)
            span = _module_getattr_span(path)
            if span and span[0] <= ln <= span[1]:
                # A changed name that the module __getattr__ reads (lazy attributes, PEP 562)
                # changes what "from module import x" binds. Those imports run at the import
                # of test modules and of other sqlglot modules, and the result is stored in the
                # module globals, so the line maps show only the first lookup: every test that
                # loads the module is a candidate, and it runs whole.
                added = _lazy_attr_users(maps, path)
                plan.cands.update(added)
                plan.note.append("lazy __getattr__ reads %s: %d tests" % (name, len(added)))
                whole = True
    if whole or not FRESH_REDUCE:
        plan.note.append("import state: no reduced runs")
        plan.lines = {"?": None}
        plan.keeps = compute_keeps(plan, maps, tests)
        return
    # a reader line stands for its whole statement, as for edits in function bodies
    full = {}
    for path, lns in lines.items():
        rec = maps.file(path)
        acc = full.setdefault(path, set())
        for ln in lns:
            if rec is not None and ln < len(rec[9]) and rec[9][ln]:
                acc.update(range(rec[9][ln], rec[10][ln] + 1))
            else:
                acc.add(ln)
    for src in (plan.stmt_lines, plan.def_units, call_lines):
        for path, lns in src.items():
            full.setdefault(path, set()).update(lns)
            plan.plain_lines.update((path, ln) for ln in lns)
    # Every changed value is an attribute of a known class: a part that creates no object of
    # such a class (or of a dialect that hands the class out) cannot read the new value.
    if owners and None not in owners and CLASS_FILTER:
        wanted = set()
        for owner in owners:
            users = maps.class_users(owner)
            if users is None:
                wanted = None
                break
            wanted.update(users)
        plan.class_filter = wanted
    plan.lines = full
    plan.note.append("import state: %d names" % len(names))
    if plan.infect_fresh is not None and not keys and not plan.gone and not plan.added and INFECT_FRESH:
        # the import left nothing different, and the edit is one mapped comparison: the
        # candidates shrink to the tests that saw an evaluation whose result changes
        before = len(plan.cands)
        plan.cands &= plan.infect_fresh | plan.data_cands
        plan.note.append("infect after import: %d -> %d" % (before, len(plan.cands)))
    plan.keeps = compute_keeps(plan, maps, tests)
    if plan.unaffected:
        plan.cands -= plan.unaffected
        plan.note.append("unaffected %d" % len(plan.unaffected))


def estimate_cost(n, plan: Plan, maps: Maps, seconds):
    """Expected CPU seconds of running test n for this commit (whole or reduced)."""
    whole = maps.cost[n] if maps.cost else seconds[n] * 1.5
    if n in plan.fixture_users:
        # the test walks a list of fixture entries, and the run keeps only the changed
        # entries (preload.reduce_class_fixtures): a typo changes one entry, or a few
        return min(whole, 0.05 + 0.05 * whole)
    keep = plan.keeps.get(n)
    if keep is None:
        return whole
    scaled, chosen = keep[3], keep[4]
    if chosen is None:
        return min(whole, 0.15) if scaled else whole
    trec = maps.test_units(n)
    traced_whole = trec[5] or 0.0
    if traced_whole <= 0:
        return whole
    base = trec[6]
    part = base + sum(max(0.0, trec[7][u] - base) for u in chosen)
    cost = whole * min(1.0, part / traced_whole)
    return min(cost, 0.15) if scaled else cost


# ---------------------------------------------------------------------------
# Packing


def pack(numbers, seconds):
    """Splits tests over 1..8 workers to minimise (largest load + 2 s per worker)."""
    if not numbers:
        return []
    items = sorted(numbers, key=lambda n: -seconds[n])
    best = None
    for k in range(1, min(8, len(items)) + 1):
        loads = [0.0] * k
        bins = [[] for _ in range(k)]
        for n in items:
            i = loads.index(min(loads))
            loads[i] += seconds[n]
            bins[i].append(n)
        cost = max(loads) + WORKER_COST * k
        if best is None or cost < best[0] - 1e-9:
            best = (cost, bins)
    return best[1]


def packed_bound(total, longest):
    """The least that a selection can cost: with k workers the largest load is at least the
    longest test and at least total / k. For many short tests pack() comes close to it, and
    it needs no list of tests (marginal_holds asks it once per candidate)."""
    if total <= 0.0:
        return 0.0
    return min(max(longest, total / k) + WORKER_COST * k for k in range(1, 9))


def packed_seconds(numbers, seconds):
    """What a selection costs: the largest worker load of pack() plus 2 s per worker."""
    bins = pack(sorted(numbers), seconds)
    if not bins:
        return 0.0
    return max(sum(seconds[n] for n in b) for b in bins) + WORKER_COST * len(bins)


# ---------------------------------------------------------------------------


def watch_readers(plan: Plan, verdicts, n_tests):
    """The tests that a watched shared container (plan.watch) can reach after a test left it
    changed (plan.leaks), or after a candidate that did not run and so may have changed it:
    the later readers of the changed entries by the key read map (keymap.bin), or of any entry
    when the entries are not known; every later test when the map does not know the table. A
    reader that ran in the child after the change has its own verdict."""
    km = keymap.load(MAPS) if keymap is not None else None
    points = [(n, key, entries, True) for n, key, entries in plan.leaks]
    unrun = sorted(n for n in plan.cands if n not in verdicts)
    if unrun:
        if plan.watched is None:
            points.append((unrun[0], None, None, False))
        for key in plan.watched or ():
            points.append((unrun[0], key, None, False))
    # A class table that a subclass copies at import ({**Parser.X, ...} in class X of the
    # subclass) copies the changed entries too when the full run imports that module after the
    # change (dialect modules load at their first use; this process loaded them all before):
    # so the readers of the same attribute in every class table count.
    same_name = {}
    if km is not None:
        for h in km.holders:
            parts = h.split("|")
            if parts[0] == "A":
                same_name.setdefault(parts[-1], []).append(h)
    out = set()
    for n, key, entries, ran in points:
        own = derived = None  # readers of the changed table; of the same attribute elsewhere
        if km is not None and key is not None:
            key = tuple(key)
            name = keymap.holder_name(key)
            own, derived = set(km.unknown), set()
            for h in [name] + (sorted(set(same_name.get(key[-1], ())) - {name}) if key[0] == "A" else []):
                tid = km.holders.get(h)
                if tid is None:
                    if h == name:
                        own = None  # the changed table itself is not in the map
                        break
                    continue
                table = km.tables[tid]
                got = own if h == name else derived
                for sk in ([keymap.WHOLE] + list(entries)) if entries is not None else list(table):
                    sid = table.get(sk)
                    if sid is not None:
                        got.update(km.tests(sid))
        if own is None:
            out.update(range(n + 1, n_tests))
            continue
        # a reader of the changed table that ran after the change has its own verdict; one of a
        # copy does not (here the copy was made before the change)
        out.update(t for t in own if t > n and not (ran and t in verdicts))
        out.update(t for t in derived if t > n)
    if out:
        plan.note.append("watched container changed: %d later readers" % len(out))
    return out


def run_order(cands, tests):
    """The order of tests.json, which is the order of a full pytest run: some tests
    leave global state behind (for instance the dialect registry), so order matters."""
    return sorted(cands)


_PIN = None  # (wall clock at the pin, CPU seconds at the pin, the CPUs allowed before it) while pinned


def pin_to_one_cpu():
    """Keeps this process and its children on the CPU that it runs on now. They work one after
    the other, so one CPU is enough, and moves between CPUs cost cache and CPU time
    (measured: about 10% less CPU for a batch). BR_PIN=0 turns it off."""
    global _PIN
    if os.environ.get("BR_PIN", "1") == "0":
        return
    try:
        with open("/proc/self/stat") as f:
            cpu = int(f.read().rsplit(")", 1)[1].split()[36])
        allowed = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {cpu} if cpu in allowed else {min(allowed)})
        _PIN = (time.monotonic(), sum(os.times()[:4]), allowed)
    except (OSError, ValueError, IndexError, AttributeError):
        pass


def pin_carry(cpu_before):
    """In a process forked from the pinned one that starts children of its own (the cone
    holder): its CPU clock starts at zero, so the CPU seconds that the batch had used before
    the fork are added to what release_pin_if_slow compares the wall time with."""
    global _PIN
    if _PIN is not None:
        _PIN = (_PIN[0], _PIN[1] - cpu_before, _PIN[2])


def release_pin_if_slow():
    """Called before each child starts. On one CPU the wall time of the batch is about its CPU
    time. When the wall time runs away from it, another program holds this CPU: then all CPUs
    are allowed again, so that the batch cannot run out of wall time."""
    global _PIN
    if _PIN is None:
        return
    wall = time.monotonic() - _PIN[0]
    cpu = sum(os.times()[:4]) - _PIN[1]
    if wall > 2.0 * cpu + 2.0:
        try:
            os.sched_setaffinity(0, _PIN[2])
        except OSError:
            pass
        log("pin released: wall", round(wall, 2), "cpu", round(cpu, 2))
        _PIN = None


def main():
    pin_to_one_cpu()
    t_start = time.monotonic()
    deadline = t_start + wall_limit()
    # Only the answers may reach stdout. Everything else that writes to fd 1 goes to stderr.
    global ANSWER_FD
    answer_fd = ANSWER_FD = os.dup(1)
    os.dup2(2, 1)
    state = {"answers": {}, "commits": [], "tests": None}
    try:
        run(t_start, deadline, state)
    except BaseException as e:  # noqa: BLE001
        log("selector failed:", repr(e))
        import traceback

        traceback.print_exc(file=sys.stderr)
    # every commit gets a line; a commit without an answer gets every test (never a miss)
    tests = state["tests"]
    commits = state["commits"]
    if tests is None:
        with open(os.path.join(DATA, "tests.json")) as f:
            tests = json.load(f)["tests"]
    if not commits:
        commits = sorted(p[:-6] for p in os.listdir(os.path.join(DATA, "commits")) if p.endswith(".patch"))
    seconds = [t["seconds"] for t in tests]
    number = state.get("number")  # set when tests.json is not the one of the maps (align_tests)
    if number is None and any(t.get("number", i) != i for i, t in enumerate(tests)):
        number = [t.get("number", i) for i, t in enumerate(tests)]
    extra = state.get("extra") or []
    everything = None
    out = []
    for c in commits:
        workers = state["answers"].get(c)
        if workers is None:
            if everything is None:
                everything = pack(list(range(len(tests))), seconds)
            workers = everything
        if number is not None:
            workers = [[number[n] for n in w if number[n] is not None] for w in workers]
            workers = [w for w in workers if w]
        if extra:
            workers = [workers[0] + extra] + workers[1:] if workers else [list(extra)]
        out.append(json.dumps({"commit": c, "workers": workers}))
    data = ("\n".join(out) + "\n").encode() if out else b""
    while data:
        n = os.write(answer_fd, data)
        data = data[n:]
    cg = ""
    try:
        with open("/sys/fs/cgroup/cpu.stat") as f:
            cg = " ".join(f.read().split("\n")[:3])
    except OSError:
        pass
    log("done in", round(time.monotonic() - t_start, 2), "s wall; cpu", round(sum(os.times()[:4]), 2), cg)
    # the answers are written (os.write): skip the interpreter teardown (module cleanup, duckdb
    # and nanobind finalizers), which only costs CPU
    sys.stderr.flush()
    os._exit(0)


def base_state_header():
    """What a saved base state depends on besides the pinned tree (see build_base_state.py)."""
    import zlib

    # the fingerprint code itself too: a saved state from another version of statefp.py
    # would compare unequal everywhere
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "statefp.py"), "rb") as f:
        code_version = zlib.crc32(f.read())
    return (sys.version, os.environ.get("PYTHONHASHSEED"), hash("blast radius base state probe"),
            hash(("G", "sqlglot", 1, 2.5, b"x", None)), code_version)


def resolve_base_objects(okeys):
    """{key: the live dict, set or frozenset that the statefp key names} for the table keys of a
    saved base state, from the loaded (pinned) sqlglot modules; None when one does not resolve."""
    out = {}
    missing = object()
    for key in okeys:
        m = sys.modules.get(key[1])
        if m is None:
            return None
        if key[0] == "G" and len(key) == 3:
            v = vars(m).get(key[2], missing)
        elif key[0] == "A" and len(key) == 4:
            d = vars(m)
            for part in key[2].split("."):
                obj = d.get(part)
                if not isinstance(obj, type):
                    return None
                d = obj.__dict__
            v = d.get(key[3], missing)
        else:
            return None
        if type(v) not in (dict, set, frozenset):
            return None
        out[key] = v
    return out


def load_base_state():
    """(state, function keys, table objects) of the pinned import from the map that
    build_base_state.py made at image build time, or None (then the snapshot is computed)."""
    if os.environ.get("BR_NO_BASE_MAP") or os.environ.get("PYTHONHASHSEED") != "0":
        return None
    import preload

    for d in (MAPS, preload.EXEC_CACHE, os.path.join(HERE, "cache")):
        try:
            with open(os.path.join(d, "base_state.bin"), "rb") as f:
                data = marshal.loads(f.read())  # (marshal.load reads a file object in small pieces)
        except (OSError, ValueError, EOFError, TypeError):
            continue
        try:
            if tuple(data["header"]) != base_state_header():
                log("base state map: other interpreter or hash seed")
                return None
            objects = resolve_base_objects(data["okeys"])
            if objects is None:
                log("base state map: a table key does not resolve")
                return None
            return data["state"], frozenset(data["fkeys"]), objects
        except (KeyError, TypeError, ValueError, IndexError):
            return None
    return None


def align_tests(tests, ids, state):
    """The maps number the tests by their position in the tests.json of the pinned commit.
    When /data/tests.json has the same ids in the same order, with those numbers, nothing
    changes (the normal case). Otherwise the tests are put in the order of the maps:
    state["number"] holds the number to print for each of them, and state["extra"] the
    numbers of tests that the maps do not know, which are selected for every commit."""
    if [t["id"] for t in tests] == list(ids) and all(t.get("number", i) == i for i, t in enumerate(tests)):
        return tests
    log("tests.json differs from the maps: tests are matched by id")
    given = {t["id"]: (t, t.get("number", i)) for i, t in enumerate(tests)}
    known = set(ids)
    state["extra"] = [number for tid, (t, number) in given.items() if tid not in known]
    state["number"] = [given[tid][1] if tid in given else None for tid in ids]
    return [
        dict(given[tid][0]) if tid in given else {"id": tid, "file": tid.split("::")[0], "seconds": 0.0}
        for tid in ids
    ]


def run(t_start, deadline, state):
    with open(os.path.join(DATA, "tests.json")) as f:
        tests = json.load(f)["tests"]
    seconds = [t["seconds"] for t in tests]
    commits = sorted(p[:-6] for p in os.listdir(os.path.join(DATA, "commits")) if p.endswith(".patch"))
    state["tests"] = tests
    state["commits"] = commits
    global MAPS_OBJ, REGISTRY_CLEAR_AT, REGISTRY_START
    try:
        with open(os.path.join(MAPS, "registry_start.json")) as f:
            REGISTRY_START = frozenset(json.load(f))
    except (OSError, ValueError):
        REGISTRY_START = None
    maps = MAPS_OBJ = Maps(MAPS)
    global CONE_MAP
    try:
        with open(os.path.join(MAPS, "cone.json")) as f:
            cone_json = json.load(f)
        if not NO_CONE:
            CONE_MAP = cone_json["modules"]
            CONE_WALK[:] = cone_json.get("walk") or []
    except (OSError, ValueError, KeyError):
        CONE_MAP = None
    tests = align_tests(tests, maps.meta["test_ids"], state)
    seconds = [t["seconds"] for t in tests]
    state["tests"] = tests
    REGISTRY_CLEAR_AT = registry_clear_at([t["id"] for t in tests])
    all_tests = list(range(len(tests)))

    plans = {}
    for c in commits:
        try:
            with open(os.path.join(DATA, "commits", c + ".patch"), encoding="utf-8", errors="surrogateescape") as f:
                text = f.read()
            release_pin_if_slow()
            plans[c] = make_plan(c, text, maps)
            plans[c].keeps = compute_keeps(plans[c], maps, tests)
        except Exception as e:  # noqa: BLE001
            log(c, "plan failed", repr(e))
            p = plans[c] = Plan(c)
            p.mode = "all"
    log("planned", len(plans), "in", round(time.monotonic() - t_start, 3))

    if os.environ.get("BR_PLAN_ONLY"):
        with open(os.environ["BR_PLAN_ONLY"], "w") as f:
            json.dump({c: {"mode": p.mode, "cands": sorted(p.cands), "note": p.note, "files": sorted(p.files),
                           "iters": {str(n): len(v[0][0]) for n, v in p.iters.items()},
                           "infect_fresh": None if p.infect_fresh is None else sorted(p.infect_fresh)}
                       for c, p in plans.items()}, f)
        return

    import preload

    preload.install_environment(ROOT)
    answers = state["answers"]
    debug_rows = {}
    tests_of_file = {}
    for n, t in enumerate(tests):
        tests_of_file.setdefault(t["file"], []).append(n)

    fixture_readers = {}

    def fixtures_of(n):
        """The .sql fixture files that test n reads."""
        if not fixture_readers:
            for path, sid in maps.meta["opened"].items():
                if path.startswith("tests/fixtures/") and path.endswith(".sql"):
                    for t in maps.tests(sid):
                        fixture_readers.setdefault(t, set()).add(path)
            fixture_readers.setdefault(-1, set())
        return fixture_readers.get(n) or frozenset()

    def long_by_evidence(evidence, held, more=0):
        """The long held tests that the tests in evidence (failing tests) select: those of the
        same test class or the same .sql fixture file, or all of them with many failures.
        more: failing tests that may exist unseen (see marginal_holds)."""
        class_of = lambda n: tests[n]["id"].rsplit("::", 1)[0]
        fail_classes = {class_of(n) for n in evidence}
        fail_fixtures = set()
        for n in evidence:
            fail_fixtures.update(fixtures_of(n))
        return {
            n for n in held if seconds[n] > QUIET_HOLD_SECONDS and (
                class_of(n) in fail_classes or fail_fixtures & fixtures_of(n) or len(evidence) + more >= MANY_FAILURES)
        }

    def finish(c, plan, verdicts, status, ran, held, blind=(), dry=False):
        failing = {n for n, bad in verdicts.items() if bad}
        # tests that the CPU guard stopped (and tests left unrun after that) have no verdict:
        # they are selected, and for every rule below they count like failing tests
        suspects = set(plan.suspects)
        selected = set(failing)
        selected.update(suspects)
        failing = failing | suspects
        selected.update(blind)
        selected.update(plan.blind)
        unknown = [n for n in ran if n not in verdicts and n not in set(blind) and n not in suspects]
        if unknown:
            selected.update(unknown)  # the child ended early: nothing is known about these
        if plan.all_fail:
            selected = set(all_tests)
        # Tests that were too costly to run have no verdict. They are selected, with one
        # exception: a very long test (the TPC-DS run, for instance) is left out unless a
        # failing test is close to it (same test class, or same fixture file) or many tests fail.
        # (the stage check, when it ran for a held long test: on "passes" the test is not held
        # any more for the rules about verdicts and evidence; on "fails" it is selected)
        info = plan.stage_info
        held_all = held
        if info and info.get("verdict") is False and not plan.all_fail:
            held = [n for n in held if n != info["n"]]
        if "import state unchanged" in plan.note and not failing and not plan.broken and not unknown:
            # the import left nothing different and the cheapest candidates pass; a held test
            # that reads a data file of the patch (a fixture edited beside the code) can still
            # fail, so it stays selected
            selected.update(n for n in held if n in plan.data_cands)
        elif (plan.broken and not (plan.coll_dead and set(plan.broken) <= plan.coll_names)) or not verdicts or unknown:
            selected.update(held)
        else:
            # On labeled mutants, a long test fails together with another test of its class
            # (or of its fixture file) in nearly every case, and almost never without one.
            # The tests of its class did run (see protect below), so their verdicts decide.
            selected.update(n for n in held if seconds[n] <= QUIET_HOLD_SECONDS)
            # (with many failures: a commit that breaks this much can break any test)
            # (one more failure is counted when marginal_holds held tests: the first failing
            # test of the runs before that rule may be one of them)
            selected.update(long_by_evidence(failing, held, 1 if plan.econ else 0))
        # held tests that read changed test data directly (see test_data_candidates), or whose
        # token streams changed (run_tokcheck)
        selected.update(n for n in held if n in plan.must)
        # long held tests whose own evaluations of the edited comparison change
        selected.update(n for n in held if n in plan.sure_infected and seconds[n] > QUIET_HOLD_SECONDS)
        # a test with a small-scale variant that did not run is selected: no evidence rule
        # (neither the long-test rule nor "import state unchanged") drops it
        selected.update(n for n in held if tests[n]["id"] in SCALED_IDS)
        held = held_all
        if info and info.get("verdict") is True:
            selected.add(info["n"])
        if plan.watch:
            selected.update(watch_readers(plan, verdicts, len(tests)))
        if plan.global_leak is not None:
            # a run left the interpreter with other settings: every later test of a full run
            # (the order of tests.json) sees them
            # A test that marginal_holds held ran before that rule, unless a test before it
            # had failed (see marginal_holds). When such a test comes before the first test
            # that was seen to change the settings, it may have changed them too.
            first_failure = min((n for n, bad in verdicts.items() if bad), default=len(tests))
            first = min([plan.global_leak] + [n for n in plan.econ if n < plan.global_leak and n < first_failure])
            selected.update(range(first, len(tests)))
            plan.note.append("shared state or interpreter settings changed in test %d" % plan.global_leak)
        for n in plan.pool_died:
            cls_id = tests[n]["id"].rsplit("::", 1)[0]
            for t in maps.tests(maps.meta.get("pool_users", 0)):
                if t > n and tests[t]["id"].rsplit("::", 1)[0] == cls_id:
                    selected.add(t)
        selected.update(patchrules.importer_tests(maps, plan.broken))
        for name in plan.broken:  # a test file that cannot be imported: all its tests fail
            selected.update(tests_of_file.get(name.replace(".", "/") + ".py", ()))
        if STOP_AT_COLLECTION:
            # When a test file cannot be imported, pytest reports the collection error and stops
            # before it runs any test ("Interrupted: errors during collection"). So the failing
            # tests of such a commit are the tests of the files that cannot be imported, and no
            # other test: none of them ran. The child imported every test file of the patched
            # tree and named the ones that raise (plan.broken).
            unimportable = set()
            for name in plan.broken:
                if name in plan.class_broken:
                    continue  # the file imports; setUpClass raises when the tests run
                unimportable.update(tests_of_file.get(name.replace(".", "/") + ".py", ()))
            if (unimportable and status == "END" and not plan.all_fail and "import does not end" not in plan.note
                    and not any(n.startswith("import-time effect outside") for n in plan.note)):
                # (not when import-time code changes outside state, sys.modules for one: the
                # child's preloaded modules can then import where a fresh pytest process does not)
                plan.note.append("collection stops: %d tests of files that do not import (of %d)" % (len(unimportable), len(selected)))
                selected = unimportable
        if dry:
            return selected  # (short_rerun: the answer as it would be now)
        answers[c] = pack(sorted(selected), seconds)
        if DEBUG:
            debug_rows[c] = {
                "mode": plan.mode, "status": status, "cands": sorted(plan.cands), "cpu": round(sum(os.times()[:4]), 3),
                "verdicts": {str(k): int(v) for k, v in verdicts.items()}, "note": plan.note,
                "reduced": sorted(plan.keeps), "iters": {str(n): len(v[0][0]) for n, v in plan.iters.items()}, "held": sorted(held), "ran": len(ran), "diff": plan.diff,
                "must": sorted(plan.must), "sure": sorted(plan.sure_infected.intersection(held)), "tokcheck": plan.tokcheck,
                "times": {str(k): round(v, 4) for k, v in plan.times.items()},
                "body": {str(k): round(v, 4) for k, v in plan.body.items()},
                "suspects": sorted(plan.suspects), "skipped": sorted(plan.skipped), "coll_dead": sorted(plan.coll_dead), "protect": sorted(plan.protect),
                "guard": len(plan.guard), "blind": sorted(plan.blind), "econ": sorted(plan.econ),
                "costs": {str(n): round(estimate_cost(n, plan, maps, seconds), 4) for n in ran},
                "est": round(sum(estimate_cost(n, plan, maps, seconds) for n in ran), 3),
                "cone": len(plan.cone) if plan.mode == "fresh_cone" else ("redo" if plan.cone == [] else None),
                "dkeys": DEBUG_DKEYS.get(c), "swap": len(plan.swap_work) if plan.mode == "fresh_cone" else None,
            }
            if plan.stage is not None or plan.stage_info is not None:
                debug_rows[c]["stage"] = plan.stage_info
                debug_rows[c]["stage_plan"] = None if plan.stage is None else {
                    k: (len(v) if k in ("later", "deciders", "others") else round(v, 3) if type(v) is float else v)
                    for k, v in plan.stage.items()}
            if plan.watch:
                debug_rows[c]["leaks"] = plan.leaks[:20]
                debug_rows[c]["watched"] = len(plan.watched or ())

    def choose(plan):
        """Splits the candidates into tests to run and tests that are held (not run)."""
        if plan.watch and not resolve_watch(plan.watch):
            # no class or module of the pinned tree holds such a container (an instance
            # attribute, for instance): nothing to watch
            plan.note.append("no shared container to watch")
            plan.watch = []
        cands = sorted(plan.cands - plan.blind)
        if COLL_RUN and plan.broken and "__getattr__" not in plan.gone and not any("__getattr__" in x for x in plan.note):
            # (not with a changed module __getattr__: the child holds submodules as attributes
            # that the pytest run imports later or never, so its runs can pass where pytest fails)
            # the tests of the files that do not import, and the tests that load a module that
            # does not import: they fail in any case and are selected without a run
            dead = set(patchrules.importer_tests(maps, [b for b in plan.broken if b not in plan.class_broken]))
            for name in plan.broken:
                if name not in plan.class_broken:
                    dead.update(tests_of_file.get(name.replace(".", "/") + ".py", ()))
            plan.coll_dead = dead
            plan.coll_names = set(plan.broken)
            if plan.coll_extra - plan.cands:
                plan.note.append("%d run-time callers of import-time units" % len(plan.coll_extra - plan.cands))
                plan.cands |= plan.coll_extra
                cands = sorted(plan.cands - plan.blind)
            cands = [n for n in cands if n not in dead]
        cost = {n: estimate_cost(n, plan, maps, seconds) for n in cands}
        if plan.mode != "hot" and "import state unchanged" not in plan.note:
            for cls_id, setup in CLASS_SETUP_SECONDS.items():
                members = [n for n in cands if tests[n]["id"].startswith(cls_id + "::")]
                for n in members:
                    cost[n] += setup / len(members)
        if RUN_ALL:
            return [n for n in cands if cost[n] <= 60 or RUN_SLOW], [], cost
        # a test is held (not run, selected as it is) when a run costs more CPU than
        # its selection costs in the worst case
        limit = {n: max(HOLD_SECONDS, HOLD_FACTOR * (WORKER_COST + seconds[n])) for n in cands}
        ran = [n for n in cands if cost[n] <= limit[n]]
        held = [n for n in cands if cost[n] > limit[n]]
        if plan.watch:
            # a candidate that does not run may leave a watched container changed unseen, and
            # then every later reader of it is selected (see watch_readers): all but the long
            # tests run
            ran = [n for n in cands if cost[n] <= limit[n] or seconds[n] <= QUIET_HOLD_SECONDS]
            held = [n for n in cands if not (cost[n] <= limit[n] or seconds[n] <= QUIET_HOLD_SECONDS)]
        if STAGE == "force":
            # validation: the test with a stage map never runs as a whole, so that the check sees it
            moved = [n for n in ran if tests[n]["id"] in maps.stages()]
            ran = [n for n in ran if n not in moved]
            held = held + moved
        budget = COMMIT_SECONDS if plan.mode == "hot" else FRESH_SECONDS
        if "import state unchanged" in plan.note:
            budget = INERT_SECONDS
        if plan.watch:
            # every candidate runs: one that does not run may leave a watched container changed
            # for the later tests, and finish() must then select all their readers
            budget = float("inf")
        if plan.coll_dead:
            budget = COLL_BUDGET  # the runs are priced beside the dead tests below
        # the verdicts of these tests decide about the long held tests: they run first
        long_held = [n for n in held if seconds[n] > QUIET_HOLD_SECONDS]
        protect_classes = {tests[n]["id"].rsplit("::", 1)[0] for n in long_held}
        protect_fixtures = set()
        for n in long_held:
            protect_fixtures.update(fixtures_of(n))
        plan.protect = {
            n for n in ran
            if tests[n]["id"].rsplit("::", 1)[0] in protect_classes
            or (protect_fixtures and protect_fixtures & fixtures_of(n) and cost[n] <= 2.0)
        }
        if sum(cost[n] for n in ran) > budget:
            # The CPU limit does not allow every run. A held test is selected as it is, which
            # costs its recorded seconds: so the runs that save the most recorded seconds per
            # CPU second go first, and the tests that are cheap to select are the ones held.
            keep_set = set()
            acc = 0.0
            for n in sorted(ran, key=lambda n: (n not in plan.protect and tests[n]["id"] not in SCALED_IDS,
                                                -(seconds[n] + 0.002) / (cost[n] + 0.002), cost[n])):
                if acc + cost[n] > budget and tests[n]["id"] not in SCALED_IDS:
                    continue  # a test with a small-scale variant always runs (see compute_keeps)
                acc += cost[n]
                keep_set.add(n)
            held += [n for n in ran if n not in keep_set]
            ran = [n for n in ran if n in keep_set]
        plan.econ = set()
        if plan.coll_dead and not plan.watch and not plan.settings:
            plan.econ = marginal_holds(plan, ran, held, cost, COLL_LOOSE)
            if plan.econ:
                held += [n for n in ran if n in plan.econ]
                ran = [n for n in ran if n not in plan.econ]
                plan.note.append("%d tests held beside the dead tests" % len(plan.econ))
        elif MARGINAL and not plan.watch and not plan.settings and "import state unchanged" not in plan.note:
            # (with "import state unchanged", finish() drops the held tests when the runs pass:
            # there a held test costs nothing, and the runs are the evidence)
            plan.econ = marginal_holds(plan, ran, held, cost)
            if plan.econ:
                held += [n for n in ran if n in plan.econ]
                ran = [n for n in ran if n not in plan.econ]
                plan.note.append("%d tests held beside the selection" % len(plan.econ))
        return ran, held, cost

    settings_files = {}
    pool_users = set()  # tests that use a class-level process pool (filled at the first use)

    def names_settings(rel):
        """True when the test file (of the pinned commit) names an interpreter-wide setting."""
        known = settings_files.get(rel)
        if known is None:
            try:
                with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
                    text = f.read()
                known = any(word in text for word in SETTINGS_WORDS)
            except (OSError, UnicodeDecodeError):
                known = True
            settings_files[rel] = known
        return known

    def marginal_holds(plan, ran, held, cost, loose=False):
        """The tests of 'ran' that are held after all, because their runs cost more CPU than
        their selection costs inside the answer of this commit.

        The rule above prices each test alone: a run against its seconds and one more worker.
        But the answer is packed (see pack), and it holds tests in any case: plan.blind and
        the held tests that finish() selects. Beside those, one more test costs less than
        alone, and nothing in the free time next to a longer test. So the candidates are
        taken in the order of CPU per recorded second, the worst first, and the cut that
        makes (CPU_PRICE * CPU of the runs + packed seconds of the selection) smallest is
        taken, for the case that every run passes.

        Only tests that today's rules already leave unrun after a first failure qualify (the
        'skip' set of prepare_plan: cost above half the recorded seconds, not in
        plan.protect). So after its first failure, a child still runs every test that it
        ran before this rule, and finish() sees the same failing tests, except one: the
        first failing test itself may now be a held one. finish() counts one more failure
        for that (see long_by_evidence). Not held by this rule: a test longer than
        QUIET_HOLD_SECONDS (finish() selects those by evidence only), a test with a
        small-scale variant, a test that shares a class or a .sql fixture with a long held
        test (its verdict is evidence for that test), and a test that uses a class-level
        process pool (its run shows whether the pool dies), and a test whose test file names an
        interpreter-wide setting (its run shows whether it leaves the setting changed)."""
        free = [
            n for n in ran
            if (loose or cost[n] > max(0.03, AFTER_FAIL_RATIO * seconds[n])) and n not in plan.protect
            and seconds[n] <= QUIET_HOLD_SECONDS and tests[n]["id"] not in SCALED_IDS
        ]
        if not free:
            return set()
        long_held = [n for n in held if seconds[n] > QUIET_HOLD_SECONDS]
        long_fixtures = set()
        for n in long_held:
            long_fixtures.update(fixtures_of(n))
        if not pool_users:
            pool_users.update(maps.tests(maps.meta.get("pool_users", 0)))
            pool_users.add(-1)
        free = [
            n for n in free
            if n not in pool_users and not (long_fixtures and long_fixtures & fixtures_of(n))
            and not names_settings(tests[n]["file"])
        ]
        if not free:
            return set()
        # what the answer holds in any case (a long held test only when it is sure to be selected)
        base = set(plan.blind) | plan.coll_dead
        base.update(
            n for n in held
            if seconds[n] <= QUIET_HOLD_SECONDS or n in plan.must or n in plan.sure_infected or tests[n]["id"] in SCALED_IDS
        )
        total = sum(seconds[n] for n in base)
        longest = max((seconds[n] for n in base), default=0.0)
        free.sort(key=lambda n: (-(cost[n] + 1e-4) / (seconds[n] + 1e-3), n))
        run_cost = sum(cost[n] for n in ran)
        best, cut = CPU_PRICE * run_cost + packed_bound(total, longest), 0
        for i, n in enumerate(free):
            total += seconds[n]
            longest = max(longest, seconds[n])
            run_cost -= cost[n]
            value = CPU_PRICE * run_cost + packed_bound(total, longest)
            if value < best - 1e-9:
                best, cut = value, i + 1
        if cut == len(ran):
            # One test always runs. Without any verdict, finish() selects every held test, the
            # long ones too. And a run is what shows a change of interpreter-wide settings
            # that the patched statement makes in every test that reaches it (see finish).
            cut -= 1
        return set(free[:cut])

    shared_info = []
    attr_reads = {}

    def class_fixture_rule(plan):
        """A commit that changes only data files, one of them a .sql fixture whose entries a
        shared class fixture holds in a list (TestExecutor: the TPC-H and TPC-DS queries).
        setUpClass reads the file for every test of the class, but only the tests that read
        that list can see the change: the others leave the candidates, and the readers run
        with the changed entries only."""
        changed = {rel for rel, text in plan.files.items() if rel.startswith("tests/fixtures/") and text is not None}
        if not changed or len(changed) != sum(1 for rel in plan.files if rel.startswith("tests/fixtures/")):
            return
        if not shared_info:
            shared_info.extend(preload.shared_class_info(ROOT) or [None])
            try:
                # which of those lists each test of the class reads (build_attr_reads.py)
                with open(os.path.join(MAPS, "class_attr_reads.json")) as f:
                    attr_reads.update(json.load(f))
            except (OSError, ValueError):
                pass
        for info in shared_info:
            if info is None:
                continue
            prefix, cls, by_file, other = info
            hit = changed & set(by_file)
            if not hit or changed & other:
                continue
            for n in sorted(plan.cands):
                test_id = tests[n]["id"]
                if not test_id.startswith(prefix):
                    continue
                read = attr_reads.get(test_id)
                if read is None:
                    continue  # not in the map: nothing is known about this test
                if any(by_file[f] in read for f in hit):
                    plan.fixture_users.add(n)
                elif not (fixtures_of(n) & changed) - hit:
                    plan.cands.discard(n)

    def prepare_plan(plan):
        """What the child of this commit runs: (tests to run, held tests, seconds allowed per
        test, tests that do not run after a failure, cap on the seconds of all runs)."""
        ran, held, cost = choose(plan)
        plan.tokcheck = []
        if TOKCHECK and not RUN_ALL and TOKENIZER_FILES.intersection(plan.files):
            import tokcheck

            checkable = tokcheck.tests_of(MAPS)
            plan.tokcheck = sorted(t for t in held if t in checkable and seconds[t] > QUIET_HOLD_SECONDS)
            if plan.watch and plan.tokcheck:
                # The check tokenizes in the child before the tests run, and patched
                # tokenizer code could then write to a watched container before the first
                # test: the watch would no longer show which test leaves it changed. With
                # a watch, the check does not run and its tests count as changed.
                plan.must.update(plan.tokcheck)
                plan.note.append("watch: token stream check not run, %s selected" % plan.tokcheck)
                plan.tokcheck = []
        allowed = {n: max(TEST_TIMEOUT, 25.0 * cost[n]) * SLOWDOWN for n in ran}
        # The CPU guard (see GUARD). A stopped test is selected and counts like a failing test
        # in finish(), so for a test whose verdict decides about long held tests (protect) the
        # limit also covers the seconds of the longest of those. With a watch, a test without
        # a run selects every later reader of the watched containers: no guard there.
        plan.guard = {}
        if GUARD and not RUN_ALL and not plan.watch:
            long_secs = [seconds[t] for t in held if seconds[t] > QUIET_HOLD_SECONDS]
            for n in ran:
                stake = seconds[n] + WORKER_COST
                if n in plan.protect and long_secs:
                    stake += max(long_secs)
                limit = max(GUARD_MIN, GUARD_FACTOR * cost[n] + GUARD_ADD, stake / GUARD_PRICE)
                if limit < allowed[n] / SLOWDOWN:
                    plan.guard[n] = limit
        # Once a test has failed, the commit is known to break something. A further
        # verdict then only saves the recorded seconds of a passing test, so only runs
        # that are cheap next to those seconds go on; the others are selected as they are.
        skip = set() if RUN_ALL or plan.watch else {
            n for n in ran if cost[n] > max(0.03, AFTER_FAIL_RATIO * seconds[n]) and n not in plan.protect
            and not (plan.coll_dead and tests[n]["id"] in SCALED_IDS)
        }
        # A failing test of the class (or fixture) of a long held test selects that long
        # test (see finish). When failures have selected every long held test, and the
        # longest of them is longer than all tests that run together, the tests that are
        # still to run fit beside it on one more worker, which costs 2 s at most. So when
        # the runs that are left cost more CPU than that, they end there and the tests
        # are selected as they are. The answer then holds every candidate without a pass.
        plan.enders = {}
        long_held = [t for t in held if seconds[t] > QUIET_HOLD_SECONDS]
        if long_held and not RUN_ALL and not plan.watch and max(seconds[t] for t in long_held) >= sum(seconds[n] for n in ran):
            plan.enders["held"] = set(long_held)
            left_cost = sum(cost[n] for n in ran)
            for n in ran:
                left_cost -= cost[n]
                n_class, n_fixtures = tests[n]["id"].rsplit("::", 1)[0], fixtures_of(n)
                decides = {
                    t for t in long_held
                    if tests[t]["id"].rsplit("::", 1)[0] == n_class or n_fixtures & fixtures_of(t)
                }
                if decides:
                    plan.enders[n] = (decides, left_cost > WORKER_COST)
        # A guard against runs that cost many times their estimate (a patch that makes every
        # test slow without hanging): past the cap, the tests without a verdict are selected.
        # The longest per-test limit is added, so that one hung test still counts as a failure.
        cap = None
        if ran and not RUN_ALL:
            cap = SLOWDOWN * (2.0 + 3.0 * sum(cost[n] for n in ran)) + max(allowed.values())
        plan.prep = (allowed, skip, cap, cost)
        stage_plan(plan, ran, held)
        if plan.stage is not None:
            # the check runs after the tests of the commit, with its own CPU limit (see GUARD)
            # and its own time limit; the cap on the other runs ends where it begins (run_child)
            n = plan.stage["n"]
            limit = max(GUARD_MIN, 1.5 * plan.stage["most"] + 3.0)  # (the child sets a smaller one for its budget)
            if os.environ.get("BR_STAGE_AUDIT") == "1":
                limit = 400.0  # validation: every task runs to its end
            if GUARD:
                plan.guard[n] = limit
            allowed[n] = max(TEST_TIMEOUT, 1.5 * limit + 5.0) * SLOWDOWN
        return ran, held, allowed, skip, cap

    def stage_plan(plan, ran, held):
        """Plans the stage check (stages.py) for a held long test that has a stage map: sets
        plan.stage, or leaves it None. The child runs the check after the other tests, and
        only when their verdicts would select the held test without a run (run_stage_check).

        The check applies when the commit changes nothing but sqlglot code (the test code, the
        class fixture and DuckDB's recorded answers are then the pinned ones). The tasks whose
        later work runs a changed line run to their end in the check ("later"): with known
        lines (the changed statements; for an import-time commit also the statements that
        read a changed name: plan.lines, the same lines that decide which parts of a test
        run in reduced form) those are the tasks that the stage map names for these lines;
        without known lines, all tasks. The check is planned when the estimated cost of its
        sure part (CPU seconds as measured when the map was built: every task up to the plan,
        the "later" tasks to their end) is not more than what leaving the whole test out can
        pay (STAGE_GAIN); the child decides with the seconds that it really saves."""
        plan.stage = None
        if STAGE == "0" or RUN_ALL or not held or plan.watch or plan.all_fail or plan.mode not in RESTART_SECONDS:
            return
        if any(text is None or not (p.startswith("sqlglot/") and p.endswith(".py")) for p, text in plan.files.items()):
            return
        if not preload.exec_cache_present():
            return  # the costs of the map were measured with the row cache of the class fixture
        if plan.mode != "hot" and not preload.exec_fresh_ok(plan.files):
            return
        for n in held:
            if seconds[n] <= QUIET_HOLD_SECONDS or n not in plan.cands:
                continue
            record = maps.stages().get(tests[n]["id"])
            if record is None:
                continue
            cpu, cost = record["cpu"], record["cost"]
            tasks = set(record["tasks"])
            keep = plan.keeps.get(n)
            active = tasks if keep is None or keep[2] is None else tasks & keep[2]
            if plan.lines and None not in plan.lines.values():
                import stages

                later = stages.late_tasks(record, plan.lines)
            else:
                later = set(tasks)
            fixed = max(0.0, cost["check"] - sum(v[0] for v in cpu.values()))  # parsing, pickling, the DuckDB texts
            to_end = sum(cpu[i][0] + cpu[i][1] for i in active & later)
            est = STAGE_STATE_SECONDS + fixed + sum(cpu[i][0] for i in active - later) + to_end
            memo_info = memo_dirty(plan, record) if MEMO and plan.mode == "hot" and later else None
            if memo_info is not None:
                est = STAGE_STATE_SECONDS + fixed + memo_cost(memo_info, cpu, active, later)
            most = STAGE_GAIN * (seconds[n] + WORKER_COST)  # what a check may cost when it saves the whole test
            if est > most and STAGE != "force":
                continue
            if deadline - time.monotonic() < 20.0 + 1.5 * most * SLOWDOWN:
                continue  # not when the batch is short of time
            cls_id, fixtures = tests[n]["id"].rsplit("::", 1)[0], fixtures_of(n)
            plan.stage = {
                "n": n, "later": sorted(later), "est": round(est, 3), "to_end": to_end, "most": round(most, 3),
                "known": len(later) < len(tasks) or not tasks, "active": len(active),
                "differ_most": sum(cpu[i][1] for i in active - later),  # when every other task runs on to its end
                # the tests that finish() selects besides the failing ones, whatever the check says
                "others": sorted(plan.blind | {k for k in held if k != n and (
                    seconds[k] <= QUIET_HOLD_SECONDS or k in plan.must or tests[k]["id"] in SCALED_IDS)}),
                "inert": "import state unchanged" in plan.note, "more": 1 if plan.econ else 0,
                "forced": n in plan.must or n in plan.sure_infected,
                # the tests whose failure selects this one in finish()
                "deciders": {k for k in ran if tests[k]["id"].rsplit("::", 1)[0] == cls_id or fixtures & fixtures_of(k)},
                "memo": memo_info,
            }
            return

    def execute(c, plan, index):
        box = {}
        if plan.mode == "hot" and not plan.work and not RUN_ALL:
            class_fixture_rule(plan)

        def prepare():
            ran, held, allowed, skip, cap = prepare_plan(plan)
            box["ran"], box["held"] = ran, held
            return ran, allowed, skip, cap

        left = deadline - time.monotonic()
        if left < 3.0:
            if plan.mode != "hot" and any(p.endswith(".py") for p in plan.files):
                # an import-time change that never ran: a broken import can fail any test
                plan.all_fail = True
            ran, held, _ = choose(plan)
            finish(c, plan, {}, "DEADLINE", ran, held)
            return
        use_diff = plan.mode != "hot" and plan.diffable and BASE_STATE is not None and not NO_DIFF
        if use_diff and small_plan(plan):
            use_diff = False
        if plan.mode != "hot" and not use_diff:
            # no comparison of the import state: every candidate runs whole (the small-scale
            # variants stay: see compute_keeps)
            plan.keeps = {n: k for n, k in plan.keeps.items() if k == (None, None, None, True, None)}
            plan.lines = {"?": None}
        if use_diff:
            def on_diff(keys):
                refine_with_diff(plan, keys, maps, tests)
                ran, allowed, skip, cap = prepare()
                return {"order": ran, "keeps": plan.keeps, "skip": skip, "allowed": allowed, "cap": cap, "enders": plan.enders,
                        "tokcheck": plan.tokcheck, "guard": plan.guard, "stage": plan.stage}

            verdicts, status, hung = run_child(plan, None, tests, index, {}, left - 2.0, (), on_diff)
            if status == "REDO" and "ran" not in box:
                return "REDO"
            if "ran" not in box:
                # the child never reported its import state: nothing is known
                ran, held, _ = choose(plan)
                if status in ("TIMEOUT", "DIED"):
                    for p in plan.files:
                        if p.endswith(".py"):
                            plan.broken.append(p[:-3].replace("/", "."))
                    plan.note.append("import does not end")
                finish(c, plan, {}, status, ran, held)
                return

        else:
            ran, allowed, skip, cap = prepare()
            if not ran:
                # (with a stage check planned, the child that runs no test still runs the check)
                stage_allowed = allowed if plan.stage is not None else {}
                if plan.mode == "hot" and (plan.tokcheck or plan.stage is not None):
                    run_child(plan, [], tests, index, stage_allowed, left - 2.0)  # only the token stream check
                elif plan.mode != "hot" and any(p.endswith(".py") for p in plan.files):
                    # No test to run, but the patched tree can still fail to import (a broken
                    # import line, a name that is gone): the child reports the modules that
                    # raise, and finish() selects the tests that load them.
                    _, status, _ = run_child(plan, [], tests, index, stage_allowed, left - 2.0)
                    if status == "REDO":
                        return "REDO"
                    if status in ("TIMEOUT", "DIED"):
                        plan.broken.extend(p[:-3].replace("/", ".") for p in plan.files if p.endswith(".py"))
                        plan.note.append("import does not end")
                finish(c, plan, {}, "END", ran, box["held"])
                return
            verdicts, status, hung = run_child(plan, ran, tests, index, allowed, left - 2.0, skip, None, cap)
            if status in ("REDO", "REDOC"):
                return status
            if status in ("TIMEOUT", "DIED") and hung is None and not verdicts and plan.mode != "hot":
                for p in plan.files:
                    if p.endswith(".py"):
                        plan.broken.append(p[:-3].replace("/", "."))
                plan.note.append("import does not end")
        conclude(c, plan, verdicts, status, hung, box["ran"], box["held"], index)

    def rest_after_stop(plan, verdicts, todo, held):
        """After the CPU guard stopped a test, the child is gone. Returns what a new child
        runs: (tests, the tests among them that do not run after a failure, enders). The
        tests that the first child would not have run any more after its verdicts are selected
        as they are (plan.skipped), as that child would have done. The others run in a new
        child when that costs less than their selection; otherwise they are selected and
        count like failing tests (plan.suspects), because nothing is known about them."""
        _, skip, _, cost = plan.prep
        bad = {n for n, failed in verdicts.items() if failed}
        if bad:
            plan.skipped.update(n for n in todo if n in skip)
            todo = [n for n in todo if n not in skip]
        enders = plan.enders
        if enders:
            # the state of runner.Verdicts after the verdicts so far (a stopped test selects
            # the long held tests of its class like a failing one)
            undecided = set(enders.get("held", ()))
            for n in sorted(bad | plan.suspects):
                decides = enders.get(n)
                if decides:
                    undecided -= decides[0]
                    if not undecided and decides[1]:
                        plan.skipped.update(todo)
                        return [], set(), enders
            enders = dict(enders)
            enders["held"] = undecided
        if not todo:
            return [], set(), enders
        evidence = bad | plan.suspects
        base = evidence | plan.blind | plan.coll_dead | (plan.skipped & set(cost)) | {n for n in held if seconds[n] <= QUIET_HOLD_SECONDS}
        base |= long_by_evidence(evidence, held)
        unrun = base | set(todo) | long_by_evidence(evidence | set(todo), held)
        run_cost = RESTART_SECONDS.get(plan.mode, 1.5) + sum(cost.get(n, 0.0) for n in todo)
        if CPU_PRICE * run_cost >= packed_seconds(unrun, seconds) - packed_seconds(base, seconds):
            plan.suspects.update(todo)
            plan.note.append("%d tests selected unrun after a stopped test" % len(todo))
            return [], set(), enders
        return todo, skip.intersection(todo), enders

    def short_rerun(c, plan, verdicts, ran, held, index):
        """Runs the short tests that the child left unrun after a failure (plan.skipped) in a
        new child, when their selection costs at least SHORT_MIN packed seconds in the answer.
        The tests that prove the most recorded seconds per CPU second go first, and the run
        stops where SHORT_PASS times the packed seconds saved no longer pays for the CPU. Their
        verdicts join 'verdicts': a passing test leaves the answer, a failing one stays. A test
        that does not end in the new child (and the ones after it) stays selected unrun."""
        if plan.mode != "hot" or RUN_ALL or plan.watch or plan.prep is None or plan.all_fail or plan.pool_died:
            return
        if not any(verdicts.values()):
            return
        allowed, _, _, cost = plan.prep
        blind = [n for n in ran if n in plan.skipped and n not in verdicts]
        short = [n for n in blind if seconds[n] <= QUIET_HOLD_SECONDS and n in allowed and n in cost]
        if DEBUG and blind:
            log(c, "short rerun: blind %d short %d" % (len(blind), len(short)))
        if not short:
            return
        left = deadline - time.monotonic()
        if left < 20.0:
            return
        note = list(plan.note)
        try:
            selected = finish(c, plan, verdicts, "END", ran, held, blind, dry=True)
        finally:
            plan.note[:] = note
        short = [n for n in short if n in selected]
        if not short:
            return
        full = packed_seconds(selected, seconds)
        if full - packed_seconds(selected.difference(short), seconds) < SHORT_MIN:
            return
        order = sorted(short, key=lambda n: (-(seconds[n] + 1e-3) / (cost[n] + 1e-3), n))
        rest = selected.difference(short)
        total = sum(seconds[n] for n in rest)
        longest = max((seconds[n] for n in rest), default=0.0)
        base_total, base_longest = total + sum(seconds[n] for n in order), max([longest] + [seconds[n] for n in order])
        whole = packed_bound(base_total, base_longest)
        # the prefix of 'order' that runs: the one with the best expected gain (packed_bound
        # for the seconds that stay selected: one call per test)
        left_total, run_cost = base_total, RESTART_SECONDS["hot"]
        best, cut = 0.0, 0
        for i, n in enumerate(order):
            left_total -= seconds[n]
            run_cost += cost[n]
            longest_left = max([longest] + [seconds[k] for k in order[i + 1:]])
            gain = SHORT_PASS * (whole - packed_bound(left_total, longest_left)) - SHORT_PRICE * run_cost
            if gain > best + 1e-9:
                best, cut = gain, i + 1
        if DEBUG:
            log(c, "short rerun plan: %d tests, packed %.2f -> %.2f, bound %.2f, cut %d gain %.3f, cost %.3f" % (
                len(short), full, packed_seconds(selected.difference(short), seconds), whole, cut, best,
                sum(cost[n] for n in order)))
        if not cut:
            return
        chosen = set(order[:cut])
        todo = [n for n in ran if n in chosen]
        saved = (plan.tokcheck, plan.enders, plan.stage)
        plan.tokcheck, plan.enders, plan.stage = [], {}, None
        try:
            cap = SLOWDOWN * (2.0 + 3.0 * sum(cost[n] for n in todo)) + max(allowed[n] for n in todo)
            v, status, hung = run_child(plan, todo, tests, index, allowed, left - 2.0, (), None, cap)
        finally:
            plan.tokcheck, plan.enders, plan.stage = saved
        if hung is not None:
            v.pop(hung, None)  # no verdict: it stays selected unrun
        if status not in ("END", "COST", "TIMEOUT", "OVER"):
            return  # (the child died or failed: nothing new is known)
        verdicts.update((n, bad) for n, bad in v.items() if n in chosen)
        plan.note.append("short rerun: %d of %d tests, %d pass, %d fail (status %s)" % (
            len(todo), len(short), sum(1 for n in todo if v.get(n) is False), sum(1 for n in todo if v.get(n)), status))

    def conclude(c, plan, verdicts, status, hung, ran, held, index):
        """The answer of a commit from the verdicts that its child reported."""
        if status == "OVER":
            plan.note.append("over the time cap")
        # a test that does not end is a timeout in the full run; the tests after it run again
        hangs = restarts = 0
        pool_users = None
        while hung is not None:
            plan.stage = None  # (a stage check belongs to the first child of a commit only)
            if status == "COST":
                # The CPU guard stopped this test: it is selected, and it has no verdict. In
                # finish() it counts like a failing test (plan.suspects).
                plan.suspects.add(hung)
                plan.note.append("test %d stopped by the CPU guard" % hung)
            elif hangs >= 2:
                break  # (the third test that does not end stays without a verdict)
            else:
                verdicts[hung] = True
                hangs += 1
            if pool_users is None:
                pool_users = set(maps.tests(maps.meta.get("pool_users", 0)))
            if hung in pool_users:
                # A pool task that does not end keeps a worker of the class-level process pool
                # busy (and spinning) for the rest of the full run: the later tests of the class
                # wait for it or run starved, and time out too. The direct runs here use an
                # in-process pool, so they cannot show that: those tests are selected unrun.
                # A test that the CPU guard stopped may be such a test, so the rule holds for it too.
                cls_id = tests[hung]["id"].rsplit("::", 1)[0]
                plan.blind.update(
                    n for n in range(hung + 1, len(tests)) if tests[n]["id"].rsplit("::", 1)[0] == cls_id
                )
            hung = None
            todo = [n for n in ran if n not in verdicts and n not in plan.skipped and n not in plan.suspects]
            left = deadline - time.monotonic()
            if not todo or left < 3.0:
                break
            if status == "COST":
                if restarts >= GUARD_RESTARTS or plan.prep is None:
                    break
                restarts += 1
                todo, skip, enders = rest_after_stop(plan, verdicts, todo, held)
                if not todo:
                    break
                allowed, _, cap, _ = plan.prep
                saved_enders = plan.enders
                plan.enders = enders
                try:
                    v, status, hung = run_child(plan, todo, tests, index, allowed, left - 2.0, skip, None, cap)
                finally:
                    plan.enders = saved_enders
            else:
                cost_allowed = {n: TEST_TIMEOUT * SLOWDOWN for n in todo}
                v, status, hung = run_child(plan, todo, tests, index, cost_allowed, left - 2.0, set(todo))
            verdicts.update(v)
        if SHORT_RERUN and status == "END":
            short_rerun(c, plan, verdicts, ran, held, index)
        blind = [n for n in ran if n in plan.skipped and n not in verdicts]
        if plan.pool_died:
            first = min(plan.pool_died)
            cls_id = tests[first]["id"].rsplit("::", 1)[0]
            users = set(maps.tests(maps.meta.get("pool_users", 0)))
            blind += [n for n in ran if n not in verdicts and n > first and n in users
                      and tests[n]["id"].rsplit("::", 1)[0] == cls_id]
        finish(c, plan, verdicts, status, ran, held, blind)

    # 1. answers that need no run
    hot, fresh = [], []
    for c in commits:
        plan = plans[c]
        if plan.mode == "all":
            answers[c] = pack(all_tests, seconds)
        elif plan.mode == "blind":
            answers[c] = pack(sorted(plan.blind), seconds)
            if DEBUG:
                debug_rows[c] = {"mode": "blind", "cands": sorted(plan.blind), "verdicts": {}, "note": plan.note}
        elif plan.mode == "noop":
            answers[c] = []
            if DEBUG:
                debug_rows[c] = {"mode": "noop", "cands": [], "verdicts": {}, "note": plan.note}
        elif plan.mode == "hot":
            hot.append(c)
        else:
            fresh.append(c)

    def small_plan(plan):
        """An import-time commit whose candidates all run whole for less CPU than the
        comparison of the import state costs in the child (about 0.2 s)."""
        if RUN_ALL or plan.mode == "hot":
            return False
        return sum(maps.cost[n] if maps.cost else seconds[n] * 1.5 for n in plan.cands - plan.blind) <= SMALL_PLAN

    def planned_cost(c):
        plan = plans[c]
        return sum(min(estimate_cost(n, plan, maps, seconds), HOLD_SECONDS * 2) for n in plan.cands)

    hot.sort(key=planned_cost)
    fresh.sort(key=planned_cost)
    need_heavy = any(
        tests[n]["file"] in ("tests/test_optimizer.py", "tests/test_executor.py")
        for c in hot + fresh for n in plans[c].cands
    )
    global PANDAS_USERS
    lazy_pandas = False
    duck_replay = False
    if need_heavy:
        users = None
        if not os.environ.get("BR_NO_LAZY_PANDAS"):
            try:
                with open(os.path.join(MAPS, "pandas_tests.json")) as f:
                    ids = set(json.load(f)["tests"])
                users = frozenset(n for n, t in enumerate(tests) if t["id"] in ids)
            except (OSError, ValueError, KeyError, TypeError):
                users = None
        recording = None
        if users is not None and not NO_DUCK_REPLAY:
            # The DuckDB queries of the optimizer tests are answered from a recording of the
            # pinned run (duckreplay.py, made by build_duck_replay.py at image build time).
            import duckreplay

            recording = duckreplay.read_file(duckreplay.default_path())
        if recording is None:
            import duckdb  # noqa: F401
        if users is not None:
            import lazymods

            lazy_pandas = lazymods.install()
        if recording is not None:
            duck_replay = lazy_pandas and duckreplay.install(recording)
            if not duck_replay:
                import duckdb  # noqa: F401
        if duck_replay:
            # duckdb is now lazy like numpy and pandas, and the optimizer tests need none of the
            # three unless a query really has to run. The few other users (the TPC tests of the
            # executor, mostly held back; one test with numpy values) load what they touch in
            # their own child: this process never loads the libraries. (Measured on the labeled
            # sets: about one such child in a batch of 60 commits.)
            pass
        elif lazy_pandas:
            # numpy and pandas load when a commit that runs one of their users comes (see
            # pandas_turn below), or never
            PANDAS_USERS = users
        else:
            import pandas  # noqa: F401
            import numpy  # noqa: F401

    def needs_pandas(c):
        p = plans[c]
        return any(n in PANDAS_USERS for n in p.cands - p.blind)

    pandas_turn = None  # the first commit before which the parent loads numpy and pandas
    if lazy_pandas:
        # the commits that need them run last, so that the forks before them are cheaper
        hot = [c for c in hot if not needs_pandas(c)] + [c for c in hot if needs_pandas(c)]
        needers = [c for c in hot + fresh if needs_pandas(c)]
        if len(needers) >= 2:
            pandas_turn = needers[0]  # with one, its child loads them (same cost, smaller forks)

    def exec_prebuild(hot_commits):
        """TestExecutor attributes that more than one child may read are built once here
        (see preload.py); the rest is built by the child that reads it, if any."""
        cls = getattr(sys.modules.get(preload.EXEC_KEY[0]), preload.EXEC_KEY[1], None)
        users = preload.exec_attribute_users(cls) if cls is not None else {}
        need = collections.Counter()
        for c in hot_commits:
            plan = plans[c]
            names = set()
            for n in plan.cands:
                path, cls_name, meth = tests[n]["id"].split("::")
                attrs = users.get(meth) if (path[:-3].replace("/", "."), cls_name) == preload.EXEC_KEY else None
                if attrs and (RUN_ALL or estimate_cost(n, plan, maps, seconds) <= max(
                        HOLD_SECONDS, HOLD_FACTOR * (WORKER_COST + seconds[n]))):
                    names.update(attrs)
            need.update(names)
        out = ["tpch_sqls", "tpcds_sqls"]
        out += [a for a in preload.EXEC_LAZY if a not in out and need[a] >= 2]
        return out

    # 2. One parent process serves every commit. It imports sqlglot and the needed test
    #    modules once. Commits that only change function bodies run first (cheap: the child
    #    replaces code objects). Commits that change import-time code run after them: their
    #    child drops the loaded sqlglot modules and imports the patched tree again.
    index = None
    import runner  # noqa: F401  (each child imports it: once here, before the forks)
    global BASE_STATE, BASE_FUNCTION_KEYS, BASE_OBJECTS, BASE_CODES
    want_diff = bool(fresh) and not NO_DIFF and any(plans[c].diffable and not small_plan(plans[c]) for c in fresh)
    if hot or want_diff:
        preload.import_sqlglot()
    if want_diff:
        # The import state of the pinned tree, to compare each patched import with. It is
        # taken before any test module is loaded, because tests change module state.
        saved = load_base_state()
        if saved is not None:
            BASE_STATE, BASE_FUNCTION_KEYS, BASE_OBJECTS = saved
        else:
            import statefp

            BASE_STATE, base_fpr = statefp.snapshot()
            BASE_FUNCTION_KEYS = base_fpr.function_keys
            BASE_OBJECTS = base_fpr.objects
            BASE_CODES = base_fpr.base_codes() if os.environ.get("BR_NO_BASE_CODES") != "1" else None
    # modules that every child imports: once here instead of in each child
    import reduce  # noqa: F401
    import runner  # noqa: F401

    if want_diff:
        import statefp  # noqa: F401

    def execute_safe(c, plan, index):
        """A failure on the parent side ends only this commit, with a safe answer: its
        candidates, or every test when an import-time change was not seen to its end."""
        try:
            return execute(c, plan, index)
        except Exception as e:  # noqa: BLE001
            log(c, "execute failed", repr(e))
            if plan.mode == "hot":
                safe = plan.cands | plan.blind | patchrules.importer_tests(maps, plan.broken)
            else:
                safe = all_tests
            answers[c] = pack(sorted(safe), seconds)

    pandas_here = []

    def load_pandas_here():
        if pandas_here:
            return
        pandas_here.append(True)
        import lazymods

        lazymods.load()
        gc.freeze()  # (no collection first: it would walk every object, and little is garbage)

    preimported = []

    def preimport_once():
        # the libraries and the code of the test modules, for the children of import-time commits
        if not preimported and len(fresh) >= 2 and os.environ.get("BR_NO_PREIMPORT") != "1":
            preimported.append(True)
            preimport_for_fresh(fresh, plans, tests)

    # 2a. Import-time commits whose patched modules have a small importer cone (cone.py) run
    #     now, while this process holds the pinned import of sqlglot and no test module: their
    #     child imports only the cone again. When its checks fail (REDO), the commit runs again
    #     below with the other import-time commits, whose child imports the whole tree.
    cone_first = []
    if (hot or want_diff) and not NO_CONE and fresh:
        cone_first = [c for c in fresh if cone_plan(plans[c])]
    if cone_first:
        t_rec = time.process_time()
        cone.record(CONE_WALK)
        log("cone record", len(cone.HEAP.recs), "tables in", round(time.process_time() - t_rec, 3))
        if hot:
            preload._single_thread_duckdb()  # as import_tests() does before the other children
        preimport_once()
        if lazy_pandas:
            # the order of the commits is now: cone commits, function-body commits, the other
            # import-time commits. The same rule as above for that order.
            cone_first = [c for c in cone_first if not needs_pandas(c)] + [c for c in cone_first if needs_pandas(c)]
            later = [c for c in fresh if c not in set(cone_first)]
            needers = [c for c in cone_first + hot + later if needs_pandas(c)]
            pandas_turn = needers[0] if len(needers) >= 2 else None
    cone_done = set()

    def cone_loop(todo):
        """Runs the cone commits of todo, each in its own child. Returns the commits that got
        their answer; the others (REDO) are imported whole below."""
        done = set()
        cone_index = None
        if any(plans[c].swap_work for c in todo):
            # the live functions, for the cone children that replace code objects (gc.get_objects
            # leaves out frozen objects: preimport_for_fresh froze them)
            t_idx = time.process_time()
            gc.unfreeze()
            cone_index = engine.FunctionIndex(ROOT)
            gc.freeze()
            log("cone function index in", round(time.process_time() - t_idx, 3))
        for c in todo:
            if c == pandas_turn:
                load_pandas_here()
            plan = plans[c]
            plan.mode = "fresh_cone"
            if execute_safe(c, plan, cone_index if plan.swap_work else None) == "REDO":
                plan.cone = []  # the checks failed: the whole tree is imported below
                plan.mode = "fresh"
            else:
                done.add(c)
        return done

    def cone_in_holder(todo):
        """Runs cone_loop(todo) in a forked holder process, while this process waits without
        running any Python code. Why: the work of this process while a child lives (the
        import-state comparison of each cone commit) writes to memory that it shares with
        that child, and every such write splits a 2 MB page of this process into 4 kB pages
        for good. Then each later fork copies, and each child exit tears down, about ten
        times more page table entries (measured: about 10 ms more CPU per later child).
        In the holder, those writes split only the holder's pages. Nothing is loaded twice:
        the holder inherits everything that this process loaded before the cone step, and
        this process continues from the same state as after cone_loop. Returns the commits
        that the holder answered (their answers are filled in here) and the commits whose
        cone checks failed (REDO: imported whole below, as after cone_loop). Commits without
        a report from the holder are left to cone_loop in this process."""
        r, w = os.pipe()
        cpu_before = sum(os.times()[:4])
        pid = os.fork()
        if pid == 0:
            try:
                os.close(r)
                pin_carry(cpu_before)
                done = cone_loop(todo)
                out = {"done": {c: answers[c] for c in done if c in answers},
                       "redo": [c for c in todo if c not in done]}
                if DEBUG:
                    out["debug"] = json.dumps({c: debug_rows[c] for c in done if c in debug_rows})
                data = marshal.dumps(out)
                while data:
                    data = data[os.write(w, data):]
            except BaseException as e:  # noqa: BLE001
                try:
                    log("cone holder failed", repr(e))
                except BaseException:  # noqa: BLE001
                    pass
            finally:
                os._exit(0)
        os.close(w)
        chunks = []
        try:
            while True:
                left = deadline - time.monotonic() - 5.0
                if left <= 0:
                    break
                ready, _, _ = select.select([r], [], [], left)
                if not ready:
                    break
                chunk = os.read(r, 1 << 16)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(r)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)
            try:
                while os.waitpid(-1, os.WNOHANG)[0] > 0:
                    pass
            except ChildProcessError:
                pass
        try:
            out = marshal.loads(b"".join(chunks))
        except (EOFError, ValueError, TypeError):
            log("cone holder: no report")
            return set(), set()
        done = set(out["done"])
        for c, workers in out["done"].items():
            answers[c] = workers
        if DEBUG and "debug" in out:
            debug_rows.update(json.loads(out["debug"]))
        redo = set(out["redo"]) - done
        log("cone holder:", len(todo), "commits,", len(done), "answered,", len(redo), "to import whole")
        for c in redo:
            plans[c].cone = []
            plans[c].mode = "fresh"
        return done, redo

    if cone_first and _holder_pays():
        if pandas_turn in cone_first and any(needs_pandas(c) for c in hot + [f for f in fresh if f not in set(cone_first)]):
            # numpy and pandas are needed after the cone step too: loaded here, once, before
            # the holder starts (it inherits them)
            load_pandas_here()
        held_done, held_redo = cone_in_holder(cone_first)
        cone_done |= held_done
        rest = [c for c in cone_first if c not in held_done and c not in held_redo]
        if pandas_turn in cone_first and not pandas_here:
            # the holder loaded numpy and pandas for itself: this process loads them before
            # its own first commit that needs them
            pandas_turn = next((c for c in rest + hot + [f for f in fresh if f not in held_done]
                                if needs_pandas(c)), None)
        cone_done |= cone_loop(rest)
    else:
        cone_done |= cone_loop(cone_first)
    if cone_done:
        fresh = [c for c in fresh if c not in cone_done]
    if hot:
        mods = sorted({tests[n]["file"][:-3].replace("/", ".") for c in hot for n in plans[c].cands})
        preload.import_tests(mods)
        import reduce

        reduce.install_loaders()
        wanted = set()
        for c in hot:
            if not any(p.startswith("tests/") and p.endswith(".py") for p in plans[c].files):
                for n in plans[c].cands:
                    path, cls_name, _ = tests[n]["id"].split("::")
                    wanted.add((path[:-3].replace("/", "."), cls_name))
        shared = preload.build_class_fixtures(wanted, exec_lazy=True, exec_prebuild=exec_prebuild(hot))
        log("shared fixtures", shared)
        # the index lists the live functions with gc.get_objects(), which leaves out frozen
        # objects (the steps above may have frozen them: preimport_for_fresh, load_pandas_here)
        gc.unfreeze()
        index = engine.FunctionIndex(ROOT)
        gc.freeze()  # (no collection first: it would walk every object, and little is garbage)
    purge = bool(hot) or want_diff

    # 2b. Commits that change nothing but entries of .sql fixture files need no patched code:
    #     their tests run on the pinned code of this process, with the patched file text. One
    #     child serves all of them, one after the other (fixture_batch_child), instead of one
    #     fork each. The commits whose tests load numpy or pandas form a group of their own:
    #     as today, their child runs after this process has loaded those libraries.
    prepared = {}  # commit -> what prepare_plan gave

    def fixture_only(c):
        """True for a commit that the shared child can serve: only existing .sql files under
        tests/fixtures/ change, none of them read by a shared class fixture; no code to
        replace, no special handling of any kind (watch, added tests, blind selection, token
        check, tests known to be needed); every candidate runs (none is held), none is a long
        test, and none belongs to the class with the lazy fixture (TestExecutor: its tests
        build class attributes at their first use and set module globals through a pool
        initializer)."""
        plan = plans[c]
        if plan.mode != "hot" or plan.work or plan.watch or plan.added or plan.blind or plan.must or plan.all_fail:
            return False
        if plan.broken or plan.lines or plan.def_units or plan.fpairs or plan.gone or plan.infect_files or plan.sure_infected:
            return False
        if not plan.files or not plan.cands:
            return False
        for rel, text in plan.files.items():
            if text is None or not rel.startswith("tests/fixtures/") or not rel.endswith(".sql"):
                return False
            if not os.path.isfile(os.path.join(ROOT, rel)):
                return False
        class_fixture_rule(plan)  # as execute() does first for such a commit
        for info in shared_info:
            if info is not None and (set(info[2]) | info[3]).intersection(plan.files):
                return False
        lazy_class = preload.EXEC_KEY[0].replace(".", "/") + ".py::" + preload.EXEC_KEY[1] + "::"
        if any(seconds[n] > QUIET_HOLD_SECONDS or tests[n]["id"].startswith(lazy_class) for n in plan.cands):
            return False
        ran, held, allowed, skip, cap = prepare_plan(plan)
        if not ran or held or plan.tokcheck or plan.enders or set(ran) != plan.cands:
            return False
        prepared[c] = (ran, held, allowed, skip, cap)
        return True

    def run_fixture_batch(members):
        """One child for these commits. Returns the commits that got their answer; the others
        (a check of the shared child failed) run in their own child, as before."""
        left = deadline - time.monotonic()
        if left < 20.0:
            return set()
        batch = [(plans[m],) + tuple(prepared[m][i] for i in (0, 2, 3, 4)) for m in members]
        done = run_fixture_batch_child(batch, tests, left - 2.0)
        answered = set()
        for i, m in enumerate(members):
            if i not in done:
                continue
            verdicts, skipped, times = done[i]
            plan = plans[m]
            plan.skipped.update(skipped)
            plan.times.update(times)
            try:
                conclude(m, plan, verdicts, "END", None, prepared[m][0], prepared[m][1], index)
                answered.add(m)
                if DEBUG:
                    debug_rows[m]["batch"] = len(members)
            except Exception as e:  # noqa: BLE001  (its own child, and execute_safe, decide)
                log(m, "fixture batch: no answer", repr(e))
                plan.skipped.clear()
                plan.times.clear()
        log("fixture batch:", len(members), "commits,", len(answered), "answered")
        return answered

    groups = {}  # commit -> its tests load numpy or pandas
    if FIXTURE_BATCH and not RUN_ALL and not NO_REDUCE and len(hot) >= FIXTURE_BATCH_MIN:
        for c in hot:
            try:
                if fixture_only(c):
                    groups[c] = needs_pandas(c)
            except Exception as e:  # noqa: BLE001  (its own child, as before)
                log(c, "fixture batch: not planned", repr(e))
        log("fixture-only commits:", sum(1 for g in groups.values() if not g), "and", sum(1 for g in groups.values() if g), "with pandas")
        for key in (False, True):
            if sum(1 for g in groups.values() if g == key) < FIXTURE_BATCH_MIN:
                groups = {c: g for c, g in groups.items() if g != key}
    answered = set()
    for c in hot:
        if c in answered:
            continue
        if c == pandas_turn:
            load_pandas_here()
        if c in groups:
            members = [m for m in hot if groups.get(m) == groups[c]]
            for m in members:
                del groups[m]  # one try
            try:
                answered.update(run_fixture_batch(members))
            except Exception as e:  # noqa: BLE001
                log("fixture batch failed", repr(e))
            if c in answered:
                continue
        redo = execute_safe(c, plans[c], index)
        if redo in ("REDO", "REDOC"):
            fresh.append(c)
            plan = plans[c]
            plan.note.append("redo" if redo == "REDO" else "redo: closure cells")
            plan.diffable = False
            for p in (plan.files if redo == "REDO" else ()):
                rec = maps.file(p)
                if rec is not None:
                    plan.cands.update(maps.tests(rec[7]))
                    plan.cands.update(maps.tests(rec[8]))
            plan.keeps = {n: k for n, k in plan.keeps.items() if k[4] is None}
    preimport_once()
    for c in fresh:
        if c == pandas_turn:
            load_pandas_here()
        plan = plans[c]
        plan.mode = "fresh_purge" if purge else "fresh"
        execute_safe(c, plan, None)

    if DEBUG:
        with open(DEBUG, "w") as f:
            json.dump(debug_rows, f)


if __name__ == "__main__":
    main()
