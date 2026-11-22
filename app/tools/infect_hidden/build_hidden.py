#!/usr/bin/env python3
"""Builds maps/infect_hidden.bin: for every site of the operand outcome maps (infect.bin,
infect2.bin), the tests that a cache hides from those maps.

Why: the outcome maps were recorded in runs of the whole suite in one process. Code that runs
once per process and whose result sqlglot keeps (generator._build_dispatch fills
_DISPATCH_CACHE for the first user of each generator class) shows its evaluations only in the
first test that needed it; every later reader of the kept result depends on them without a
record of its own.

What is recorded (reach probes, see instrument_reach.py and plugin/_brprobe.py; only runs of the
pinned commit's own suite, no patches, labels or edit kinds). For every mapped site: how often
each test phase REACHES the site's statement and how often it STARTS the site's expression,
  warm  in runs of the whole suite in one process, once per PYTHONHASHSEED (as for the outcome maps);
  cold  with every test alone in a process of its own ("pytest <test id>").
Records made while a sqlglot module body runs (imports) are not counted on either side: the
outcome maps hold import-time evaluations as records of no single test, and the selector does
not use the maps when one of those changes.

A test is HIDDEN at a site when, alone, it does more there than in its own phases of the
whole-suite run: something kept from an earlier test stands in for the difference. In detail
  call phase     more reaches of the statement, or more starts of the site, alone than in the
                 test's own call phase of the whole-suite run (pool tasks count for the test
                 that submitted them; other threads count only for the cold side);
  fixture phases (setup, teardown: setUpClass and the like) the same, against the largest
                 count of a fixture phase of the test's class in the whole-suite run (the
                 outcome maps attribute fixture-phase evaluations to every test of the file);
  line map       a test that the selector's line coverage map (MAPS_DIR) shows on the
                 statement's lines, while its cold run here never got there: hidden too,
                 unless its process got there outside the test's phases (in a module body or
                 during collection: import-time and collection records of the outcome maps),
                 or the site is in a lambda body (the line also belongs to the statement that
                 defines the lambda), or the test got to the statement in its call phase of
                 the whole-suite run.
The cold runs are compared with the warm run of the same hash seed (the first WARM_PROBE_DIR;
at 16 sites the counts depend on the seed: loops over a set that stop at the first match). A
site where the seeds disagree on WHICH tests get there gets no row: the selector then keeps
its wider rule (every test that runs the statement's lines without a recorded evaluation).

Steps (all inside the pinned image; build_hidden.sh runs them in containers):
  build_hidden.py prepare --src CHECKOUT --repo TREE --maps MAPS_DIR
  build_hidden.py run --repo TREE --probe DIR --seed N           (one warm suite run)
  build_hidden.py cold --repo TREE --probe DIR --tests tests.json --part I/N [--seed 0]
  build_hidden.py merge --repo TREE --maps MAPS_DIR --tests tests.json --out MAPS_DIR/infect_hidden.bin
                        --cold COLD_DIR [--report R.json] [--detail D.json] WARM_PROBE_DIR...
                        (the warm run with the hash seed of the cold runs first)
"""
import argparse
import json
import marshal
import os
import shutil
import subprocess
import sys
import zlib
from array import array
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.join(HERE, "plugin")
GLOBAL_TAGS = {"?", "|collect", "|between", "|end"}
COPY = ("sqlglot", "tests", "setup.py", "setup.cfg", "pyproject.toml", "README.md", "Makefile", "MANIFEST.in",
        "sqlglotc", "benchmarks", "pdoc", "posts", "sqlglot-integration-tests")


def prepare(src, repo, maps):
    if os.path.exists(repo):
        shutil.rmtree(repo)
    os.makedirs(repo)
    for name in COPY:
        p = os.path.join(src, name)
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(repo, name), ignore=shutil.ignore_patterns("__pycache__"))
        elif os.path.exists(p):
            shutil.copy2(p, os.path.join(repo, name))
    subprocess.run([sys.executable, os.path.join(HERE, "instrument_reach.py"), repo, maps], check=True)


def _env(repo, probe, seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed), SKIP_INTEGRATION="0", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=repo + os.pathsep + PLUGIN, BR_PROBE_DIR=probe, BR_OUT=os.path.join(probe, "_result.json"))
    env.pop("BR_PROBE_TAG", None)
    env.pop("BR_ONLY", None)
    return env


PYTEST = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "brplugin", "-p", "brinfect_plugin", "tests"]


def run(repo, probe, seed):
    os.makedirs(probe, exist_ok=True)
    return subprocess.run(PYTEST, cwd=repo, env=_env(repo, probe, seed)).returncode


def cold(repo, probe, tests_path, part, seed):
    """Every test of this part alone in a fresh process (pytest gets the one test id)."""
    with open(tests_path) as f:
        tests = json.load(f)["tests"]
    i, n = (int(x) for x in part.split("/"))
    for t in tests:
        if t["number"] % n != i:
            continue
        d = os.path.join(probe, str(t["number"]))
        if os.path.exists(os.path.join(d, "rc")):
            continue
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
        p = subprocess.run(PYTEST[:-1] + [t["id"]], cwd=repo, env=_env(repo, d, seed), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with open(os.path.join(d, "tail.txt"), "w") as f:
            f.write(p.stdout[-400:])
        with open(os.path.join(d, "rc"), "w") as f:
            f.write(str(p.returncode))


def read_probe(probe):
    """One probe directory -> ({(tag, slot, number): count}, {(kind, number, tag)}): the counts
    per test phase (slot 0 regions, 1 sites, 2 regions in other threads, 3 sites in other
    threads) and the import-time records (kind "R" or "s")."""
    counts = defaultdict(int)
    imp = set()
    for fn in os.listdir(probe):
        if not fn.endswith(".txt") or fn.startswith("_") or fn == "tail.txt":
            continue
        tag = "?"
        with open(os.path.join(probe, fn)) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                c = line[0]
                if c == "P":
                    parts = line.split(" ", 3)
                    tag = "W" if parts[2] == "worker" else parts[3]
                elif c == "T":
                    tag = line[2:]
                elif c == "C":
                    for slot, d in enumerate(json.loads(line[2:])):
                        for k, v in d.items():
                            counts[(tag, slot, int(k))] += v
                elif (c == "R" or c == "s") and line[1] == "G":
                    imp.add((c, int(line[2:], 16), tag))
    return counts, imp


def result_ok(probe, want_ran=None):
    rp = os.path.join(probe, "_result.json")
    if not os.path.exists(rp):
        return False, "no result"
    with open(rp) as f:
        res = json.load(f)
    if res.get("exitstatus") != 0 or res.get("failed") or res.get("collect_errors"):
        return False, "exit %s, %d failed" % (res.get("exitstatus"), len(res.get("failed", {})))
    if want_ran is not None and want_ran not in res.get("durations", {}):
        return False, "the test did not run"
    return True, len(res.get("durations", {}))


def _phase(tag):
    """(test id, "c" | "s" | "t") of a phase tag: call, setup or teardown phase of a test."""
    nodeid, _, phase = tag.rpartition("|")
    if phase in ("c", "s", "t"):
        return nodeid, phase
    return None, None


def warm_counts(probe, tests):
    """One warm run -> {(slot, number): ({test: count in its call phase}, {(class, "s" | "t"):
    largest count of one fixture phase of that kind in the class})}; slot 0 regions, 1 sites.
    Only main-thread counts of a test's own phases: worker start-up, other threads,
    collection and the time between tests belong to nobody here."""
    num = {t["id"]: t["number"] for t in tests}
    out = {}
    counts, _ = read_probe(probe)
    for (tag, slot, n), v in counts.items():
        if slot > 1 or tag in GLOBAL_TAGS or tag == "W":
            continue
        nodeid, phase = _phase(tag)
        if nodeid not in num:
            continue
        rec = out.get((slot, n))
        if rec is None:
            rec = out[(slot, n)] = ({}, {})
        if phase == "c":
            rec[0][num[nodeid]] = rec[0].get(num[nodeid], 0) + v
        else:
            k = (nodeid.rpartition("::")[0], phase)
            if v > rec[1].get(k, 0):
                rec[1][k] = v
    return out


def cold_counts(cold_dir, tests):
    """The cold runs -> {(slot, number): {test: [call count, setup count, teardown count]}} and
    {(slot, number): tests whose process got there outside the test's own phases: inside a
    module body (an import, whenever it happened) or while pytest started and collected}.
    Everything else the test's process did counts for the test: its phases, its pool
    workers, its threads."""
    out = {}
    imp = defaultdict(set)
    for t in tests:
        d = os.path.join(cold_dir, str(t["number"]))
        ok, why = result_ok(d, t["id"]) if os.path.isdir(d) else (False, "missing")
        if not ok:
            raise SystemExit("cold run of test %d: %s" % (t["number"], why))
        counts, im = read_probe(d)
        for (tag, slot, n), v in counts.items():
            if tag in GLOBAL_TAGS:
                imp[(slot % 2, n)].add(t["number"])  # start of pytest and collection
                continue
            phase = _phase(tag)[1] or "c"
            rec = out.get((slot % 2, n))
            if rec is None:
                rec = out[(slot % 2, n)] = {}
            row = rec.get(t["number"])
            if row is None:
                row = rec[t["number"]] = [0, 0, 0]
            row["cst".index(phase)] += v
        for kind, n, tag in im:
            imp[(0 if kind == "R" else 1, n)].add(t["number"])
    return out, imp


class LineMap:
    """The selector's cold line coverage map (maps/f, sets.idx, sets.bin, meta.bin)."""

    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "meta.bin"), "rb") as f:
            self.files = marshal.load(f)["files"]
        with open(os.path.join(path, "sets.idx"), "rb") as f:
            self.offsets = array("I")
            self.offsets.frombytes(f.read())
        with open(os.path.join(path, "sets.bin"), "rb") as f:
            self.blob = f.read()
        self.recs = {}

    def tests(self, set_id):
        a = array("H")
        a.frombytes(self.blob[self.offsets[set_id]:self.offsets[set_id + 1]])
        return a

    def statement_runners(self, rel, l1, l2):
        """Tests that run the statement lines l1..l2 (as plan_python_file_infect reads them)."""
        rec = self.recs.get(rel)
        if rec is None:
            with open(os.path.join(self.path, "f", "%d.bin" % self.files[rel]), "rb") as f:
                rec = self.recs[rel] = marshal.loads(zlib.decompress(f.read()))
        stmt_set = rec[3]
        out = set()
        for ln in range(l1, l2 + 1):
            if 0 <= ln < len(stmt_set) and stmt_set[ln]:
                out.update(self.tests(stmt_set[ln]))
        return out


def hidden_by_counts(cold, warm, klass):
    """Tests that do more alone than in their own phases of the whole-suite run.
    cold: {test: [call, setup, teardown]}; warm: ({test: call count}, {(class, phase): count})."""
    w_call, w_fix = warm
    call, fix = set(), set()
    for t, (c, s, td) in cold.items():
        if c > w_call.get(t, 0):
            call.add(t)
        if s > w_fix.get((klass[t], "s"), 0) or td > w_fix.get((klass[t], "t"), 0):
            fix.add(t)
    return call, fix


def _who(warm):
    """Which tests (call phase) and which class fixture phases have a count at all."""
    return set(warm[0]), set(warm[1])


def merge(repo, maps, tests_path, out_path, cold_dir, warm, report=None, detail=None):
    with open(tests_path) as f:
        tests = json.load(f)["tests"]
    klass = {t["number"]: t["id"].rpartition("::")[0] for t in tests}
    with open(os.path.join(repo, "_reach_sites.json")) as f:
        meta = json.load(f)
    sites, regions = meta["sites"], meta["regions"]
    runs = []
    for p in warm:
        ok, why = result_ok(p)
        if not ok or why < len(tests):
            raise SystemExit("a warm run did not pass like the plain suite: %s (%s)" % (p, why))
        runs.append(warm_counts(p, tests))
    cold, cold_imp = cold_counts(cold_dir, tests)
    lm = LineMap(maps)
    files = defaultdict(list)
    stats = defaultdict(int)
    rows_detail = []
    none2 = ({}, {})
    for sid, (rel, l1, c1, l2, c2, kind, rid) in enumerate(sites):
        stats["sites"] += 1
        if rid < 0:
            stats["no_region"] += 1
            continue
        wr = [r.get((0, rid), none2) for r in runs]
        ws = [r.get((1, sid), none2) for r in runs]
        if any(_who(x) != _who(wr[0]) for x in wr) or any(_who(x) != _who(ws[0]) for x in ws):
            # which tests get there depends on the hash seed: no row, the selector's wide rule stays
            stats["unstable"] += 1
            if detail:
                rows_detail.append({"file": rel, "key": [l1, c1, l2, c2], "kind": kind, "region": regions[rid], "unstable": True})
            continue
        if any(x != wr[0] for x in wr) or any(x != ws[0] for x in ws):
            # the same tests get there, but how often depends on the hash seed (a loop over a
            # set that stops at the first match): the cold runs are compared with the warm
            # run of their own seed, which is the first one given
            stats["counts_differ_between_seeds"] += 1
        wr, ws = wr[0], ws[0]
        c_reach = cold.get((0, rid), {})
        c_start = cold.get((1, sid), {})
        why = {}
        why["call"], why["fixture"] = hidden_by_counts(c_reach, wr, klass)
        why["site_call"], why["site_fixture"] = hidden_by_counts(c_start, ws, klass)
        lines = lm.statement_runners(rel, l1, l2)
        if regions[rid][1] != "L":
            why["lines"] = lines - set(c_reach) - cold_imp.get((0, rid), set()) - set(wr[0])
        hidden = set().union(*why.values())
        stats["rows"] += 1
        if hidden:
            stats["rows_with_hidden"] += 1
            stats["hidden_pairs"] += len(hidden)
        for k, v in why.items():
            if v:
                stats["rows_" + k] += 1
                stats["pairs_" + k] += len(v)
        files[rel].append((l1, c1, l2, c2, array("H", sorted(hidden)).tobytes()))
        if detail:
            w_call = set(wr[0])
            rows_detail.append({"file": rel, "key": [l1, c1, l2, c2], "kind": kind, "region": regions[rid],
                                "lines": len(lines), "w_call": len(w_call), "ws_call": len(ws[0]),
                                "c_reach": len(c_reach), "c_start": len(c_start), "c_imp": len(cold_imp.get((0, rid), ())),
                                "why": {k: sorted(v) for k, v in why.items() if v},
                                "plain": len(lines - w_call), "hidden": len(hidden),
                                "zero_warm": len({t for t in why["call"] if t not in w_call}),
                                "cold_not_in_lines": len(set(c_reach) - lines), "warm_not_cold": len(w_call - set(c_reach))})
    m = {"version": 1, "files": {rel: zlib.compress(marshal.dumps(rows), 9) for rel, rows in sorted(files.items())}}
    data = marshal.dumps(m)
    with open(out_path, "wb") as f:
        f.write(data)
    stats = dict(stats, bytes=len(data), regions=len(regions), warm_runs=len(runs), cold_runs=len(tests))
    print(json.dumps(stats), file=sys.stderr)
    if report:
        with open(report, "w") as f:
            json.dump(stats, f, indent=1)
    if detail:
        with open(detail, "w") as f:
            json.dump(rows_detail, f)
    return stats


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--src", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--maps", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo", required=True)
    r.add_argument("--probe", required=True)
    r.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("cold")
    c.add_argument("--repo", required=True)
    c.add_argument("--probe", required=True)
    c.add_argument("--tests", required=True)
    c.add_argument("--part", default="0/1")
    c.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("merge")
    m.add_argument("--repo", required=True)
    m.add_argument("--maps", required=True)
    m.add_argument("--tests", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--cold", required=True)
    m.add_argument("--report")
    m.add_argument("--detail")
    m.add_argument("warm", nargs="+")
    o = ap.parse_args()
    if o.cmd == "prepare":
        prepare(o.src, o.repo, o.maps)
    elif o.cmd == "run":
        sys.exit(run(os.path.abspath(o.repo), os.path.abspath(o.probe), o.seed))
    elif o.cmd == "cold":
        cold(os.path.abspath(o.repo), os.path.abspath(o.probe), o.tests, o.part, o.seed)
    else:
        merge(o.repo, o.maps, o.tests, o.out, o.cold, o.warm, o.report, o.detail)


if __name__ == "__main__":
    main()
