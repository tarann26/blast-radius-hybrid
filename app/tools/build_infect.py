#!/usr/bin/env python3
"""Builds maps/infect.bin, the operand outcome map (condition coverage with operand relations)
of the pinned sqlglot commit. See infect.py for what the map holds and how it is used.

The map holds only facts about runs of the pinned commit's own test suite: for every
comparison site in sqlglot/ and every test, which operand relations (signatures) occurred.
Nothing in it is computed from patches, mutants, labels or edit kinds.

Steps (all inside the pinned image: python:3.13-slim + pip install -r requirements.lock):
  1. copy the pinned checkout, keeping modification times;
  2. instrument.py writes instrumented .pyc files next to the unchanged sources;
  3. run the full suite (python -m pytest tests, SKIP_INTEGRATION=0) with the probe
     (plugin/_brprobe.py, plugin/brinfect_plugin.py) once per PYTHONHASHSEED;
  4. merge: the union of all runs, with every evaluation attributed to the test phase that
     made it (call phase: the test; setup/teardown: every test of that file; pool workers:
     the test that submitted the task; module import, collection, between tests: global).

usage:
  build_infect.py all --src SQLGLOT_CHECKOUT --work DIR --seeds 0,1,2 --out maps/infect.bin --tests tests.json
  build_infect.py run --repo DIR --probe DIR --seed N        (one instrumented suite run)
  build_infect.py merge --out maps/infect.bin --tests tests.json --sites SITES.json RUN_PROBE_DIR...
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
OP_INDEX = {n: i for i, n in enumerate(("Is", "IsNot", "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "In", "NotIn"))}
GLOBAL_TAGS = {"?", "|collect", "|between", "|end", "G"}
COPY = ("sqlglot", "tests", "setup.py", "setup.cfg", "pyproject.toml", "README.md", "Makefile", "MANIFEST.in",
        "sqlglotc", "benchmarks", "pdoc", "posts", "sqlglot-integration-tests")


def prepare(src, repo):
    if os.path.exists(repo):
        shutil.rmtree(repo)
    os.makedirs(repo)
    for name in COPY:
        p = os.path.join(src, name)
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(repo, name), ignore=shutil.ignore_patterns("__pycache__"))
        elif os.path.exists(p):
            shutil.copy2(p, os.path.join(repo, name))
    subprocess.run([sys.executable, os.path.join(HERE, "instrument.py"), repo], check=True)


def run(repo, probe, seed):
    os.makedirs(probe, exist_ok=True)
    env = dict(os.environ, PYTHONHASHSEED=str(seed), SKIP_INTEGRATION="0", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=repo + os.pathsep + PLUGIN, BR_PROBE_DIR=probe, BR_OUT=os.path.join(probe, "_result.json"))
    env.pop("BR_PROBE_TAG", None)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "brplugin",
                        "-p", "brinfect_plugin", "tests"], cwd=repo, env=env)
    return p.returncode


def read_run(probe):
    """{key: set of tags}, set of tags that used a process pool, result json."""
    keys = defaultdict(set)
    pool_tags = set()
    for fn in os.listdir(probe):
        if not fn.endswith(".txt"):
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
                    kind = parts[2]
                    tag = "W" if kind == "worker" else parts[3]
                elif c == "T":
                    tag = line[2:]
                elif c == "S":
                    pool_tags.add(line[2:])
                elif c == "G":
                    keys[int(line[1:], 16)].add("G")
                elif c == "X":
                    keys[int(line[1:], 16)].add("X")
                else:
                    keys[int(line, 16)].add(tag)
    res = None
    rp = os.path.join(probe, "_result.json")
    if os.path.exists(rp):
        with open(rp) as f:
            res = json.load(f)
    return keys, pool_tags, res


class Attribution:
    def __init__(self, tests):
        self.num = {t["id"]: t["number"] for t in tests}
        self.by_file = defaultdict(set)
        for t in tests:
            self.by_file[t["file"]].add(t["number"])
        self.unknown = defaultdict(int)

    def tag(self, tag, pool):
        """Returns (set of tests, global flag) for one phase tag."""
        if tag in GLOBAL_TAGS:
            return set(), True
        if tag in ("W", "X"):
            return pool
        nodeid, _, phase = tag.rpartition("|")
        if phase == "c":
            n = self.num.get(nodeid)
            if n is None:
                self.unknown[nodeid] += 1
                return set(), True
            return {n}, False
        if phase in ("s", "t"):
            f = nodeid.split("::")[0]
            s = self.by_file.get(f)
            if not s:
                self.unknown[nodeid] += 1
                return set(), True
            return set(s), False
        self.unknown[tag] += 1
        return set(), True


def attribute(keys, pool_tags, att):
    pool = set()
    pg = False
    for t in pool_tags:
        s, g = att.tag(t, (set(), True))
        pool |= s
        pg |= g
    out = {}
    cache = {}
    for k, tags in keys.items():
        tests = set()
        g = False
        for t in tags:
            r = cache.get(t)
            if r is None:
                r = cache[t] = att.tag(t, (pool, pg))
            tests |= r[0]
            g |= r[1]
        out[k] = (tests, g)
    return out


def equality_code(repo):
    """Line ranges of the __eq__ / __ne__ / __hash__ methods in sqlglot/. The probe takes ==
    of two sqlglot expressions from Expression.__eq__ of the pinned commit; an edit inside
    that code changes what == means, so comparisons inside these methods are left out of
    the map (their edits take the selector's normal path)."""
    import ast

    ranges = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(os.path.join(repo, "sqlglot")):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, repo)
            with open(path, "rb") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ("__eq__", "__ne__", "__hash__"):
                    ranges[rel].append((node.lineno, node.end_lineno))
    return ranges


def merge(out_path, tests_path, sites_path, probes, report=None, repo=None):
    with open(tests_path) as f:
        tests = json.load(f)["tests"]
    with open(sites_path) as f:
        sites = json.load(f)
    att = Attribution(tests)
    runs = []
    for p in probes:
        keys, pool_tags, res = read_run(p)
        bad = res is None or res.get("exitstatus") != 0 or res.get("failed") or res.get("collect_errors")
        runs.append(attribute(keys, pool_tags, att))
        print(p, "keys", len(keys), "pool tags", len(pool_tags), "result", None if res is None else
              (res.get("exitstatus"), len(res.get("failed", {})), len(res.get("durations", {}))), "BAD RUN" if bad else "", file=sys.stderr)
        if bad:
            raise SystemExit("a map run did not pass like the plain suite: " + p)
    # union over runs
    union = {}
    for r in runs:
        for k, (ts, g) in r.items():
            u = union.get(k)
            if u is None:
                union[k] = (set(ts), g)
            else:
                u[0].update(ts)
                union[k] = (u[0], u[1] or g)
    # seed-to-seed differences
    stats = {}
    if len(runs) > 1:
        pair_sets = []
        for r in runs:
            ps = set()
            for k, (ts, g) in r.items():
                for t in ts:
                    ps.add((k, t))
                if g:
                    ps.add((k, -1))
            pair_sets.append(ps)
        allp = set().union(*pair_sets)
        common = set.intersection(*pair_sets)
        diff = allp - common
        stats["pairs_total"] = len(allp)
        stats["pairs_not_in_every_run"] = len(diff)
        stats["sites_with_difference"] = len({k >> 16 for k, _ in diff})
        stats["keys_not_in_every_run"] = len(set.union(*[set(r) for r in runs]) - set.intersection(*[set(r) for r in runs]))
        stats["tests_with_difference"] = len({t for _, t in diff})
        unstable = {k >> 16 for k, _ in diff}
        unstable |= {k >> 16 for k in set.union(*[set(r) for r in runs]) - set.intersection(*[set(r) for r in runs])}
    else:
        unstable = set()
    # build the map
    set_ids = {(): 0}
    set_list = [b""]
    by_site = defaultdict(list)
    for k, (ts, g) in union.items():
        by_site[k >> 16].append((k & 0xFFFF, ts, g))
    files = defaultdict(list)
    excluded = equality_code(repo or os.path.dirname(os.path.abspath(sites_path)))
    n_excluded = 0
    n_unstable = 0
    for sid, (rel, l1, c1, l2, c2, op) in enumerate(sites):
        if any(a <= l1 <= b for a, b in excluded.get(rel, ())):
            n_excluded += 1
            continue
        if sid in unstable:
            # the operand relations at this site differ between runs (hash seed, ordering):
            # a map from a few runs cannot be trusted there, so the site is left out
            n_unstable += 1
            continue
        recs = []
        for sig, ts, g in sorted(by_site.get(sid, ()), key=lambda x: x[0]):
            key = tuple(sorted(ts))
            i = set_ids.get(key)
            if i is None:
                i = set_ids[key] = len(set_list)
                set_list.append(array("H", key).tobytes())
            recs += [sig, i, 1 if g else 0]
        files[rel].append((l1, c1, l2, c2, OP_INDEX[op], tuple(recs)))
    m = {
        "version": 1,
        "files": {rel: zlib.compress(marshal.dumps(rows), 9) for rel, rows in files.items()},
        "sets": zlib.compress(marshal.dumps(set_list), 9),
    }
    data = marshal.dumps(m)
    with open(out_path, "wb") as f:
        f.write(data)
    stats.update(bytes=len(data), sites=len(sites), excluded_equality_sites=n_excluded, excluded_unstable_sites=n_unstable, evaluated_sites=len(by_site), sets=len(set_list),
                 unknown_tags=dict(att.unknown), runs=len(runs))
    print(json.dumps(stats), file=sys.stderr)
    if report:
        with open(report, "w") as f:
            json.dump(stats, f, indent=1)
    return stats


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("all")
    a.add_argument("--src", required=True)
    a.add_argument("--work", required=True)
    a.add_argument("--seeds", default="0,1,2")
    a.add_argument("--out", required=True)
    a.add_argument("--tests", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo", required=True)
    r.add_argument("--probe", required=True)
    r.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.add_argument("--tests", required=True)
    m.add_argument("--sites", required=True)
    m.add_argument("--report")
    m.add_argument("probes", nargs="+")
    o = ap.parse_args()
    if o.cmd == "all":
        repo = os.path.join(o.work, "repo")
        prepare(o.src, os.path.join(o.work, "tree"))
        probes = []
        for seed in o.seeds.split(","):
            rd = os.path.join(o.work, "run%s" % seed)
            if os.path.exists(rd):
                shutil.rmtree(rd)
            shutil.copytree(os.path.join(o.work, "tree"), rd + "/repo", symlinks=True)
            rc = run(rd + "/repo", rd + "/probe", int(seed))
            print("seed", seed, "pytest exit", rc, file=sys.stderr)
            probes.append(rd + "/probe")
        merge(o.out, o.tests, os.path.join(o.work, "tree", "_infect_sites.json"), probes)
    elif o.cmd == "run":
        sys.exit(run(os.path.abspath(o.repo), os.path.abspath(o.probe), o.seed))
    else:
        merge(o.out, o.tests, o.sites, o.probes, o.report)


if __name__ == "__main__":
    main()
