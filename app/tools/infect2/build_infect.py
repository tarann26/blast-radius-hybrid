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
  build_infect.py all2 --src SQLGLOT_CHECKOUT --work DIR --seeds 0,1,2 --out maps/infect2.bin --tests tests.json
      (phase 2: count run, hot-site skip list, one probed suite run per seed, merge2; about 4 + 8 min per seed)
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


GROUPS = {
    # phase 2 probes are split over several suite runs (each run probes one group of files),
    # so that every run stays well below the machine's 25 minute limit for test containers
    "g1": r"^sqlglot/(parser\.py|parsers/|tokenizer_core\.py|tokens\.py)",
    "g2": r"^sqlglot/(generator\.py|generators/)",
    "g3": r"^sqlglot/(optimizer/|executor/|planner\.py)",
    "g4": r"^sqlglot/(?!parser\.py|parsers/|tokenizer_core\.py|tokens\.py|generator\.py|generators/|optimizer/|executor/|planner\.py)",
}


def prepare(src, repo, inst_args=()):
    if os.path.exists(repo):
        shutil.rmtree(repo)
    os.makedirs(repo)
    for name in COPY:
        p = os.path.join(src, name)
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(repo, name), ignore=shutil.ignore_patterns("__pycache__"))
        elif os.path.exists(p):
            shutil.copy2(p, os.path.join(repo, name))
    subprocess.run([sys.executable, os.path.join(HERE, "instrument.py"), repo, *(inst_args or ("--no-phase2",))], check=True)


def run(repo, probe, seed, tests=("tests",)):
    os.makedirs(probe, exist_ok=True)
    env = dict(os.environ, PYTHONHASHSEED=str(seed), SKIP_INTEGRATION="0", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=repo + os.pathsep + PLUGIN, BR_PROBE_DIR=probe, BR_OUT=os.path.join(probe, "_result.json"),
               BR_BSITES=os.path.join(repo, "_infect_bsites.json"))
    env.pop("BR_PROBE_TAG", None)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "brplugin",
                        "-p", "brinfect_plugin", *tests], cwd=repo, env=env)
    return p.returncode


def read_run(probe, extra=None):
    """{key: set of tags}, set of tags that used a process pool, result json. extra: dict that
    receives the phase 2 records: "K" {(site, code): tags}, "B" {(site, sig): tags}, "Z" stats."""
    keys = defaultdict(set)
    pool_tags = set()
    if extra is not None:
        extra.setdefault("K", defaultdict(set))
        extra.setdefault("B", defaultdict(set))
        extra.setdefault("Z", [])
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
                elif c == "K" or c == "B":
                    if extra is None:
                        continue
                    if line[1] in "GX":
                        t2, body = line[1], line[2:]
                    else:
                        t2, body = tag, line[1:]
                    s, _, v = body.partition(":")
                    extra[c][(int(s, 16), v if c == "K" else int(v, 16))].add(t2)
                elif c == "Z":
                    if extra is not None:
                        extra["Z"].append(json.loads(line[2:]))
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
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ("__eq__", "__ne__", "__hash__", "__bool__", "__len__"):
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


def _union_runs(runs):
    union = {}
    for r in runs:
        for k, (ts, g) in r.items():
            u = union.get(k)
            if u is None:
                union[k] = (set(ts), g)
            else:
                u[0].update(ts)
                union[k] = (u[0], u[1] or g)
    return union


def _unstable(runs, site_of):
    """Sites whose (record, test) pairs are not the same in every run."""
    if len(runs) < 2:
        return set()
    pair_sets = []
    for r in runs:
        ps = set()
        for k, (ts, g) in r.items():
            for t in ts:
                ps.add((k, t))
            if g:
                ps.add((k, -1))
        pair_sets.append(ps)
    diff = set().union(*pair_sets) - set.intersection(*pair_sets)
    return {site_of(k) for k, _ in diff}


def merge2(out_path, tests_path, repo, probes, report=None):
    """The phase 2 map (infect2.bin): and/or trees and constant comparisons. Same attribution,
    union over runs and exclusion of seed-dependent sites as the phase 1 map."""
    with open(tests_path) as f:
        tests = json.load(f)["tests"]
    with open(os.path.join(repo, "_infect_sites.json")) as f:
        sites = json.load(f)
    with open(os.path.join(repo, "_infect_bsites.json")) as f:
        bsites = json.load(f)
    with open(os.path.join(repo, "_infect_inst.json")) as f:
        inst = json.load(f)
    # only sites that the runs probed: a site left unprobed (too hot) has no records, which must
    # not read as "never evaluated"
    k_probed = set(inst["K"])
    b_probed = set(inst["B"])
    att = Attribution(tests)
    kruns, bruns, zs = [], [], []
    for p in probes:
        extra = {}
        keys, pool_tags, res = read_run(p, extra)
        bad = res is None or res.get("exitstatus") != 0 or res.get("failed") or res.get("collect_errors")
        if bad:
            raise SystemExit("a map run did not pass like the plain suite: " + p)
        kruns.append(attribute(extra["K"], pool_tags, att))
        bruns.append(attribute(extra["B"], pool_tags, att))
        tot = defaultdict(int)
        for z in extra["Z"]:
            for a, v in z.items():
                tot[a] += v
        zs.append(dict(tot))
        print(p, "K records", len(extra["K"]), "B records", len(extra["B"]), "probe stats", dict(tot), file=sys.stderr)
    kun = _unstable(kruns, lambda k: k[0])
    bun = _unstable(bruns, lambda k: k[0])
    kunion = _union_runs(kruns)
    bunion = _union_runs(bruns)
    excluded = equality_code(repo)
    set_ids = {(): 0}
    set_list = [b""]

    def set_id(ts):
        key = tuple(sorted(ts))
        i = set_ids.get(key)
        if i is None:
            i = set_ids[key] = len(set_list)
            set_list.append(array("H", key).tobytes())
        return i

    kby = defaultdict(list)
    for (sid, code), (ts, g) in kunion.items():
        kby[sid].append((code, ts, g))
    bby = defaultdict(list)
    for (sid, sig), (ts, g) in bunion.items():
        bby[sid].append((sig, ts, g))
    kfiles = defaultdict(list)
    bfiles = defaultdict(list)
    n_ex = n_un = 0
    for sid in sorted(k_probed):
        recs = kby.get(sid, ())
        rel, l1, c1, l2, c2, op = sites[sid]
        if any(a <= l1 <= b for a, b in excluded.get(rel, ())):
            n_ex += 1
            continue
        if sid in kun:
            n_un += 1
            continue
        flat = []
        for code, ts, g in sorted(recs, key=lambda x: x[0]):
            flat += [code, set_id(ts), 1 if g else 0]
        kfiles[rel].append((l1, c1, l2, c2, OP_INDEX[op], tuple(flat)))
    nb_ex = nb_un = 0

    def tup(d):
        return tuple(tup(x) if isinstance(x, list) else x for x in d)

    for sid, (rel, l1, c1, l2, c2, desc, ctx, srcs) in enumerate(bsites):
        if sid not in b_probed:
            continue
        if any(a <= l1 <= b for a, b in excluded.get(rel, ())):
            nb_ex += 1
            continue
        if sid in bun:
            nb_un += 1
            continue
        flat = []
        for sig, ts, g in sorted(bby.get(sid, ()), key=lambda x: x[0]):
            flat += [sig, set_id(ts), 1 if g else 0]
        bfiles[rel].append((l1, c1, l2, c2, tup(desc), ctx, tuple(flat)))
    m = {
        "version": 2,
        "kfiles": {rel: zlib.compress(marshal.dumps(rows), 9) for rel, rows in kfiles.items()},
        "bfiles": {rel: zlib.compress(marshal.dumps(rows), 9) for rel, rows in bfiles.items()},
        "sets": zlib.compress(marshal.dumps(set_list), 9),
    }
    data = marshal.dumps(m)
    with open(out_path, "wb") as f:
        f.write(data)
    assert not (set(kby) - k_probed) and not ({s for s, _ in bunion} - b_probed), "records of unprobed sites"
    stats = dict(bytes=len(data), ksites_probed=len(k_probed), bsites_probed=len(b_probed), ksites=len(kby), ksites_excluded_equality=n_ex, ksites_unstable=n_un,
                 kfiles_rows=sum(len(v) for v in kfiles.values()), bfiles_rows=sum(len(v) for v in bfiles.values()),
                 bsites=len(bsites), bsites_evaluated=len(bby), bsites_excluded_equality=nb_ex, bsites_unstable=nb_un,
                 sets=len(set_list), probe_stats=zs, unknown_tags=dict(att.unknown))
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
    r.add_argument("--tests", nargs="*", default=["tests"])
    pr = sub.add_parser("prepare")
    pr.add_argument("--src", required=True)
    pr.add_argument("--repo", required=True)
    pr.add_argument("--group", help="g1..g4 or all: probe only this group (phase 2 only, no phase 1 comparisons)")
    pr.add_argument("--count", action="store_true", help="only count evaluations of the phase 2 sites")
    pr.add_argument("--skip", help="json {K: [...], B: [...]}: sites left unprobed")
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.add_argument("--tests", required=True)
    m.add_argument("--sites", required=True)
    m.add_argument("--report")
    m.add_argument("probes", nargs="+")
    m2 = sub.add_parser("merge2")
    m2.add_argument("--out", required=True)
    m2.add_argument("--tests", required=True)
    m2.add_argument("--repo", required=True)
    m2.add_argument("--report")
    m2.add_argument("probes", nargs="+")
    a2 = sub.add_parser("all2")
    a2.add_argument("--src", required=True)
    a2.add_argument("--work", required=True)
    a2.add_argument("--seeds", default="0,1,2")
    a2.add_argument("--out", required=True)
    a2.add_argument("--tests", required=True)
    a2.add_argument("--hot", type=float, default=1e6, help="sites evaluated more often than this in the count run are not probed")
    o = ap.parse_args()
    if o.cmd == "all2":
        cnt_repo = os.path.join(o.work, "tree_count")
        prepare(o.src, cnt_repo, ("--group", "^sqlglot/", "--no-cmp", "--count"))
        cnt_probe = os.path.join(o.work, "count_probe")
        if os.path.exists(cnt_probe):
            shutil.rmtree(cnt_probe)
        rc = run(cnt_repo, cnt_probe, 0)
        if rc != 0:
            raise SystemExit("count run failed")
        K = defaultdict(int)
        B = defaultdict(int)
        for fn in os.listdir(cnt_probe):
            if fn.endswith(".txt"):
                with open(os.path.join(cnt_probe, fn)) as f:
                    for line in f:
                        if line.startswith("C "):
                            d = json.loads(line[2:])
                            for k, v in d["K"].items():
                                K[int(k)] += v
                            for k, v in d["B"].items():
                                B[int(k)] += v
        skip = {"K": sorted(k for k, v in K.items() if v > o.hot), "B": sorted(k for k, v in B.items() if v > o.hot),
                "threshold": o.hot}
        skip_path = os.path.join(o.work, "skip_hot.json")
        with open(skip_path, "w") as f:
            json.dump(skip, f)
        tree = os.path.join(o.work, "tree_all")
        prepare(o.src, tree, ("--group", "^sqlglot/", "--no-cmp", "--skip", skip_path))
        probes = []
        for seed in o.seeds.split(","):
            rd = os.path.join(o.work, "run%s" % seed)
            if os.path.exists(rd):
                shutil.rmtree(rd)
            shutil.copytree(tree, rd + "/repo", symlinks=True)
            rc = run(rd + "/repo", rd + "/probe", int(seed))
            print("seed", seed, "pytest exit", rc, file=sys.stderr)
            probes.append(rd + "/probe")
        merge2(o.out, o.tests, tree, probes)
    elif o.cmd == "all":
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
        merge(o.out, o.tests, os.path.join(o.work, "tree", "_infect_sites.json"), probes, repo=os.path.join(o.work, "tree"))
    elif o.cmd == "run":
        sys.exit(run(os.path.abspath(o.repo), os.path.abspath(o.probe), o.seed, o.tests))
    elif o.cmd == "prepare":
        a = ("--group", GROUPS.get(o.group, "^sqlglot/"), "--no-cmp") if o.group else ()
        if o.count:
            a += ("--count",)
        if o.skip:
            a += ("--skip", o.skip)
        prepare(o.src, o.repo, a)
    elif o.cmd == "merge2":
        merge2(o.out, o.tests, o.repo, o.probes, o.report)
    else:
        merge(o.out, o.tests, o.sites, o.probes, o.report)


if __name__ == "__main__":
    main()
