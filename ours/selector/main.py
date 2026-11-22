"""Blast Radius selector: for each patch, print which tests to run and how to split them.

Pipeline per commit:
  1. parse the patch and apply it in memory to the pinned sqlglot checkout;
  2. find a sound superset of tests that could fail (selector.candidates), using
     per-test coverage maps built from the pinned commit;
  3. when it is cheaper in expectation, actually run the cheap candidates
     against the patched tree (selector.verify) and keep only those that fail;
  4. pack the final tests onto workers (selector.pack).
Only answer lines go to stdout; everything else goes to stderr.
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from selector import keyrefine, policy  # noqa: E402
from selector.candidates import CandidateFinder  # noqa: E402
from selector.maps import Maps  # noqa: E402
from selector.pack import pack  # noqa: E402
from selector.patch import parse_patch  # noqa: E402

START = time.monotonic()
DATA = os.environ.get("BR_DATA", "/data")
ROOT = os.environ.get("BR_ROOT", "/opt/sqlglot")
TMP = os.environ.get("BR_TMP", "/tmp")
TIME_LIMIT = float(os.environ.get("BR_TIME_LIMIT", "0") or 0)
# Hybrid mode: commits this selector does not judge risky go to another engine (its main.py,
# run once as a subprocess on a data folder holding only those patches); the risky ones, and any
# commit the engine gives no valid line for, are verified and answered here.
HYBRID_MAIN = os.environ.get("BR_HYBRID_MAIN", "")
HYBRID_ROOT = os.environ.get("BR_HYBRID_ROOT", "") or ROOT
RISKY_NOTES = ("state check", "may leak", "unknown change", "certain only")
HYBRID_MARGIN = float(os.environ.get("BR_HYBRID_MARGIN", "0.5"))
# (planned costs run low against real ones: only a clear gap is worth a second verification)
HYBRID_FACTOR = float(os.environ.get("BR_HYBRID_FACTOR", "0.6"))
RISKY_PARTS = ("import check", "copy-only", "leak", "everything", "analysis failed", "unparsed")


def log(message: str):
    print(f"[{time.monotonic() - START:6.2f}] {message}", file=sys.stderr, flush=True)


def _risky(plan, commit_dir, everything) -> bool:
    note = plan.note or ""
    if not note or note.startswith(RISKY_NOTES) or any(part in note for part in RISKY_PARTS):
        return True
    if getattr(plan, "leak_check", False) or (plan.include and set(plan.include) >= everything):
        return True
    try:
        text = open(os.path.join(commit_dir, plan.cid + ".patch"), "rb").read().decode("utf-8", "replace")
    except OSError:
        return True
    return _annotation_edit(text)


def _annotation_edit(text) -> bool:
    """A class-level name that gains or loses its annotation (`x: int = 0` <-> `x = 0`): for a
    dataclass, NamedTuple or TypedDict that adds or removes a field."""
    import re

    ann = re.compile(r"^([-+])\s+([A-Za-z_]\w*)\s*:\s*[^=]+=")
    plain = re.compile(r"^([-+])\s+([A-Za-z_]\w*)\s*=(?!=)")
    seen = {"-ann": set(), "+ann": set(), "-plain": set(), "+plain": set()}
    for line in text.split("\n"):
        m = ann.match(line)
        if m:
            seen[m.group(1) + "ann"].add(m.group(2))
            continue
        m = plain.match(line)
        if m:
            seen[m.group(1) + "plain"].add(m.group(2))
    return bool((seen["-ann"] & seen["+plain"]) or (seen["+ann"] & seen["-plain"]))


def _run_engine(cids, commit_dir, budget, valid) -> dict:
    """Run the other engine on these commits; cid -> tests for every valid answer line."""
    import shutil
    import subprocess

    folder = os.path.join(TMP, "hybrid_data")
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(os.path.join(folder, "commits"))
    for cid in cids:
        os.symlink(os.path.join(commit_dir, cid + ".patch"), os.path.join(folder, "commits", cid + ".patch"))
    os.symlink(os.path.join(DATA, "tests.json"), os.path.join(folder, "tests.json"))
    env = dict(os.environ, BR_DATA=folder, BR_ROOT=HYBRID_ROOT, BR_WALL=f"{budget:.0f}")
    for key in ("BR_HYBRID_MAIN", "BR_HYBRID_ROOT", "BR_TIME_LIMIT", "BR_PLAN_DUMP", "BR_TMP"):
        env.pop(key, None)
    out = {}
    started = time.monotonic()
    try:
        done = subprocess.run([sys.executable, HYBRID_MAIN], env=env, cwd=HYBRID_ROOT, stdout=subprocess.PIPE,
                              stderr=None, timeout=budget + 20)
        lines = done.stdout.decode("utf-8", "replace").splitlines()
    except Exception as error:  # noqa: BLE001 - every commit is then answered here
        log(f"hybrid: engine failed ({error!r})")
        lines = []
    wanted = set(cids)
    for line in lines:
        try:
            record = json.loads(line)
            cid = record["commit"]
            tests = [n for worker in record["workers"] for n in worker]
        except Exception:  # noqa: BLE001
            continue
        if cid in wanted and all(isinstance(n, int) and n in valid for n in tests):
            out[cid] = sorted(set(tests))
    log(f"hybrid: engine answered {len(out)} of {len(cids)} in {time.monotonic() - started:.1f}s")
    shutil.rmtree(folder, ignore_errors=True)
    return out


def main():
    tests = json.loads(open(os.path.join(DATA, "tests.json")).read())["tests"]
    number_of = {t["id"]: t["number"] for t in tests}
    ms_of = {t["number"]: int(t["seconds"] * 1000 + 0.5) for t in tests}
    seconds_of = {t["number"]: t["seconds"] for t in tests}
    commit_dir = os.path.join(DATA, "commits")
    commits = sorted(name[:-6] for name in os.listdir(commit_dir) if name.endswith(".patch"))
    log(f"{len(tests)} tests, {len(commits)} commits")

    answers: dict[str, list[int]] = {}
    # Fallback first: if anything below crashes we still print valid lines.
    try:
        maps = Maps()
        policy._MAPS = maps
        finder = CandidateFinder(maps, ROOT)
        # Map map indexes to this run's test numbers (ids not in our maps are unknown).
        map_to_number = [number_of.get(node) for node in maps.tests]
        unknown_numbers = sorted(set(number_of.values()) - {n for n in map_to_number if n is not None})
        if unknown_numbers:
            log(f"{len(unknown_numbers)} tests are not in the maps; they are added whenever a commit is risky")

        total_limit = TIME_LIMIT or policy.default_time_limit(len(commits))
        # Analysis is not interruptible like verification: past this budget the
        # remaining commits are not analysed and run everything (never reached in
        # normal batches: analysis takes ~0.1 s per commit).
        analysis_deadline = START + total_limit * 0.5
        plans = []
        for cid in commits:
            if time.monotonic() > analysis_deadline:
                log(f"{cid}: analysis time budget spent; running everything")
                plans.append(policy.Plan(cid, include=set(number_of.values())))
                continue
            try:
                text = open(os.path.join(commit_dir, cid + ".patch"), "rb").read().decode("utf-8", "replace")
                changes = parse_patch(text)
                paths = [c.path for c in changes]
                if (text.strip() and not changes) or len(set(paths)) != len(paths):
                    # text we could not read as a diff, or two sections for one file:
                    # do not reason about it
                    raise ValueError("unparsed patch or repeated file section")
                found = finder.analyze(changes)
                key_note = ""
                if HYBRID_MAIN and not found.unknown:
                    # the state comparison only narrows this selector's own plan: skip it for a
                    # commit the engine will answer (judged on the plan without it)
                    first = policy.plan_commit(cid, changes, found, maps, map_to_number, seconds_of, unknown_numbers)
                    if not _risky(first, commit_dir, set(number_of.values())):
                        plans.append(first)
                        continue
                if not found.unknown and any(getattr(f, "keyrec_sites", None) or getattr(f, "modrec_sites", None)
                                             or getattr(f, "classrec_sites", None) for f in found.files) \
                        and time.monotonic() - START < total_limit * 0.3 \
                        and policy.state_check_worthwhile(cid, changes, found, maps, map_to_number, seconds_of,
                                                     unknown_numbers, finder, log):
                    try:
                        # never let the fingerprints eat into the analysis budget
                        budget = START + total_limit * 0.4 - time.monotonic()
                        key_note = keyrefine.refine(found, changes, ROOT, TMP, maps, log,
                                                    timeout=max(5.0, min(60.0, budget)), finder=finder)
                    except Exception as error:  # noqa: BLE001 - only a narrowing: keep the masks
                        key_note = f"keys: failed ({error!r})"
                plan = policy.plan_commit(cid, changes, found, maps, map_to_number, seconds_of, unknown_numbers)
                if key_note:
                    plan.note = (plan.note + " | " + key_note) if plan.note else key_note
            except Exception as error:  # noqa: BLE001 - never lose a commit
                log(f"{cid}: analysis failed ({error!r}); running everything")
                plan = policy.Plan(cid, include=set(number_of.values()))
            plans.append(plan)

        deadline = START + total_limit
        id_of = {t["number"]: t["id"] for t in tests}
        delegated: dict = {}
        if HYBRID_MAIN:
            everything = set(number_of.values())
            safe = [p.cid for p in plans if not _risky(p, commit_dir, everything)]
            log(f"hybrid: {len(plans) - len(safe)} commits judged here, {len(safe)} to the engine")
            if safe:
                budget = max(20.0, 0.6 * (deadline - time.monotonic()))
                delegated = _run_engine(safe, commit_dir, budget, set(number_of.values()))
        # A delegated commit whose own plan is expected to cost clearly less than the engine's
        # answer is verified here too, and the cheaper of the two answers is kept.
        rival: dict = {}
        for p in plans:
            chosen = delegated.get(p.cid)
            if chosen is None or p.est is None:
                continue
            engine_cost = pack(sorted(chosen), ms_of, fast=True)[1] / 1000.0
            if p.est < HYBRID_FACTOR * engine_cost - HYBRID_MARGIN:
                rival[p.cid] = (chosen, engine_cost)
                del delegated[p.cid]
        if rival:
            log(f"hybrid: {len(rival)} commits also verified here (planned cost below the engine's)")
        # A plan that runs nothing here has an exact cost already: keep it where it beats the
        # engine's answer (no verification needed).
        exact = 0
        for p in plans:
            chosen = delegated.get(p.cid)
            if chosen is None or p.verify or p.leak_check:
                continue
            own = sorted(p.final_tests())
            own_cost = pack(own, ms_of, fast=True)[1] / 1000.0 if own else 0.0
            engine_cost = pack(sorted(chosen), ms_of, fast=True)[1] / 1000.0
            if own_cost < engine_cost:
                delegated[p.cid] = own
                exact += 1
        if exact:
            log(f"hybrid: {exact} commits keep this selector's exact plan (cheaper than the engine's)")
        to_verify = [p for p in plans if p.cid not in delegated]
        policy.verify_plans(to_verify, ROOT, TMP, deadline, log, id_of, number_of, seconds_of)
        for cid, chosen in delegated.items():
            answers[cid] = chosen
            log(f"{cid}: {len(chosen)} tests; engine")
        kept = []
        for p in to_verify:
            if p.cid in rival:
                chosen, engine_cost = rival[p.cid]
                own = sorted(p.final_tests())
                own_cost = pack(own, ms_of, fast=True)[1] / 1000.0 if own else 0.0
                if own_cost >= engine_cost:
                    answers[p.cid] = chosen
                    log(f"{p.cid}: {len(chosen)} tests; engine ({engine_cost:.2f}s <= own {own_cost:.2f}s)")
                    continue
            kept.append(p)
        plans = kept
        dump = os.environ.get("BR_PLAN_DUMP")
        for plan in plans:
            answers[plan.cid] = sorted(plan.final_tests())
            log(f"{plan.cid}: {len(answers[plan.cid])} tests; {plan.note}")
        if dump:  # dev diagnostics only (never set in the image)
            with open(dump, "w") as handle:
                for plan in plans:
                    handle.write(json.dumps({"commit": plan.cid, "why": {str(k): v for k, v in plan.explain().items()},
                                             "note": plan.note, "verify_cpu": getattr(plan, "verify_cpu", 0.0),
                                             "verify_est": plan.verify_cpu_estimate,
                                             "lite_rest_est": plan.lite_rest_cpu}) + "\n")
    except Exception as error:  # noqa: BLE001
        log(f"fatal: {error!r}; falling back to running everything")
        for cid in commits:
            answers.setdefault(cid, [t["number"] for t in tests])

    total_ms = 0
    out = []
    packed: dict = {}  # identical selections (e.g. every test) are packed once
    for cid in commits:
        selected = answers.get(cid, [t["number"] for t in tests])
        key = tuple(sorted(selected))
        if key not in packed:
            packed[key] = pack(list(key), ms_of)
        workers, cost = packed[key]
        total_ms += cost
        out.append(json.dumps({"commit": cid, "workers": workers}, separators=(",", ":")))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))
    sys.stdout.flush()
    log(f"done: simulated cost of selections {total_ms / 1000:.3f}s")
    sys.stderr.flush()
    # Skip interpreter teardown (it costs CPU, and native extensions loaded for
    # the verifier print leak reports at exit).
    os._exit(0)


if __name__ == "__main__":
    main()
