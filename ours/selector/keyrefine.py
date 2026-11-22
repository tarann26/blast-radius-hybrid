"""Key-level narrowing of edits of string-keyed dict entries (see selector/keyrec.py).

For a commit whose class-body or module-level edits change entries of string-keyed
dicts, fingerprint the state of every sqlglot class and module after importing the
pinned tree (twice, two hash seeds; once per run) and the patched tree (once per
commit), each in its own interpreter. If the only differences are entries of string-keyed
dicts that the build-time trace recorded (tools/trace_keys.py), each such site's
candidates are cut down to the tests that looked up a changed key of a changed dict, or
enumerated it, plus every test without a record. Anything else (another value changed,
a dict without records, an import that fails, a timeout) leaves the candidates as they
were.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_PINNED: list = []  # [pinned (seed 1), unstable keys] or [None, None] when unavailable
_CODE = ("import sys, json\n"
         "sys.path.insert(0, {parent!r})\n"
         "from selector import keyrec\n"
         "json.dump(keyrec.state_fingerprint(), sys.stdout)\n")
# Build-time only: allocate a few throwaway classes and buffers before each import so
# that objects hashed by address (classes) land elsewhere; values ordered by such
# hashes (tuples or dicts built from sets of classes) then show up as unstable even
# where addresses are reproducible (no ASLR, e.g. an emulated container), where the
# build-time and run-time layouts still differ.
_PAD = ("import builtins, random\n"
        "_rng, _keep, _real = random.Random({pad}), [], builtins.__import__\n"
        "def _padded(*a, **k):\n"
        "    for _ in range(_rng.randrange(0, 6)):\n"
        "        _keep.append(type('Pad', (), {{}}))\n"
        "        _keep.append(bytearray(_rng.randrange(1, 4000)))\n"
        "    return _real(*a, **k)\n"
        "builtins.__import__ = _padded\n")


def _start(tree: str, seed: int, pad: int = 0):
    env = dict(os.environ, PYTHONPATH=tree, PYTHONHASHSEED=str(seed), PYTHONDONTWRITEBYTECODE="1",
               SKIP_INTEGRATION="0")
    code = (_PAD.format(pad=pad) if pad else "") + _CODE.format(parent=os.path.dirname(HERE))
    return subprocess.Popen([sys.executable, "-c", code], cwd=tree, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)


def _finish(proc, timeout):
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return None
    if proc.returncode != 0:
        return None
    try:
        return json.loads(out)
    except ValueError:
        return None


def _loop_leftovers(root: str):
    """key -> whether holder\tattr is the target of a `for` loop directly in that class
    body (or at module level) in the pinned source."""
    import ast

    cache: dict = {}

    def check(key: str) -> bool:
        holder, attr = key.split("\t", 1)
        module, _, qual = holder.partition(":")
        if module not in cache:
            base = os.path.join(root, *module.split("."))
            path = base + ".py" if os.path.exists(base + ".py") else os.path.join(base, "__init__.py")
            targets: dict = {}
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except (OSError, SyntaxError, ValueError):
                tree = None

            def visit(body, prefix):
                for node in body:
                    if isinstance(node, (ast.For, ast.AsyncFor)):
                        for name in ast.walk(node.target):
                            if isinstance(name, ast.Name):
                                targets.setdefault(prefix, set()).add(name.id)
                    elif isinstance(node, ast.ClassDef):
                        visit(node.body, (prefix + "." if prefix else "") + node.name)

            if tree is not None:
                visit(tree.body, "")
            cache[module] = targets
        return attr in cache[module].get(qual, set())

    return check


def _pinned_from_file():
    """Pinned fingerprints computed when the image was built (Dockerfile), if any."""
    import gzip

    path = os.environ.get("BR_PINNED_STATE") or os.path.join(os.path.dirname(HERE), "maps", "pinned_state.json.gz")
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt") as handle:
            data = json.load(handle)
        return [data["seed1"], frozenset(data["unstable"])]
    except Exception:  # noqa: BLE001
        return None


def refine(found, changes, root: str, tmp: str, maps, log, timeout: float = 60.0, finder=None) -> str:
    """Narrow the key-record and module-record sites of `found` in place, if the state
    fingerprints allow it. Returns a note for the log."""
    from .keyrec import state_changes
    from .verify import build_overlay

    key_sites = [(f, i) for f in found.files for i in getattr(f, "keyrec_sites", [])]
    mod_sites = [(f, i, alt) for f in found.files for i, alt in getattr(f, "modrec_sites", {}).items()]
    class_sites = [(f, i) for f in found.files for i in getattr(f, "classrec_sites", [])]
    if not key_sites and not mod_sites and not class_sites:
        return ""
    if any(c.binary for c in changes):
        return "state: binary change"
    files = {}
    for change in changes:
        report = next((f for f in found.files if f.path == change.path), None)
        if change.old_path and change.is_rename:
            files[change.old_path] = None
        if change.is_deleted:
            files[change.old_path] = None
            continue
        if report is None or report.new_text is None:
            return "state: change without text"
        files[change.path] = report.new_text.encode("utf-8")
    overlay = os.path.join(tmp, "keys_overlay")
    shutil.rmtree(overlay, ignore_errors=True)
    try:
        build_overlay(overlay, root, files, copy_dirs=())
    except Exception as error:  # noqa: BLE001
        return f"state: overlay failed ({error!r})"
    if not _PINNED:
        stored = _pinned_from_file()
        if stored is not None:
            _PINNED.extend(stored)
    procs = {"patched": _start(overlay, 1)}
    if not _PINNED:
        procs["p1"] = _start(root, 1)
        procs["p2"] = _start(root, 2)
    results = {name: _finish(proc, timeout) for name, proc in procs.items()}
    shutil.rmtree(overlay, ignore_errors=True)
    if not _PINNED:
        p1, p2 = results.get("p1"), results.get("p2")
        _PINNED.extend([p1, _unstable(p1, p2) if p1 is not None and p2 is not None else None])
    pinned1, unstable, patched = _PINNED[0], _PINNED[1], results["patched"]
    if pinned1 is None or unstable is None or patched is None:
        return "state: fingerprint unavailable"
    changed_paths = {c.path for c in changes} | {c.old_path for c in changes if c.old_path}
    res = state_changes(pinned1, patched, changed_paths, _loop_leftovers(root), unstable)
    if res is None:
        return "state: unreadable change"
    dicts, others = res
    allowed = {f"{m}\t{n}" for _, _, (_, group) in mod_sites for m, n in group}
    # Every other change of the program state, and who can see it at runtime:
    # changed class attributes through the class-attribute rule without import-time
    # lines (their import-time readers' effects are themselves in this comparison),
    # changed dict entries through the key records. Module bindings other than the
    # module sites' own, or a class the class map does not know, keep the old masks.
    extra = 0
    notes = []
    attrs = 0
    for key in sorted(others - allowed):
        holder, attr = key.split("\t", 1)
        if ":" not in holder or finder is None or not class_sites:
            return "state: " + holder + "." + attr + " changed"
        if attr.startswith("_") and attr.endswith("_") and attr not in ("__firstlineno__", "__static_attributes__"):
            # dunder/sunder state (an Enum's _value2member_map_, ...) is read by the
            # machinery behind lookups like DType("INT"), not by name (Python 3.13's
            # compile metadata __firstlineno__/__static_attributes__ is read by nothing
            # in sqlglot or its tests: it moves when lines above a class move)
            return "state: " + holder + "." + attr + " changed"
        m = finder.runtime_attr_mask(holder, attr)
        if m is None:
            return f"state: {holder} not in the class map"
        extra |= m
        attrs += 1
    if attrs:
        notes.append(f"{attrs} class attributes")
    rec = getattr(maps, "keyrec", None) or {}
    unrecordable = set(rec.get("unrecordable", ()))
    holders = set(rec.get("holders", ()))
    lookups = rec.get("lookups", {})
    narrowed = 0
    keyed = 0
    for holder, keys in dicts.items():
        if holder in unrecordable or holder not in holders:
            if holder not in allowed:
                return f"state: {holder} has no usable records"
            continue
        keyed += 1
        by_key = lookups.get(holder, {})
        for k in set(keys) | {"*"}:
            narrowed |= maps.mask(by_key.get(k))
    if keyed:
        extra |= narrowed | (maps.all_mask & ~maps.mask(rec.get("traced")))
        notes.append(f"{sum(len(k) for h, k in dicts.items() if h not in allowed)} entries of {keyed} dicts")
    done = set()
    for report, index in key_sites + class_sites:
        if (id(report), index) not in done:
            report.site_masks[index] &= extra
            done.add((id(report), index))
    for report, index, (alt, _group) in mod_sites:
        if (id(report), index) not in done:
            report.site_masks[index] &= alt | extra
            done.add((id(report), index))
    if mod_sites:
        notes.append(f"{len(mod_sites)} module bindings through records")
    if not notes:
        notes.append("nothing else changed")
    for report in found.files:
        imp = 0
        for j in getattr(report, "imp_sites", []):
            imp |= report.site_masks[j]
        report.imp = imp
    return "state: " + ", ".join(notes)


def _unstable(p1: dict, p2: dict) -> frozenset:
    """Values whose fingerprint (non-dict values) or key order (dicts) differs between
    the two hash seeds on the pinned commit (their order comes from a set)."""
    s1, s2 = p1["state"], p2["state"]
    out = set()
    for k in set(s1) | set(s2):
        a, b = s1.get(k) or {}, s2.get(k) or {}
        if "v" in a and "v" in b and a != b:
            out.add(k)
        elif "d" in a and "d" in b and list(a["d"]) != list(b["d"]) \
                and [x for x in a["d"] if x in b["d"]] != [x for x in b["d"] if x in a["d"]]:
            out.add(k)
    return frozenset(out)


def build_pinned(root: str, out: str) -> None:
    """Fingerprint the pinned tree four times (hash seeds 1-3, two of them with shifted
    memory layouts), one at a time to bound memory; store the first and the values
    whose order differs in any other run (run when the image is built, so a run only
    fingerprints patched trees)."""
    import gzip

    first = _finish(_start(root, 1), 1800)
    if first is None:
        raise SystemExit("pinned fingerprint failed: run (1, 0)")
    unstable = set()
    for seed, pad in [(2, 0), (2, 1), (3, 2)]:
        other = _finish(_start(root, seed, pad=pad), 1800)
        if other is None:
            raise SystemExit(f"pinned fingerprint failed: run ({seed}, {pad})")
        if other["imports"] != first["imports"] or other["import_errors"] != first["import_errors"]:
            raise SystemExit("pinned imports differ between runs")
        unstable |= _unstable(first, other)
        del other
    with gzip.open(out, "wt", compresslevel=9) as handle:
        json.dump({"seed1": first, "unstable": sorted(unstable)}, handle)
