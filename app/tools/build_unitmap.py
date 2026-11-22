"""Builds the maps for reduced test runs from the unit traces (see trace_units.py, reduce.py).

usage (inside the pinned image): python build_unitmap.py REPO UNITS_RAW_DIR MAPS_DIR
Writes MAPS_DIR/t/<test>.bin and MAPS_DIR/u/<file id>_<chunk>.bin.
"""
from __future__ import annotations

import glob
import gzip
import json
import marshal
import os
import sys
import zlib
from array import array

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reduce

CHUNK = 256
BASE = 0xFFFF  # the line runs in the part of the test that always runs


def union(a, b):
    out = {f: set(v) for f, v in a.items()}
    for f, v in b.items():
        out.setdefault(f, set()).update(v)
    return out


def minus(a, b):
    out = {}
    for f, v in a.items():
        left = set(v) - set(b.get(f, ()))
        if left:
            out[f] = left
    return out


def class_tables(root):
    """Class names of sqlglot, and for each class K the classes whose instances can read an
    attribute that K defines: K, its subclasses, and the dialects that bundle one of them
    (a dialect object hands out its tokenizer, parser and generator classes)."""
    import preload

    preload.install_environment(root)
    preload.import_sqlglot()
    classes = {}

    def visit(cls):
        name = "%s.%s" % (cls.__module__, cls.__qualname__)
        if name in classes:
            return
        classes[name] = cls
        for v in list(cls.__dict__.values()):
            if isinstance(v, type) and (v.__module__ or "").startswith("sqlglot"):
                visit(v)

    for mname, m in list(sys.modules.items()):
        if m is not None and (mname == "sqlglot" or mname.startswith("sqlglot.")):
            for v in list(vars(m).values()):
                if isinstance(v, type) and (v.__module__ or "").startswith("sqlglot"):
                    visit(v)
    names = sorted(classes)
    ids = {n: i for i, n in enumerate(names)}
    objs = [classes[n] for n in names]

    def is_sub(a, b):  # TypedDict classes refuse the check
        try:
            return issubclass(a, b)
        except TypeError:
            return a is b

    def has_init(cls):
        for k in cls.__mro__:
            if (k.__module__ or "").startswith("sqlglot") and callable(k.__dict__.get("__init__")):
                return True
        return False

    bundles = {}
    for n, cls in classes.items():
        members = [v for v in vars(cls).values() if isinstance(v, type) and (v.__module__ or "").startswith("sqlglot")]
        for attr in ("tokenizer_class", "parser_class", "generator_class", "jsonpath_tokenizer_class"):
            v = getattr(cls, attr, None)
            if isinstance(v, type):
                members.append(v)
        if members:
            bundles[n] = members
    users = {}
    for n, cls in classes.items():
        subs = [o for o in objs if is_sub(o, cls)]
        if not all(has_init(o) for o in subs):
            continue  # instantiation of some user is not visible: no class filter for K
        req = {ids["%s.%s" % (o.__module__, o.__qualname__)] for o in subs}
        for owner, members in bundles.items():
            if any(is_sub(mm, cls) for mm in members):
                oc = classes[owner]
                req.update(ids["%s.%s" % (o.__module__, o.__qualname__)] for o in objs if is_sub(o, oc))
        users[n] = tuple(sorted(req))
    return ids, users


def main():
    root, raw_dir, maps_dir = sys.argv[1:4]
    class_dir = sys.argv[4] if len(sys.argv) > 4 else None
    class_ids, class_users = ({}, {})
    if class_dir and os.path.isdir(class_dir) and os.listdir(class_dir):
        class_ids, class_users = class_tables(os.path.abspath(root))
        with open(os.path.join(maps_dir, "classes.bin"), "wb") as f:
            f.write(zlib.compress(marshal.dumps(class_users), 9))
    with open(os.path.join(maps_dir, "meta.bin"), "rb") as f:
        meta = marshal.load(f)
    files = meta["files"]
    os.makedirs(os.path.join(maps_dir, "t"), exist_ok=True)
    os.makedirs(os.path.join(maps_dir, "u"), exist_ok=True)
    for p in glob.glob(os.path.join(maps_dir, "t", "*.bin")) + glob.glob(os.path.join(maps_dir, "u", "*.bin")):
        os.remove(p)
    shapes_cache = {}
    line_entries = {}  # (file, line) -> {test: pattern id or BASE}
    n_red = n_tests = 0
    stats = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json.gz"))):
        rec = json.load(gzip.open(path))
        n_tests += 1
        if not rec.get("reducible") or not rec.get("units"):
            continue
        number = rec["number"]
        rel, cls_name, meth = rec["id"].split("::")
        if rel not in shapes_cache:
            shapes_cache[rel] = reduce.file_shapes(open(os.path.join(root, rel), encoding="utf-8").read())
        shape = shapes_cache[rel].get((cls_name, meth))
        if shape is None:
            continue
        units = rec["units"]
        by_key = {tuple(u["key"]): u for u in units}
        warm = {(tuple(k) if k else None): cov for k, cov in rec.get("warm", [])}
        covs = []
        for u in units:
            cov = {f: set(v) for f, v in u["cov"].items()}
            if u["key"][0] == "t":
                i = u["key"][1]
                parent = u.get("parent")
                own = by_key[("s", parent)]["cov"] if parent is not None and ("s", parent) in by_key else {}
                windows = {}
                for k, c in warm.items():
                    if k is not None and k[0] == "w":
                        windows = union(windows, c)
                own_always = minus(own, windows)  # runs for every task: a change there needs all tasks
                cov = minus(cov, own)
                cov = union(cov, warm.get(("w", i), {}))
                cov = union(cov, warm.get(("t", i), {}))
                cov = union(cov, own_always)
            covs.append(cov)
        # line -> set of unit indices, for this test
        per_line = {}
        for ui, cov in enumerate(covs):
            for f, lines in cov.items():
                for ln in lines:
                    per_line.setdefault((f, ln), []).append(ui)
        patterns = {}
        plist = []
        for key, uis in per_line.items():
            t = tuple(uis)
            pid = patterns.get(t)
            if pid is None:
                pid = patterns[t] = len(plist)
                plist.append(t)
            line_entries.setdefault(key, {})[number] = pid
        for f, lines in rec.get("base", {}).items():
            for ln in lines:
                line_entries.setdefault((f, ln), {})[number] = BASE
        if len(plist) >= BASE:
            raise SystemExit("too many patterns for test %d" % number)
        unit_rows = []
        for u in units:
            k = u["key"]
            parent = u.get("parent")
            if k[0] == "f":
                unit_rows.append(("f", k[1], k[2], -1 if parent is None else parent))
            else:
                unit_rows.append((k[0], "", k[1], -1 if parent is None else parent))
        unit_classes = None
        base_classes = None
        if class_ids:
            cpath = os.path.join(class_dir, "%d.json.gz" % number)
            if os.path.exists(cpath):
                crec = json.load(gzip.open(cpath))
                if crec.get("base") is not None and len(crec.get("units", [])) == len(units):
                    conv = lambda names: None if names is None else tuple(sorted(class_ids[c] for c in names if c in class_ids))
                    base_classes = conv(crec["base"])
                    unit_classes = [conv(u) for u in crec["units"]]
        trec = (shape[0], shape[1], [tuple(s) for s in shape[2]], unit_rows, plist,
                rec.get("full_cpu") or 0.0, rec.get("base_cpu") or 0.0, [u.get("cpu") or 0.0 for u in units],
                unit_classes, base_classes)
        with open(os.path.join(maps_dir, "t", "%d.bin" % number), "wb") as f:
            f.write(zlib.compress(marshal.dumps(trec), 9))
        n_red += 1
        stats.append((number, len(units), len(plist)))
    # per source file, chunked by line
    chunks = {}
    for (f, ln), ent in line_entries.items():
        fid = files.get(f)
        if fid is None:
            continue
        arr = array("H")
        for t in sorted(ent):
            arr.append(t)
            arr.append(ent[t])
        chunks.setdefault((fid, ln // CHUNK), {})[ln] = arr.tobytes()
    total = 0
    for (fid, c), d in chunks.items():
        blob = zlib.compress(marshal.dumps(d), 6)
        total += len(blob)
        with open(os.path.join(maps_dir, "u", "%d_%d.bin" % (fid, c)), "wb") as f:
            f.write(blob)
    print("tests traced", n_tests, "reducible with units", n_red, "line entries", len(line_entries),
          "chunks", len(chunks), "compressed bytes", total)


if __name__ == "__main__":
    main()
