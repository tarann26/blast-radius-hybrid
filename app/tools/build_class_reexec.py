"""Build-time check for the class route (classroute.py), on the pinned tree: for every class that
maps/attr_import_reads.json lists as hooked, in a module without a small importer cone, the
class statement of the PINNED text is created again beside the pinned class. The class is listed
in "reexec_ok" when that gives a class equal to the pinned one in every attribute, changes nothing
outside it (cone.reimport checks with an empty cone), and the new class is gone after a collection.

usage: python build_class_reexec.py MAPS_DIR   (rewrites attr_import_reads.json with "reexec_ok")"""
from __future__ import annotations

import gc
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.environ.get("BR_ROOT", "/repo")


def check_module(mod_name, names, reads, base, base_codes):
    import classroute
    import cone

    rel = mod_name.replace(".", "/")
    path = os.path.join(ROOT, rel + ".py")
    if not os.path.exists(path):
        path = os.path.join(ROOT, rel, "__init__.py")
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    ok, why = [], {}
    gc.freeze()
    for name in names:
        key = mod_name + ":" + name
        r = frozenset(a for a in reads.get(key, {}) if a == "*")  # only the whole-dict rule here
        try:
            old, new, attrs, changed, state, f = classroute.prepare(mod_name, name, text, path, r, base, base_codes, {})
            if changed:
                raise cone.Redo("differs from the pinned class: %r" % (changed[:3],))
            cone.reimport([], lambda: None)
            ref = classroute.apply(old, new, attrs)
            new = state = f = None
            if not classroute.gone(ref):
                raise cone.Redo("still alive")
            ok.append(key)
        except cone.Redo as e:
            why[key] = str(e)[:200]
            new = state = f = None
    return {"ok": ok, "why": why}


def main():
    maps_dir = sys.argv[1]
    import build_cone
    import preload

    preload.install_environment(ROOT)
    preload.import_sqlglot()
    import cone
    import statefp

    with open(os.path.join(maps_dir, "attr_import_reads.json")) as f:
        data = json.load(f)
    with open(os.path.join(maps_dir, "cone.json")) as f:
        cj = json.load(f)
    base, fpr = statefp.snapshot()
    base_codes = fpr.base_codes()
    cone.record(cj.get("walk") or [])
    by_mod = {}
    for key in data["hooked"]:
        mod, _, name = key.partition(":")
        if "." in name or mod in cj["modules"]:
            continue
        by_mod.setdefault(mod, []).append(name)
    ok, why = [], {}
    for mod, names in sorted(by_mod.items()):
        r = build_cone.in_child(check_module, mod, names, data["reads"], base, base_codes)
        if not r or "ok" not in r:
            print(mod, "FAILED", str(r)[:300], file=sys.stderr)
            continue
        ok.extend(r["ok"])
        why.update(r["why"])
    data["reexec_ok"] = sorted(ok)
    with open(os.path.join(maps_dir, "attr_import_reads.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"), sort_keys=True)
    # what the selector loads: for each class that passed, the attributes that something other
    # than the creation of the class reads during the import
    small = {key: sorted(data["reads"].get(key, {})) for key in sorted(ok)}
    with open(os.path.join(maps_dir, "class_route.json"), "w") as f:
        json.dump({"version": 1, "classes": small}, f, separators=(",", ":"))
    print("class reexec: %d ok, %d not" % (len(ok), len(why)), file=sys.stderr)
    import collections

    c = collections.Counter(v.split(":")[1].strip()[:50] if ":" in v else v[:50] for v in why.values())
    print(c.most_common(12), file=sys.stderr)
    for k, v in list(why.items())[:25]:
        print("  ", k, v, file=sys.stderr)


if __name__ == "__main__":
    main()
