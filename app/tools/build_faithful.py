"""Build step: maps/faithful.json, the tests whose method compiles alone (engine.compile_lines,
as reduce.apply_reduction does it) to a code object equal to the pinned method. For those the
child uses the pinned code object instead of compiling the method again (selector.apply_keeps).
usage: python build_faithful.py MAPS_DIR OUT_JSON  (run with BR_ROOT pointing at the pinned tree)"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import selector  # noqa: E402  (installs the code cache: pinned code objects by path)
import engine  # noqa: E402
import preload  # noqa: E402
import reduce  # noqa: E402

ROOT = selector.ROOT
maps = selector.Maps(sys.argv[1])
ids = maps.meta["test_ids"]
preload.install_environment(ROOT)
preload.import_sqlglot()
out = []
for n, test_id in enumerate(ids):
    rel, cls_name, meth = test_id.split("::")
    trec = maps.test_units(n)
    shape = (trec[0], trec[1], trec[2]) if trec is not None else maps.meta["shapes"].get(n)
    frec = maps.file(rel)
    if shape is None or frec is None:
        continue
    try:
        cls = getattr(importlib.import_module(rel[:-3].replace("/", ".")), cls_name)
    except Exception:  # noqa: BLE001
        continue
    full = os.path.join(ROOT, rel)
    with open(full, encoding="utf-8") as f:
        lines = f.read().split("\n")
    first = shape[0]
    old_mod = engine.compile_lines(lines[first - 1:shape[1]], first, (cls_name,), full, frec[1], frec[2])
    old = reduce.find_code(old_mod, meth, first) if old_mod is not None else None
    pinned = selector.pinned_code(full, meth, first)
    if old is None or pinned is None or old != pinned:
        continue
    if reduce.live_function(cls, meth, old) is None or reduce.live_function(cls, meth, pinned) is None:
        continue
    out.append(n)
with open(sys.argv[2], "w") as f:
    json.dump(out, f)
print("faithful", len(out), "of", len(ids), file=sys.stderr)
