"""The import state of the pinned tree (statefp.snapshot, as the selector's parent takes it
right after importing sqlglot), saved at image build time so that the selector loads it
instead of computing it (about 0.2 s of CPU per batch). The fingerprints hash strings, so the
map is valid only under the same PYTHONHASHSEED (the image sets 0); the selector checks the
seed, the Python version and a probe hash, and computes the snapshot when any differs.
Usage: PYTHONHASHSEED=0 python build_base_state.py REPO OUT.bin"""
import marshal
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(repo, out):
    import preload

    preload.install_environment(repo)
    preload.import_sqlglot()
    import statefp

    state, f = statefp.snapshot()
    import selector

    data = {
        "header": selector.base_state_header(),
        "state": state,
        "fkeys": sorted(f.function_keys),
        "okeys": sorted(f.objects),
    }
    objects = selector.resolve_base_objects(data["okeys"])
    if objects is None or any(objects[k] is not f.objects[k] for k in f.objects):
        raise SystemExit("base state: the table keys do not resolve to the snapshot's objects")
    with open(out, "wb") as fh:
        marshal.dump(data, fh)
    print("base state:", len(state), "keys,", len(f.objects), "tables,", len(f.function_keys), "functions", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
