"""Image build: saves patchrules.tests_read_docs for the pinned tree (it reads every test file).

usage: python build_docs_read.py ROOT MAPS_DIR OUT_JSON
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(root, maps_dir, out):
    os.environ["BR_ROOT"] = root
    import patchrules
    import selector

    patchrules.ROOT = root
    maps = selector.Maps(maps_dir)
    files = list(patchrules.test_files(maps))
    value = any(patchrules.DOC_READ_RE.search(patchrules.source(rel)) for rel in files)
    rows = [[rel, patchrules._file_size(os.path.join(root, rel))] for rel in files]
    with open(out, "w") as f:
        json.dump({"value": value, "files": rows}, f)
    print("docs_read:", value, len(rows), "files", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:4])
