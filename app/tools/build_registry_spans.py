"""Builds maps/registry_spans.json from the pinned tree: the line spans of the registry lookup
methods in sqlglot/dialects/dialect.py (selector._registry_spans), with the length and checksum
of the text they belong to. The selector then does not parse that file at run time.

usage: build_registry_spans.py REPO OUT_DIR"""
import json
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BR_MAPS", "/nonexistent")
import selector  # noqa: E402


def main():
    repo, out = sys.argv[1:3]
    with open(os.path.join(repo, "sqlglot/dialects/dialect.py"), encoding="utf-8") as f:
        text = f.read()
    selector._REGISTRY_SPANS.clear()
    selector.MAPS = "/nonexistent"
    spans = selector._registry_spans(text)
    rec = {"len": len(text), "crc": zlib.crc32(text.encode("utf-8", "surrogateescape")), "spans": [list(s) for s in spans]}
    with open(os.path.join(out, "registry_spans.json"), "w") as f:
        json.dump(rec, f)
    print(rec)


if __name__ == "__main__":
    main()
