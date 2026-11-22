"""Map of the "if TYPE_CHECKING:" body spans of every pinned .py file (see
selector.type_checking_groups). Usage: python build_tc_spans.py REPO OUT.bin
Each entry: rel path -> (len(text), crc32(text), spans as _type_checking_spans returns them)."""
import marshal
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selector import _type_checking_spans  # noqa: E402


def main(repo, out):
    rows = {}
    for top, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(top, name)
            rel = os.path.relpath(path, repo)
            try:
                with open(path, encoding="utf-8", newline="") as f:
                    text = f.read()
            except (OSError, UnicodeDecodeError):
                continue
            if "TYPE_CHECKING" not in text:
                continue
            spans = _type_checking_spans(text)
            rows[rel] = (len(text), zlib.crc32(text.encode("utf-8", "surrogateescape")), spans)
    with open(out, "wb") as f:
        marshal.dump(rows, f)
    print("tc spans:", len(rows), "files,", sum(1 for r in rows.values() if r[2]), "with spans", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
