"""Image build: stores the zlib-compressed map blobs raw (see rawz.py).

usage: python rawmaps.py MAPS_DIR

Each blob is written as RAW_MAGIC + zlib.decompress(blob), so rawz.unz returns the same
bytes as before. Every converted file is read back and compared; a file that fails a
check keeps its compressed form (unz reads both).
"""
import marshal
import os
import sys
import zlib

from rawz import RAW_MAGIC, unz

# files read as marshal.loads(unz(file bytes))
WHOLE = ("readers.bin", "classes.bin", "classmap.bin", "tight_lines.bin", "tok_inputs.bin")
KEYMAP = "keymap.bin"
SHARD_DIRS = ("f", "t", "u")
# files read as marshal.loads(file bytes) whose values hold compressed blobs (infect.py)
NESTED = ("infect.bin", "infect2.bin")


def inflate(blob):
    """The decompressed bytes of a complete zlib stream with nothing after it, else None."""
    if not isinstance(blob, bytes) or blob[:1] != b"\x78":
        return None
    d = zlib.decompressobj()
    try:
        out = d.decompress(blob) + d.flush()
    except zlib.error:
        return None
    if not d.eof or d.unused_data:
        return None
    return out


def convert_whole(path):
    with open(path, "rb") as f:
        old = f.read()
    raw = inflate(old)
    if raw is None:
        return 0
    new = RAW_MAGIC + raw
    if unz(new) != zlib.decompress(old):
        return 0
    with open(path, "wb") as f:
        f.write(new)
    return 1


def convert_keymap(path):
    """keymap.bin: raw, and each table as its own marshal blob (keymap._LazyTables)."""
    with open(path, "rb") as f:
        old = f.read()
    raw = inflate(old)
    if raw is None:
        return 0
    d = marshal.loads(raw)
    if not isinstance(d, dict) or d.get("lazy_tables") or not isinstance(d.get("tables"), list):
        return convert_whole(path)
    new = dict(d)
    new["tables"] = [marshal.dumps(t) for t in d["tables"]]
    new["lazy_tables"] = 1
    data = RAW_MAGIC + marshal.dumps(new)
    back = marshal.loads(unz(data))
    ok = (
        list(back) == list(d) + ["lazy_tables"]
        and all(back[k] == d[k] for k in d if k != "tables")
        and len(back["tables"]) == len(d["tables"])
        and all(marshal.loads(b) == t for b, t in zip(back["tables"], d["tables"]))
    )
    if not ok:
        return convert_whole(path)
    with open(path, "wb") as f:
        f.write(data)
    return 1


def _convert_value(v):
    """(new value, number of blobs converted) for a value of a nested map."""
    if isinstance(v, bytes):
        raw = inflate(v)
        return (v, 0) if raw is None else (RAW_MAGIC + raw, 1)
    if isinstance(v, dict):
        out, n = {}, 0
        for k, x in v.items():
            if isinstance(x, bytes):
                raw = inflate(x)
                if raw is not None:
                    out[k] = RAW_MAGIC + raw
                    n += 1
                    continue
            out[k] = x
        return out, n
    return v, 0


def _same(old, new):
    """new reads like old: every blob gives the same bytes through unz, everything else equal."""
    if isinstance(old, bytes) and isinstance(new, bytes):
        if old == new:
            return True
        return inflate(old) is not None and unz(new) == zlib.decompress(old)
    if isinstance(old, dict) and isinstance(new, dict):
        return list(old) == list(new) and all(_same(old[k], new[k]) for k in old)
    return type(old) is type(new) and old == new


def convert_nested(path):
    with open(path, "rb") as f:
        old_bytes = f.read()
    try:
        m = marshal.loads(old_bytes)
    except (ValueError, EOFError, TypeError):
        return 0
    if not isinstance(m, dict):
        return 0
    new, n = {}, 0
    for k, v in m.items():
        new[k], c = _convert_value(v)
        n += c
    if not n:
        return 0
    data = marshal.dumps(new)
    if not _same(m, marshal.loads(data)):
        return 0
    with open(path, "wb") as f:
        f.write(data)
    return n


def main(maps):
    done = 0
    for name in WHOLE:
        p = os.path.join(maps, name)
        if os.path.isfile(p):
            done += convert_whole(p)
    p = os.path.join(maps, KEYMAP)
    if os.path.isfile(p):
        done += convert_keymap(p)
    for d in SHARD_DIRS:
        p = os.path.join(maps, d)
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith(".bin"):
                    done += convert_whole(os.path.join(p, name))
    for name in NESTED:
        p = os.path.join(maps, name)
        if os.path.isfile(p):
            done += convert_nested(p)
    print("rawmaps: %d blobs stored raw" % done, file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
