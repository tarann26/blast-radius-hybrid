"""Build step, run by the added interpreter build: proves that it can stand in for the pinned
image's interpreter. Every fact below that a test could observe must be equal in both
(same CPython version and bytecode, same hashes under a fixed seed, same limits, and above
all the same SQLite library behind the sqlite3 module). Exits non-zero on any difference;
the Dockerfile then keeps the pinned image's interpreter.
Usage: /opt/python/bin/python3.13 check_python.py [/usr/local/bin/python3.13]"""
import json
import os
import subprocess
import sys

OFFICIAL = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/usr/local/bin/python3.13"


def facts():
    import importlib.util
    import sqlite3
    import sysconfig

    import _sqlite3

    con = sqlite3.connect(":memory:")
    return {
        "version": list(sys.version_info),
        "magic": importlib.util.MAGIC_NUMBER.hex(),
        "soabi": sysconfig.get_config_var("SOABI"),
        "hash_info": list(sys.hash_info),
        "hashes": [hash("abc"), hash(b"abc"), hash(1.5), hash(("a", 1))],
        "set_order": list({"select", "from", "where", "join", "x" * 30}),
        "recursion_limit": sys.getrecursionlimit(),
        "float_repr": [repr(0.1 + 0.2), repr(1e23), "%.17g" % 2.675, float.__getformat__("double")],
        "int_info": list(sys.int_info),
        "encodings": [sys.getdefaultencoding(), sys.getfilesystemencoding()],
        "sqlite_version": sqlite3.sqlite_version,
        "sqlite_file": _sqlite3.__file__,
        "sqlite_options": [r[0] for r in con.execute("pragma compile_options")],
        "sqlite_query": con.execute("select sqlite_version(), 1.0 / 3, typeof(1 / 2), 'a' || 1").fetchall(),
    }


def main():
    if "--facts" in sys.argv:
        print(json.dumps(facts()))
        return
    mine = json.loads(json.dumps(facts()))
    env = dict(os.environ)
    theirs = json.loads(subprocess.run([OFFICIAL, os.path.abspath(__file__), "--facts"], env=env, check=True, capture_output=True, text=True).stdout)
    bad = [k for k in sorted(set(mine) | set(theirs)) if mine.get(k) != theirs.get(k)]
    for k in bad:
        print(f"differs: {k}: {mine.get(k)!r} != {theirs.get(k)!r}", file=sys.stderr)
    # the sqlite3 extension must bind to this interpreter: no second Python library in the process
    with open("/proc/self/maps") as fh:
        libs = sorted({line.split()[-1] for line in fh if "libpython" in line})
    if libs:
        print(f"a second Python library is loaded: {libs}", file=sys.stderr)
    if bad or libs:
        sys.exit(1)
    print(f"interpreter check ok: {sys.version.split()[0]}, sqlite {mine['sqlite_version']} from {mine['sqlite_file']}", file=sys.stderr)


if __name__ == "__main__":
    main()
