"""Build step: downloads sqlglot at the pinned commit from GitHub (no git, no apt), checks
its content against a hash taken from a git checkout of that commit, and unpacks it to /repo."""
import hashlib
import io
import os
import sys
import tarfile
import time
import urllib.request

COMMIT = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
URL = f"https://github.com/tobymao/sqlglot/archive/{COMMIT}.tar.gz"
# sha256 over every file of the tree: sorted "path\0mode\0sha256(content)\n" lines
# (mode = 755 for executables, 644 otherwise). The tarball's own bytes are not pinned,
# since GitHub does not promise byte-stable archives; the content is what counts.
TREE_SHA256 = sys.argv[2] if len(sys.argv) > 2 else None
DEST = sys.argv[1] if len(sys.argv) > 1 else "/repo"


def tree_digest(root):
    lines = []
    for d, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(d, f)
            rel = os.path.relpath(p, root)
            mode = "755" if os.stat(p).st_mode & 0o100 else "644"
            with open(p, "rb") as fh:
                h = hashlib.sha256(fh.read()).hexdigest()
            lines.append(f"{rel}\0{mode}\0{h}\n")
    lines.sort()
    return hashlib.sha256("".join(lines).encode()).hexdigest(), len(lines)


def main():
    data = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(URL, timeout=120) as r:
                data = r.read()
            break
        except Exception as e:  # noqa: BLE001
            print(f"download attempt {attempt + 1} failed: {e!r}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    if data is None:
        sys.exit("could not download sqlglot")
    prefix = f"sqlglot-{COMMIT}/"
    os.makedirs(DEST, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = []
        for m in tf.getmembers():
            if m.name.rstrip("/") + "/" == prefix:
                continue
            if not m.name.startswith(prefix):
                sys.exit(f"unexpected member {m.name}")
            m.name = m.name[len(prefix):]
            members.append(m)
        tf.extractall(DEST, members=members, filter="data")
    digest, n = tree_digest(DEST)
    print(f"sqlglot {COMMIT}: {n} files, tree sha256 {digest}", file=sys.stderr)
    if TREE_SHA256 and digest != TREE_SHA256:
        sys.exit(f"tree sha256 mismatch: got {digest}, want {TREE_SHA256}")
    for sub in ("sqlglot", "tests", "sqlglotc", "sqlglot-integration-tests"):
        if not os.path.isdir(os.path.join(DEST, sub)):
            sys.exit(f"missing {sub}/")


if __name__ == "__main__":
    main()
