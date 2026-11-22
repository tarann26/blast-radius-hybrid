"""Core mechanics: patch parsing, in-memory code replacement, fresh import, direct test runs."""
from __future__ import annotations

import gc
import marshal
import os
import re
import sys
import types

CO_OPTIMIZED = 0x1
FUTURE_ANNOTATIONS = 0x1000000

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class FilePatch:
    __slots__ = ("old_path", "new_path", "status", "hunks", "groups", "new_text", "binary", "ok", "why")

    def __init__(self):
        self.old_path = None
        self.new_path = None
        self.status = "modify"  # modify | add | delete | rename | copy
        self.hunks = []  # (old_start, [lines with prefix])
        self.groups = []  # (old_line, n_removed, new_line, n_added, removed_lines, added_lines)
        self.new_text = None
        self.binary = False
        self.ok = True
        self.why = None  # what the parser did not understand (then ok is False)

    def fail(self, why):
        if self.ok:
            self.ok = False
            self.why = why


# Header lines of a git diff section that the parser understands. Every other line between
# "diff --git" and the first hunk makes the section unknown: the plan is then "all".
_MODES = ("100644", "100755")  # regular files; a symlink (120000) or a submodule (160000) is unknown


def _strip_prefix(p):
    """git apply's default -p1: the first path component goes (a/, b/, or any other)."""
    k = p.find("/")
    return p[k + 1:] if k > 0 else None


def _header_path(p, sec):
    """The path of a ---/+++ line, or "" for /dev/null, or None when it is not understood."""
    if "\t" in p:
        p = p.split("\t", 1)[0]  # git adds a tab after a name with spaces; diff -u a time stamp
    if p.endswith("\r") or p.startswith('"') or not p:
        sec.fail("path form %r" % p[:80])
        return None
    if p == "/dev/null":
        return ""
    q = _strip_prefix(p)
    if q is None:
        sec.fail("path without a prefix %r" % p[:80])
    return q


def parse_patch(text: str) -> list[FilePatch]:
    """The file sections of a unified diff: git sections ("diff --git") and plain ones (only
    ---/+++ lines, as diff -u writes them). A section with anything this parser does not
    fully understand gets ok False (see FilePatch.why), and the commit is then planned "all":
    quoted paths, CR line ends in headers, symlinks or submodules, binary data, hunks whose
    line counts do not add up, header lines it does not know, or one path in two sections."""
    files: list[FilePatch] = []
    cur = None
    state = None  # "head": git header lines; "body": after ---/+++; "hunks": after a hunk; "tail": mail signature
    info = {}  # per section: names from the diff --git line, the ---/+++ lines, rename/copy lines
    lines = text.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        # header lines may end with CR (a patch saved with CRLF line ends): git apply reads
        # them without it; hunk lines keep it (they are consumed below, in the hunk loop)
        line = lines[i][:-1] if lines[i].endswith("\r") else lines[i]
        if line.startswith("diff --git "):
            cur = FilePatch()
            files.append(cur)
            info[id(cur)] = {"git": line[len("diff --git "):]}
            state = "head"
            i += 1
            continue
        if line.startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ "):
            if cur is None or state != "head" or "minus" in info[id(cur)]:
                # a plain section: no diff --git line before it
                cur = FilePatch()
                files.append(cur)
                info[id(cur)] = {}
            d = info[id(cur)]
            d["minus"] = _header_path(line[4:], cur)
            d["plus"] = _header_path(lines[i + 1][4:].removesuffix("\r"), cur)
            state = "body"
            i += 2
            continue
        if cur is None or state == "tail":
            i += 1  # text before the first section (a commit message) or after the signature
            continue
        if line.startswith("@@"):
            m = HUNK_RE.match(line)
            if not m or state not in ("body", "hunks"):
                cur.fail("hunk header %r" % line[:60])
                i += 1
                continue
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_count = int(m.group(4)) if m.group(4) is not None else 1
            body = []
            i += 1
            seen_old = seen_new = 0
            while i < n and (seen_old < old_count or seen_new < new_count):
                h = lines[i]
                c = h[:1]
                if c == " " or h == "":
                    seen_old += 1
                    seen_new += 1
                    body.append(" " + h[1:])
                elif c == "-":
                    seen_old += 1
                    body.append(h)
                elif c == "+":
                    seen_new += 1
                    body.append(h)
                elif c == "\\":
                    body.append(h)
                else:
                    break
                i += 1
            if seen_old != old_count or seen_new != new_count:
                cur.fail("hunk line counts")
            if i < n and lines[i].startswith("\\"):
                body.append(lines[i])
                i += 1
            cur.hunks.append((old_start, old_count, body))
            state = "hunks"
            continue
        if state == "head":
            d = info[id(cur)]
            if line.startswith("index "):
                pass
            elif line.startswith(("old mode ", "new mode ")):
                d["mode"] = True
                if line.split()[-1] not in _MODES:
                    cur.fail("file mode %r" % line)
            elif line.startswith("new file mode "):
                cur.status = "add"
                if line.split()[-1] not in _MODES:
                    cur.fail("file mode %r" % line)
            elif line.startswith("deleted file mode "):
                cur.status = "delete"
                if line.split()[-1] not in _MODES:
                    cur.fail("file mode %r" % line)
            elif line.startswith(("similarity index ", "dissimilarity index ")):
                pass
            elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
                kind, side, p = line.split(" ", 2)
                if p.startswith('"') or p.endswith("\r") or not p:
                    cur.fail("path form %r" % p[:80])
                d[side] = p
                cur.status = kind
            elif line.startswith("GIT binary patch") or line.startswith("Binary files "):
                cur.binary = True
                cur.fail("binary")
            elif line == "":
                pass
            else:
                cur.fail("header line %r" % line[:60])
        elif line == "-- ":
            state = "tail"  # git format-patch: the mail signature after the last section
        elif line[:1] in ("+", "-", " ") or line.startswith("GIT binary patch") or line.startswith("Binary files "):
            cur.fail("line outside a hunk %r" % line[:60])
        i += 1
    seen = {}
    for fp in files:
        d = info[id(fp)]
        minus, plus = d.get("minus"), d.get("plus")
        if "git" in d:
            g = d["git"]
            same = None
            half = (len(g) - 1) // 2
            if len(g) % 2 == 1 and g[half] == " ":
                a, b = _strip_prefix(g[:half]), _strip_prefix(g[half + 1:])
                if a is not None and a == b:
                    same = a
            if g.startswith('"') or g.endswith('"'):
                fp.fail("quoted path")
        else:
            same = None
            if not fp.hunks:
                fp.fail("plain section without hunks")
        if minus == "" and plus == "":
            fp.fail("both sides /dev/null")
        if minus == "":
            if fp.status not in ("modify", "add"):
                fp.fail("status")
            fp.status = "add"
        if plus == "":
            if fp.status not in ("modify", "delete"):
                fp.fail("status")
            fp.status = "delete"
        if fp.status in ("rename", "copy"):
            fp.old_path, fp.new_path = d.get("from"), d.get("to")
            if not fp.old_path or not fp.new_path or fp.old_path == fp.new_path:
                fp.fail("rename/copy names")
            if minus and minus != fp.old_path or plus and plus != fp.new_path:
                fp.fail("rename/copy names differ from ---/+++")
        else:
            name = plus or minus or same
            if not name or (minus and plus and minus != plus) or (same and name != same) or (
                "git" in d and same is None
            ):
                fp.fail("file names")
            fp.old_path = fp.new_path = name
        if not fp.hunks and not fp.binary:
            # a section without hunks changes no content: a mode change, a rename or copy at
            # 100% similarity, an empty new file, a deletion written without its lines (-D)
            if "minus" in d or (fp.status == "modify" and not d.get("mode")):
                fp.fail("section without hunks")
        # the paths the section writes (a copy only reads its source, from the pinned tree)
        for p in {fp.new_path} if fp.status in ("copy", "add") else {fp.old_path, fp.new_path}:
            if p is None:
                continue
            if p in seen and seen[p] is not fp:
                fp.fail("path in two sections")
                seen[p].fail("path in two sections")
            seen[p] = fp
    return files


def apply_file_patch(fp: FilePatch, old_text: str | None) -> None:
    """Fills fp.new_text and fp.groups. Sets fp.ok False when the hunks do not fit."""
    if fp.binary:
        fp.ok = False
        return
    if fp.status == "delete":
        fp.new_text = None
        return
    old_lines = old_text.split("\n") if old_text is not None else []
    old_trailing_nl = old_text is not None and old_text.endswith("\n")
    if old_trailing_nl:
        old_lines.pop()
    out: list[str] = []
    pos = 0  # index into old_lines
    new_trailing_nl = old_trailing_nl if old_text else True
    for old_start, old_count, body in fp.hunks:
        start = old_start - 1 if old_count > 0 else old_start
        if old_text is None:
            start = 0
        if start < pos or start > len(old_lines):
            fp.ok = False
            return
        out.extend(old_lines[pos:start])
        pos = start
        g = None
        last = None
        for h in body:
            c = h[:1]
            t = h[1:]
            if c == " ":
                if pos >= len(old_lines) or old_lines[pos] != t:
                    fp.ok = False
                    return
                out.append(t)
                pos += 1
                g = None
            elif c == "-":
                if pos >= len(old_lines) or old_lines[pos] != t:
                    fp.ok = False
                    return
                if g is None:
                    g = [pos + 1, 0, len(out) + 1, 0, [], []]
                    fp.groups.append(g)
                g[1] += 1
                g[4].append(t)
                pos += 1
            elif c == "+":
                if g is None:
                    g = [pos + 1, 0, len(out) + 1, 0, [], []]
                    fp.groups.append(g)
                g[3] += 1
                g[5].append(t)
                out.append(t)
            elif c == "\\":
                if last == "+":
                    new_trailing_nl = False
                elif last == "-":
                    new_trailing_nl = True
                elif last == " ":
                    new_trailing_nl = False
            last = c if c != "\\" else last
    out.extend(old_lines[pos:])
    fp.new_text = "\n".join(out) + ("\n" if new_trailing_nl and out else "")


# ---------------------------------------------------------------------------
# Unit comparison


def _const_equal(x, y) -> bool:
    return type(x) is type(y) and repr(x) == repr(y)


def func_code_equal(x: types.CodeType, y: types.CodeType) -> bool:
    """Same behaviour: everything but line numbers."""
    if (
        x.co_code != y.co_code
        or x.co_names != y.co_names
        or x.co_varnames != y.co_varnames
        or x.co_freevars != y.co_freevars
        or x.co_cellvars != y.co_cellvars
        or x.co_argcount != y.co_argcount
        or x.co_posonlyargcount != y.co_posonlyargcount
        or x.co_kwonlyargcount != y.co_kwonlyargcount
        or x.co_flags != y.co_flags
        or x.co_exceptiontable != y.co_exceptiontable
        or x.co_name != y.co_name
        or x.co_qualname != y.co_qualname
        or len(x.co_consts) != len(y.co_consts)
    ):
        return False
    for a, b in zip(x.co_consts, y.co_consts):
        ia = isinstance(a, types.CodeType)
        if ia != isinstance(b, types.CodeType):
            return False
        if ia:
            if not func_code_equal(a, b):
                return False
        elif not _const_equal(a, b):
            return False
    return True


def changed_lines(x: types.CodeType, y: types.CodeType):
    """The line numbers of the instructions that differ between two versions x (pinned) and y
    (patched) of one function, or None when that cannot be told.

    It can be told when both have the same instructions in the same places except for some
    operations or arguments (same count, same exception table, same names of arguments and
    cells), so that the instructions pair up one to one. A run of x that executes none of
    the returned lines (no LINE event there) executes none of the differing instructions, so
    it runs the same instructions as y would, in the same order."""
    import dis

    if (
        x.co_varnames != y.co_varnames
        or x.co_freevars != y.co_freevars
        or x.co_cellvars != y.co_cellvars
        or x.co_argcount != y.co_argcount
        or x.co_posonlyargcount != y.co_posonlyargcount
        or x.co_kwonlyargcount != y.co_kwonlyargcount
        or x.co_flags != y.co_flags
        or x.co_exceptiontable != y.co_exceptiontable
        or x.co_name != y.co_name
        or x.co_qualname != y.co_qualname
        or len(x.co_consts) != len(y.co_consts)
        or len(x.co_code) != len(y.co_code)
    ):
        return None
    ix = list(dis.get_instructions(x))
    iy = list(dis.get_instructions(y))
    if len(ix) != len(iy):
        return None
    out = set()
    used = set()
    for a, b in zip(ix, iy):
        if a.offset != b.offset:
            return None
        if a.opcode in dis.hasconst:
            used.add(a.arg)
        same = a.opname == b.opname
        if same:
            if isinstance(a.argval, types.CodeType) or isinstance(b.argval, types.CodeType):
                if not (isinstance(a.argval, types.CodeType) and isinstance(b.argval, types.CodeType)):
                    same = False
                else:
                    inner = changed_lines(a.argval, b.argval)
                    if inner is None:
                        return None
                    out |= inner
            elif a.opcode in dis.hasconst:
                same = _const_equal(a.argval, b.argval)
            elif a.opcode in dis.hasname:
                # the low bits of these arguments are flags (method call, NULL push)
                same = _const_equal(a.argval, b.argval) and (a.arg & 3) == (b.arg & 3)
            elif a.opcode in dis.haslocal or a.opcode in dis.hasfree:
                same = _const_equal(a.argval, b.argval)
            elif a.opcode in dis.hasjump:
                same = a.argval == b.argval
            else:
                same = a.arg == b.arg
        if not same:
            la, lb = a.positions.lineno, b.positions.lineno
            if la is None or lb is None:
                return None
            out.add(la)
            out.add(lb)
    # constants that no instruction loads (a docstring) count as a change of the whole function
    for i, (a, b) in enumerate(zip(x.co_consts, y.co_consts)):
        if i not in used and not isinstance(a, types.CodeType) and not _const_equal(a, b):
            return None
    return out


def deftime_equal(a: types.CodeType, b: types.CodeType, pairs: list) -> bool:
    """True when the module-level / class-level code of a and b does the same thing.

    Function code objects found on the way are appended to pairs as (old, new)."""
    if (
        a.co_code != b.co_code
        or a.co_names != b.co_names
        or a.co_exceptiontable != b.co_exceptiontable
        or len(a.co_consts) != len(b.co_consts)
        or a.co_flags != b.co_flags
    ):
        return False
    for x, y in zip(a.co_consts, b.co_consts):
        ix = isinstance(x, types.CodeType)
        if ix != isinstance(y, types.CodeType):
            return False
        if ix:
            fx = x.co_flags & CO_OPTIMIZED
            if fx != (y.co_flags & CO_OPTIMIZED):
                return False
            if fx:
                if (
                    x.co_name != y.co_name
                    or x.co_qualname != y.co_qualname
                    or len(x.co_freevars) != len(y.co_freevars)
                ):
                    return False
                pairs.append((x, y))
            elif not deftime_equal(x, y, pairs):
                return False
        elif not _const_equal(x, y):
            return False
    return True


def compile_unit(lines: list[str], start: int, end: int, class_path, filename: str, flags: int, import_names=()):
    """Compiles file lines start..end (1-based, inclusive) as they stand inside their classes.

    The unit keeps its real line numbers and its indentation; one-line class wrappers give
    the enclosing class scopes (name mangling, qualified names, __class__ cells)."""
    return compile_lines(lines[start - 1:end], start, class_path, filename, flags, import_names)


def compile_lines(unit_lines, start: int, class_path, filename: str, flags: int, import_names=()):
    """Like compile_unit, for lines that are already cut out; start = line number of the first."""
    depth = len(class_path)
    pad = start - 1 - depth
    if pad < 0:
        return None
    head = "".join(" " * k + "class " + name + ":\n" for k, name in enumerate(class_path))
    src = "\n" * pad + head + "\n".join(unit_lines) + "\n"
    if import_names:
        # same module-level import bindings as the real file (they change how calls compile)
        src += "import " + ", ".join(import_names) + "\n"
    return compile(src, filename, "exec", flags=flags, dont_inherit=True)


class FunctionIndex:
    """All live function objects by (file, first line, name)."""

    def __init__(self, root):
        idx: dict = {}
        ft = types.FunctionType
        for o in gc.get_objects():
            if type(o) is ft:
                c = o.__code__
                if c.co_filename.startswith(root):
                    idx.setdefault((c.co_filename, c.co_firstlineno, c.co_name), []).append(o)
        self.idx = idx

    def live(self, code):
        return [
            f
            for f in self.idx.get((code.co_filename, code.co_firstlineno, code.co_name), ())
            if f.__code__ == code
        ]

    def any_at(self, code):
        return self.idx.get((code.co_filename, code.co_firstlineno, code.co_name), ())


SWAP_FAIL = None  # why the last plan_swaps call returned None: "cells" (see below) or "other"


def plan_swaps(pairs, index: FunctionIndex):
    """Returns the list of (function, new_code) or None when a safe swap is not possible."""
    global SWAP_FAIL
    SWAP_FAIL = "other"
    swaps = []
    stack = list(pairs)
    while stack:
        x, y = stack.pop()
        if func_code_equal(x, y):
            continue
        cands = index.any_at(x)
        live = [f for f in cands if f.__code__ == x]
        if len(live) != len(cands):
            return None  # a function sits there with code we cannot reproduce
        if live and x.co_freevars != y.co_freevars:
            if len(x.co_freevars) == len(y.co_freevars):
                SWAP_FAIL = "cells"
            # A live closure keeps the cells of the pinned code's free variables, in the order
            # of their names. New code that reads other free variables (same count, other
            # names) would read the wrong cells: only a fresh import builds the right closure.
            return None
        for f in live:
            swaps.append((f, y))
        xc = [c for c in x.co_consts if isinstance(c, types.CodeType)]
        yc = [c for c in y.co_consts if isinstance(c, types.CodeType)]
        if len(xc) != len(yc) or any(a.co_name != b.co_name for a, b in zip(xc, yc)):
            # nested shape changed: fine only when no nested function is alive now
            todo = list(xc)
            while todo:
                c = todo.pop()
                if index.any_at(c):
                    return None
                todo.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
            continue
        stack.extend(zip(xc, yc))
    return swaps


# ---------------------------------------------------------------------------
# Fresh import with patched sources


class OverlayFinder:
    """Meta path finder that serves patched or new modules from memory and hides deleted ones."""

    def __init__(self, root: str, files: dict):
        self.root = root
        self.files = files  # repo-relative path -> new text, or None when deleted

    def find_spec(self, name, path=None, target=None):
        import importlib.util

        top = name.partition(".")[0]
        if top not in ("sqlglot", "tests"):
            return None
        rel = name.replace(".", "/")
        for cand, is_pkg in ((rel + "/__init__.py", True), (rel + ".py", False)):
            if cand in self.files:
                text = self.files[cand]
                if text is None:
                    continue
                full = os.path.join(self.root, cand)
                loader = _MemLoader(full, text)
                return importlib.util.spec_from_file_location(
                    name, full, loader=loader,
                    submodule_search_locations=[os.path.dirname(full)] if is_pkg else None,
                )
            if os.path.exists(os.path.join(self.root, cand)):
                return None  # the normal machinery loads it from disk
        return None

    def hidden(self, name):
        rel = name.replace(".", "/")
        return self.files.get(rel + ".py", 1) is None or self.files.get(rel + "/__init__.py", 1) is None


# True when the process keeps the module code objects that it loaded from pycs (selector.py
# CODE_CACHE). Then this loader keeps its code object too: a constant that a module value
# holds has one more reference either way, and marshal.dumps (statefp's fast path) writes
# the same bytes for it (marshal marks objects with more than one reference).
KEEP_CODE = False


class _MemLoader:
    def __init__(self, path, text):
        self.path = path
        self.text = text
        self.code = None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Python compiles a source file from its bytes: the coding declaration of line 1 or 2
        # (PEP 263) decides how they decode, a byte order mark is dropped, and an unknown or
        # wrong encoding is a SyntaxError at import. The patch was read as UTF-8 with
        # surrogateescape, so this gives back the exact bytes of the patched file.
        code = compile(self.text.encode("utf-8", "surrogateescape"), self.path, "exec", dont_inherit=True)
        # Built the way a code object from a .pyc is (the pinned modules come from pycs): a
        # frozenset constant that marshal loads iterates in another order than the one that
        # compile() made, and a table built by iterating it (parsers/clickhouse.py
        # _AGG_FUNC_MAPPING) would then differ from the pinned table in its order alone.
        code = marshal.loads(marshal.dumps(code))
        if KEEP_CODE:
            self.code = code
        exec(code, module.__dict__)

    def get_source(self, name):
        return self.text

    def get_filename(self, name):
        return self.path


class _HideFinder:
    """Raises for modules whose file a patch deleted."""

    def __init__(self, names):
        self.names = names

    def find_spec(self, name, path=None, target=None):
        if name in self.names:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


def install_overlay(root: str, files: dict):
    deleted = set()
    for rel, text in files.items():
        if text is None and rel.endswith(".py"):
            mod = rel[:-3].replace("/", ".")
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            deleted.add(mod)
    sys.meta_path.insert(0, OverlayFinder(root, files))
    if deleted:
        sys.meta_path.insert(0, _HideFinder(deleted))
    import importlib

    importlib.invalidate_caches()


def install_open_overlay(root: str, files: dict):
    """Makes open() see the patched files. Returns a context manager factory that
    shows the pinned files again for a moment."""
    import builtins
    import contextlib
    import io

    table = {os.path.join(root, rel): text for rel, text in files.items()}
    real_open = builtins.open
    active = [True]

    def patched_open(file, mode="r", *args, **kwargs):
        try:
            key = os.path.abspath(os.fspath(file)) if isinstance(file, (str, os.PathLike)) else None
        except TypeError:
            key = None
        if active[0] and key is not None and key in table:
            text = table[key]
            if text is None:
                raise FileNotFoundError(2, "No such file or directory", str(file))
            if "w" in mode or "a" in mode or "+" in mode or "x" in mode:
                return real_open(file, mode, *args, **kwargs)
            if "b" in mode:
                return io.BytesIO(text.encode("utf-8", "surrogateescape"))
            newline = kwargs.get("newline")
            encoding = kwargs.get("encoding", args[1] if len(args) > 1 else None)
            if (encoding or "utf-8").lower().replace("_", "-") in ("utf-8", "utf8") and text.isascii():
                return io.StringIO(text, newline=newline)
            # other encodings, or bytes that are not plain ASCII: decode the file's bytes the
            # way open() does (a byte that is not valid UTF-8 raises there too)
            errors = kwargs.get("errors", args[2] if len(args) > 2 else None)
            return io.TextIOWrapper(
                io.BytesIO(text.encode("utf-8", "surrogateescape")),
                encoding=encoding or "utf-8", errors=errors, newline=newline,
            )
        return real_open(file, mode, *args, **kwargs)

    builtins.open = patched_open
    io.open = patched_open

    # runpy reads through io.open_code (test_sqlglotc runs sqlglotc/setup.py with runpy.run_path)
    real_open_code = io.open_code

    def patched_open_code(path):
        try:
            key = os.path.abspath(os.fspath(path))
        except TypeError:
            key = None
        if active[0] and key is not None and key in table:
            text = table[key]
            if text is None:
                raise FileNotFoundError(2, "No such file or directory", str(path))
            return io.BytesIO(text.encode("utf-8", "surrogateescape"))
        return real_open_code(path)

    io.open_code = patched_open_code

    @contextlib.contextmanager
    def read_old():
        active[0] = False
        try:
            yield
        finally:
            active[0] = True

    return read_old


# ---------------------------------------------------------------------------
# Direct test runs


def run_test(test_id: str, failfast: bool = True) -> bool:
    """Runs one test the way unittest does. Returns True when it fails."""
    import importlib
    import unittest

    path, cls_name, meth = test_id.split("::")
    mod_name = path[:-3].replace("/", ".")
    try:
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        case = cls(meth)
    except BaseException as e:  # import error, missing class, missing method
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        return True
    result = unittest.TestResult()
    result.failfast = failfast
    suite = unittest.TestSuite([case])
    try:
        suite.run(result)
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        return True
    return bool(result.failures or result.errors or result.unexpectedSuccesses)
