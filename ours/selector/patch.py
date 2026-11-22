"""Parse a git unified diff into per-file changes and apply it to the pinned tree."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list[str] = field(default_factory=list)  # each starts with ' ', '-', '+' or '\\'


@dataclass
class FileChange:
    old_path: str | None  # None for a new file
    new_path: str | None  # None for a deleted file
    hunks: list[Hunk] = field(default_factory=list)
    binary: bool = False
    mode_only: bool = False
    old_mode: str | None = None
    new_mode: str | None = None

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""

    @property
    def is_new(self) -> bool:
        return self.old_path is None

    @property
    def is_deleted(self) -> bool:
        return self.new_path is None

    @property
    def is_rename(self) -> bool:
        return self.old_path is not None and self.new_path is not None and self.old_path != self.new_path


def _strip_prefix(path: str) -> str | None:
    path = path.strip()
    if path == "/dev/null":
        return None
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape").encode("latin-1").decode("utf-8", "replace")
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def parse_patch(text: str) -> list[FileChange]:
    changes: list[FileChange] = []
    current: FileChange | None = None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        if line.startswith("diff --git "):
            match = re.match(r'^diff --git ("?a/.*?"?) ("?b/.*"?)$', line)
            old = new = None
            if match:
                old, new = _strip_prefix(match.group(1)), _strip_prefix(match.group(2))
            current = FileChange(old_path=old, new_path=new)
            changes.append(current)
            i += 1
            continue
        if line.startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ "):
            old = _strip_prefix(line[4:].split("\t")[0])
            new = _strip_prefix(lines[i + 1][4:].split("\t")[0])
            if current is None or current.hunks:
                current = FileChange(old_path=old, new_path=new)
                changes.append(current)
            else:
                current.old_path, current.new_path = old, new
            i += 2
            continue
        if line.startswith("@@") and current is not None:
            match = HUNK.match(line)
            if not match:
                i += 1
                continue
            hunk = Hunk(
                int(match.group(1)),
                1 if match.group(2) is None else int(match.group(2)),
                int(match.group(3)),
                1 if match.group(4) is None else int(match.group(4)),
            )
            current.hunks.append(hunk)
            i += 1
            old_seen = new_seen = 0
            while i < n and (old_seen < hunk.old_len or new_seen < hunk.new_len):
                body = lines[i]
                tag = body[:1]
                if body == "":
                    body, tag = " ", " "  # an empty context line whose space was stripped
                if tag == " ":
                    old_seen += 1
                    new_seen += 1
                elif tag == "-":
                    old_seen += 1
                elif tag == "+":
                    new_seen += 1
                elif tag != "\\":
                    break
                hunk.lines.append(body)
                i += 1
            while i < n and lines[i].startswith("\\"):
                hunk.lines.append(lines[i])
                i += 1
            continue
        if current is not None:
            if line.startswith("new file mode"):
                current.old_path = None
                current.new_mode = line.split()[-1]
            elif line.startswith("deleted file mode"):
                current.new_path = None
                current.old_mode = line.split()[-1]
            elif line.startswith("rename from "):
                current.old_path = line[len("rename from "):]
            elif line.startswith("rename to "):
                current.new_path = line[len("rename to "):]
            elif line.startswith("copy from "):
                current.old_path = line[len("copy from "):]
            elif line.startswith("copy to "):
                current.new_path = line[len("copy to "):]
            elif line.startswith("old mode "):
                current.old_mode = line.split()[-1]
            elif line.startswith("new mode "):
                current.new_mode = line.split()[-1]
            elif line.startswith("Binary files") or line.startswith("GIT binary patch"):
                current.binary = True
        i += 1
    for change in changes:
        if not change.hunks and not change.binary and not change.is_new and not change.is_deleted and not change.is_rename:
            if change.old_mode is not None or change.new_mode is not None:
                change.mode_only = True
            else:
                # neither hunks nor a mode change: a content change we cannot see
                # (truncated diff, unknown format); handled like a binary change
                change.binary = True
    return changes


def _hunk_full(hunk: Hunk) -> bool:
    old = sum(1 for l in hunk.lines if l[:1] in (" ", "-"))
    new = sum(1 for l in hunk.lines if l[:1] in (" ", "+"))
    return old >= hunk.old_len and new >= hunk.new_len


class PatchError(Exception):
    pass


def apply_hunks(old_text: str | None, change: FileChange) -> str:
    """Apply the change's hunks to old_text and return the new text."""
    if change.is_deleted:
        return ""
    old_lines = (old_text or "").splitlines(keepends=True)
    out: list[str] = []
    pos = 0  # index into old_lines
    for hunk in change.hunks:
        expected_old = [l[1:] for l in hunk.lines if l[:1] in (" ", "-")]
        start = hunk.old_start - 1 if hunk.old_len > 0 else hunk.old_start
        start = _locate(old_lines, expected_old, start, pos)
        out.extend(old_lines[pos:start])
        pos = start
        new_piece: list[str] = []
        previous = None
        for raw in hunk.lines:
            tag, body = raw[:1], raw[1:]
            if tag == "\\":
                # "\ No newline at end of file" applies to the previous line.
                if previous == "+" and new_piece:
                    new_piece[-1] = new_piece[-1].rstrip("\n")
                elif previous == " " and new_piece:
                    new_piece[-1] = new_piece[-1].rstrip("\n")
                continue
            if tag == " ":
                new_piece.append(old_lines[pos] if pos < len(old_lines) else body + "\n")
                pos += 1
            elif tag == "-":
                pos += 1
            elif tag == "+":
                new_piece.append(body + "\n")
            previous = tag
        out.extend(new_piece)
    out.extend(old_lines[pos:])
    return "".join(out)


def _norm(line: str) -> str:
    return line.rstrip("\r\n")


def _locate(old_lines, expected, guess, floor):
    def matches(at):
        if at < floor or at + len(expected) > len(old_lines):
            return False
        return all(_norm(old_lines[at + k]) == _norm(expected[k]) for k in range(len(expected)))

    if not expected:
        return max(floor, min(guess, len(old_lines)))
    if matches(guess):
        return guess
    for delta in range(1, len(old_lines) + 1):
        for at in (guess - delta, guess + delta):
            if matches(at):
                return at
        if guess - delta < floor and guess + delta > len(old_lines):
            break
    raise PatchError("hunk does not apply")


def changed_old_lines(change: FileChange) -> tuple[set[int], set[int]]:
    """Old-file line numbers removed/replaced, and insertion points.

    An insertion point k means new lines were inserted between old lines k and k+1
    (k = 0: before the first line).
    """
    removed: set[int] = set()
    inserts: set[int] = set()
    for hunk in change.hunks:
        old_no = hunk.old_start if hunk.old_len > 0 else hunk.old_start + 1
        for raw in hunk.lines:
            tag = raw[:1]
            if tag == " ":
                old_no += 1
            elif tag == "-":
                removed.add(old_no)
                old_no += 1
            elif tag == "+":
                inserts.add(old_no - 1)
    return removed, inserts
