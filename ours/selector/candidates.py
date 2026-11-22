"""From a patch to the set of tests that could possibly fail (a sound superset).

Each candidate carries the reason it was picked; the planner uses reasons to
decide what to verify by running and what to include outright.
"""

from __future__ import annotations

import os
import re
import tokenize
import importlib.machinery
import io
import sys
from dataclasses import dataclass, field

from .analyze import analyze_python
from .maps import Maps
from .patch import FileChange, apply_hunks
from . import tokcore

PYTEST_CONFIG = {"pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml", "conftest.py"}
# Class-level dicts whose string keys are SQL text (function names, keywords...).


@dataclass
class FileReport:
    path: str
    kind: str  # "py" | "data" | "new" | "deleted" | ...
    detail: str = ""
    certain: int = 0  # tests that fail for sure (e.g. module no longer imports)
    func: int = 0  # function-scope coverage candidates
    imp: int = 0  # import-time candidates
    data: int = 0  # data-file readers
    pydata: int = 0  # tests that read a changed .py file as text
    other: int = 0
    sites: list = field(default_factory=list)
    new_text: str | None = None
    old_text: str | None = None  # data files only
    site_masks: list = field(default_factory=list)  # one mask per Python change site (func/imp)
    # A change that runs when the module is imported (class/module level, decorators,
    # defaults) and might raise there: the verifier imports the module, and if that
    # fails every importer is included.
    import_check: bool = False
    # indexes into site_masks: import-time sites, and those that edit entries of a
    # string-keyed dict (selector/keyrefine.py may narrow them)
    imp_sites: list = field(default_factory=list)
    keyrec_sites: dict = field(default_factory=dict)  # index -> (dict holder, changed key descriptors)
    # class-body assignments whose readers keyrefine can re-derive from the state
    # fingerprints (selector/keyrefine.py, runtime_attr_mask)
    classrec_sites: dict = field(default_factory=dict)  # index -> (class key, assigned names)
    # index into site_masks -> (narrower mask, {(module, name)} bindings it may change):
    # a module-level value edit whose runtime readers through module objects are taken
    # from the build-time records; keyrefine applies it when the state fingerprint shows
    # that nothing but those bindings changed
    modrec_sites: dict = field(default_factory=dict)


@dataclass
class CommitCandidates:
    files: list[FileReport] = field(default_factory=list)
    unknown: bool = False  # something we cannot reason about
    leak: str = ""  # why the commit may leak state between tests: everything (selector/leak.py)
    leak_check: str = ""  # why the commit needs the state check (selector/leak.py)
    leak_write: bool = False  # ... for a reason other than copying alone

    @property
    def certain(self) -> int:
        m = 0
        for f in self.files:
            m |= f.certain
        return m

    @property
    def possible(self) -> int:
        m = 0
        for f in self.files:
            m |= f.certain | f.func | f.imp | f.data | f.pydata | f.other
        return m


def decode_source(raw: bytes) -> str | None:
    """Decode Python source the way the import system would; None if it cannot."""
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        text = raw.decode(encoding)
    except (SyntaxError, UnicodeDecodeError, LookupError):
        return None
    if text.startswith("﻿"):
        text = text[1:]
    return text


def _shadowed_module(rel: str, repo_root: str) -> str | None:
    """Name of the standard-library or installed module that a new repository-root
    module or package would shadow, or None."""
    parts = rel.split("/")
    if len(parts) == 1 and rel.endswith(".py"):
        name = rel[:-3]
    elif len(parts) == 2 and parts[1] == "__init__.py":
        name = parts[0]
    else:
        return None
    if not name.isidentifier():
        return None
    if name in ("sitecustomize", "usercustomize") or name in sys.stdlib_module_names:
        return name
    # Look only where the test run looks besides the repository: not in the repository
    # itself and not in the selector's own directory (sys.path[0] here).
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(repo_root)
    paths = []
    for entry in sys.path:
        full = os.path.abspath(entry or ".")
        if full != here and full != root and not full.startswith(root + os.sep):
            paths.append(entry)
    try:
        found = importlib.machinery.PathFinder.find_spec(name, paths) is not None
    except Exception:  # noqa: BLE001 - cannot tell: assume it is shadowed
        found = True
    return name if found else None


def _collected_by_pytest(rel: str) -> bool:
    base = os.path.basename(rel)
    return rel.startswith("tests/") and rel.endswith(".py") and (base.startswith("test_") or base.endswith("_test.py"))


def _inert_test_module(text: str | None) -> bool:
    """A module whose import only binds names: imports, functions, classes whose bodies only
    define functions or bind constants, constant assignments, a docstring, a __main__ block."""
    import ast as _ast

    if text is None:
        return False
    try:
        tree = _ast.parse(text)
    except (SyntaxError, ValueError):
        return False

    def constant(node) -> bool:
        if isinstance(node, _ast.Constant):
            return True
        if isinstance(node, (_ast.Tuple, _ast.List, _ast.Set)):
            return all(constant(e) for e in node.elts)
        if isinstance(node, _ast.Dict):
            return all(k is not None and constant(k) for k in node.keys) and all(constant(v) for v in node.values)
        return False

    def simple_decorator(node) -> bool:
        target = node.func if isinstance(node, _ast.Call) else node
        while isinstance(target, _ast.Attribute):
            target = target.value
        return isinstance(target, _ast.Name) and (not isinstance(node, _ast.Call) or all(
            constant(a) for a in list(node.args) + [k.value for k in node.keywords]))

    def stmt_ok(node, in_class=False) -> bool:
        if isinstance(node, (_ast.Import, _ast.ImportFrom, _ast.Pass)):
            return not in_class or isinstance(node, _ast.Pass)
        if isinstance(node, _ast.Expr) and isinstance(node.value, _ast.Constant):
            return True
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            return all(simple_decorator(d) for d in node.decorator_list) and all(
                constant(d) for d in list(node.args.defaults) + [k for k in node.args.kw_defaults if k is not None])
        if isinstance(node, _ast.ClassDef):
            return all(simple_decorator(d) for d in node.decorator_list) and all(
                isinstance(b, (_ast.Name, _ast.Attribute)) for b in node.bases) and not node.keywords and all(
                stmt_ok(b, in_class=True) for b in node.body)
        if isinstance(node, (_ast.Assign, _ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, _ast.Assign) else [node.target]
            return all(isinstance(t, _ast.Name) for t in targets) and (value is None or constant(value))
        if isinstance(node, _ast.If) and not in_class:
            test = node.test
            return isinstance(test, _ast.Compare) and isinstance(test.left, _ast.Name) and test.left.id == "__name__"
        return False

    return all(stmt_ok(n) for n in tree.body)


class CandidateFinder:
    def __init__(self, maps: Maps, repo_root: str):
        self.maps = maps
        self.root = repo_root
        self._py_files = None

    def _read(self, rel: str) -> bytes | None:
        try:
            with open(os.path.join(self.root, rel), "rb") as handle:
                return handle.read()
        except OSError:
            return None

    def _module_tests(self, rel: str) -> int:
        """Tests whose process imported this module."""
        return self.maps.import_mask(rel)

    def analyze(self, changes: list[FileChange]) -> CommitCandidates:
        result = CommitCandidates()
        try:
            self._tokcore_on = self._tokcore_enabled(changes)
        except Exception:  # noqa: BLE001 - only a narrowing: off
            self._tokcore_on = False
        for change in changes:
            result.files.append(self._file(change, result))
        return result

    # ------------------------------------------------------------------
    def _file(self, change: FileChange, result: CommitCandidates) -> FileReport:
        path = change.path
        old_rel = change.old_path
        maps = self.maps
        report = FileReport(path=path, kind="data")

        old_raw = self._read(old_rel) if old_rel else None
        if old_rel and old_raw is None:
            # Patch refers to a file we do not have: cannot reason precisely.
            report.kind = "unknown"
            report.other = maps.all_mask
            result.unknown = True
            return report

        new_text = None
        if not change.is_deleted and not change.binary:
            try:
                old_text = old_raw.decode("utf-8") if old_raw is not None else None
                new_text = apply_hunks(old_text, change)
            except Exception as error:  # noqa: BLE001
                report.kind = "unapplicable"
                report.detail = repr(error)
                report.other = maps.all_mask
                result.unknown = True
                return report
        report.new_text = new_text

        base = os.path.basename(path)
        if base in PYTEST_CONFIG and (base != "pyproject.toml" or self._pytest_config_changed(old_raw, new_text)):
            if base != "setup.cfg" or self._setup_cfg_pytest_changed(old_raw, new_text):
                report.kind = "pytest-config"
                report.other = maps.all_mask
                if base == "conftest.py":
                    # A conftest only affects tests below its directory.
                    directory = os.path.dirname(path)
                    mask = 0
                    for i, node in enumerate(maps.tests):
                        if node.startswith(directory + "/") or directory == "":
                            mask |= 1 << i
                    report.other = mask
                return report

        if change.is_rename and not change.hunks and not change.is_deleted:
            # A pure rename: the old path disappears and a new path appears.
            self._deleted(old_rel, report)
            self._new_file(path, new_text, report)
            report.kind = "rename"
            return report

        if change.is_deleted:
            self._deleted(old_rel, report)
            return report

        if change.is_new:
            self._new_file(path, new_text, report)
            return report

        if change.is_rename:
            self._deleted(old_rel, report)
            self._new_file(path, new_text, report)

        if change.binary:
            report.kind = "binary"
            report.data = maps.open_mask(old_rel) | self._module_tests(old_rel)
            return report

        if change.mode_only:
            report.kind = "mode"
            return report

        if path.endswith(".py"):
            self._python(old_rel, old_raw, new_text, report)
            # An edit that may change what later tests in the same process see
            # (selector/leak.py): "hard" ones include every test; "soft" ones keep
            # their narrowing but get the state check (policy.plan_commit). Only
            # edits that change the AST can change behaviour.
            trees = getattr(report, "trees", None)
            if report.sites and trees is not None:
                try:
                    from .leak import change_lines, classify

                    old_lines, new_lines, removed, added = change_lines(change)
                    old_text = old_raw.decode("utf-8") if old_raw is not None else None
                    hard, soft = classify(old_text, new_text, removed, added, old_lines, new_lines, trees=trees)
                except Exception as error:  # noqa: BLE001 - cannot tell: treat as leaking
                    hard, soft = f"leak check failed ({error!r})", None
                if hard:
                    report.kind = "may-leak"
                    report.detail = hard
                    report.other = maps.all_mask
                    result.unknown = True
                    result.leak = hard
                elif soft:
                    from .leak import COPY_ONLY

                    result.leak_check = (result.leak_check + "; " if result.leak_check else "") + f"{path}: {soft}"
                    if not soft.startswith(COPY_ONLY):
                        result.leak_write = True
            report.trees = None
        else:
            report.kind = "data"
            try:
                report.old_text = old_raw.decode("utf-8") if old_raw is not None else None
            except UnicodeDecodeError:
                report.old_text = None
            if self._sql_fixture_noop(old_rel, old_raw, new_text):
                report.detail = "only blank or -- comment lines changed"
                return report
            report.data = maps.open_mask(old_rel)
        return report

    @staticmethod
    def _sql_fixture_noop(rel, old_raw, new_text) -> bool:
        """tests/helpers.py reads every .sql fixture through _filter_comments,
        which drops empty lines and lines starting with "--"; a change that
        leaves the filtered text identical cannot change any test."""
        if not (rel.startswith("tests/fixtures/") and rel.endswith(".sql")) or old_raw is None or new_text is None:
            return False

        def filtered(text):
            return [line for line in text.splitlines() if line and not line.startswith("--")]

        try:
            return filtered(old_raw.decode("utf-8")) == filtered(new_text)
        except UnicodeDecodeError:
            return False

    def _pytest_config_changed(self, old_raw, new_text) -> bool:
        try:
            import tomllib

            old = tomllib.loads(old_raw.decode("utf-8")) if old_raw else {}
            new = tomllib.loads(new_text or "")
        except Exception:  # noqa: BLE001 - unparsable config: assume it matters
            return True
        return old.get("tool", {}).get("pytest") != new.get("tool", {}).get("pytest")

    def _setup_cfg_pytest_changed(self, old_raw, new_text) -> bool:
        import configparser

        def section(text):
            parser = configparser.ConfigParser(interpolation=None)
            try:
                parser.read_string(text or "")
            except Exception:  # noqa: BLE001
                return None
            return {k: dict(parser[k]) for k in parser.sections() if "pytest" in k}

        return section(old_raw.decode("utf-8") if old_raw else "") != section(new_text)

    def _deleted(self, rel: str, report: FileReport):
        maps = self.maps
        report.kind = "deleted"
        mask = maps.open_mask(rel) | self._module_tests(rel)
        # Tests defined in a deleted test file.
        for i, node in enumerate(maps.tests):
            if node.split("::", 1)[0] == rel:
                mask |= 1 << i
        report.certain |= mask
        report.other |= maps.list_mask(os.path.dirname(rel) or ".")

    def _new_file(self, rel: str, new_text: str | None, report: FileReport):
        maps = self.maps
        report.kind = "new"
        shadowed = _shadowed_module(rel, self.root)
        if shadowed:
            # The repository root is on sys.path ahead of the installed packages: a new
            # top-level module or package named like a standard-library or installed
            # module (or sitecustomize) replaces it for every importer, including the
            # third-party packages tests import (pandas importing dateutil).
            report.detail = f"new top-level module shadows {shadowed}"
            report.other |= maps.all_mask
            return
        if _collected_by_pytest(rel) and not _inert_test_module(new_text):
            # pytest imports every test module before the first test runs: code that a new
            # test module runs at import (or a module that does not even import) can change
            # what every test sees.
            report.detail = "new test module that runs code at import"
            report.other |= maps.all_mask
            return
        mask = 0
        directory = os.path.dirname(rel) or "."
        mask |= maps.list_mask(directory)
        if rel.endswith(".py") or os.path.isdir(os.path.join(self.root, rel[:-3] if rel.endswith(".py") else rel)):
            # Something may import it by name (e.g. an optional import that used
            # to fail). Find repository lines that mention the module's name.
            mask |= self._name_references(rel)
        report.other |= mask

    def _name_references(self, rel: str) -> int:
        maps = self.maps
        module = rel[:-3] if rel.endswith(".py") else rel
        if module.endswith("/__init__"):
            module = module[: -len("/__init__")]
        dotted = module.replace("/", ".")
        leaf = dotted.rsplit(".", 1)[-1]
        pattern = re.compile(r"\b" + re.escape(leaf) + r"\b")
        mask = 0
        for py in self._all_py_files():
            raw = self._read(py)
            if raw is None or leaf.encode() not in raw:
                continue
            text = raw.decode("utf-8", "replace")
            hits = [i + 1 for i, line in enumerate(text.split("\n")) if pattern.search(line)]
            if hits:
                mask |= maps.func_mask(py, hits) | maps.mod_mask(py, hits)
        return mask

    def _all_py_files(self):
        if self._py_files is None:
            files = []
            for base in ("sqlglot", "tests", "sqlglotc"):
                for dirpath, dirnames, filenames in os.walk(os.path.join(self.root, base)):
                    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                    for name in filenames:
                        if name.endswith(".py"):
                            files.append(os.path.relpath(os.path.join(dirpath, name), self.root))
            self._py_files = files
        return self._py_files

    def _python(self, rel: str, old_raw: bytes, new_text: str, report: FileReport):
        maps = self.maps
        report.kind = "py"
        old_src = decode_source(old_raw)
        new_raw = new_text.encode("utf-8")
        new_src = decode_source(new_raw)
        readers = maps.open_mask(rel)
        report.pydata |= readers
        if new_src is None:
            report.detail = "undecodable"
            report.certain |= self._module_tests(rel)
            return
        analysis = analyze_python(old_src, new_src, rel)
        report.trees = {"old": analysis.old_tree, "new": analysis.new_tree}
        if analysis.error and not analysis.syntax_error:
            report.detail = analysis.error
            report.other |= self._module_tests(rel)
            return
        if analysis.syntax_error:
            report.detail = "syntax error: " + (analysis.error or "")
            report.certain |= self._module_tests(rel)
            # A broken test file fails all of its own tests too.
            for i, node in enumerate(maps.tests):
                if node.split("::", 1)[0] == rel:
                    report.certain |= 1 << i
            return
        if analysis.noop:
            report.detail = "no AST change"
            return
        report.sites = analysis.sites
        module = rel[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        for site in analysis.sites:
            if site.kind == "docstring":
                continue
            if site.scope == "function":
                if site.kind == "lambda":
                    # The lambda's first line is also run by the class body or
                    # module that creates it; only the lambda's own code counts.
                    mask = maps.func_mask(rel, site.lines)
                else:
                    mask = maps.func_mask(rel, site.lines) | maps.mod_mask(rel, site.lines)
                if site.entries:
                    mask |= maps.entry_mask(rel, site.entries)
                report.func |= mask
            else:
                mask = self._import_site_mask(rel, module, site)
                report.imp |= mask
                report.imp_sites.append(len(report.site_masks))
                alt = self._module_site_alternative(rel, module, site)
                if alt is not None:
                    report.modrec_sites[len(report.site_masks)] = alt
                if site.scope == "class" and site.kind == "stmt" and site.names \
                        and not any(n.startswith("__") and n.endswith("__") for n in site.names) \
                        and getattr(site, "same_binding", False):
                    report.classrec_sites[len(report.site_masks)] = (f"{module}:{site.class_qual}", tuple(sorted(site.names)))
                if site.dict_keys and site.kind != "docstring" and getattr(site, "dict_order_same", False):
                    holder = None
                    if site.names and len(site.names) == 1:
                        name = next(iter(site.names))
                        holder = f"{module}:{site.class_qual}\t{name}" if site.scope == "class" and site.class_qual \
                            else (f"{module}\t{name}" if site.scope == "module" else None)
                    report.keyrec_sites[len(report.site_masks)] = (holder, tuple(site.dict_keys))
                if not site.value_safe:
                    report.import_check = True
            if site.kind == "def-header" and site.scope == "function" and not site.lines:
                report.import_check = True  # defaults and annotations run at import
            report.site_masks.append(mask)

    # -- import-time changes ---------------------------------------------------
    def _import_site_mask(self, rel, module, site) -> int:
        """Tests that can observe a changed class-body or module-level statement.

        Start from the tests that executed the statement at all (for an
        unconditional statement: every test that imported the module). For a
        class body, keep only tests that used the class: instantiated it or a
        subclass, or entered one of their methods (dialect families count as
        one class). Then, when the statement just binds names, keep only tests
        that ran a line mentioning one of those names, or a line that reads
        attributes by computed name that could match.
        """
        maps = self.maps
        base = maps.mod_mask(rel, site.lines) | maps.func_mask(rel, site.lines)
        if not site.lines:
            base = self._module_tests(rel)
        if not base:
            return 0
        names = set(site.names)
        dunder = any(n.startswith("__") and n.endswith("__") for n in names)
        if site.kind == "class-header" and len(names) > 1:
            # A renamed class: its __name__ feeds keys computed at import time (the
            # dialect registry keys "athena" off `class Athena`, expression keys and
            # SQL function names come from class names), read with no mention of
            # either name. Every test that executed the class statement.
            return base
        if site.kind == "class-header" and site.scope != "class":
            # Bases, keywords or decorators of a module-level class: every user of
            # the class (instances, methods, reads through its name), not only tests
            # that mention it (a dialect is used through its registry key).
            key = f"{module}:{next(iter(names))}" if names else None
            usage = self._class_usage(key) if key else None
            cross = self._class_name_reads(key, set()) if usage is not None else None
            if usage is None or cross is None:
                return base
            return base & (usage | cross | self._name_refs(names, rel, site.lines, ("object", "module", "instance"),
                                                           module=module))
        if site.scope == "class" and rel.startswith("tests/") and site.class_qual and "." not in site.class_qual \
                and names and all(n.startswith("test") for n in names) and site.kind != "class-header":
            # Only test methods of a test class changed (added, removed, redecorated):
            # that can affect only those tests themselves.
            prefix = f"{rel}::{site.class_qual}::"
            mask = 0
            for n in names:
                k = maps.index.get(prefix + n)
                if k is not None:
                    mask |= 1 << k
            return mask
        if site.scope == "class":
            usage = self._class_usage(f"{module}:{site.class_qual}")
            if usage is not None and os.environ.get("BR_CLASS_USAGE", "1") == "0":
                usage = maps.all_mask  # measurement switch: no class-usage narrowing
            if usage is not None:
                cross = self._class_name_reads(f"{module}:{site.class_qual}", names)
                usage = None if cross is None else usage | cross
            if usage is None:
                return base
            if site.kind == "class-header":
                return base & (usage | self._name_refs(names, rel, site.lines, ("object", "module", "instance")))
            cand = base & usage
            if names and not dunder:
                owner = f"{module}:{site.class_qual}"
                cand &= self._name_refs(names, rel, site.lines, ("object",),
                                        owner=owner if owner in maps.classes else None,
                                        truth_same=getattr(site, "truth_same", False),
                                        same_keys=getattr(site, "same_keys", False))
            if cand and site.dict_keys:
                keyed = self._dict_key_usage(site.dict_keys, names, cand)
                if keyed is not None:
                    cand &= keyed
            return cand
        if not names or dunder:
            return base
        # A module-level name is read by this module's own code, or by modules
        # that reach it through this module (import it, or name it).
        return base & self._name_refs(names, rel, site.lines, ("object", "module", "instance"), module=module,
                                      same_binding=getattr(site, "same_binding", False),
                                      plain_value=getattr(site, "plain_value", False))

    def _module_site_alternative(self, rel, module, site):
        """For a module-level edit that only changes the value bound to plain names in
        a sqlglot module: (mask, bindings). The mask is the M12 rule with one change:
        attribute reads `<x>.NAME` inside functions count only for tests without a
        module-read record; tests with one count when they read one of the bindings
        through a module object. Bindings are (module, name) pairs that hold the value:
        the name itself and every `from M import NAME [as A]` / star import of it,
        transitively. Valid only if nothing else in the program state changes, which
        keyrefine checks with the state fingerprints."""
        maps = self.maps
        rec = getattr(maps, "modrec", None)
        if not rec or site.scope != "module" or not getattr(site, "same_binding", False):
            return None
        if not rel.startswith("sqlglot/") or not site.names or site.kind == "docstring":
            return None
        names = set(site.names)
        if any(n.startswith("__") and n.endswith("__") for n in names):
            return None
        bindings = self._binding_group(module, names)
        base = maps.mod_mask(rel, site.lines) | maps.func_mask(rel, site.lines)
        if not site.lines:
            base = self._module_tests(rel)
        # re-exports under another name (`from M import NAME as A`) are read as `R.A`
        refs = self._name_refs(names | {alias for _, alias in bindings}, rel, site.lines,
                               ("object", "module", "instance"), module=module,
                               same_binding=True, plain_value=getattr(site, "plain_value", False),
                               module_records=bindings)
        return base & refs, bindings

    def _binding_group(self, module, names):
        """{(module, name)} bound to the same value as `module`.NAME through
        `from X import NAME [as A]` and `from X import *`, transitively."""
        group = {(module, n) for n in names}
        changed = True
        while changed:
            changed = False
            modules = {m for m, _ in group}
            for py in self._all_py_files():
                if not py.startswith("sqlglot/"):
                    continue
                raw = self._source(py)
                index = self._file_index(py, raw) if raw else None
                if index is None:
                    continue
                mod = py[:-3].replace("/", ".")
                if mod.endswith(".__init__"):
                    mod = mod[: -len(".__init__")]
                for base, aliases, _ in index["imports"]:
                    if base not in modules:
                        continue
                    for name, asname in aliases:
                        if name == "*":
                            new = {(mod, n) for m, n in group if m == base}
                        elif (base, name) in group:
                            new = {(mod, asname or name)}
                        else:
                            continue
                        if not new <= group:
                            group |= new
                            changed = True
        return group

    def _dict_key_usage(self, keys, names=(), within=None):
        """Tests that could look up these dict entries.

        * Every key a sqlglot class (TRANSFORMS = {exp.Foo: ...}): an entry keyed
          by a class is only consulted for instances of that class or subclasses.

        Dicts keyed by SQL text (FUNCTIONS, KEYWORDS, ...) are not narrowed: the SQL
        a test runs is often built at runtime (f"CAST(x AS {t.value})" in
        tests/dialects/test_mysql.py::test_types, DataType.build("int") inside the
        library), so the test's text need not contain the key. (A rule that
        required the word in the test's source or data files missed
        test_mysql.py::test_types for a KEYWORDS "INT" edit, labelled with pytest.)
        """
        if keys and all(k[0] == "str" for k in keys):
            return None
        if not hasattr(self, "_by_simple_name"):
            index = {}
            for k in self.maps.classes:
                index.setdefault(k.rsplit(".", 1)[-1].split(":")[-1], []).append(k)
            self._by_simple_name = index
        mask = 0
        for key in keys:
            if key[0] not in ("attr", "name"):
                return None
            matches = self._by_simple_name.get(key[1])
            if not matches:
                return None
            for k in matches:
                usage = self._class_usage(k, family=False)
                if usage is None:
                    return None
                mask |= usage
        return mask

    def _mod(self, py, lines) -> int:
        """Coverage of module/class-level (import-time) execution of `lines`; nothing
        in runtime-only mode, where import-time effects are known from the state
        fingerprints instead."""
        if getattr(self, "_runtime_only", False):
            return 0
        return self.maps.mod_mask(py, lines)

    def _hits_mask(self, py, raw, hits) -> int:
        """Coverage of lines that read a changed name. A read at import time
        inside an entry of a class-level dict literal (FUNCTIONS = {"DATEADD":
        build(..., unit_mapping=NAME)}) only matters where that entry is used,
        so it is narrowed like a change to that entry."""
        maps = self.maps
        mask = maps.func_mask(py, hits)
        if getattr(self, "_runtime_only", False):
            return mask
        mod_hits = [h for h in hits if maps.mod_mask(py, [h])]
        if not mod_hits:
            return mask
        entries = self._dict_entries(py, raw)
        module = py[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        for h in mod_hits:
            refined = None
            for start, end, class_qual, dict_name, key in entries:
                if start <= h <= end:
                    usage = self._class_usage(f"{module}:{class_qual}")
                    if usage is not None:
                        keyed = self._dict_key_usage([key], {dict_name}, usage)
                        if keyed is not None:
                            refined = usage & keyed
                    break
            mask |= maps.mod_mask(py, [h]) if refined is None else refined
        return mask

    def _dict_entries(self, py, raw):
        import ast as _ast

        from .analyze import _key_descriptor

        cache = self.__dict__.setdefault("_entries", {})
        if py in cache:
            return cache[py]
        trees = self.__dict__.setdefault("_trees", {})
        tree = trees.get(py)
        if tree is None:
            try:
                tree = trees[py] = _ast.parse(raw)
            except SyntaxError:
                cache[py] = []
                return []
        out = []

        def walk(body, prefix):
            for node in body:
                if isinstance(node, _ast.ClassDef):
                    qual = prefix + node.name
                    for child in node.body:
                        if isinstance(child, (_ast.Assign, _ast.AnnAssign)) and isinstance(child.value, _ast.Dict):
                            targets = child.targets if isinstance(child, _ast.Assign) else [child.target]
                            if len(targets) != 1 or not isinstance(targets[0], _ast.Name):
                                continue
                            for k, v in zip(child.value.keys, child.value.values):
                                start = k.lineno if k is not None else v.lineno
                                out.append((start, v.end_lineno, qual, targets[0].id, _key_descriptor(k)))
                    walk(node.body, qual + ".")

        walk(tree.body, "")
        cache[py] = out
        return out

    def _tree(self, py, raw):
        import ast as _ast

        cache = self.__dict__.setdefault("_trees", {})
        tree = cache.get(py)
        if tree is None:
            try:
                tree = _ast.parse(raw)
            except SyntaxError:
                return None
            cache[py] = tree
        return tree

    @staticmethod
    def _import_base(py, node):
        base = node.module or ""
        if node.level:
            package = py[:-3].replace("/", ".").rsplit(".", 1)[0]
            parts = package.split(".")
            base = ".".join(parts[: len(parts) - node.level + 1] + ([base] if base else []))
        return base

    def _file_index(self, py, raw):
        """One pass over a file's AST, cached: its `from M import ...` statements
        (resolved base, [(name, asname)], line), the lines of every Name id, and
        every attribute access (attr -> [(first line, last line, leaf of the value)])."""
        import ast as _ast

        cache = self.__dict__.setdefault("_indexes", {})
        if py in cache:
            return cache[py]
        tree = self._tree(py, raw)
        if tree is None:
            cache[py] = None
            return None
        imports, names, attrs, strings = [], {}, {}, []
        stack = [(tree, False)]
        while stack:
            node, in_function = stack.pop()
            if isinstance(node, _ast.Constant) and isinstance(node.value, str):
                strings.append((node.lineno, node.end_lineno or node.lineno, node.value))
            elif isinstance(node, _ast.ImportFrom):
                imports.append((self._import_base(py, node), [(a.name, a.asname) for a in node.names], node.lineno))
            elif isinstance(node, _ast.Attribute):
                value = node.value
                leaf = value.id if isinstance(value, _ast.Name) else (value.attr if isinstance(value, _ast.Attribute) else None)
                attrs.setdefault(node.attr, []).append((node.lineno, node.end_lineno or node.lineno, leaf, in_function))
            elif isinstance(node, _ast.Name):
                names.setdefault(node.id, []).append(node.lineno)
            inner = in_function or isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.Lambda))
            for child in _ast.iter_child_nodes(node):
                # decorators, defaults and annotations run where the def statement runs
                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and (
                        child in node.decorator_list or child is node.args or child is node.returns):
                    stack.append((child, in_function))
                elif isinstance(node, _ast.Lambda) and child is node.args:
                    stack.append((child, in_function))
                else:
                    stack.append((child, inner))
        cache[py] = index = {"imports": imports, "names": names, "attrs": attrs, "strings": strings}
        return index

    def _external_module_refs(self, py, raw, names, keep_import_lines=True, split=False):
        """Lines of another file that can read a module-level binding of one of
        `names`, whatever module object they reach it through: the module may be
        imported under an alias (`from sqlglot import expressions as exp`, then
        `exp.NAME`) or re-export the name by a star import, so module identity is
        not tracked. Counted: every attribute access `<anything>.NAME`; every use
        of a name bound by `from <anything> import NAME [as A]`; in a file with a
        star import, every bare use of NAME; every string constant mentioning NAME
        (getattr(m, "NAME"), __all__). With keep_import_lines=False (only the value
        bound to the names changed) a `from M import NAME` line is not itself a
        read: it binds the same object, and the later reads are lines of their own;
        star imports stay reads."""
        index = self._file_index(py, raw)
        if index is None:
            everything = set(range(1, raw.count(b"\n") + 2))  # cannot tell: everything
            return (everything, set()) if split else everything
        imported = set()
        hits = set()
        runtime_attrs = set()
        star = False
        for base, aliases, lineno in index["imports"]:
            for name, asname in aliases:
                if name == "*":
                    star = True
                    hits.add(lineno)
                elif name in names:
                    imported.add(asname or name)
                    if keep_import_lines:
                        hits.add(lineno)
        if star:
            imported.update(names)
        for name in names:
            for first, last, _leaf, in_function in index["attrs"].get(name, ()):
                if split and in_function:
                    runtime_attrs.update(range(first, last + 1))
                else:
                    hits.update(range(first, last + 1))
        for name in imported:
            hits.update(index["names"].get(name, ()))
        word = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in sorted(names)) + r")\b")
        for first, last, value in index["strings"]:
            if word.search(value):
                hits.update(range(first, last + 1))
        return (hits, runtime_attrs) if split else hits

    def runtime_attr_mask(self, holder, attr):
        """Tests that can read class attribute holder.attr at runtime, given that its
        import-time readers' effects are known (keyrefine compares the whole program
        state): the class-attribute rule (class users, or tests that read it through
        the class name, AND lines that mention the name, typed where recorded,
        computed-name readers) with every module/class-level line left out. None when
        the class is unknown to the class map."""
        maps = self.maps
        module = holder.split(":", 1)[0]
        base = module.replace(".", "/")
        rel = base + ".py" if os.path.exists(os.path.join(self.root, base + ".py")) else base + "/__init__.py"
        self._runtime_only = True
        try:
            usage = self._class_usage(holder)
            if usage is None:
                return None
            cross = self._class_name_reads(holder, {attr})
            if cross is None:
                return None
            refs = self._name_refs({attr}, rel, [], ("object",), owner=holder if holder in maps.classes else None)
        finally:
            self._runtime_only = False
        return maps.import_mask(rel) & (usage | cross) & refs

    def _class_name_reads(self, key, names):
        """Tests that ran a line reading one of `names` through the class `key` (or a
        subclass) spelled by name (`MySQL.TIME_MAPPING` in Presto's class body,
        `Hive.INVERSE_TIME_MAPPING` in PrestoGenerator): such a read needs no instance,
        so it is a use of the class the instantiation records do not show. A read of
        another attribute through the class (`MySQL.TIME_FORMAT`) cannot read NAME
        (nested chains are recorded per class: `MySQL.Tokenizer.KEYWORDS` is a read of
        `Tokenizer` through MySQL and of `KEYWORDS` through MySQL.Tokenizer). Without
        names (a class header) every such read counts. None when the table of such
        reads is missing (then the class usage is not used)."""
        maps = self.maps
        if maps.xreads is None:
            return None
        group = maps.descendants(key) | {key}
        mask = 0
        for k in group:
            for rel, line, attr in maps.xreads.get(k, ()):
                if names and attr not in names:
                    continue
                mask |= maps.func_mask(rel, [line]) | self._mod(rel, [line])
        return mask

    def _class_usage(self, key, family=True):
        maps = self.maps
        classes = maps.classes
        if key not in classes:
            return None
        group = maps.descendants(key)
        if not hasattr(self, "_families"):
            families = {}
            for k, v in classes.items():
                families.setdefault(v.get("family"), set()).add(k)
            self._families = families
        for k in list(group) if family else ():
            fam = classes.get(k, {}).get("family")
            if fam is not None:
                group |= self._families.get(fam, set())
        mask = 0
        for k in group:
            info = classes.get(k)
            if info is None or not info.get("tracked"):
                return None
            mask |= maps.inst_mask(k)
            if info.get("file") and info.get("methods"):
                mask |= maps.entry_mask(info["file"], info["methods"])
        return mask

    def _source(self, rel):
        cache = getattr(self, "_source_cache", None)
        if cache is None:
            cache = self._source_cache = {}
        if rel not in cache:
            cache[rel] = self._read(rel) or b""
        return cache[rel]

    def _typed_mask(self, py, line, owner, table=None) -> int:
        """Tests that ran `line` (which reads the changed names only as self.NAME /
        cls.NAME, or only as self.dialect.NAME with table=typed_dialect) with a
        receiver that is `owner` or a subclass, plus every test that ran it without
        a typed record for it."""
        maps = self.maps
        table = maps.typed if table is None else table
        ran = maps.func_mask(py, [line]) | self._mod(py, [line])
        recorded = 0
        hit = 0
        for key, k in table[py][str(line)].items():
            m = maps.mask(k)
            recorded |= m
            ancestors = maps.typed_mro.get(key)
            if ancestors is None or owner in ancestors:
                hit |= m
        return (ran & ~recorded) | (ran & hit)

    @staticmethod
    def _only_attribute_reads(line, names) -> bool:
        """The line mentions the names, and every mention is `self.NAME` or `cls.NAME`."""
        mentioned = False
        for name in names:
            total = len(re.findall(r"\b" + re.escape(name) + r"\b", line))
            if total and total != len(re.findall(r"\b(?:self|cls)\." + re.escape(name) + r"\b", line)):
                return False
            mentioned = mentioned or total > 0
        return mentioned

    def _dynamic_mask(self, rel, line) -> int:
        """Tests that ran the computed-name read at rel:line. When every call on
        that line that reads by computed name (getattr/vars/__dict__) sits inside a
        function or lambda body, only runs of that function count: the module or
        class body executing the same line merely defines the function
        (executor/env.py's EXTRACT lambda shares its line with the dict literal)."""
        import ast as _ast

        maps = self.maps
        cache = self.__dict__.setdefault("_dynamic_inside", {})
        key = (rel, line)
        if key not in cache:
            inside = None
            raw = self._source(rel)
            tree = self._tree(rel, raw) if raw else None
            if tree is not None:
                inside = True
                found = False

                def visit(node, in_func):
                    nonlocal inside, found
                    is_reader = (isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
                                 and node.func.id in ("getattr", "vars", "hasattr", "setattr")) \
                        or (isinstance(node, _ast.Attribute) and node.attr == "__dict__")
                    if is_reader and hasattr(node, "lineno") and node.lineno <= line <= (node.end_lineno or node.lineno):
                        found = True
                        if not in_func:
                            inside = False
                    child_in = in_func or isinstance(node, (_ast.Lambda, _ast.FunctionDef, _ast.AsyncFunctionDef))
                    for child in _ast.iter_child_nodes(node):
                        visit(child, child_in)

                visit(tree, False)
                if not found:
                    inside = None  # cannot tell which call reads: count every run of the line
            cache[key] = inside
        if cache[key]:
            return maps.func_mask(rel, [line])
        return maps.func_mask(rel, [line]) | self._mod(rel, [line])

    def _class_only_reader(self, rel, line) -> bool:
        """Whether the computed-name attribute read at rel:line can only use a
        class: `inspect.getmembers(m, pred)` whose predicate starts with
        `inspect.isclass(...)` (and), or `getattr(m, name)(...)` (the value is
        called). A change between two plain-data values (neither a class nor
        callable) cannot change what such a reader does."""
        import ast as _ast

        cache = self.__dict__.setdefault("_class_only", {})
        key = (rel, line)
        if key in cache:
            return cache[key]
        result = False
        raw = self._source(rel)
        tree = self._tree(rel, raw) if raw else None
        if tree is not None:
            for node in _ast.walk(tree):
                if not isinstance(node, _ast.Call) or not (node.lineno <= line <= (node.end_lineno or node.lineno)):
                    continue
                func = node.func
                name = func.attr if isinstance(func, _ast.Attribute) else getattr(func, "id", None)
                if name == "getmembers" and len(node.args) >= 2 and isinstance(node.args[1], _ast.Lambda):
                    body = node.args[1].body
                    first = body.values[0] if isinstance(body, _ast.BoolOp) and isinstance(body.op, _ast.And) else body
                    if isinstance(first, _ast.Call):
                        f = first.func
                        if (f.attr if isinstance(f, _ast.Attribute) else getattr(f, "id", None)) == "isclass":
                            result = True
                if isinstance(func, _ast.Call):
                    inner = func.func
                    if getattr(inner, "id", None) == "getattr":
                        result = True
        cache[key] = result
        return result

    # -- tokenizer settings (selector/tokcore.py) ------------------------------
    def _tokcore_enabled(self, changes) -> bool:
        """The owner records of the tokenizer core hold for this commit: the core
        and Tokenizer._init_core are unchanged and nothing else builds a core."""
        if not self.maps.tokcore.get("lines"):
            return False
        for change in changes:
            paths = {change.path, change.old_path}
            if tokcore.CORE_FILE in paths:
                return False
            if tokcore.TOKENS_FILE in paths:
                if change.path != change.old_path or change.is_deleted or change.binary:
                    return False
                old = self._source(tokcore.TOKENS_FILE).decode("utf-8")
                new = apply_hunks(old, change)
                before = tokcore.init_core_dump(old)
                if before is None or before != tokcore.init_core_dump(new):
                    return False
                continue
            if change.binary:
                continue
            old_raw = self._read(change.old_path) if change.old_path else None
            new = apply_hunks(old_raw.decode("utf-8") if old_raw is not None else None, change) \
                if not change.is_deleted else ""
            if "TokenizerCore" in new or "_init_core" in new:
                return False
        return True

    def _tokcore_plumbing(self):
        cache = self.__dict__.get("_plumbing_cache")
        if cache is None:
            cache = tokcore.plumbing(self._source(tokcore.TOKENS_FILE).decode("utf-8"))
            self._plumbing_cache = cache
        return cache

    def _tokcore_observers(self):
        cache = self.__dict__.get("_observers_cache")
        if cache is None:
            cache = tokcore.observers(self._source(tokcore.CORE_FILE).decode("utf-8"))
            self._observers_cache = cache
        return cache

    @staticmethod
    def _only_plumbed(line, names, plumbs) -> bool:
        """Every mention of the names on this _init_core line is one `field=self.NAME`."""
        settings = [setting for setting, _, _ in plumbs]
        mentioned = False
        for name in names:
            total = len(re.findall(r"\b" + re.escape(name) + r"\b", line))
            if total and (total != 1 or settings.count(name) != 1):
                return False
            mentioned = mentioned or total > 0
        return mentioned

    def _blind_lines(self, py, raw, hits, names, same_keys) -> set:
        """Hit lines whose every mention of the names is used only for its truth
        value (or, with same_keys, also as the right operand of in / not in)."""
        tree = self._tree(py, raw)
        if tree is None:
            return set()
        cache = self.__dict__.setdefault("_parents_cache", {})
        parents = cache.get(py)
        if parents is None:
            parents = cache[py] = tokcore.parents_of(tree)
        allowed = {"truth", "member"} if same_keys else {"truth"}
        text = raw.decode("utf-8", "replace").split("\n")
        index = self.__dict__.setdefault("_mention_index", {}).get(py)
        if index is None:
            index = self._mention_index[py] = tokcore.mention_index(tree)
        by_line = {}
        for name in names:
            for line, nodes in index.get(name, {}).items():
                by_line.setdefault(line, []).extend(nodes)
        out = set()
        for i in hits:
            if not 0 < i <= len(text):
                continue
            kinds = tokcore.mention_kinds(tree, parents, i, set(names), text[i - 1], by_line)
            if kinds and kinds <= allowed:
                out.add(i)
        return out

    def _tokcore_mask(self, py, line, owner, names, plumbs, truth_same, same_keys=False) -> int:
        """Tests that can see a changed tokenizer setting through the core: those
        whose record shows a core line observing the field run for a tokenizer of
        class `owner` or a subclass (or a core of unknown origin), plus every test
        that ran this _init_core line without a record."""
        maps = self.maps
        tc = maps.tokcore
        if str(line) in maps.typed.get(py, {}):
            ran = self._typed_mask(py, line, owner)
        else:
            ran = maps.func_mask(py, [line]) | self._mod(py, [line])
        out = ran & ~maps.mask(tc["traced"])
        observers = self._tokcore_observers()
        for setting, field_name, through_bool in plumbs:
            if setting not in names:
                continue
            if through_bool and truth_same:
                continue  # bool(SETTING) is the same before and after
            for oline, kind in observers.get(field_name, ()):
                if (kind == "truth" and (truth_same or same_keys)) or (kind == "member" and same_keys):
                    continue
                for okey, k in tc["lines"].get(str(oline), {}).items():
                    ancestors = maps.typed_mro.get(okey)
                    if okey == "?" or ancestors is None or owner in ancestors:
                        out |= maps.mask(k)
        return out

    def _bare_annotations(self, py, raw):
        """{(line, name)} of one-line `name: annotation` statements without a value,
        at module level or directly in the body of a plain class (no decorators, no
        class keywords, no bases other than object/Protocol). Such a statement only
        records an annotation. (In a dataclass-like class the field machinery would
        look up an inherited `name` as the default: those classes are left out.)"""
        import ast as _ast

        cache = self.__dict__.setdefault("_bare_cache", {})
        if py not in cache:
            found = set()
            tree = self._tree(py, raw)

            def plain_class(node):
                if node.decorator_list or node.keywords:
                    return False
                for base in node.bases:
                    name = base.attr if isinstance(base, _ast.Attribute) else getattr(base, "id", None)
                    if name not in ("object", "Protocol"):
                        return False
                return True

            def visit(body):
                for node in body:
                    if isinstance(node, _ast.AnnAssign) and node.value is None and isinstance(node.target, _ast.Name) \
                            and node.lineno == node.end_lineno:
                        found.add((node.lineno, node.target.id))
                    elif isinstance(node, _ast.ClassDef) and plain_class(node):
                        visit(node.body)

            if tree is not None:
                visit(tree.body)
            cache[py] = found
        return cache[py]

    @staticmethod
    def _only_dialect_reads(line, names) -> bool:
        """The line mentions the names, and every mention is `self.dialect.NAME`."""
        mentioned = False
        for name in names:
            total = len(re.findall(r"\b" + re.escape(name) + r"\b", line))
            if total and total != len(re.findall(r"\bself\.dialect\." + re.escape(name) + r"\b", line)):
                return False
            mentioned = mentioned or total > 0
        return mentioned

    def _name_refs(self, names, site_rel, site_lines, kinds, module=None, owner=None, same_binding=False,
                   plain_value=False, truth_same=False, same_keys=False, module_records=None) -> int:
        maps = self.maps
        needles = [n.encode() for n in names]
        traced_mask = 0
        if module_records is not None:
            traced_mask = maps.mask(maps.modrec.get("traced"))  # tests that have a record
        alternatives = "|".join(re.escape(n) for n in sorted(names))
        pattern = re.compile(r"\b(?:" + alternatives + r")\b")
        definition = re.compile(r"^\s*(?:" + alternatives + r")\s*(?::[^=]*)?=(?!=)")
        mask = 0
        for py in self._all_py_files():
            raw = self._source(py)
            if not any(needle in raw for needle in needles):
                continue
            if module is not None and py != site_rel:
                if module_records is None:
                    hits = self._external_module_refs(py, raw, names, keep_import_lines=not same_binding)
                    if hits:
                        mask |= self._hits_mask(py, raw, hits)
                else:
                    hits, runtime_attrs = self._external_module_refs(py, raw, names, keep_import_lines=not same_binding,
                                                                     split=True)
                    if hits:
                        mask |= self._hits_mask(py, raw, hits)
                    if runtime_attrs:
                        mask |= self._hits_mask(py, raw, runtime_attrs) & ~traced_mask
                continue
            hits = set()
            bare = self._bare_annotations(py, raw)
            for i, line in enumerate(raw.decode("utf-8", "replace").split("\n"), start=1):
                found = pattern.findall(line)
                if not found:
                    continue
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if definition.match(line) and len(found) == 1:
                    continue  # `NAME = ...` (re)defines the name; it does not read it
                if len(found) == 1 and (i, found[0]) in bare:
                    continue  # `NAME: <annotation>` without a value reads nothing
                hits.add(i)
                if stripped.startswith("."):
                    # A chained attribute is reported on the line where the chain starts.
                    hits.update((i - 1, i - 2, i - 3))
            if py == site_rel:
                hits -= set(site_lines)
            if owner is not None and (truth_same or same_keys) and hits:
                # Only the truth value (and, for a dict whose keys stay the same, only
                # `key in VALUE`) of a class attribute is used on a line: such a line
                # cannot tell the old value from the new one (selector/tokcore.py rules).
                hits -= self._blind_lines(py, raw, hits, names, same_keys)
            if owner is not None and (maps.typed.get(py) or getattr(maps, "typed_dialect", {}).get(py)):
                # A class attribute changed in class `owner`: a line that only reads
                # it as self.NAME / cls.NAME (or as self.dialect.NAME) sees the change
                # only when the receiver (or the dialect object) is `owner` or a
                # subclass (typed coverage); other lines as usual.
                typed_lines = maps.typed.get(py, {})
                dialect_lines = getattr(maps, "typed_dialect", {}).get(py, {})
                text = raw.decode("utf-8", "replace").split("\n")
                narrowed = set()
                plumbs = self._tokcore_plumbing() if (py == tokcore.TOKENS_FILE and self._tokcore_on) else {}
                for i in hits:
                    if not 0 < i <= len(text):
                        continue
                    if i in plumbs and self._only_plumbed(text[i - 1], names, plumbs[i]):
                        mask |= self._tokcore_mask(py, i, owner, names, plumbs[i], truth_same, same_keys)
                        narrowed.add(i)
                    elif str(i) in typed_lines and self._only_attribute_reads(text[i - 1], names):
                        mask |= self._typed_mask(py, i, owner)
                        narrowed.add(i)
                    elif str(i) in dialect_lines and self._only_dialect_reads(text[i - 1], names):
                        mask |= self._typed_mask(py, i, owner, table=maps.typed_dialect)
                        narrowed.add(i)
                hits -= narrowed
            mask |= self._hits_mask(py, raw, hits)
        if module_records is not None:
            reads = maps.modrec.get("reads", {})
            for mod, name in module_records:
                mask |= maps.mask(reads.get(f"{mod}\t{name}"))
        for rel, line, regex, kind in maps.dynamic:
            if kind not in kinds:
                continue
            if kind == "module" and module is not None:
                # getattr(<module>, name) loaders read sibling modules of their
                # own package (sqlglot.dialects.__getattr__ reads dialect modules);
                # serde/helper read expression classes.
                same_package = os.path.dirname(rel) == os.path.dirname(site_rel)
                expressions = site_rel.startswith("sqlglot/expressions/") and rel in ("sqlglot/serde.py", "sqlglot/helper.py")
                if not (same_package or expressions):
                    continue
            if plain_value and self._class_only_reader(rel, line):
                continue
            if regex is None or any(re.fullmatch(regex, n) for n in names):
                mask |= self._dynamic_mask(rel, line)
        return mask
