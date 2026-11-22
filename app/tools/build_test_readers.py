#!/usr/bin/env python3
"""build_test_readers.py ROOT TESTS_JSON OUT: a static map of the pinned test sources.

tests/helpers.py defines module-level data (TPCH_SCHEMA, TPCDS_SCHEMA, ...) that test files
import. The coverage maps cannot say which tests read such a value: it is built when the
module is imported, before any test runs. This map lists, for each module-level statement of
tests/helpers.py, the names it defines, and for each name the tests whose code refers to it:
in the test method, in setUp / setUpClass / the class body / decorators, in functions of the
same test file or of tests/helpers.py that the test calls, or through a module-level name of
the test file that is computed from it. "direct" holds the tests whose own method (or what it
calls) refers to the name; "readers" adds the tests that only meet it through class set-up.

Output: {"tests/helpers.py": {"stmts": [[first line, last line, [names], is a literal]],
                              "readers": {name: [test numbers]}, "direct": {name: [...]}}}"""
import ast
import json
import os
import sys

HELPERS = ["tests/helpers.py"]
SETUP = ("setUp", "setUpClass", "tearDown", "tearDownClass", "setUpModule")


def refs(node):
    """Every name and attribute name that the code below node mentions."""
    s = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            s.add(n.id)
        elif isinstance(n, ast.Attribute):
            s.add(n.attr)
    return s


def target_names(stmt):
    names = set()
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
    for t in targets:
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                names.add(n.id)
    return names


def main():
    root, tests_json, out = sys.argv[1:4]
    tests = json.load(open(tests_json))["tests"]
    by_file = {}
    for t in tests:
        by_file.setdefault(t["file"], []).append(t)
    res = {}
    for hp in HELPERS:
        tree = ast.parse(open(os.path.join(root, hp)).read())
        stmts, defined, helper_funcs, deps = [], set(), {}, []
        for s in tree.body:
            if isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                names = target_names(s)
                if names:
                    literal = isinstance(s.value, (ast.Dict, ast.List, ast.Set, ast.Tuple, ast.Constant))
                    stmts.append([s.lineno, s.end_lineno, sorted(names), literal])
                    defined |= names
                    deps.append((names, refs(s.value) if s.value is not None else set()))
            elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                helper_funcs[s.name] = refs(s)
        # helper functions that reach a defined name, also through other helper functions
        reach = {f: r & defined for f, r in helper_funcs.items()}
        changed = True
        while changed:
            changed = False
            for f, r in helper_funcs.items():
                add = set()
                for g in r & set(helper_funcs):
                    add |= reach[g]
                if not add <= reach[f]:
                    reach[f] |= add
                    changed = True
        readers = {n: set() for n in defined}
        direct = {n: set() for n in defined}
        for tf, ts in by_file.items():
            try:
                ft = ast.parse(open(os.path.join(root, tf)).read())
            except (OSError, SyntaxError):
                continue
            file_names, classes = {}, {}
            for s in ft.body:
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    file_names[s.name] = refs(s)
                elif isinstance(s, ast.ClassDef):
                    classes[s.name] = s
                elif isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and s.value is not None:
                    # a module-level name of the test file computed from a helpers name
                    for n in target_names(s):
                        file_names.setdefault(n, set()).update(refs(s.value))

            def closure(start, local):
                seen, stack, found = set(), list(start), set()
                while stack:
                    x = stack.pop()
                    if x in seen:
                        continue
                    seen.add(x)
                    if x in defined:
                        found.add(x)
                    if x in reach:
                        found |= reach[x]
                    if x in local:
                        stack.extend(local[x])
                return found

            for t in ts:
                parts = t["id"].split("::")
                if len(parts) < 3:
                    continue
                cls = classes.get(parts[1])
                if cls is None:
                    continue
                meth_name = parts[2].split("[")[0]
                local = dict(file_names)
                chain, i = [cls], 0  # the class and its bases in the same file
                while i < len(chain):
                    for b in chain[i].bases:
                        if isinstance(b, ast.Name) and b.id in classes and classes[b.id] not in chain:
                            chain.append(classes[b.id])
                    i += 1
                setup_refs = set()
                for c in chain:
                    for d in c.decorator_list:
                        setup_refs |= refs(d)
                    for s in c.body:
                        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            local.setdefault(s.name, refs(s))
                            if s.name == meth_name:
                                for d in s.decorator_list:
                                    setup_refs |= refs(d)
                            if s.name in SETUP:
                                setup_refs |= refs(s)
                        else:
                            setup_refs |= refs(s)
                meth = local.get(meth_name, set())
                for n in closure(meth | setup_refs, local):
                    readers[n].add(t["number"])
                for n in closure(meth, local):
                    direct[n].add(t["number"])
        # a name computed from another one: the readers of the result read the source too
        for _ in range(3):
            for names, src in deps:
                for m in src & defined:
                    for n in names:
                        readers[m] |= readers[n]
                        direct[m] |= direct[n]
        res[hp] = {
            "stmts": stmts,
            "readers": {n: sorted(v) for n, v in readers.items()},
            "direct": {n: sorted(v) for n, v in direct.items()},
        }
    with open(out, "w") as f:
        json.dump(res, f)
    print({hp: {n: (len(v), r["direct"][n]) for n, v in r["readers"].items()} for hp, r in res.items()})


if __name__ == "__main__":
    main()
