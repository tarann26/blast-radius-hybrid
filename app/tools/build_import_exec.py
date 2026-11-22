"""Builds maps/import_exec.json on the pinned tree: for every line of a sqlglot function body
that runs while the tree is imported, the modules whose import was running (the innermost module
body on the stack). A function whose lines run only during the import of modules M1..Mk takes no
part in the import of any other module, so a change of its body can change the import state only
through M1..Mk and their importers (selector.cone_plan, swap route).

Recorded: every 'line' event and the first line of the code object at every 'call' event, in
frames of function code (CO_OPTIMIZED: functions, lambdas, comprehensions, generators); module and
class bodies are not recorded (they run at their own import, which re-imports them anyway).

usage: python build_import_exec.py OUT.json   (cwd and sys.path as for the selector)"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.environ.get("BR_ROOT", "/repo")
CO_OPTIMIZED = 0x1


def main():
    out_path = sys.argv[1]
    import preload

    preload.install_environment(ROOT)
    from importlib.machinery import SourceFileLoader

    stack = []
    orig = SourceFileLoader.exec_module

    def exec_module(self, module):
        stack.append(module.__name__)
        try:
            return orig(self, module)
        finally:
            stack.pop()

    SourceFileLoader.exec_module = exec_module
    rec = {}
    prefix = os.path.join(ROOT, "sqlglot", "")
    outside = [0]

    def tracer(frame, event, arg):
        code = frame.f_code
        fn = code.co_filename
        if not fn.startswith(prefix) or not code.co_flags & CO_OPTIMIZED:
            return None
        d = rec.get(fn)
        if d is None:
            d = rec[fn] = {}
        if not stack:
            outside[0] += 1
            return None
        d.setdefault(code.co_firstlineno, set()).add(stack[-1])

        def local(frame, event, arg):
            if event == "line":
                if stack:
                    d.setdefault(frame.f_lineno, set()).add(stack[-1])
                else:
                    outside[0] += 1
            return local

        return local

    sys.settrace(tracer)
    try:
        preload.import_sqlglot()
    finally:
        sys.settrace(None)
    SourceFileLoader.exec_module = orig
    modules = sorted({m for d in rec.values() for s in d.values() for m in s})
    idx = {m: i for i, m in enumerate(modules)}
    files = {}
    for fn, d in sorted(rec.items()):
        files[os.path.relpath(fn, ROOT)] = [[ln, sorted(idx[m] for m in s)] for ln, s in sorted(d.items())]
    with open(out_path, "w") as f:
        json.dump({"version": 1, "modules": modules, "files": files, "outside": outside[0]}, f, separators=(",", ":"))
    print("import_exec:", len(files), "files", sum(len(v) for v in files.values()), "lines", len(modules), "modules; events outside an import:", outside[0], file=sys.stderr)


if __name__ == "__main__":
    main()
