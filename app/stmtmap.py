"""Statement units of a source file: the pieces that run when the module is imported."""
from __future__ import annotations

import ast


def build_units(src: str):
    """Returns (units, future_flags, import_names).

    import_names: the names that import statements bind at module level. The compiler
    treats attribute calls on such names differently, so a unit that is compiled alone
    must declare them to get the same bytecode as in the full module.


    A unit is [start, end, class_path, kind]. kind 0 = statement at module level or class level,
    kind 1 = class header (decorators, name, bases)."""
    tree = ast.parse(src)
    units = []
    flags = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(a.name == "annotations" for a in node.names):
                flags = 0x1000000

    def first_line(node):
        decos = getattr(node, "decorator_list", None)
        if decos:
            return min(node.lineno, min(d.lineno for d in decos))
        return node.lineno

    def walk(body, path):
        for node in body:
            start = first_line(node)
            if isinstance(node, ast.ClassDef):
                body_start = first_line(node.body[0])
                if body_start == node.lineno:  # one-line class
                    units.append([start, node.end_lineno, path, 0])
                    continue
                units.append([start, body_start - 1, path, 1])
                walk(node.body, path + (node.name,))
            else:
                units.append([start, node.end_lineno, path, 0])

    walk(tree.body, ())

    names = set()

    def scan(nodes):
        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    if a.name != "*":
                        names.add(a.asname or a.name.partition(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            else:
                for field in ("body", "orelse", "finalbody", "handlers", "cases"):
                    sub = getattr(node, field, None)
                    if isinstance(sub, list):
                        scan(sub)

    scan(tree.body)
    return units, flags, sorted(names)
