"""ast.walk without the generators: the same nodes in the same (breadth-first) order."""
import ast
import os

_AST = ast.AST


def walk(node):
    """The list of ast.walk(node): node, then the children of each listed node in the order of
    ast.iter_child_nodes (fields in _fields order, AST values and the AST items of lists)."""
    out = [node]
    append = out.append
    i = 0
    while i < len(out):
        n = out[i]
        i += 1
        for name in n._fields:
            try:
                v = getattr(n, name)
            except AttributeError:
                continue
            if isinstance(v, _AST):
                append(v)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, _AST):
                        append(x)
    return out


if os.environ.get("BR_NO_ASTWALK") == "1":  # validation: the standard library's walk

    def walk(node):  # noqa: F811
        return list(ast.walk(node))
