"""Stage checks: compare what the optimizer hands to the rest of a TPC-DS case.

test_execute_tpcds runs, per case: parse the query, `execute()` it with the
Python executor (which calls `optimize()`, then plans and executes the optimized
tree), run the same query in duckdb and compare. Executing is what makes the
test slow; optimizing is cheap.

When every reason the test is a candidate only reaches code that runs *inside*
`optimize()` (each case's coverage outside `optimize()` calls was traced at
build time as an "outer" pseudo-test, tools/trace_stage.py), everything outside
runs exactly as at the pinned commit as long as `optimize()` returns the same
tree. So the verifier runs each affected case only up to the end of
`optimize()`, fingerprints its input and output, and compares them with the
fingerprints recorded at the pinned commit; it then stops the case. Equal for
every case: the test cannot fail. Any difference, a missing or extra call, or a
case that was not deterministic when recorded: the test is included.

The fingerprint walks the objects directly (slots and plain attributes, no
sqlglot code), so a patch cannot change how it is computed.
"""

from __future__ import annotations

import enum
import hashlib

NODE_ATTRS = ("comments", "_meta")  # besides args, _type, and the parent links


def _expression_like(x) -> bool:
    try:
        return hasattr(x, "args") and isinstance(getattr(x, "args"), dict) and hasattr(x, "parent")
    except Exception:  # noqa: BLE001
        return False


def fingerprint(root) -> str:
    """A digest of an expression tree: node classes, args in order, types,
    comments, meta, parent/arg_key/index links and shared-node aliasing."""
    seen: dict[int, int] = {}
    out: list = []

    def value(x, depth=0):
        if depth > 10000:
            raise RecursionError("fingerprint depth")
        if x is None or isinstance(x, (bool, int, float, str, bytes)):
            out.append((type(x).__name__, repr(x)))
            return
        if isinstance(x, enum.Enum):
            out.append(("enum", type(x).__module__, type(x).__qualname__, x.name, repr(x.value)))
            return
        if isinstance(x, (list, tuple)):
            out.append((type(x).__name__, len(x)))
            for item in x:
                value(item, depth + 1)
            return
        if isinstance(x, dict):
            out.append(("dict", len(x)))
            for k, v in x.items():
                value(k, depth + 1)
                value(v, depth + 1)
            return
        if _expression_like(x):
            key = id(x)
            if key in seen:
                out.append(("ref", seen[key]))
                return
            seen[key] = len(seen)
            cls = type(x)
            parent = getattr(x, "parent", None)
            out.append(("node", cls.__module__, cls.__qualname__,
                        seen.get(id(parent), "ext" if parent is not None else None),
                        getattr(x, "arg_key", None), getattr(x, "index", None)))
            for name in NODE_ATTRS:
                out.append(name)
                value(getattr(x, name, None), depth + 1)
            out.append("_type")
            value(getattr(x, "_type", None), depth + 1)
            out.append("args")
            value(x.args, depth + 1)
            return
        out.append(("obj", type(x).__module__, type(x).__qualname__, repr(x)))

    value(root)
    return hashlib.sha256(repr(out).encode("utf-8", "backslashreplace")).hexdigest()[:32]


class StageDone(Exception):
    """Raised right after the checked call returns, to skip the rest of a case."""


class StageAbort(Exception):
    """Raised when enough cases differ that the test will be included anyway:
    stops the whole test run (the process-pool stand-in lets it through)."""

    _br_abort = True


# Early abort (set by the verification child): expected (after, after_input) per
# input fingerprint, the case ordinal of each input, per-case cost of a full
# re-run, the warm-up and the cap above which the test is included instead.
ABORT = {"by_input": {}, "ordinal": {}, "cost": {}, "warm": 0.0, "cap": -1.0}


# Set in the verification child: list of expected (input, output) fingerprints in
# call order, and what was observed.
EXPECTED: list = []
OBSERVED: list = []


def differing_cases(per_case, observed):
    """Ordinals of the cases whose recorded calls were not observed unchanged.
    Observed calls are matched to cases by the fingerprint of the call's input
    (the parsed query), so a case whose optimize() raised (no record) or whose
    input changed shows up as differing; cases sharing an input fingerprint are
    judged together."""
    by_input: dict = {}
    for k, calls in per_case:
        for call in calls:
            by_input.setdefault(call[0], []).append((k, tuple(call)))
    seen: dict = {}
    for call in observed:
        seen.setdefault(call[0], []).append(tuple(call))
    differing = set()
    for before, items in by_input.items():
        if seen.get(before, []) != [c for _, c in items]:
            differing.update(k for k, _ in items)
    expected_inputs = set(by_input)
    if any(call[0] not in expected_inputs for call in observed):
        differing.update(k for k, _ in per_case)  # a call we cannot attribute
    return sorted(differing)


def install(module, attr="optimize"):
    """Wrap module.<attr> so each call records fingerprints of its input and
    output, compares them with EXPECTED in order, and then stops the case."""
    original = getattr(module, attr)
    if getattr(original, "_br_stage", False):
        return

    def checked(*args, **kwargs):
        before = fingerprint(args[0]) if args else fingerprint(kwargs.get("expression"))
        result = original(*args, **kwargs)
        after = fingerprint(result)
        after_input = fingerprint(args[0]) if args else fingerprint(kwargs.get("expression"))
        OBSERVED.append((before, after, after_input))
        if ABORT["cap"] >= 0:
            differing = set()
            for call in OBSERVED:
                expected = ABORT["by_input"].get(call[0])
                if expected is None or (call[1], call[2]) not in expected:
                    differing.add(ABORT["ordinal"].get(call[0], -1))
            cost = ABORT["warm"] + sum(ABORT["cost"].get(k, ABORT["cap"] + 1.0) for k in differing)
            if differing and cost > ABORT["cap"]:
                raise StageAbort()
        raise StageDone()

    checked._br_stage = True
    checked._br_original = original
    setattr(module, attr, checked)


def uninstall(module, attr="optimize"):
    current = getattr(module, attr)
    if getattr(current, "_br_stage", False):
        setattr(module, attr, current._br_original)
