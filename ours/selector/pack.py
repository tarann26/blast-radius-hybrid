"""Split the chosen tests over workers to minimise busiest-worker time + 2 s/worker."""

from __future__ import annotations

WORKER_MS = 2000
MAX_WORKERS = 8


def _lpt(items: list[tuple[int, int]], k: int):
    loads = [0] * k
    bins: list[list[int]] = [[] for _ in range(k)]
    for ms, number in items:
        i = loads.index(min(loads))
        loads[i] += ms
        bins[i].append(number)
    return loads, bins


def _improve(items_ms: dict[int, int], loads: list[int], bins: list[list[int]]):
    """Move or swap single tests off the busiest worker while that helps."""
    improved = True
    while improved:
        improved = False
        top = max(range(len(loads)), key=loads.__getitem__)
        for other in range(len(loads)):
            if other == top:
                continue
            for a in list(bins[top]):
                da = items_ms[a]
                # move a
                if loads[other] + da < loads[top]:
                    new_top = loads[top] - da
                    new_other = loads[other] + da
                    if max(new_top, new_other) < loads[top]:
                        bins[top].remove(a)
                        bins[other].append(a)
                        loads[top], loads[other] = new_top, new_other
                        improved = True
                        break
                # swap a with b
                for b in list(bins[other]):
                    db = items_ms[b]
                    if db >= da:
                        continue
                    new_top = loads[top] - da + db
                    new_other = loads[other] - db + da
                    if max(new_top, new_other) < loads[top]:
                        bins[top].remove(a)
                        bins[top].append(b)
                        bins[other].remove(b)
                        bins[other].append(a)
                        loads[top], loads[other] = new_top, new_other
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return loads, bins


def pack(numbers: list[int], ms_of: dict[int, int], fast: bool = False) -> tuple[list[list[int]], int]:
    """Return (workers, cost_ms). An empty selection costs nothing. fast: skip the
    move/swap pass for large selections (planner estimates; LPT is within one test
    of the improved result there, and the pass is quadratic)."""
    numbers = sorted(set(numbers))
    if not numbers:
        return [], 0
    items = sorted(((ms_of[n], n) for n in numbers), reverse=True)
    total = sum(ms for ms, _ in items)
    biggest = items[0][0]
    best = None
    for k in range(1, min(MAX_WORKERS, len(items)) + 1):
        lower = max(biggest, -(-total // k)) + WORKER_MS * k
        if best is not None and lower >= best[0]:
            if biggest >= total / k:
                break
            continue
        loads, bins = _lpt(items, k)
        if not (fast and len(items) > 64):
            loads, bins = _improve(ms_of, loads, bins)
        cost = max(loads) + WORKER_MS * k
        if best is None or cost < best[0]:
            best = (cost, [sorted(b) for b in bins if b])
    return best[1], best[0]
