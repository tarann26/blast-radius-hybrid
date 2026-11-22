"""Validation only (BR_STAGE_AUDIT=1; not needed in the image): the stage check of stages.py
with every claim tested. Each task is judged as the check judges it (same input / other input
/ later work touched / another SQL text), but every task then runs to its end, the test's
real comparison with DuckDB's answer runs for every task, and its result is recorded instead
of ending the test. Wrong claims would be: a task that the check stops as "same" and whose
comparison fails; a task whose rows equal the pinned rows (with the pinned SQL text) and whose
comparison fails."""
from __future__ import annotations

import contextlib
import time

import stages


class AuditCheck(stages.StageCheck):
    def __init__(self, record, later, budget, active=None):
        super().__init__(record, later, float("inf"), active)
        self.order = []
        self.kind = {}  # task -> "same" | what differs | "later" | "path"
        self.sql = {}  # task -> the patched code makes another DuckDB text
        self.rows = {}  # task -> its rows equal the pinned rows
        self.outcome = {}  # task -> "pass" | "fail" | "error" | "none"
        self.used = {}  # task -> [CPU s up to the point, CPU s after it]
        self.t = None

    def install(self, test_module, cls):
        if not super().install(test_module, cls):
            return False
        real = self.helper
        audit = self

        def helper(case, i, table, tpch=True):
            want = audit.tasks.get(i)
            audit.rows[i] = want is not None and stages.rows_fingerprint(table) == want[3]
            case.subTest = lambda *a, **k: contextlib.nullcontext()
            try:
                real(case, i, table, tpch=tpch)
                audit.outcome[i] = "pass"
            except Exception:  # noqa: BLE001
                audit.outcome[i] = "fail"
            finally:
                del case.subTest
                audit.close(i)

        cls.subtestHelper = helper
        return True

    def close(self, index):
        now = time.process_time()
        if index in self.used and self.t is not None:
            self.used[index][1] += now - self.t
        self.t = None

    def at_executor(self, args, kwargs, plan_input):
        plan, executor = self.real
        index = self.task
        now = time.process_time()
        if index is not None:
            self.used[index] = [now - self.t, 0.0]
            self.t = now
        if not isinstance(plan_input, stages._PlanInput):
            if index is not None:
                self.kind[index] = "call"
                self.task = None
            return executor(*args, **kwargs).execute(plan_input)
        self.kind[index] = self.compare(index, args, kwargs, plan_input) or "same"
        self.task = None
        return executor(*args, **kwargs).execute(plan(plan_input.expression))

    def run_task(self, index, fn, args, kwargs):
        if self.tasks.get(index) is None:
            return fn(*args, **kwargs)
        self.order.append(index)
        self.sql[index] = self.sql_text_differs(index)
        self.t = time.process_time()
        try:
            if index in self.later:
                self.kind[index] = "later"
                self.used[index] = [0.0, 0.0]
                value = fn(*args, **kwargs)
            else:
                self.task = index
                try:
                    value = fn(*args, **kwargs)
                finally:
                    if self.task is not None:
                        self.kind[index] = "path"
                        self.used.setdefault(index, [0.0, 0.0])
                    self.task = None
        except Exception:  # noqa: BLE001  (in the real test this error ends the test: it fails)
            self.outcome[index] = "error"
            self.used.setdefault(index, [0.0, 0.0])
            self.close(index)
            return None
        if value is None:
            self.outcome[index] = "none"
            self.close(index)
        return value

    def probe(self, case):
        return True  # (no probe: every task is judged once, in the test)

    def complete(self):
        return False

    def report(self):
        return {
            "order": self.order,
            "kind": {str(i): k for i, k in self.kind.items()},
            "sql": [i for i, d in self.sql.items() if d],
            "rows": [i for i, same in self.rows.items() if not same],
            "out": {str(i): o for i, o in self.outcome.items()},
            "used": {str(i): [round(a, 3), round(b, 3)] for i, (a, b) in self.used.items()},
        }
