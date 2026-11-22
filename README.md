# Blast Radius selector (team hybrid)

Test selection for sqlglot at eb11aec4be8cb2122afe713a4816662c3f0fb2fa: for each patch in
`/data/commits`, one JSON line with the tests to run, packed onto workers.

Two engines from the team run in one container:

- `ours/` (the checking selector) parses and analyses every commit first (its state comparison,
  which only narrows its own plans, runs only for the commits it keeps). It answers and
  verifies the risky ones itself: edits that can leave state behind for later tests, copy-only
  edits, import-time and module-level edits, new, deleted or special files, unknown changes, and a
  class-level name that gains or loses its annotation. It also answers any commit where its own
  plan is expected to cost well under the other engine's answer (below 60 % of it, by 0.5 s), and
  keeps the cheaper of the two answers there; a plan of its own that runs nothing (its cost is
  exact) replaces a dearer engine answer outright.
- `app/` (the run-the-affected-tests engine) answers every other commit. `ours` runs it once, as a
  subprocess, on a data folder that holds only those patches. If it fails, times out or gives no
  valid line for a commit, `ours` verifies and answers that commit too.

Build-time steps (see the `Dockerfile`): the engine's interpreter, caches and maps, then the
pinned-commit state fingerprint of `ours`. The engine's image compiles sqlglot to unchecked-hash
bytecode, so the overlays of `ours` never show the compiled files of modules a patch changes.

Local results (simulated seconds, misses):

| set | hybrid | `ours` alone | engine alone |
| --- | ---: | ---: | ---: |
| kit sample (30) | 94.351, 0 | 94.4, 0 | 94.351, 0 |
| 29 labelled sets (1,043 commits) | 6,122, 0 | 7,092, 0 | 7,765, 6 |
| val9 in the container (60) | 191.551, 0 | 210.5, 0 | 233.8, 0 |
| val5 in the container (64) | 175.230, 0 | 176.4, 0 | 236.8, 0 |

CPU over the 29 sets: hybrid 545 s, `ours` alone 593 s, engine alone about 497 s.
