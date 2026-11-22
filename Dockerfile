# Blast Radius entry image. Base: the pinned image the organizers labeled in.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    SKIP_INTEGRATION=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    BR_ROOT=/repo \
    BR_DATA=/data \
    BR_MAPS=/app/maps

# 0. a faster build of the same interpreter (CPython 3.13.18 built with clang, profile-guided and
#    link-time optimization; one file from a GitHub release at a fixed tag, sha256 checked). Test
#    verdicts must be those of the pinned image, so its sqlite3 module is made to use the image's own
#    SQLite (build/sitecustomize.py) and build/check_python.py compares the two interpreters. If the
#    download or the check fails, "python" stays the pinned image's interpreter: same answers, more CPU.
COPY build/fetch_python.py build/sitecustomize.py build/check_python.py /tmp/build/
RUN mkdir -p /opt/br/bin \
    && ( PYTHONHASHSEED=0 sh -c '/usr/local/bin/python3.13 /tmp/build/fetch_python.py /opt \
           && cp /tmp/build/sitecustomize.py /opt/python/lib/python3.13/site-packages/sitecustomize.py \
           && /opt/python/bin/python3.13 /tmp/build/check_python.py /usr/local/bin/python3.13' \
         && ln -s /opt/python/bin/python3.13 /opt/br/bin/python \
       || ( echo "keeping the image interpreter" >&2; rm -rf /opt/python; ln -s /usr/local/bin/python3.13 /opt/br/bin/python ) ) \
    && rm -rf /tmp/build
ENV PATH=/opt/br/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin

# 1. third-party packages (PyPI only, exact versions, no dependency resolution)
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --no-deps --no-compile --disable-pip-version-check --root-user-action=ignore \
        -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# 2. sqlglot at the pinned commit (GitHub archive of sqlglot's own repo; no git, no apt)
COPY build/fetch_sqlglot.py /tmp/fetch_sqlglot.py
RUN /usr/local/bin/python3.13 /tmp/fetch_sqlglot.py /repo b5e23dd1faff7c2a14ef374d02be8bab60c9343d6d3140690cbad12dbcc7764e \
    && rm /tmp/fetch_sqlglot.py

# 3. bytecode for everything that can be imported (stdlib, site-packages, sqlglot, tests), as
#    unchecked-hash pycs: the run disk is read-only, so nothing may be compiled or stat-checked there.
#    Neither interpreter ships usable stdlib pycs; pip's own pycs are timestamp-based (-f redoes them).
#    Left out, to keep the build short: folders that no run imports (the test suites of the
#    standard library, numpy and pandas, pip, and the GUI / demo packages).
RUN STDLIB=$(python -c "import sysconfig; print(sysconfig.get_path('stdlib'))") \
    && python -m compileall -f -q -j 4 --invalidation-mode unchecked-hash \
         -x '/lib/python3[.]13/(test|idlelib|tkinter|turtledemo|lib2to3|ensurepip)/|/site-packages/(pandas|numpy)/(.*/)?tests/|/site-packages/pip/' \
         "$STDLIB" /repo \
    && find /repo /opt "$STDLIB" \( -type d ! -perm -005 -o -type f ! -perm -004 \) -exec chmod a+rX {} +

# CPU settings for the run (see the selector notes): a fixed hash seed (the saved base state of
# the import fingerprint depends on it) and mimalloc with transparent huge pages (cheaper fork,
# exit and page faults; no effect where the host has THP off).
ENV PYTHONHASHSEED=0 \
    PYTHONMALLOC=mimalloc \
    MIMALLOC_LARGE_OS_PAGES=1 \
    MIMALLOC_RESERVE_OS_MEMORY=256MiB \
    MIMALLOC_ARENA_EAGER_COMMIT=1 \
    MIMALLOC_PURGE_DELAY=-1

# 4. our program and maps (last, so that a code change rebuilds only this). The TestExecutor row
#    cache (preload.py) is built from the pinned csv files and checked against the real
#    setUpClass; if the check fails, the cache is removed and the selector runs setUpClass as before.
#    DuckDB's answers to the SQL texts of the pinned tests are recorded too (build_duck_replay.py,
#    duckreplay.py): a test that sends a recorded text gets the recorded answer, any other text
#    runs in DuckDB. The recording is committed as a plain file (app/prebuilt/duck_replay.bin, made
#    by the same script on the pinned tree; byte-identical across builds), so the build only
#    rehearses it; if that fails, it is recorded here, and if that fails too, every query runs.
#    Two more facts of the pinned tree are prepared here: whether any test file reads docstrings
#    (build_docs_read.py; the selector otherwise reads all test files to find out), and the map
#    files are stored without zlib compression (rawmaps.py; rawz.py reads both forms, and every
#    converted file is read back and compared). If a step fails, the selector works as before.
COPY app /app
# The step memo of the stage check of test_execute_tpcds (build_memo.py, memo.py): the outputs of
# the executor steps of every TPC-DS query on the pinned commit. Which outputs are stored and the step
# costs come from a fixed file (app/maps/memo_fixed.bin), not from this machine's timing. If it fails,
# the check runs as before.
RUN echo "step build_memo start $(date +%s)" && (cd /repo && SKIP_INTEGRATION=0 timeout 240 python /app/build_memo.py /repo /app/maps/stages.bin /app/maps 4) \
    || (echo "no step memo" >&2; rm -f /app/maps/memo.bin /app/maps/memo.dat); echo "step build_memo end $(date +%s)"
RUN echo "step exec_cache start $(date +%s)" && (timeout 120 python /app/build_exec_cache.py /repo /app/cache --check \
     || (echo "exec cache check failed; running without it" >&2; rm -rf /app/cache)) \
    && mkdir -p /app/cache && echo "step duck_replay start $(date +%s)" \
    && (timeout 120 python /app/build_duck_replay.py /repo /app/cache/duck_replay.bin --from /app/prebuilt/duck_replay.bin --check \
     || (echo "the committed DuckDB recording failed its check; recording it here" >&2; rm -f /app/cache/duck_replay.bin; \
         timeout 240 python /app/build_duck_replay.py /repo /app/cache/duck_replay.bin --check) \
     || (echo "no DuckDB answer recording; every query runs" >&2; rm -f /app/cache/duck_replay.bin)) \
    && echo "step base_state start $(date +%s)" && (python /app/build_base_state.py /repo /app/cache/base_state.bin \
     || (echo "no saved base state; the selector computes it" >&2; rm -f /app/cache/base_state.bin)) \
    && echo "step docs_read start $(date +%s)" && (python /app/build_docs_read.py /repo /app/maps /app/cache/docs_read.json \
     || (echo "no saved docs answer; the selector reads the test files" >&2; rm -f /app/cache/docs_read.json)) \
    && echo "step rawmaps start $(date +%s)" && (python /app/rawmaps.py /app/maps || echo "maps stay compressed" >&2) \
    && echo "step compile start $(date +%s)" \
    && python -m compileall -f -q --invalidation-mode unchecked-hash /app \
    && chmod -R a+rX /app && echo "step all end $(date +%s)"

# 5. the checking selector (/ours): it analyses every commit, answers and verifies the risky
#    ones itself (state that can leak, copies, import-time edits, special files, annotation
#    edits) and hands the others to the engine above (/app/main.py) in one subprocess. Its
#    pinned-commit state fingerprint is built here, with this interpreter. Its overlays hide
#    the unchecked-hash bytecode of the modules a patch changes.
COPY ours /ours
RUN cd /repo && python -c "import sys; sys.path.insert(0, '/ours'); from selector import keyrefine; keyrefine.build_pinned('/repo', '/ours/maps/pinned_state.json.gz')" \
    && python -m compileall -f -q --invalidation-mode unchecked-hash /ours && chmod -R a+rX /ours
ENV BR_HYBRID_MAIN=/app/main.py

WORKDIR /repo
CMD ["python", "/ours/selector/main.py"]
