# Loaded at interpreter start (PYTHONPATH contains this directory) in every process of the
# map-building run: the pytest process, its spawn pool workers and subprocesses. It installs
# the comparison probe that the instrumented sqlglot code calls (see _brprobe.py).
import _brprobe  # noqa: F401
