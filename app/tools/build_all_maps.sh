#!/bin/sh
# Rebuilds /root/project/work/maps from the traces (coverage maps, then unit maps).
set -e
CORES="${CORES:-0,1}"
docker run --rm --cpuset-cpus "$CORES" --cpus 2 -e PYTHONDONTWRITEBYTECODE=1 \
  -v /root/project/work/sqlglot:/repo:ro -v /root/project/work/rt:/rt:ro -v /root/project/work/trace:/trace:ro \
  -v /root/project/nap-time/sample:/data:ro -v /root/project/work/maps:/out -v /root/project/work/deftime:/deftime:ro -v /root/project/work/units2/raw:/units:ro -v /root/project/work/fullcov2/raw:/fullcov:ro -w /repo br-label \
  python /rt/build_maps.py /repo /trace/agg.pkl /data/tests.json /out /deftime/defmap.pkl /units /fullcov 2>&1 | grep -v "nanobind\|leaked\|skipped" | tail -3
docker run --rm --cpuset-cpus "$CORES" --cpus 2 -e PYTHONDONTWRITEBYTECODE=1 \
  -v /root/project/work/sqlglot:/repo:ro -v /root/project/work/rt:/rt:ro -v /root/project/work/units2/raw:/units:ro \
  -v /root/project/work/maps:/out -v /root/project/work/classes/raw:/classes:ro -w /repo br-label \
  python /rt/build_unitmap.py /repo /units /out /classes 2>&1 | grep -v "nanobind\|leaked\|skipped" | tail -2
docker run --rm -v /root/project/work/rt:/rt:ro -v /root/project/work/timing:/timing:ro -v /root/project/nap-time/sample:/data:ro -v /root/project/work/maps:/out br-label python /rt/build_costs.py /timing/full.json /data/tests.json /out
du -sh /root/project/work/maps
