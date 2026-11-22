#!/bin/sh
# Runs cold per-test runs of build_hidden.py in short containers (each well below 20 minutes).
# usage: cold_all.sh WORK_DIR "CORE CORE ..." N "PART PART ..."   (WORK_DIR holds tree/ from the prepare step;
# the suite is split into N parts by test number; the listed parts are spread over the cores)
WORK="$1"; CORES="$2"; N="$3"; PARTS="$4"
BUILD="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORK/cold"
NC=$(echo $CORES | wc -w)
k=0
for core in $CORES; do
  [ -d "$WORK/coldrepo$core" ] || cp -a "$WORK/tree" "$WORK/coldrepo$core"
  (
    j=0
    for p in $PARTS; do
      if [ $((j % NC)) -eq $k ]; then
        docker run --rm --network none --cpuset-cpus $core --cpus 1 --name i3u_cold$p -v "$WORK":/work -v "$BUILD":/build:ro \
          -v /root/project/nap-time/sample/tests.json:/tests.json:ro -w /work br-label \
          python /build/build_hidden.py cold --repo /work/coldrepo$core --probe /work/cold --tests /tests.json --part $p/$N
        echo "part $p rc $? $(date -u +%H:%M:%S)" >> "$WORK/cold_parts.log"
      fi
      j=$((j + 1))
    done
  ) &
  k=$((k + 1))
done
wait
