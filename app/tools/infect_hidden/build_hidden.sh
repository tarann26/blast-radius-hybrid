#!/bin/sh
# Builds MAPS_DIR/infect_hidden.bin from the pinned checkout (see build_hidden.py).
# usage: build_hidden.sh WORK_DIR MAPS_DIR "C0 C1 C2 C3 C4 C5"   (six cores; about 30 minutes)
# Needs MAPS_DIR/infect.bin, infect2.bin (the mapped sites) and the line coverage map (f/, sets.*, meta.bin).
set -e
WORK="$1"; MAPS="$2"; set -- $3
BUILD="$(cd "$(dirname "$0")" && pwd)"
SRC=/root/project/work/sqlglot
TESTS=/root/project/nap-time/sample/tests.json
mkdir -p "$WORK"
docker run --rm --network none --cpuset-cpus $1 -v $SRC:/src:ro -v "$BUILD":/build:ro -v "$MAPS":/maps:ro -v "$WORK":/work br-label \
  python /build/build_hidden.py prepare --src /src --repo /work/tree --maps /maps
rm -rf "$WORK"/run0 "$WORK"/run1 "$WORK"/run2 "$WORK"/cold "$WORK"/coldrepo* "$WORK"/cold_parts.log
(
  for s in 0 1 2; do
    core=$(eval echo \$$((s + 1)))
    mkdir -p "$WORK/run$s"; cp -a "$WORK/tree" "$WORK/run$s/repo"
    docker run --rm --network none --cpuset-cpus $core --cpus 1 --name i3u_warm$s -v "$WORK":/work -v "$BUILD":/build:ro -w /work br-label \
      sh -c "python /build/build_hidden.py run --repo /work/run$s/repo --probe /work/run$s/probe --seed $s > /work/run$s/log.txt 2>&1; echo \$? > /work/run$s/rc" &
  done
  wait
  "$BUILD"/cold_all.sh "$WORK" "$1 $2 $3" 18 "9 10 11 12 13 14 15 16 17"
) &
"$BUILD"/cold_all.sh "$WORK" "$4 $5 $6" 18 "0 1 2 3 4 5 6 7 8"
wait
docker run --rm --network none --cpuset-cpus $1 -v "$BUILD":/build:ro -v "$MAPS":/maps -v "$WORK":/work -v $TESTS:/tests.json:ro br-label \
  python /build/build_hidden.py merge --repo /work/tree --maps /maps --tests /tests.json --out /maps/infect_hidden.bin \
  --cold /work/cold --report /work/hidden_report.json --detail /work/hidden_detail.json /work/run0/probe /work/run1/probe /work/run2/probe
echo "build done $(date -u +%H:%M:%S)" >> "$WORK/cold_parts.log"
