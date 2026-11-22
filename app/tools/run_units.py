#!/usr/bin/env python3
"""Host driver for trace_units.py: each worker takes shards from a shared queue and runs them pinned to one core.
usage: run_units.py OUT_DIR NSHARDS CORE [CORE...]"""
import os, subprocess, sys, threading
out, nshards, cores = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
os.makedirs(out + '/locks', exist_ok=True)
os.makedirs(out + '/logs', exist_ok=True)
def worker(core):
    for shard in range(nshards):
        try:
            os.mkdir(f'{out}/locks/{shard}')
        except FileExistsError:
            continue
        with open(f'{out}/logs/{shard}.log', 'w') as log:
            subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--cpuset-cpus', core, '--cpus', '1', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=1',
                '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'BR_CLASSES_FULL=' + os.environ.get('BR_CLASSES_FULL', ''), '-e', 'BR_FULL_ONLY=' + os.environ.get('BR_FULL_ONLY', ''), '-e', 'BR_CLASSES=' + ('/units2' if os.environ.get('BR_CLASSES') else ''),
                '-v', '/root/project/work/units2/raw:/units2:ro',
                '-v', '/root/project/work/runs/base/repo:/repo:ro', '-v', '/root/project/work/defgap/rt_trace:/rt:ro',
                '-v', '/root/project/nap-time/sample:/data:ro', '-v', f'{out}/raw:/out', '-w', '/repo',
                'br-label', 'python', '/rt/trace_units.py', '/repo', '/data/tests.json', '/out', str(shard), str(nshards)],
                stdout=log, stderr=subprocess.STDOUT)
        open(f'{out}/locks/{shard}/done', 'w').close()
ths = [threading.Thread(target=worker, args=(c,)) for c in cores]
for t in ths: t.start()
for t in ths: t.join()
