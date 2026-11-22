#!/usr/bin/env python3
"""Aggregate raw/<n>.json.gz + suite_out/suite_order.json.gz into agg.pkl (protocol 4) and suite_order.json.gz.
usage: python3 build_agg.py"""
import json, gzip, pickle, os, glob, shutil
TRACE = '/root/project/work/trace'
tests = json.load(open(TRACE + '/tests.json'))['tests']
call, import_call, lazy_def, import_def_tests, import_def_lines = {}, {}, {}, {}, {}
opened, opened_import, listed, classes, outcome, funcs = {}, {}, {}, {}, {}, {}
iso_call = {}
def add(d, f, l, n):
    d.setdefault(f, {}).setdefault(l, set()).add(n)
missing = []
for t in tests:
    n = t['number']
    p = f'{TRACE}/raw/{n}.json.gz'
    if not os.path.exists(p):
        missing.append(n); continue
    d = json.load(gzip.open(p))
    outcome[n] = d['outcome']
    for f, ls in d['test']['call'].items():
        for l in ls:
            add(call, f, l, n); add(iso_call, f, l, n)
    for f, ls in d['import']['call'].items():
        import_call.setdefault(f, set()).update(ls)
    for f, ls in d['import']['def'].items():
        import_def_tests.setdefault(f, set()).add(n)
        import_def_lines.setdefault(f, set()).update(ls)
    for f, ls in d['test']['def'].items():
        for l in ls: add(lazy_def, f, l, n)
    for x in d['test']['opened']: opened.setdefault(x, set()).add(n)
    for x in d['import']['opened']: opened_import.setdefault(x, set()).add(n)
    for x in d['test']['listed'] + d['import']['listed']: listed.setdefault(x, set()).add(n)
    for c in d['classes'] or []: classes.setdefault(c, set()).add(n)
    for f, fs in d['test']['funcs'].items():
        for a, b, q in fs: funcs.setdefault(f, {}).setdefault((a, b, q), set()).add(n)
suite_only = 0
so = json.load(gzip.open(TRACE + '/suite_out/suite_order.json.gz'))
for ns, fl in so.items():
    n = int(ns)
    for f, ls in fl.items():
        for l in ls:
            s = call.setdefault(f, {}).setdefault(l, set())
            if n not in s:
                suite_only += 1; s.add(n)
shutil.copy(TRACE + '/suite_out/suite_order.json.gz', TRACE + '/suite_order.json.gz')
S = lambda d: {k: sorted(v) for k, v in d.items()}
SS = lambda d: {f: {l: sorted(v) for l, v in m.items()} for f, m in d.items()}
agg = {
    'call': SS(call), 'import_call': S(import_call), 'lazy_def': SS(lazy_def),
    'import_def_tests': S(import_def_tests), 'opened': S(opened), 'opened_import': S(opened_import),
    'classes': S(classes), 'outcome': outcome,
    # extras
    'import_def_lines': S(import_def_lines),
    'listed': S(listed),
    'funcs': {f: {k: sorted(v) for k, v in m.items()} for f, m in funcs.items()},
    'call_isolated_only': SS(iso_call),
    'meta': {'missing_raw': missing, 'suite_only_pairs': suite_only,
             'doc': 'call = isolated test phase (incl. traced child processes) U suite_order; funcs = {file: {(first_line, last_line, qualname): tests}} for call-time code objects hit in the isolated test phase; listed = os.listdir/os.scandir/glob targets (both phases)'},
}
pickle.dump(agg, open(TRACE + '/agg.pkl', 'wb'), protocol=4)
print('tests', len(outcome), 'missing', len(missing), 'files in call', len(call), 'suite-only (file,line,test) pairs', suite_only,
      'classes', len(classes), 'opened', len(opened), 'size', os.path.getsize(TRACE + '/agg.pkl'))
