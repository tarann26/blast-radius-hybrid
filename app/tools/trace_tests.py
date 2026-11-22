#!/usr/bin/env python3
"""Trace one sqlglot test in a fresh process (runs inside the br-label container).

usage: python /trace/trace_one.py NUMBER TEST_ID OUT.json.gz [--no-classes] [--no-lines]
  TEST_ID like tests/dialects/test_athena.py::TestAthena::test_athena
Repo must be at /repo (cwd /repo, PYTHONPATH=/repo:/trace/child).
Phases: "import" (import of the test module, monitoring starts before sqlglot is imported)
and "test" (unittest.TestSuite([TestCase(method)]) run with a plain TestResult).
Child python processes (multiprocessing spawn, subprocess) are traced by /trace/child/sitecustomize.py
when BR_CHILD_DIR is set; their lines are merged into the test phase and listed under "child".
"""
import sys, os, time, json, gzip, resource

REPO = '/repo/'
CO_OPTIMIZED = 0x1
mon = sys.monitoring
LT = mon.COVERAGE_ID
CT = mon.PROFILER_ID
DISABLE = mon.DISABLE


_rel = {}
def relpath(fn):
    r = _rel.get(fn)
    if r is None:
        if fn.startswith('/repo/sqlglot/') or fn.startswith('/repo/tests/'):
            r = fn[6:]
        else:
            r = ''
        _rel[fn] = r
    return r

class Phase:
    def __init__(self):
        self.deft = {}
        self.call = {}
        self.codes = set()
        self.opened = set()
        self.listed = set()

P = Phase()

def on_line(code, line):
    r = relpath(code.co_filename)
    if r:
        d = P.call if code.co_flags & CO_OPTIMIZED else P.deft
        s = d.get(r)
        if s is None:
            s = d[r] = set()
        s.add(line)
        P.codes.add(code)
    return DISABLE

WANT = {'open', 'os.listdir', 'os.scandir', 'glob.glob', 'glob.glob/2'}
def _abs(p):
    try:
        p = os.fspath(p)
    except TypeError:
        return None
    if isinstance(p, bytes):
        p = os.fsdecode(p)
    if not isinstance(p, str):
        return None
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    return os.path.normpath(p)

def audit(ev, a):
    if ev not in WANT:
        return
    try:
        if ev == 'open':
            p = a[0]
            if isinstance(p, int):
                return
            p = _abs(p)
            if not p or not p.startswith(REPO):
                return
            # skip files opened by the import system (module sources / pyc): def lines cover those
            f = sys._getframe(1)
            n = 0
            while f is not None and n < 40:
                if f.f_code.co_filename.startswith('<frozen importlib'):
                    return
                f = f.f_back; n += 1
            P.opened.add(p[len(REPO):])
        else:
            p = a[0] if a else None
            if p is None or p == '':
                p = '.'
            if isinstance(p, int):
                return
            p = _abs(p)
            if p and (p + '/').startswith(REPO):
                P.listed.add(p[len(REPO):] or '.')
    except Exception:
        pass

classes = set()
_seen_cls = set()
def on_start(code, offset):
    fn = code.co_filename
    if not fn.startswith('/repo/sqlglot/') or code.co_argcount == 0:
        return DISABLE
    try:
        obj = sys._getframe(1).f_locals.get(code.co_varnames[0])
    except Exception:
        return
    t = obj if isinstance(obj, type) else type(obj)
    if t in _seen_cls:
        return
    _seen_cls.add(t)
    m = getattr(t, '__module__', '') or ''
    if m.startswith('sqlglot') or m.startswith('tests'):
        classes.add(m + '.' + t.__qualname__)

def dump_phase(p):
    funcs = {}
    for c in p.codes:
        if not (c.co_flags & CO_OPTIMIZED):
            continue
        r = relpath(c.co_filename)
        last = c.co_firstlineno
        for _, _, l in c.co_lines():
            if l is not None and l > last:
                last = l
        funcs.setdefault(r, []).append([c.co_firstlineno, last, c.co_qualname])
    for v in funcs.values():
        v.sort()
    return {
        'def': {k: sorted(v) for k, v in sorted(p.deft.items())},
        'call': {k: sorted(v) for k, v in sorted(p.call.items())},
        'funcs': dict(sorted(funcs.items())),
        'opened': sorted(p.opened),
        'listed': sorted(p.listed),
    }

def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime + c.ru_utime + c.ru_stime

def main():
    global P
    import unittest, importlib, traceback, shutil  # stdlib only before monitoring starts
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = {a for a in sys.argv[1:] if a.startswith('--')}
    NUMBER, TEST_ID, OUT = int(args[0]), args[1], args[2]
    DO_CLASSES = '--no-classes' not in flags
    DO_LINES = '--no-lines' not in flags

    sys.addaudithook(audit)
    if DO_LINES:
        mon.use_tool_id(LT, 'brtrace')
        mon.register_callback(LT, mon.events.LINE, on_line)
        mon.set_events(LT, mon.events.LINE)

    out = {'number': NUMBER, 'id': TEST_ID}
    path, cls_name, meth = TEST_ID.split('::')
    modname = path[:-3].replace('/', '.')
    t0 = time.perf_counter()
    err = None
    try:
        mod = importlib.import_module(modname)
    except BaseException:
        err = traceback.format_exc()[-3000:]
    out['import_seconds'] = round(time.perf_counter() - t0, 4)
    imp = P
    out['import'] = None  # filled below

    P = Phase()
    if DO_LINES:
        mon.restart_events()
    child_dir = '/tmp/brchild_%d' % NUMBER
    shutil.rmtree(child_dir, ignore_errors=True)
    os.makedirs(child_dir)
    os.environ['BR_CHILD_DIR'] = child_dir

    result = unittest.TestResult()
    if err is None:
        try:
            tc = getattr(mod, cls_name)(meth)
            suite = unittest.TestSuite([tc])
        except BaseException:
            err = traceback.format_exc()[-3000:]
    if DO_CLASSES and err is None:
        mon.use_tool_id(CT, 'brclasses')
        mon.register_callback(CT, mon.events.PY_START, on_start)
        mon.set_events(CT, mon.events.PY_START)
    c0 = cpu()
    t0 = time.perf_counter()
    if err is None:
        try:
            suite.run(result)
        except BaseException:
            err = traceback.format_exc()[-3000:]
    secs = time.perf_counter() - t0
    cpus = cpu() - c0
    if DO_CLASSES and err is None:
        mon.set_events(CT, 0)
    if DO_LINES:
        mon.set_events(LT, 0)
    test = P
    P = Phase()
    os.environ.pop('BR_CHILD_DIR', None)

    # merge child process traces
    children = []
    for fn in sorted(os.listdir(child_dir)):
        try:
            with open(os.path.join(child_dir, fn)) as f:
                ch = json.load(f)
        except Exception as e:
            children.append({'file': fn, 'error': repr(e)})
            continue
        children.append({'file': fn, 'argv': ch.get('argv'), 'n_call_files': len(ch['call']), 'n_def_files': len(ch['def'])})
        for key, dst in (('call', test.call), ('def', test.deft)):
            for k, v in ch[key].items():
                dst.setdefault(k, set()).update(v)
        test.opened.update(ch.get('opened', []))
        test.listed.update(ch.get('listed', []))
    shutil.rmtree(child_dir, ignore_errors=True)

    if err is not None:
        outcome = 'error'
    elif result.errors:
        outcome = 'error'
    elif result.failures or result.unexpectedSuccesses:
        outcome = 'failed'
    elif result.skipped:
        outcome = 'skipped'
    elif result.testsRun == 1:
        outcome = 'passed'
    else:
        outcome = 'error'
    out['outcome'] = outcome
    if outcome != 'passed':
        det = []
        if err: det.append(err)
        for _, tb in result.errors + result.failures:
            det.append(tb[-2000:])
        for _, why in result.skipped:
            det.append('skip: ' + str(why))
        out['detail'] = det[:5]
    out['import'] = dump_phase(imp)
    out['test'] = dump_phase(test)
    out['children'] = children
    out['classes'] = sorted(classes) if DO_CLASSES else None
    out['seconds'] = round(secs, 4)
    out['cpu_seconds'] = round(cpus, 4)
    out['flags'] = sorted(flags)
    tmp = OUT + '.tmp'
    with gzip.open(tmp, 'wt') as f:
        json.dump(out, f, separators=(',', ':'))
    os.replace(tmp, OUT)
    sys.stderr.write('%d %s %.2fs\n' % (NUMBER, outcome, secs))
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
