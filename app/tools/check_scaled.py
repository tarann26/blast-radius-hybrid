"""Checks on the pinned commit that every small-scale test variant passes, and times it."""
import importlib, json, marshal, os, sys, time, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preload, reduce, runner, stmtmap
root = os.path.abspath(sys.argv[1])
tests = json.load(open(sys.argv[2]))["tests"]
preload.install_environment(root); preload.import_sqlglot()
for t in tests:
    if t["id"] in reduce.SCALED:
        rel, cls_name, meth = t["id"].split("::")
        preload.import_tests([rel[:-3].replace("/", ".")])
        src = open(os.path.join(root, rel)).read()
        units, flags, inames = stmtmap.build_units(src)
        shape = reduce.file_shapes(src)[(cls_name, meth)]
        cls = getattr(sys.modules[rel[:-3].replace("/", ".")], cls_name)
        ok = reduce.apply_reduction(cls, t["id"], os.path.join(root, rel), shape, src.split("\n"), flags, inames, None, True)
        res = {}
        import gc; gc.collect(); gc.freeze()
        c0 = time.process_time()
        r = runner.run_tests([(t["number"], t["id"])], lambda n, bad: res.__setitem__(n, bad))
        for _, tb in r.failures + r.errors: print(tb[-1200:])
        print(t["id"], "applied", ok, "failed", res, "cpu", round(time.process_time() - c0, 3), flush=True)
