"""Builds maps/tok_inputs.bin for tokcheck.py from recorded tokenize inputs (tight1106/xt:
tokrec.py through sitecustomize.py and xtplugin.py, one pytest run per test and hash seed,
spawn workers included). Each input is tokenized here with the pinned tree; an input whose
digest differs between two fresh processes, or whose tokenizer cannot be built again, makes
its test unknown (left out of the map).

usage (inside the pinned image, /repo on sys.path):
    python build_tok_inputs.py TOKDIR TESTS_JSON MAPS_DIR N[,N...]
TOKDIR holds t<N>_s<seed>/<process>.json."""
import glob
import json
import marshal
import os
import subprocess
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def digests_here(keys, inputs):
    import tokcheck

    out = []
    cache = {}
    for k, sql in inputs:
        try:
            tok = cache.get(k)
            if tok is None:
                tok = cache[k] = tokcheck.build_tokenizer(keys[k])
            out.append(tokcheck.digest(tok, sql).hex())
        except Exception as e:  # noqa: BLE001
            out.append("unbuildable %r" % (e,))
    return out


def main():
    if sys.argv[1] == "--digests":
        data = json.load(open(sys.argv[2]))
        keys = [tuple(tuple(x) if isinstance(x, list) else x for x in k) for k in data["keys"]]
        json.dump(digests_here(keys, [tuple(x) for x in data["inputs"]]), open(sys.argv[3], "w"))
        return
    tokdir, tests_path, out_dir, numbers = sys.argv[1:5]
    numbers = [int(x) for x in numbers.split(",")]
    keys = []
    key_id = {}
    inputs = []
    input_id = {}
    per_test = {n: set() for n in numbers}
    for n in numbers:
        dirs = glob.glob(os.path.join(tokdir, "t%d_s*" % n))
        if not dirs:
            raise SystemExit("no recording for test %d" % n)
        for d in dirs:
            for p in glob.glob(os.path.join(d, "*.json")):
                for k, sqls in json.load(open(p)):
                    k = tuple(tuple(x) if isinstance(x, list) else x for x in json.loads(k))
                    if k not in key_id:
                        key_id[k] = len(keys)
                        keys.append(k)
                    for sql in sqls:
                        x = (key_id[k], sql)
                        if x not in input_id:
                            input_id[x] = len(inputs)
                            inputs.append(x)
                        per_test[n].add(input_id[x])
    tmp = os.path.join(out_dir, "_tok_tmp.json")
    json.dump({"keys": keys, "inputs": inputs}, open(tmp, "w"))
    runs = []
    for r in range(2):
        res = tmp + ".%d" % r
        subprocess.run([sys.executable, os.path.abspath(__file__), "--digests", tmp, res], check=True)
        runs.append(json.load(open(res)))
        os.remove(res)
    os.remove(tmp)
    bad = {i for i, (a, b) in enumerate(zip(*runs)) if a != b or a.startswith("unbuildable")}
    tests = {}
    for n in numbers:
        if per_test[n] & bad:
            print("test %d: %d inputs not reproducible, left out" % (n, len(per_test[n] & bad)))
            continue
        tests[n] = sorted(per_test[n])
    out = {"keys": keys, "inputs": inputs, "digests": [bytes.fromhex(h) if not h.startswith("unbuildable") else b"" for h in runs[0]], "tests": tests}
    with open(os.path.join(out_dir, "tok_inputs.bin"), "wb") as f:
        f.write(zlib.compress(marshal.dumps(out), 9))
    print("inputs", len(inputs), "keys", len(keys), "tests", {n: len(v) for n, v in tests.items()}, "bad", len(bad))


if __name__ == "__main__":
    main()
