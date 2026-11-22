"""Writes maps/cost.bin: the CPU seconds of each whole test under the direct runner (pinned commit),
scaled so that the cheap tests sum to their recorded seconds (the measuring machine was loaded).
usage: python build_costs.py TIMING_JSON TESTS_JSON MAPS_DIR"""
import json, marshal, os, sys
d = json.load(open(sys.argv[1]))
tests = json.load(open(sys.argv[2]))["tests"]
tm = {int(k): v[0] for k, v in d["times"].items()}
small = [n for n in tm if tests[n]["seconds"] < 0.5]
factor = sum(tm[n] for n in small) / sum(tests[n]["seconds"] for n in small)
cost = [round(tm[t["number"]] / factor, 5) if t["number"] in tm else float(t["seconds"]) * 1.5 for t in tests]
with open(os.path.join(sys.argv[3], "cost.bin"), "wb") as f:
    marshal.dump(cost, f)
print("load factor", round(factor, 3), "sum cost", round(sum(cost), 1))
