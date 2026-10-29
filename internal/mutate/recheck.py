"""Re-run the mutants that survived the per-file subsets against a broad
fast subset, to tell equivalent mutants from weak assertions."""

import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mutate import OPERATORS, ROOT  # noqa: E402

BROAD = ("TestEngine|TestWorkspace|TestStream|TestPlan|TestAnalytic|TestCompile|TestTransform(Tail|Compound|Panjer|Wrap|Tilt)"
         "|TestMetamorphic(Constant|Split)|TestSweep(Holding|Transform)|TestInvariantInfinite|TestDifferentialWorkers|TestMutant|TestFaultLimits")
CONFIG = "TestUnits|TestLoaderErrorPaths|TestLoadExamples|TestProbabilityScope|TestTemplates|TestSettings|TestPerTime|TestMerge"

survivors = []
for line in open(Path(__file__).with_name("run.log")):
    m = re.match(r"SURVIVED\s+(\S+):(\d+) \[([^\]]+)\]", line)
    if m:
        survivors.append((m.group(1), int(m.group(2)), m.group(3)))

results = []
for rel, lineno, desc in survivors:
    path = ROOT / rel
    original = path.read_text()
    lines = original.split("\n")
    src = lines[lineno - 1]
    new = None
    for rx, rep, d in OPERATORS:
        if d != desc:
            continue
        m = re.search(rx, src)
        if m:
            new = src[: m.start()] + rep + src[m.end():]
        break
    if new is None or new == src:
        print(f"skip      {rel}:{lineno} [{desc}]", flush=True)
        continue
    lines[lineno - 1] = new
    path.write_text("\n".join(lines))
    pkg = "./config" if rel.startswith("config/") else "."
    pattern = CONFIG if pkg == "./config" else BROAD
    t0 = time.time()
    run = subprocess.run(["go", "test", "-count=1", "-failfast", "-timeout", "1200s", "-run", pattern, pkg], cwd=ROOT, capture_output=True, text=True)
    path.write_text(original)
    status = "killed" if run.returncode != 0 else "SURVIVED"
    killer = ""
    if status == "killed":
        km = re.search(r"--- FAIL: (\w+)", run.stdout)
        killer = km.group(1) if km else "build"
    print(f"{status:9s} {rel}:{lineno} [{desc}] {new.strip()[:80]} {killer} ({time.time()-t0:.0f}s)", flush=True)
    results.append((status, rel, lineno, desc, new.strip(), killer))
print(f"\n== {len(results)} rechecked, {sum(1 for r in results if r[0]=='SURVIVED')} still survive")
for r in results:
    if r[0] == "SURVIVED":
        print(f"  {r[1]}:{r[2]} [{r[3]}] {r[4][:100]}")
