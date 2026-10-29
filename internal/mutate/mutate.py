"""Hand-rolled mutation testing: for each target file, generate mutants
by applying one operator at one site, run a fast test subset with
-failfast, and record killed/survived. Restores the file after each."""

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = {
    "simulate.go": "TestEngine|TestWorkspaceDeterminism|TestCompileDiamond|TestStreamCompleteness|TestMetamorphicConstantForms|TestSweepHolding",
    "finalise.go": "TestEngine|TestWorkspaceDeterminism|TestStreamCompleteness|TestSweepHolding",
    "criteria.go": "TestPlan|TestEngineAdaptive|TestTransformTailCriterion|TestCalibration",
    "tbounds.go": "TestTransformCompound|TestTransformPanjer|TestTransformWrap|TestTransformTilt|TestSweepTransform",
    "analytic.go": "TestAnalytic|TestCompileDiamond|TestMetamorphicSplitSource|TestInvariantInfinite",
    "config/dist.go": "TestUnits|TestLoaderErrorPaths|TestLoadExamples|TestProbabilityScope",
}
# (regex, replacement, description); applied to one match at a time.
OPERATORS = [
    (r"(?<![<>=!])<(?![<=])", "<=", "lt->le"),
    (r"(?<![<>=!])>(?![>=])", ">=", "gt->ge"),
    (r"<=", "<", "le->lt"),
    (r">=", ">", "ge->gt"),
    (r"(?<![+])\+=", "-=", "pluseq->minuseq"),
    (r"\*=", "/=", "muleq->diveq"),
    (r"(?<=[\w)\]]) \+ (?=[\w(])", " - ", "plus->minus"),
    (r"(?<=[\w)\]]) \* (?=[\w(])", " / ", "mul->div"),
    (r"return true", "return false", "true->false"),
    (r"return false", "return true", "false->true"),
    (r"(?<=\bif )(!)(?=[\w.])", "", "drop-not"),
    (r"\b1e-9\b", "1e-3", "tol-loose"),
]
MAX_PER_FILE = int(sys.argv[1]) if len(sys.argv) > 1 else 24


def code_lines(src):
    out = []
    for i, line in enumerate(src.split("\n")):
        s = line.strip()
        if s.startswith("//") or not s or s.startswith("func ") or "bound:" in s:
            continue
        out.append(i)
    return set(out)


def mutants(src):
    lines = src.split("\n")
    ok = code_lines(src)
    found = []
    for i in sorted(ok):
        for rx, rep, desc in OPERATORS:
            for m in re.finditer(rx, lines[i]):
                new = lines[i][: m.start()] + rep + lines[i][m.end():]
                if new == lines[i] or '"' in lines[i][: m.start()] and lines[i].count('"') % 2 == 1:
                    continue
                found.append((i, desc, new))
                break  # one mutant per line per operator family
    # spread mutants evenly over the file
    if len(found) > MAX_PER_FILE:
        step = len(found) / MAX_PER_FILE
        found = [found[int(k * step)] for k in range(MAX_PER_FILE)]
    return found, lines


def main():
    results = []
    for rel, pattern in TARGETS.items():
        path = ROOT / rel
        original = path.read_text()
        pkg = "./config" if rel.startswith("config/") else "."
        muts, lines = mutants(original)
        print(f"== {rel}: {len(muts)} mutants, tests {pattern}", flush=True)
        for i, desc, new in muts:
            mutated = lines[:]
            mutated[i] = new
            path.write_text("\n".join(mutated))
            t0 = time.time()
            build = subprocess.run(["go", "vet", pkg], cwd=ROOT, capture_output=True, text=True)
            if build.returncode != 0:
                status = "invalid"
            else:
                run = subprocess.run(["go", "test", "-count=1", "-failfast", "-timeout", "900s", "-run", pattern, pkg],
                                     cwd=ROOT, capture_output=True, text=True)
                status = "killed" if run.returncode != 0 else "SURVIVED"
            path.write_text(original)
            dt = time.time() - t0
            print(f"{status:9s} {rel}:{i+1} [{desc}] {new.strip()[:90]}  ({dt:.0f}s)", flush=True)
            results.append((status, rel, i + 1, desc, new.strip()))
    path.write_text(original)
    survived = [r for r in results if r[0] == "SURVIVED"]
    print(f"\n== {len(results)} mutants, {len(survived)} survived, {sum(1 for r in results if r[0]=='invalid')} invalid")
    for r in survived:
        print(f"  {r[1]}:{r[2]} [{r[3]}] {r[4][:100]}")


if __name__ == "__main__":
    main()
