// Smoke test of the wasm engine under node: run the compound example
// through variateRun and compare the report with the native CLI's JSON
// written to the path given as the first argument (variate run -f
// examples/compound.yaml -n 2000 --seed 5 --block-periods 50 --format
// json --analytic). The wasm report adds "histogram" and "warnings";
// everything else must be identical.
//
//   node web/smoke.js native.json
"use strict";
const fs = require("fs");
const path = require("path");

require(path.join(__dirname, "wasm_exec.js"));
const go = new Go();

// Numbers agree to a relative tolerance: the native build and the wasm
// engine differ in the last unit of values the math library derives,
// since Go's Exp, Log and Pow use CPU-specific instructions natively.
const TOLERANCE = 1e-9;

function deepEqual(a, b, where) {
  if (typeof a !== typeof b) throw new Error(`${where}: ${typeof a} vs ${typeof b}`);
  if (a === null || typeof a !== "object") {
    if (typeof a === "number" && typeof b === "number") {
      if (a !== b && !(Number.isNaN(a) && Number.isNaN(b)) && Math.abs(a - b) > TOLERANCE * Math.max(Math.abs(a), Math.abs(b))) throw new Error(`${where}: ${a} vs ${b}`);
      return;
    }
    if (a !== b) throw new Error(`${where}: ${a} vs ${b}`);
    return;
  }
  if (Array.isArray(a) !== Array.isArray(b)) throw new Error(`${where}: array vs object`);
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) throw new Error(`${where}: ${ka.length} keys vs ${kb.length}`);
  for (const k of ka) deepEqual(a[k], b[k], `${where}.${k}`);
}

async function main() {
  const native = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
  const wasm = fs.readFileSync(path.join(__dirname, "variate.wasm"));
  const { instance } = await WebAssembly.instantiate(wasm, go.importObject);
  go.run(instance); // registers variateRun; returns when the program blocks
  await new Promise((r) => setTimeout(r, 0));
  const yaml = fs.readFileSync(path.join(__dirname, "..", "examples", "compound.yaml"), "utf8");
  const t0 = Date.now();
  const out = variateRun(yaml, { periods: "2000", seed: "5", "block-periods": "50", transform: "true" });
  if (out.error !== undefined) throw new Error(out.error);
  const report = JSON.parse(out.ok);
  if (!report.histogram || !report.warnings) throw new Error("wasm report lacks histogram or warnings");
  delete report.histogram;
  delete report.warnings;
  deepEqual(native, report, "report");
  console.log(`wasm report matches the native CLI to ${TOLERANCE} relative; version ${variateVersion()}; ${report.periods} periods in ${Date.now() - t0} ms`);
  process.exit(0);
}

main().catch((err) => { console.error(err.message || err); process.exit(1); });
