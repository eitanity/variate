// The engine lives in this worker so the page stays responsive while a
// run goes. wasm_exec.js is Go's runtime shim; variate.wasm registers
// variateRun, variatePlan, variateVersion, variateHistogram and
// variateRelease on the global object.
importScripts("wasm_exec.js");

const go = new Go();
const ready = WebAssembly.instantiateStreaming(fetch("variate.wasm"), go.importObject)
  .then((result) => { go.run(result.instance); return true; });

onmessage = async (event) => {
  const { id, kind, yaml, settings, keep, top, bands } = event.data;
  try {
    await ready;
    if (kind === "version") {
      postMessage({ id, ok: variateVersion() });
      return;
    }
    if (kind === "release") {
      variateRelease(keep);
      postMessage({ id, ok: true });
      return;
    }
    const t0 = performance.now();
    const out = kind === "plan" ? variatePlan(yaml, settings)
      : kind === "histogram" ? variateHistogram(keep, top, bands)
      : kind === "quantity" ? variateQuantity(keep, event.data.qkind, event.data.qname, top, bands)
      : kind === "series" ? variateSeries(keep)
      : variateRun(yaml, settings);
    if (out.error !== undefined) {
      postMessage({ id, error: out.error });
    } else {
      postMessage({ id, ok: JSON.parse(out.ok), ms: performance.now() - t0, keep: out.keep, blocks: out.blocks ? JSON.parse(out.blocks) : [], graph: out.graph ? JSON.parse(out.graph) : null });
    }
  } catch (err) {
    postMessage({ id, error: String(err) });
  }
};
