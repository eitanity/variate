"use strict";

const $ = (id) => document.getElementById(id);
const pending = new Map();
let nextId = 1;
let worker = null;
let lastResult = null;
let lastKeep = null;   // the engine's handle on the last run's workspace
let lastGraph = null;  // the compiled graph of the last run, for the flow diagram
let baseline = null;   // { keep, result, label } pinned for comparison
const BANDS = 50;
const DECADES = 1e4;   // the log histogram starts this far below its top
const COLOURS = { current: "#1f6feb", baseline: "#e08a1e" };

// The engine cannot be interrupted once a call is running, so Cancel
// terminates the worker and starts a fresh one.
function startWorker() {
  worker = new Worker("worker.js");
  worker.onmessage = (event) => {
    const { id, ok, error, ms, keep, blocks, graph } = event.data;
    const p = pending.get(id);
    if (!p) return;
    pending.delete(id);
    if (error !== undefined) p.reject(new Error(error)); else p.resolve({ result: ok, ms, keep, blocks, graph });
  };
}

function cancelAll() {
  worker.terminate();
  for (const p of pending.values()) p.reject(new Error("cancelled"));
  pending.clear();
  startWorker();
  // The kept workspaces died with the worker.
  lastKeep = null;
  if (baseline) {
    baseline = null;
    $("unpin").hidden = true;
    $("compareNote").textContent = "The baseline was lost with the cancelled run.";
  }
}

startWorker();

function call(kind, yaml, settings, extra = {}) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, kind, yaml, settings, ...extra });
  });
}

let models = [];
let current = null;
let edited = false;

async function init() {
  models = await (await fetch("models/index.json")).json();
  const select = $("model");
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.title;
    select.appendChild(opt);
  }
  select.onchange = () => selectModel(select.value);
  // Every template is fetched now, so changing models needs nothing
  // from the server once the page is up.
  await Promise.all(models.map(async (m) => { m.template = await (await fetch("models/" + m.file)).text(); }));
  $("run").onclick = run;
  $("cancel").onclick = () => { cancelAll(); $("status").textContent = "cancelled"; };
  $("logx").onchange = () => { redraw().catch((err) => { $("error").textContent = String(err); }); };
  $("logp").onchange = () => { redraw().catch((err) => { $("error").textContent = String(err); }); };
  $("quantity").onchange = () => { redraw().catch((err) => { $("error").textContent = String(err); }); };
  $("tailrp").onchange = () => { if (lastResult) drawTailContributions($("tailcontrib"), lastResult, Number($("tailrp").value)); };
  $("countsource").onchange = () => { drawCountsFor($("countsource").value).catch((err) => { $("error").textContent = String(err); }); };
  $("pin").onclick = pin;
  $("unpin").onclick = () => { unpin(); redraw().catch(() => {}); };
  $("yaml").oninput = () => { edited = true; };
  call("version").then(({ result }) => { $("version").textContent = "engine " + result; }).catch(() => {});
  await selectModel(models[0].id);
}

async function selectModel(id) {
  current = models.find((m) => m.id === id);
  if (current.template === undefined) current.template = await (await fetch("models/" + current.file)).text();
  $("about").textContent = current.about;
  edited = false;
  const knobs = $("knobs");
  knobs.innerHTML = "";
  for (const k of current.knobs) {
    const label = document.createElement("label");
    label.textContent = k.label;
    const row = document.createElement("div");
    row.className = "knob";
    const range = document.createElement("input");
    range.type = "range";
    range.min = k.min; range.max = k.max; range.step = k.step; range.value = k.value;
    const num = document.createElement("input");
    num.type = "number";
    num.min = k.min; num.max = k.max; num.step = k.step; num.value = k.value;
    range.oninput = () => { num.value = range.value; k.current = Number(range.value); refreshYAML(); };
    num.oninput = () => { range.value = num.value; k.current = Number(num.value); refreshYAML(); };
    k.current = k.value;
    row.append(range, num);
    knobs.append(label, row);
  }
  refreshYAML();
}

// refreshYAML substitutes the knobs into the template, unless the YAML
// box has been edited by hand.
function refreshYAML() {
  if (edited) return;
  let text = current.template;
  for (const k of current.knobs) {
    text = text.split("{{" + k.key + "}}").join(String(k.current));
  }
  $("yaml").value = text;
}

async function run() {
  const button = $("run");
  button.disabled = true;
  $("cancel").hidden = false;
  $("error").textContent = "";
  $("status").textContent = $("transform").checked ? "running, then computing the exact distribution…" : "running…";
  const settings = { periods: $("periods").value, seed: $("seed").value, transform: $("transform").checked ? "true" : "false" };
  if ($("keepseries").checked && Number($("periods").value) <= 400000) settings.series = "true";
  try {
    const { result, ms, keep, blocks, graph } = await call("run", $("yaml").value, settings);
    $("blockNote").textContent = (blocks || []).map((b) => `Reliability block ${b.source}: ${fmt(b.rate, 3)} failures per period; system MTBF ${fmt(b.mtbf, 4)} hours; unavailability ${pct(b.unavailability)}.`).join(" ");
    if (lastKeep !== null && (!baseline || baseline.keep !== lastKeep)) call("release", null, null, { keep: lastKeep }).catch(() => {});
    lastKeep = keep;
    lastGraph = graph;
    show(result, ms);
    $("pin").hidden = false;
    fillQuantities(result, graph);
    await redraw();
    drawContributions($("contrib"), result);
    drawTailContributions($("tailcontrib"), result, Number($("tailrp").value));
    drawFlow($("flow"), graph, result);
    await drawCountsFor($("countsource").value);
    await drawSeriesIfKept(settings.series === "true");
    $("status").textContent = `${result.periods.toLocaleString()} periods in ${(ms / 1000).toFixed(2)} s: ${Math.round(result.periods / (ms / 1000)).toLocaleString()} periods/s, ${Math.round(result.steps / (ms / 1000) / 1e6)} M steps/s. Seed ${result.seed}: enter it to repeat this run exactly; leave 0 for an independent run each time.`;
  } catch (err) {
    if (err.message !== "cancelled") {
      $("status").textContent = "";
      $("error").textContent = err.message;
    }
  }
  $("cancel").hidden = true;
  button.disabled = false;
}

const fmt = (x, digits = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toLocaleString(undefined, { maximumSignificantDigits: digits });
const pct = (x) => (x === null || x === undefined) ? "–" : (100 * x).toFixed(1) + "%";
// pctTick labels a probability axis tick in percent at any scale: 100%, 10%, 0.01%.
const pctTick = (p) => Number(100 * p).toLocaleString(undefined, { maximumSignificantDigits: 2 }) + "%";

function show(r, ms) {
  $("results").hidden = false;
  $("mean").textContent = fmt(r.total.mean, 4) + " ± " + fmt(r.total.stderr, 2);
  $("exact").textContent = fmt(r.analytic.mean, 4);
  $("p50").textContent = fmt(r.total.p50, 4);
  $("p99").textContent = fmt(r.total.p99, 4);
  $("pzero").textContent = pct(r.analytic.pzero);
  lastResult = r;
  table($("breakdown"), ["quantity", "mean", "share", "P50", "P90", "P99", "P99.9"], [
    ["total", fmt(r.total.mean, 4), "", fmt(r.total.p50), fmt(r.total.p90), fmt(r.total.p99), fmt(r.total.p999)],
    ...Object.entries(r.by_category).map(([name, s]) => ["category " + name, fmt(s.mean, 4), pct(s.share), fmt(s.p50), fmt(s.p90), fmt(s.p99), fmt(s.p999)]),
    ...Object.entries(r.by_source).map(([name, s]) => ["source " + name, fmt(s.mean, 4), pct(s.share), fmt(s.p50), fmt(s.p90), fmt(s.p99), fmt(s.p999)]),
  ]);
  // Reach: for each source, the nodes its rounds pass through, in the
  // graph's order, with the share that get there. A model where every
  // share is one has no branching, and the table says only that.
  const reachRows = [];
  let branching = false;
  for (const [src, nodes] of Object.entries(r.reach)) {
    for (const [node, p] of Object.entries(nodes)) {
      if (!(p.p > 0)) continue;
      if (p.p < 1) branching = true;
      reachRows.push([src, node, pct(p.p), fmt(r.entries_per[node].mean, 3)]);
    }
  }
  $("reach").hidden = !branching;
  $("reachNote").textContent = branching ? "" : "Every round from each source passes through every node listed for it: the model does not branch.";
  table($("reach"), ["source", "node its rounds reach", "share that reach it", "entries per period"], reachRows);
  table($("plan"), ["criterion", "driver", "planned periods", "observed", "met"], r.plan.map((p) => [p.criterion, p.driver, p.periods === 0 ? "unknown" : fmt(p.periods, 6), p.observed === null ? "not yet" : fmt(p.observed, 3), p.met ? "yes" : "no"]));
  $("planNote").textContent = r.plan.some((p) => p.periods === 0 || p.observed === null)
    ? "A model with delays has no exact variance, so the plan cannot size its error criteria, and the run estimates its errors from blocks of periods: it needs 16 full blocks before it can report them. Run more periods."
    : "";
  const tr = r.analytic.transform;
  if (tr && tr.applicable) {
    const q = tr.quantiles;
    $("transformNote").textContent = `Exact distribution by FFT on ${tr.m.toLocaleString()} grid points: P50 in [${fmt(q.p50[0], 4)}, ${fmt(q.p50[1], 4)}], P99 in [${fmt(q.p99[0], 4)}, ${fmt(q.p99[1], 4)}], P99.9 in [${fmt(q["p99.9"][0], 4)}, ${fmt(q["p99.9"][1], 4)}]. The bounds are drawn on the exceedance chart.`;
  } else if (tr) {
    $("transformNote").textContent = "The exact distribution does not apply here: " + tr.reason;
  } else {
    $("transformNote").textContent = "";
  }
  $("warnings").textContent = (r.warnings || []).join("; ");
}

// label describes the run being pinned: the model and its knobs, or
// the model with a note when the YAML was edited by hand.
function label() {
  if (edited) return current.title + " (edited YAML)";
  return current.title + ": " + current.knobs.map((k) => k.label.toLowerCase() + " " + fmt(k.current, 4)).join(", ");
}

function pin() {
  if (lastKeep === null || !lastResult) return;
  if (baseline && baseline.keep !== lastKeep) call("release", null, null, { keep: baseline.keep }).catch(() => {});
  baseline = { keep: lastKeep, result: lastResult, label: label() };
  $("unpin").hidden = false;
  $("compareNote").textContent = "Baseline pinned: " + baseline.label + ". Change the knobs or the model and run again to see both on the charts.";
  redraw().catch((err) => { $("error").textContent = String(err); });
}

function unpin() {
  if (baseline && baseline.keep !== lastKeep) call("release", null, null, { keep: baseline.keep }).catch(() => {});
  baseline = null;
  $("unpin").hidden = true;
  $("compareNote").textContent = "";
}

// fillQuantities lists what the two charts can draw: the total, each
// category and source, and each node's entries per period.
function fillQuantities(r, graph) {
  const sel = $("quantity");
  sel.innerHTML = "";
  const add = (value, label) => { const o = document.createElement("option"); o.value = value; o.textContent = label; sel.appendChild(o); };
  add("total:", "period total");
  for (const name of Object.keys(r.by_category)) add("category:" + name, "category " + name);
  for (const name of Object.keys(r.by_source)) add("source:" + name, "source " + name);
  for (const name of Object.keys(r.layers || {})) add("layer:" + name, "layer " + name);
  if (graph) for (const n of graph.nodes) add("entries:" + n.id, "events per period at " + n.id);
  const cs = $("countsource");
  cs.innerHTML = "";
  if (graph) for (const s of graph.sources) { const o = document.createElement("option"); o.value = s.id; o.textContent = s.id; cs.appendChild(o); }
}

// redraw draws the last run's charts for the quantity chosen, with the
// baseline beside it when one is pinned and the quantity is the total:
// both histograms are then asked for again on the same bands, up to a
// little past the larger P99, so the bars line up.
async function redraw() {
  if (!lastResult) return;
  const log = $("logx").checked;
  const [qkind, qname] = ($("quantity").value || "total:").split(":");
  if (qkind !== "total") {
    const { result: q } = await call("quantity", null, null, { keep: lastKeep, qkind, qname, top: 0, bands: BANDS });
    const title = qkind === "entries" ? "events per period at " + qname : qkind + " " + qname;
    $("histTitle").textContent = title;
    $("ecTitle").textContent = title;
    $("quantityNote").textContent = `mean ${fmt(q.stat.mean, 4)}, median ${fmt(q.stat.p50, 4)}, P99 ${fmt(q.stat.p99, 4)}` + (q.stat.share ? `, ${pct(q.stat.share)} of the total` : "");
    $("legend").textContent = ""; $("legend2").textContent = "";
    const cur = { result: { exceedance: q.exceedance }, colour: COLOURS.current, name: "current", hist: log ? q.log_histogram : q.histogram };
    drawHistogram($("hist"), [cur], q.stat, log);
    drawExceedance($("ec"), [cur], $("logp").checked);
    return;
  }
  $("histTitle").textContent = "Period total";
  $("ecTitle").textContent = "total";
  $("quantityNote").textContent = "";
  const cur = { result: lastResult, colour: COLOURS.current, name: "current" };
  if (!baseline || baseline.keep === lastKeep) {
    cur.hist = log ? lastResult.log_histogram : lastResult.histogram;
    $("legend").textContent = "";
    $("legend2").textContent = "";
    drawHistogram($("hist"), [cur], lastResult.total, log);
    drawExceedance($("ec"), [cur], $("logp").checked);
    figures([cur]);
    return;
  }
  const top = 1.05 * Math.max(baseline.result.total.p99, lastResult.total.p99);
  const [a, b] = await Promise.all([
    call("histogram", null, null, { keep: baseline.keep, top, bands: BANDS }),
    call("histogram", null, null, { keep: lastKeep, top, bands: BANDS }),
  ]);
  const base = { result: baseline.result, colour: COLOURS.baseline, name: "baseline", hist: log ? a.result.log_histogram : a.result.histogram };
  cur.hist = log ? b.result.log_histogram : b.result.histogram;
  if (log) alignLog([base, cur], top);
  legend($("legend"), base, cur);
  legend($("legend2"), base, cur);
  drawHistogram($("hist"), [base, cur], lastResult.total, log);
  drawExceedance($("ec"), [base, cur], $("logp").checked);
  figures([base, cur]);
}

function legend(el, base, cur) {
  el.innerHTML = "";
  for (const s of [base, cur]) {
    const b = document.createElement("b");
    b.style.background = s.colour;
    el.append(b, (s.name === "baseline" ? "baseline: " + baseline.label : "current: " + label()) + "  ");
  }
}

// alignLog pads two log histograms to the same first band: both have
// bands equal in the logarithm from top/DECADES, trimmed where their
// mass starts, so the offset of each is its first edge's band number.
function alignLog(series, top) {
  const lo = top / DECADES, step = Math.log(DECADES) / BANDS;
  const first = series.map((s) => Math.round(Math.log(s.hist.edges[0] / lo) / step));
  const start = Math.min(...first);
  series.forEach((s, i) => {
    const pad = first[i] - start;
    if (pad <= 0) return;
    const edges = [], mass = [];
    for (let k = 0; k < pad; k++) { edges.push(lo * Math.exp(step * (start + k))); mass.push(0); }
    s.hist = { edges: edges.concat(s.hist.edges), mass: mass.concat(s.hist.mass) };
  });
}

function table(el, head, rows) {
  el.innerHTML = "";
  const tr = document.createElement("tr");
  for (const h of head) { const th = document.createElement("th"); th.textContent = h; tr.appendChild(th); }
  el.appendChild(tr);
  for (const row of rows) {
    const r = document.createElement("tr");
    for (const c of row) { const td = document.createElement("td"); td.textContent = c; r.appendChild(td); }
    el.appendChild(r);
  }
}

function setup(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = "#666";
  return { ctx, w, h, left: 46, right: 10, top: 36, bottom: 24 };
}

// caption writes a chart's note in the band above its plot, on one line
// or two so that it fits the width.
function caption(ctx, text, x, width, top) {
  const lines = [""];
  for (const word of text.split(" ")) {
    const next = lines[lines.length - 1] === "" ? word : lines[lines.length - 1] + " " + word;
    if (ctx.measureText(next).width <= width || lines[lines.length - 1] === "") lines[lines.length - 1] = next;
    else lines.push(word);
  }
  if (lines.length > 2) { lines.length = 2; lines[1] += "…"; }
  ctx.fillStyle = "#666";
  lines.forEach((line, i) => ctx.fillText(line, x, top - 9 - 12 * (lines.length - 1 - i)));
}

// tag writes a label on a white ground so that it reads over bars and
// bands.
function tag(ctx, text, x, y) {
  const width = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  ctx.fillRect(x - 2, y - 10, width + 4, 13);
  ctx.fillStyle = "#333";
  ctx.fillText(text, x, y);
  ctx.fillStyle = "#666";
}

// barLabel writes a bar's name below the axis, slanted and ending at
// the bar's centre, cut to fit the label band.
function barLabel(ctx, name, cx, y) {
  ctx.save(); ctx.translate(cx, y); ctx.rotate(-Math.PI / 4); ctx.textAlign = "right"; ctx.fillStyle = "#666";
  ctx.fillText(name.length > 14 ? name.slice(0, 13) + "…" : name, 0, 0); ctx.restore();
}

// drawHistogram draws one or two series of bands, equal in width since
// they are equal in their own scale, side by side within each band when
// there are two, with the quantile markers of the current run placed on
// that scale. The empty periods are in no band; they are shown as a
// share above.
function drawHistogram(canvas, series, total, log) {
  const { ctx, w, h, left, right, top, bottom } = setup(canvas);
  const ref = series[series.length - 1].hist;
  if (!ref) return;
  const edges = ref.edges;
  const n = Math.max(...series.map((s) => s.hist.edges.length - 1));
  const plotW = w - left - right, plotH = h - top - bottom;
  const peak = Math.max(1e-9, ...series.flatMap((s) => s.hist.mass.slice(0, s.hist.edges.length - 1)));
  const scale = log ? Math.log : (x) => x;
  const frac = (v) => (scale(v) - scale(edges[0])) / (scale(edges[edges.length - 1]) - scale(edges[0]) || 1);
  const slot = plotW / (n + 1), width = (slot - 2) / series.length;
  series.forEach((s, si) => {
    const mass = s.hist.mass, m = s.hist.edges.length - 1;
    for (let i = 0; i <= m; i++) {
      const x = left + slot * i + 1 + width * si;
      const barH = plotH * mass[i] / peak;
      ctx.fillStyle = i === m ? "#c77" : s.colour;
      if (i === m && series.length > 1) ctx.fillStyle = s.colour;
      ctx.globalAlpha = i === m ? 0.6 : 1;
      ctx.fillRect(x, top + plotH - barH, width, barH);
      ctx.globalAlpha = 1;
    }
  });
  ctx.fillStyle = "#666";
  ctx.strokeStyle = "#ccc";
  ctx.beginPath(); ctx.moveTo(left, top + plotH); ctx.lineTo(left + plotW, top + plotH); ctx.stroke();
  for (let i = 0; i <= n; i += 10) {
    ctx.fillText(fmt(edges[i], 2), left + plotW * i / (n + 1) - 10, h - 8);
  }
  ctx.fillText("beyond →", left + plotW * n / (n + 1) - 30, h - 8);
  caption(ctx, "share of periods with a total, " + fmt(100 * peak, 2) + "% at the tallest bar; empty periods left out", left + 4, plotW - 8, top);
  for (const [p, label] of [[total.p50, "P50"], [total.p90, "P90"], [total.p99, "P99"]]) {
    if (!(p > edges[0]) || p > edges[n]) continue;
    const x = left + plotW * frac(p) * n / (n + 1);
    ctx.strokeStyle = "#333"; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotH); ctx.stroke();
    ctx.setLineDash([]);
    tag(ctx, label, x + 3, top + 14);
  }
}

// RETURN_PERIODS are the exceedance levels marked on the chart and
// tabulated: a loss reached once in this many periods.
const RETURN_PERIODS = [10, 100, 1000];

// lossAt interpolates the loss a curve reaches with probability p, in
// the logarithm of the loss between the two points that straddle p, or
// null when the curve never falls that low.
function lossAt(pts, p) {
  for (let i = 1; i < pts.length; i++) {
    if (pts[i].p <= p && pts[i - 1].p > p) {
      const f = (pts[i - 1].p - p) / (pts[i - 1].p - pts[i].p || 1);
      return Math.exp(Math.log(pts[i - 1].x) + f * (Math.log(pts[i].x) - Math.log(pts[i - 1].x)));
    }
  }
  return null;
}

// drawExceedance draws one or two exceedance curves, the loss along a
// log axis and the probability linear by default, which shows where
// the frequent losses accumulate, or logarithmic to open up the tail.
// Each curve carries a faint band of two standard errors, and its exact
// bounds shaded when it has them; dashed guides mark the return periods.
function drawExceedance(canvas, series, logP) {
  const { ctx, w, h, left, right, top, bottom } = setup(canvas);
  const all = series.map((s) => s.result.exceedance.filter((p) => p.x > 0 && p.p > 0));
  if (all.some((pts) => pts.length < 2)) return;
  const plotW = w - left - right, plotH = h - top - bottom;
  const lx = (x) => Math.log10(x);
  // The curve is flat at P(total > 0) below the smallest loss the model
  // books, so the loss axis starts one point before the first visible
  // drop, a hundredth of the starting share, across the series drawn.
  const startOf = (pts) => {
    let i = 0;
    while (i < pts.length - 1 && pts[i].p > pts[0].p * 0.99) i++;
    return pts[Math.max(0, i - 1)].x;
  };
  const x0 = Math.min(...all.map((pts) => lx(startOf(pts)))), x1 = Math.max(...all.map((pts) => lx(pts[pts.length - 1].x)));
  const pMin = Math.max(1e-5, Math.min(...all.flatMap((pts) => pts.map((p) => p.p))));
  const X = (x) => left + plotW * (lx(x) - x0) / (x1 - x0 || 1);
  const Y = logP
    ? (p) => top + plotH * (0 - Math.log10(Math.max(p, pMin))) / (0 - Math.log10(pMin) || 1)
    : (p) => top + plotH * (1 - Math.min(1, Math.max(0, p)));
  ctx.strokeStyle = "#ddd";
  const yTicks = logP ? Array.from({ length: Math.ceil(-Math.log10(pMin)) + 1 }, (_, i) => Math.pow(10, -i)) : [1, 0.8, 0.6, 0.4, 0.2, 0];
  for (const p of yTicks) {
    const y = Y(p);
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + plotW, y); ctx.stroke();
    ctx.fillText(pctTick(p), 4, y + 4);
  }
  for (let d = Math.ceil(x0); d <= Math.floor(x1); d++) {
    const x = X(Math.pow(10, d));
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotH); ctx.stroke();
    const tick = fmt(Math.pow(10, d), 1);
    ctx.fillText(tick, Math.min(x - 8, w - ctx.measureText(tick).width - 2), h - 8);
  }
  // Return-period guides, with the current run's loss at each.
  ctx.strokeStyle = "#999"; ctx.setLineDash([4, 3]);
  const cur = all[all.length - 1];
  // Labels from the lowest guide up, each lifted clear of the one below
  // when the guides crowd the bottom of a linear axis.
  let floor = Infinity;
  for (const n of [...RETURN_PERIODS].reverse()) {
    const p = 1 / n;
    if (p < pMin) continue;
    const y = Y(p);
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + plotW, y); ctx.stroke();
    const loss = lossAt(cur, p);
    const ly = Math.min(y - 3, floor - 13);
    floor = ly;
    tag(ctx, `1 in ${n.toLocaleString()} periods` + (loss ? `: ${fmt(loss, 3)}` : ""), left + plotW - 150, ly);
    ctx.fillStyle = "#666";
  }
  ctx.setLineDash([]);
  let shaded = false;
  all.forEach((pts, si) => { all[si] = pts.filter((p) => lx(p.x) >= x0); });
  series.forEach((s, si) => {
    const pts = all[si];
    // Two standard errors of the simulated share, either side.
    ctx.fillStyle = s.colour; ctx.globalAlpha = 0.12;
    ctx.beginPath();
    ctx.moveTo(X(pts[0].x), Y(pts[0].p + 2 * pts[0].stderr));
    for (const p of pts) ctx.lineTo(X(p.x), Y(p.p + 2 * p.stderr));
    for (let i = pts.length - 1; i >= 0; i--) ctx.lineTo(X(pts[i].x), Y(Math.max(pMin, pts[i].p - 2 * pts[i].stderr)));
    ctx.closePath(); ctx.fill();
    if (pts[0].lower !== undefined) {
      ctx.globalAlpha = 0.2;
      ctx.beginPath();
      ctx.moveTo(X(pts[0].x), Y(pts[0].upper));
      for (const p of pts) ctx.lineTo(X(p.x), Y(p.upper));
      for (let i = pts.length - 1; i >= 0; i--) ctx.lineTo(X(pts[i].x), Y(pts[i].lower));
      ctx.closePath(); ctx.fill();
      shaded = true;
    }
    ctx.globalAlpha = 1;
  });
  ctx.fillStyle = "#666";
  caption(ctx, "faint band: ± 2 standard errors of the simulation" + (shaded ? "; darker band: exact bounds by FFT" : ""), left + 4, plotW - 8, top);
  series.forEach((s, si) => {
    const pts = all[si];
    ctx.strokeStyle = s.colour; ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((p, i) => { const x = X(p.x), y = Y(p.p); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.stroke();
  });
  ctx.lineWidth = 1;
}

// figures fills the table beside the exceedance chart: the figures a
// decision turns on, for the current run and the baseline when pinned.
function figures(series) {
  const head = ["figure", ...series.map((s) => s.name)];
  const row = (label, f) => [label, ...series.map((s) => f(s.result))];
  const rows = [
    row("mean per period", (r) => fmt(r.total.mean, 4) + " ± " + fmt(r.total.stderr, 2)),
    row("exact mean", (r) => fmt(r.analytic.mean, 4)),
    row("median", (r) => fmt(r.total.p50, 4)),
    row("99th percentile", (r) => fmt(r.total.p99, 4)),
    row("periods with nothing booked", (r) => pct(r.analytic.pzero)),
  ];
  for (const n of RETURN_PERIODS) {
    rows.push(row(`loss reached once in ${n.toLocaleString()} periods`, (r) => {
      const loss = lossAt(r.exceedance.filter((p) => p.x > 0 && p.p > 0), 1 / n);
      return loss === null ? "beyond the run" : fmt(loss, 4);
    }));
  }
  table($("figures"), head, rows);
}

init().catch((err) => { $("error").textContent = String(err); });

// drawContributions draws the sources' means per period as sorted
// bars with the cumulative share as a line: the Pareto chart of a
// register. The smallest are gathered as "others" past fifteen.
// labelBand is the height under a bar chart for its slanted names.
const labelBand = 70;

function drawContributions(canvas, r) {
  const { ctx, w, h, left, right, top } = setup(canvas);
  const rows = Object.entries(r.by_source).map(([name, s]) => [name, s.mean]).sort((a, b) => b[1] - a[1]);
  if (rows.length === 0) return;
  const shown = rows.slice(0, 15);
  if (rows.length > 15) shown.push(["others (" + (rows.length - 15) + ")", rows.slice(15).reduce((t, x) => t + x[1], 0)]);
  const total = rows.reduce((t, x) => t + x[1], 0) || 1;
  const plotW = w - left - right - 60, plotH = h - top - labelBand, lx = left + 60;
  const peak = Math.max(...shown.map((x) => x[1]), 1e-9);
  const slot = plotW / shown.length;
  let cum = 0;
  ctx.strokeStyle = "#e08a1e"; ctx.lineWidth = 2; ctx.beginPath();
  shown.forEach(([name, mean], i) => {
    const x = lx + slot * i, barH = plotH * mean / peak;
    ctx.fillStyle = "#1f6feb";
    ctx.fillRect(x + 2, top + plotH - barH, slot - 4, barH);
    cum += mean / total;
    const cx = x + slot / 2, cy = top + plotH * (1 - cum);
    if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    barLabel(ctx, name, x + slot / 2, top + plotH + 12);
  });
  ctx.stroke(); ctx.lineWidth = 1;
  caption(ctx, "bars: mean per period (tallest " + fmt(peak, 3) + "); line: cumulative share of the total mean", lx + 4, plotW - 8, top);
}

// drawTailContributions draws each source's share of the total over
// the periods beyond the loss reached once in n periods, read at the
// nearest tail threshold the run counted exactly.
function drawTailContributions(canvas, r, n) {
  const { ctx, w, h, left, right, top } = setup(canvas);
  const pts = r.exceedance.filter((p) => p.x > 0 && p.p > 0);
  const loss = lossAt(pts, 1 / n);
  const usable = (r.tail_shares || []).filter((t) => t.periods >= 30);
  let pick = null;
  for (const t of usable) if (loss === null || t.x <= loss) pick = t;
  if (!pick) { $("tailNote").textContent = "Not enough periods beyond that loss to say; run more periods."; return; }
  $("tailNote").textContent = `Loss reached once in ${n.toLocaleString()} periods: ${loss === null ? "beyond the run" : fmt(loss, 4)}. Shares over the ${pick.periods.toLocaleString()} periods whose total reached ${fmt(pick.x, 3)}, the nearest level the run counted exactly.`;
  const rows = Object.entries(pick.by_source).sort((a, b) => b[1] - a[1]).slice(0, 16);
  const plotW = w - left - right - 60, plotH = h - top - labelBand, lx = left + 60;
  const slot = plotW / rows.length;
  const peak = Math.max(...rows.map((x) => x[1]), 1e-9);
  rows.forEach(([name, share], i) => {
    const x = lx + slot * i, barH = plotH * share / peak;
    ctx.fillStyle = "#c77";
    ctx.fillRect(x + 2, top + plotH - barH, slot - 4, barH);
    ctx.fillStyle = "#666";
    if (ctx.measureText(pct(share)).width <= slot - 2) ctx.fillText(pct(share), x + 2, top + plotH - barH - 3);
    else { ctx.save(); ctx.translate(x + slot / 2 + 4, top + plotH - barH - 4); ctx.rotate(-Math.PI / 2); ctx.fillText(pct(share), 0, 0); ctx.restore(); }
    barLabel(ctx, name, x + slot / 2, top + plotH + 12);
  });
}

// drawCountsFor draws the distribution of events per period at the
// node a source enters, against a Poisson of the same mean, so a count
// law's narrowing or clustering shows.
async function drawCountsFor(sourceID) {
  if (!lastGraph || !lastResult || !sourceID) return;
  const src = lastGraph.sources.find((s) => s.id === sourceID);
  if (!src) return;
  const { result: q } = await call("quantity", null, null, { keep: lastKeep, qkind: "entries", qname: src.to, top: 0, bands: BANDS });
  const canvas = $("counts");
  const { ctx, w, h, left, right, top, bottom } = setup(canvas);
  const p = (q.counts && q.counts.p) || [];
  if (p.length === 0) { $("countNote").textContent = "No events reached " + src.to + "."; return; }
  const mean = lastResult.entries_per[src.to].mean;
  const pois = []; let term = Math.exp(-mean);
  for (let k = 0; k < p.length; k++) { pois.push(term); term *= mean / (k + 1); }
  const plotW = w - left - right, plotH = h - top - bottom;
  const peak = Math.max(...p, ...pois, 1e-9);
  const slot = plotW / p.length;
  ctx.fillStyle = "#1f6feb";
  p.forEach((v, k) => ctx.fillRect(left + slot * k + 1, top + plotH * (1 - v / peak), Math.max(1, slot - 2), plotH * v / peak));
  ctx.strokeStyle = "#e08a1e"; ctx.lineWidth = 2; ctx.beginPath();
  pois.forEach((v, k) => { const x = left + slot * k + slot / 2, y = top + plotH * (1 - v / peak); if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
  ctx.stroke(); ctx.lineWidth = 1;
  ctx.fillStyle = "#666";
  const every = Math.max(1, Math.round(p.length / 12));
  for (let k = 0; k < p.length; k += every) ctx.fillText(String(k), left + slot * k + 1, h - 8);
  caption(ctx, "bars: share of periods with k events at " + src.to + "; orange line: Poisson reference with the same mean " + fmt(mean, 3), left + 4, plotW - 8, top);
  const sd = lastResult.entries_per[src.to].sd;
  $("countNote").textContent = `Events per period at ${src.to}: mean ${fmt(mean, 4)}, standard deviation ${fmt(sd, 3)}; a Poisson count would have ${fmt(Math.sqrt(mean), 3)}. The index of dispersion is ${fmt(sd * sd / mean, 3)}.`;
}

// drawSeriesIfKept draws the period totals over the run when the run
// kept its series, with a rolling mean over a fiftieth of the periods.
async function drawSeriesIfKept(kept) {
  $("seriesBlock").hidden = !kept;
  if (!kept) return;
  let totals;
  try {
    const { result } = await call("series", null, null, { keep: lastKeep });
    totals = result.total;
  } catch (err) {
    $("seriesNote").textContent = err.message;
    return;
  }
  const canvas = $("series");
  const { ctx, w, h, left, right, top, bottom } = setup(canvas);
  const n = totals.length;
  if (n < 2) return;
  const plotW = w - left - right, plotH = h - top - bottom;
  let peak = 1e-9;
  for (const v of totals) if (v > peak) peak = v;
  const X = (i) => left + plotW * i / (n - 1), Y = (v) => top + plotH * (1 - v / peak);
  ctx.strokeStyle = "#9bb8e8"; ctx.beginPath();
  const stride = Math.max(1, Math.floor(n / (plotW * 2)));
  for (let i = 0; i < n; i += stride) { const x = X(i), y = Y(totals[i]); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
  ctx.stroke();
  const win = Math.max(2, Math.floor(n / 50));
  ctx.strokeStyle = "#1f6feb"; ctx.lineWidth = 2; ctx.beginPath();
  let acc = 0;
  for (let i = 0; i < n; i++) {
    acc += totals[i]; if (i >= win) acc -= totals[i - win];
    if (i >= win - 1) { const x = X(i), y = Y(acc / win); if (i === win - 1) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
  }
  ctx.stroke(); ctx.lineWidth = 1;
  ctx.fillStyle = "#666";
  caption(ctx, "period totals in order (light), and their rolling mean over " + win + " periods (dark); peak " + fmt(peak, 3), left + 4, plotW - 8, top);
  for (let i = 0; i <= 5; i++) ctx.fillText(Math.round((n - 1) * i / 5).toLocaleString(), left + plotW * i / 5 - 10, h - 8);
  $("seriesNote").textContent = `${n.toLocaleString()} periods in the order they were simulated. Clusters show a count law or delays carrying from one period to the next; a flat rolling mean shows the run has settled.`;
}

// drawFlow draws the compiled graph as an SVG: sources on the left,
// nodes in columns by their depth, edges as curves whose width is the
// events per period taking them and whose label is the share of the
// node's entries, dashed when a holding time delays them.
function drawFlow(svg, graph, r) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!graph) return;
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text !== undefined) e.textContent = text; return e; };
  const depth = {};
  for (const s of graph.sources) depth[s.to] = 1;
  let changed = true, guard = 0;
  while (changed && guard++ < graph.nodes.length + 2) {
    changed = false;
    for (const o of graph.outcomes) {
      if (depth[o.from] !== undefined && (depth[o.to] === undefined || depth[o.to] < depth[o.from] + 1)) { depth[o.to] = depth[o.from] + 1; changed = true; }
    }
  }
  const nodes = graph.nodes.filter((n) => depth[n.id] !== undefined);
  const columns = Math.max(1, ...nodes.map((n) => depth[n.id])) + 1;
  const W = Math.max(svg.clientWidth || 900, 160 * columns), H = svg.clientHeight || 360;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const colX = (d) => 20 + (W - 160) * d / Math.max(1, columns - 1);
  const byCol = {};
  for (const n of nodes) (byCol[depth[n.id]] = byCol[depth[n.id]] || []).push(n);
  const pos = {};
  for (const [d, list] of Object.entries(byCol)) list.forEach((n, i) => { pos[n.id] = { x: colX(Number(d)), y: 20 + (H - 60) * (i + 0.5) / list.length }; });
  const srcPos = {};
  graph.sources.forEach((s, i) => { srcPos[s.id] = { x: colX(0), y: 20 + (H - 60) * (i + 0.5) / graph.sources.length }; });
  const entries = (id) => (r.entries_per[id] ? r.entries_per[id].mean : 0);
  const flows = graph.outcomes.map((o) => ({ o, share: r.outcomes[o.name] ? r.outcomes[o.name].p : 0, flow: entries(o.from) * (r.outcomes[o.name] ? r.outcomes[o.name].p : 0) }));
  const maxFlow = Math.max(1e-9, ...flows.map((f) => f.flow), ...graph.sources.map((s) => entries(s.to)));
  const curve = (a, b, width, cls, label) => {
    const path = el("path", { class: cls, d: `M ${a.x + 120} ${a.y} C ${a.x + 150} ${a.y}, ${b.x - 30} ${b.y}, ${b.x} ${b.y}`, "stroke-width": (1 + 7 * width / maxFlow).toFixed(1) });
    svg.appendChild(path);
    if (label) svg.appendChild(el("text", { x: (a.x + 120 + b.x) / 2, y: (a.y + b.y) / 2 - 3, "text-anchor": "middle" }, label));
  };
  for (const s of graph.sources) {
    const p = srcPos[s.id], q = pos[s.to];
    if (q) curve({ x: p.x - 30, y: p.y }, q, entries(s.to), "edge", "");
    svg.appendChild(el("rect", { x: p.x - 10, y: p.y - 12, width: 100, height: 24, rx: 12, class: "src" }));
    svg.appendChild(el("text", { x: p.x + 40, y: p.y + 4, "text-anchor": "middle" }, s.id.length > 16 ? s.id.slice(0, 15) + "…" : s.id));
  }
  for (const f of flows) {
    const a = pos[f.o.from], b = pos[f.o.to];
    if (a && b) curve(a, b, f.flow, "edge" + (f.o.holding ? " hold" : ""), pct(f.share));
  }
  for (const n of nodes) {
    const p = pos[n.id];
    svg.appendChild(el("rect", { x: p.x, y: p.y - 18, width: 120, height: 36, rx: 4, class: n.leaf ? "leaf" : "node" }));
    svg.appendChild(el("text", { x: p.x + 60, y: p.y - 4, "text-anchor": "middle", "font-weight": "bold" }, n.id.length > 18 ? n.id.slice(0, 17) + "…" : n.id));
    const booked = r.contrib_per && r.contrib_per[n.id] ? r.contrib_per[n.id].mean : 0;
    svg.appendChild(el("text", { x: p.x + 60, y: p.y + 11, "text-anchor": "middle" }, fmt(entries(n.id), 3) + "/period" + (booked > 0 ? ", books " + fmt(booked, 3) : "")));
  }
}
