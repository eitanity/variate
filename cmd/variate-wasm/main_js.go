// Command variate-wasm is the engine for the browser: it exposes run,
// plan, version, histogram and release to JavaScript through syscall/js,
// taking a model as YAML text and settings as JSON text, and returning
// the report as JSON text. A run's workspace is kept until released so
// the page can ask for its histogram again on other bands, to draw two
// runs on one chart. The engine and config packages know nothing of the
// browser.
package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"runtime/debug"
	"strconv"
	"syscall/js"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
	"code.eitanity.com/variate/report"
)

func main() {
	s := &session{kept: map[int]keptRun{}, nextKeep: 1}
	js.Global().Set("variateRun", js.FuncOf(s.run))
	js.Global().Set("variatePlan", js.FuncOf(planFunc))
	js.Global().Set("variateVersion", js.FuncOf(versionFunc))
	js.Global().Set("variateHistogram", js.FuncOf(s.histogram))
	js.Global().Set("variateQuantity", js.FuncOf(s.quantity))
	js.Global().Set("variateSeries", js.FuncOf(s.series))
	js.Global().Set("variateRelease", js.FuncOf(s.release))
	select {}
}

// keptRun is a finished run whose workspace and result are held for
// its histograms, its quantities and its series.
type keptRun struct {
	c *variate.Compiled
	w *variate.Workspace
	r *variate.Result
}

// session is the page's state in the engine: the kept runs, at most
// maxKept; keeping one more closes the oldest.
type session struct {
	kept     map[int]keptRun
	nextKeep int
}

const maxKept = 4

func (s *session) keep(c *variate.Compiled, w *variate.Workspace, r *variate.Result) int {
	if len(s.kept) >= maxKept {
		oldest := s.nextKeep
		// unordered: the minimum is taken
		for id := range s.kept {
			oldest = min(oldest, id)
		}
		s.kept[oldest].w.Close()
		delete(s.kept, oldest)
	}
	id := s.nextKeep
	s.nextKeep++
	s.kept[id] = keptRun{c, w, r}
	return id
}

// result returns either {ok: json} or {error: text} to JavaScript.
func result(text string, err error) js.Value {
	out := js.Global().Get("Object").New()
	if err != nil {
		out.Set("error", err.Error())
		return out
	}
	out.Set("ok", text)
	return out
}

func versionFunc(js.Value, []js.Value) any {
	info, ok := debug.ReadBuildInfo()
	if !ok || info.Main.Version == "" {
		return "devel"
	}
	return info.Main.Version
}

// load parses the model text and lays the settings over it: a JSON
// object of setting key to value, in file or flag spelling.
func load(args []js.Value) (*config.Model, error) {
	if len(args) < 1 {
		return nil, errors.New("run(model, settings): model text missing")
	}
	m, err := config.Parse([]byte(args[0].String()), "")
	if err != nil {
		return nil, err
	}
	if len(args) > 1 && !args[1].IsUndefined() && !args[1].IsNull() {
		keys := js.Global().Get("Object").Call("keys", args[1])
		for i := 0; i < keys.Length(); i++ {
			key := keys.Index(i).String()
			if err := config.SetSetting(&m.Config, key, args[1].Get(key).String()); err != nil {
				return nil, err
			}
		}
	}
	m.Config.Workers = 1
	// The page draws any node's entry count, so the node histograms are
	// always kept.
	m.Config.NodeHist = true
	// The browser cannot interrupt a running transform, so unless the
	// settings say otherwise its work budget is one that answers within
	// a minute or so rather than hours.
	if m.Config.Transform && m.Config.TransformWork == 0 {
		m.Config.TransformWork = browserTransformWork
	}
	return m, nil
}

const browserTransformWork = 1 << 29

func planFunc(_ js.Value, args []js.Value) any {
	m, err := load(args)
	if err != nil {
		return result("", err)
	}
	c, warnings, err := m.Graph.Compile(m.Config)
	if err != nil {
		return result("", err)
	}
	e, err := c.Analytic()
	if err != nil {
		return result("", err)
	}
	blocks := make([]report.BlockFigure, 0, len(m.Blocks))
	for _, b := range m.Blocks {
		blocks = append(blocks, report.BlockFigure{Source: b.Source, Rate: b.Rate, MTBF: b.MTBF, Unavailability: b.Unavailability})
	}
	var buf bytes.Buffer
	w := bufio.NewWriter(&buf)
	report.PlanJSON(w, c, e, m.Config, warnings, "", blocks)
	if err := w.Flush(); err != nil {
		return result("", err)
	}
	return result(buf.String(), nil)
}

// run runs the model and returns the report JSON, extended with a
// linear and a log-scaled histogram of the period total for drawing.
func (s *session) run(_ js.Value, args []js.Value) any {
	m, err := load(args)
	if err != nil {
		return result("", err)
	}
	c, warnings, err := m.Graph.Compile(m.Config)
	if err != nil {
		return result("", err)
	}
	w, err := c.NewWorkspace(m.Config)
	if err != nil {
		return result("", err)
	}
	r, err := w.Run(context.Background(), m.Config.Seed)
	if err != nil {
		w.Close()
		return result("", err)
	}
	rep, err := report.New(c, w, r, true)
	if err != nil {
		w.Close()
		return result("", err)
	}
	top := histogramTop(rep.Sum)
	rep.Hist = report.LinearHistogram(w, top, histogramBands, c.LatticeUnit())
	rep.LogHist = report.LogHistogram(w, top/histogramDecades, top, histogramBands)
	rep.Warnings = warnings
	out := result(string(rep.JSON()), nil)
	out.Set("keep", s.keep(c, w, r))
	out.Set("blocks", blocksJSON(m.Blocks))
	out.Set("graph", structureJSON(c.Structure()))
	return out
}

// structureJSON describes the graph for drawing.
func structureJSON(st variate.Structure) string {
	var buf bytes.Buffer
	buf.WriteString("{\"nodes\":[")
	for i, n := range st.Nodes {
		if i > 0 {
			buf.WriteByte(',')
		}
		fmt.Fprintf(&buf, "{\"id\":%s,\"leaf\":%v,\"functions\":[", strconv.Quote(n.ID), n.Leaf)
		for j, f := range n.Functions {
			if j > 0 {
				buf.WriteByte(',')
			}
			fmt.Fprintf(&buf, "{\"name\":%s,\"category\":%s,\"per_time\":%v}", strconv.Quote(f.Name), strconv.Quote(f.Category), f.PerTime)
		}
		buf.WriteString("]}")
	}
	buf.WriteString("],\"outcomes\":[")
	for i, o := range st.Outcomes {
		if i > 0 {
			buf.WriteByte(',')
		}
		fmt.Fprintf(&buf, "{\"name\":%s,\"from\":%s,\"to\":%s,\"holding\":%v}", strconv.Quote(o.Name), strconv.Quote(o.From), strconv.Quote(o.To), o.Holding)
	}
	buf.WriteString("],\"sources\":[")
	for i, src := range st.Sources {
		if i > 0 {
			buf.WriteByte(',')
		}
		fmt.Fprintf(&buf, "{\"id\":%s,\"to\":%s}", strconv.Quote(src.ID), strconv.Quote(src.To))
	}
	buf.WriteString("]}")
	return buf.String()
}

// quantity(keep, kind, name, top, bands) returns a kept run's quantity
// (total, a category, a source, a node's entries or contribution): its
// statistics, histograms on bands up to top (a top of 0 takes a little
// past its P99) and exceedance curve, with the transform's bounds for a
// category or source when it has them.
func (s *session) quantity(_ js.Value, args []js.Value) any {
	if len(args) < 5 {
		return result("", errors.New("quantity(keep, kind, name, top, bands)"))
	}
	k, ok := s.kept[args[0].Int()]
	if !ok {
		return result("", errors.New("the run is no longer kept"))
	}
	kind, name := args[1].String(), args[2].String()
	row, ok := k.w.QuantityRow(kind, name)
	if !ok || !k.w.HasHistogram(row) {
		return result("", errors.New("no such quantity: "+kind+" "+name))
	}
	st := k.w.QuantityStat(row)
	top, bands := args[3].Float(), args[4].Int()
	if !(top > 0) {
		top = st.P99 * 1.05
		if !(top > 0) {
			top = math.Max(st.Max, 1)
		}
	}
	unit := 0.0
	if kind == "total" {
		unit = k.c.LatticeUnit()
	}
	if kind == "entries" {
		unit = 1
	}
	lin := report.LinearHistogramOf(k.w, row, top, bands, unit)
	log := report.LogHistogramOf(k.w, row, top/histogramDecades, top, bands)
	var buf bytes.Buffer
	buf.WriteString("{\"stat\":")
	statJSON(&buf, st)
	buf.WriteString(",\"histogram\":")
	histJSON(&buf, lin)
	buf.WriteString(",\"log_histogram\":")
	histJSON(&buf, log)
	if kind == "entries" {
		buf.WriteString(",\"counts\":")
		countsJSON(&buf, k.w, row, st.Max)
	}
	buf.WriteString(",\"exceedance\":[")
	k.exceedanceJSON(&buf, kind, name, row)
	buf.WriteString("]}")
	return result(buf.String(), nil)
}

// exceedanceJSON writes a kept run's exceedance curve for a quantity,
// with the transform's bounds for a category, source or layer when it
// has them.
func (k *keptRun) exceedanceJSON(buf *bytes.Buffer, kind, name string, row int) {
	bounds, hasBounds := k.quantityBounds(kind, name)
	layer, isLayer := k.layerOf(name)
	tr := k.c.Transformed()
	layerBounds := kind == "layer" && isLayer && tr != nil && tr.Applicable
	for i, p := range k.w.ExceedanceOf(row, nil) {
		if i > 0 {
			buf.WriteByte(',')
		}
		fmt.Fprintf(buf, "{\"x\":%s,\"p\":%s,\"stderr\":%s", num(p.X), num(p.P), num(p.StdErr))
		if hasBounds || layerBounds {
			var lo, hi float64
			if layerBounds {
				lo, hi = variate.LayerBoundsAt(tr.Total, layer, p.X)
			} else {
				lo, hi = report.BoundsAt(bounds, p.X)
			}
			fmt.Fprintf(buf, ",\"lower\":%s,\"upper\":%s", num(lo), num(hi))
		}
		buf.WriteByte('}')
	}
}

// countsJSON writes P(N = k) for k up to the largest count seen, from
// the entry histogram's exact buckets, zero included.
func countsJSON(buf *bytes.Buffer, w *variate.Workspace, row int, maxSeen float64) {
	top := int(math.Min(maxSeen, maxCountBuckets))
	edges := make([]float64, top+2)
	for k := range edges {
		edges[k] = float64(k) - 0.5
	}
	mass := w.BandsOf(row, edges, make([]float64, top+1))
	buf.WriteString("{\"p\":[")
	for k, p := range mass {
		if k > 0 {
			buf.WriteByte(',')
		}
		buf.WriteString(num(p))
	}
	buf.WriteString("]}")
}

const maxCountBuckets = 400

// quantityBounds is the transform's bracket on a category or source,
// when the transform ran and kept it.
func (k keptRun) quantityBounds(kind, name string) (variate.Bounds, bool) {
	tr := k.c.Transformed()
	if tr == nil || !tr.Applicable {
		return variate.Bounds{}, false
	}
	switch kind {
	case "total":
		return tr.Total, true
	case "category":
		for c, n := range k.c.Categories() {
			if n == name && c < len(tr.ByCategory) {
				return tr.ByCategory[c], true
			}
		}
	case "source":
		for i, n := range k.c.SourceIDs() {
			if n == name && i < len(tr.BySource) {
				return tr.BySource[i], true
			}
		}
	}
	return variate.Bounds{}, false
}

// layerOf is the model's layer by name.
func (k keptRun) layerOf(name string) (variate.Layer, bool) {
	for _, y := range k.c.Layers() {
		if y.ID == name {
			return y, true
		}
	}
	return variate.Layer{}, false
}

// statJSON writes a quantity's statistics; the percentiles only where
// the quantity has a histogram.
func statJSON(buf *bytes.Buffer, st variate.Stat) {
	fmt.Fprintf(buf, "{\"n\":%d,\"mean\":%s,\"stderr\":%s,\"sd\":%s,\"min\":%s,\"max\":%s,",
		st.N, num(st.Mean), num(st.StdErr), num(st.StdDev), num(st.Min), num(st.Max))
	if st.Quantiles {
		fmt.Fprintf(buf, "\"p50\":%s,\"p90\":%s,\"p95\":%s,\"p99\":%s,\"p999\":%s,",
			num(st.P50), num(st.P90), num(st.P95), num(st.P99), num(st.P999))
	}
	fmt.Fprintf(buf, "\"share\":%s}", num(st.Share))
}

// series(keep) returns a kept run's period totals when the run kept
// its series, for a chart over periods; up to maxSeriesPoints.
func (s *session) series(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return result("", errors.New("series(keep)"))
	}
	k, ok := s.kept[args[0].Int()]
	if !ok {
		return result("", errors.New("the run is no longer kept"))
	}
	if k.r.Series == nil {
		return result("", errors.New("the run kept no series; set series: true"))
	}
	n := k.r.Series.Periods()
	if n > maxSeriesPoints {
		return result("", errors.New("the series has "+strconv.FormatInt(n, 10)+" periods, more than the page draws"))
	}
	totals := k.r.Series.TotalColumn(make([]float64, n))
	var buf bytes.Buffer
	buf.WriteString("{\"total\":[")
	for i, x := range totals {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.WriteString(num(x))
	}
	buf.WriteString("]}")
	return result(buf.String(), nil)
}

const maxSeriesPoints = 400000

// blocksJSON lists the sources whose rates came from reliability
// blocks, with the derived rate, system MTBF and unavailability.
func blocksJSON(blocks []config.Block) string {
	var buf bytes.Buffer
	buf.WriteByte('[')
	for i, b := range blocks {
		if i > 0 {
			buf.WriteByte(',')
		}
		fmt.Fprintf(&buf, "{\"source\":%s,\"rate\":%s,\"mtbf\":%s,\"unavailability\":%s}", strconv.Quote(b.Source), num(b.Rate), num(b.MTBF), num(b.Unavailability))
	}
	buf.WriteByte(']')
	return buf.String()
}

// histogram(keep, top, bands) returns the kept run's linear and
// log-scaled histograms on bands up to top, as JSON, so two runs can be
// drawn on the same bands.
func (s *session) histogram(_ js.Value, args []js.Value) any {
	if len(args) < 3 {
		return result("", errors.New("histogram(keep, top, bands)"))
	}
	k, ok := s.kept[args[0].Int()]
	if !ok {
		return result("", errors.New("the run is no longer kept"))
	}
	top, bands := args[1].Float(), args[2].Int()
	lin := report.LinearHistogram(k.w, top, bands, k.c.LatticeUnit())
	log := report.LogHistogram(k.w, top/histogramDecades, top, bands)
	if lin == nil || log == nil {
		return result("", errors.New("histogram: top must be positive and bands at least one"))
	}
	var buf bytes.Buffer
	buf.WriteString("{\"histogram\":")
	histJSON(&buf, lin)
	buf.WriteString(",\"log_histogram\":")
	histJSON(&buf, log)
	buf.WriteByte('}')
	return result(buf.String(), nil)
}

func histJSON(buf *bytes.Buffer, h *report.Histogram) {
	if h == nil {
		buf.WriteString("null")
		return
	}
	buf.WriteString("{\"edges\":[")
	for i, x := range h.Edges {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.WriteString(num(x))
	}
	buf.WriteString("],\"mass\":[")
	for i, x := range h.Mass {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.WriteString(num(x))
	}
	buf.WriteString("]}")
}

// release(keep) closes a kept run.
func (s *session) release(_ js.Value, args []js.Value) any {
	if len(args) < 1 {
		return nil
	}
	if k, ok := s.kept[args[0].Int()]; ok {
		k.w.Close()
		delete(s.kept, args[0].Int())
	}
	return nil
}

// histogramBands is the number of bands drawn; the chart runs to a
// little past the 99th percentile, and its log-scaled form starts
// histogramDecades below that, or where the mass starts if later.
const (
	histogramBands   = 50
	histogramDecades = 1e4
)

func histogramTop(sum *variate.Summary) float64 {
	if sum.Total.P99 > 0 {
		return sum.Total.P99 * 1.05
	}
	return math.Max(sum.Total.Max, 1)
}

func num(x float64) string {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return "null"
	}
	return strconv.FormatFloat(x, 'g', -1, 64)
}
