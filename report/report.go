// Package report writes a run's summary as a table, JSON or CSV from the
// typed values, by hand. The CLI, the browser build and the language
// bindings all produce their output through it, so the JSON shape is
// one and the same everywhere.
package report

import (
	"bufio"
	"bytes"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"code.eitanity.com/variate"
)

// Report is everything the summary writers read.
type Report struct {
	C        *variate.Compiled
	R        *variate.Result
	Sum      *variate.Summary
	Analytic bool // include the exact expectations and the transform
	NodeHist bool // entries per node have histograms
	E        *variate.Expectation
	Tr       *variate.Transformed
	Hist     *Histogram        // linear bands of the period total, for drawing; nil = none
	LogHist  *Histogram        // log-spaced bands of the same, for drawing; nil = none
	Warnings []variate.Warning // compile warnings to include in the JSON
}

// New assembles the report of a finished run from its workspace: the
// summary, and with analytic the exact expectations and the transform
// computed at compile. Every entry point builds its report here.
func New(c *variate.Compiled, w *variate.Workspace, r *variate.Result, analytic bool) (*Report, error) {
	sum := &variate.Summary{}
	w.Summarize(sum)
	rep := &Report{C: c, R: r, Sum: sum, Analytic: analytic, NodeHist: w.NodeHist()}
	if analytic {
		e, err := c.Analytic()
		if err != nil {
			return nil, fmt.Errorf("analytic: %w", err)
		}
		rep.E = e
		rep.Tr = c.Transformed()
	}
	return rep, nil
}

// JSON renders the report as one JSON document.
func (rep *Report) JSON() []byte {
	var buf bytes.Buffer
	w := bufio.NewWriter(&buf)
	rep.WriteJSON(w)
	_ = w.Flush() // a bytes.Buffer never fails
	return buf.Bytes()
}

// Histogram is a band chart of the period total: the mass between
// consecutive edges, with the mass beyond the last edge as a final band.
type Histogram struct {
	Edges []float64
	Mass  []float64
}

// LinearHistogram builds a histogram of n bands from 0 to top from the
// workspace's streaming histogram, plus the open band beyond top. The
// empty periods are in no band: they are reported as a share of their
// own, and in a rare-event model they would dwarf every other band. A
// positive unit says the total is a multiple of it; the bands are then
// the same whole number of multiples each, with edges halfway between
// multiples, so a chart of a lattice does not ripple with the number
// of atoms a band happens to catch.
func LinearHistogram(w *variate.Workspace, top float64, n int, unit float64) *Histogram {
	return LinearHistogramOf(w, 0, top, n, unit)
}

// LinearHistogramOf is LinearHistogram for the quantity at a row of the
// workspace (Workspace.QuantityRow).
func LinearHistogramOf(w *variate.Workspace, row int, top float64, n int, unit float64) *Histogram {
	if !(top > 0) || n < 1 || !w.HasHistogram(row) {
		return nil
	}
	// A band from 0 to the smallest positive value holds the exact zeros.
	edges := make([]float64, n+3)
	edges[1] = math.SmallestNonzeroFloat64
	if unit > 0 {
		per := max(1, int(math.Ceil(top/float64(n)/unit)))
		for i := 1; i <= n; i++ {
			edges[i+1] = (float64(i*per) + 0.5) * unit
		}
	} else {
		for i := 1; i <= n; i++ {
			edges[i+1] = top * float64(i) / float64(n)
		}
	}
	edges[n+2] = math.Inf(1)
	mass := w.BandsOf(row, edges, make([]float64, n+2))
	edges[1] = 0
	return &Histogram{Edges: edges[1 : n+2], Mass: mass[1:]}
}

// LogHistogram builds a histogram of n bands with edges spaced equally
// in the logarithm from lo to top, plus the open band beyond top, from
// the workspace's streaming histogram. Leading bands with less than
// minMass are dropped, so the chart starts where the mass does. Periods
// with a total below lo, including the empty ones, are in no band.
func LogHistogram(w *variate.Workspace, lo, top float64, n int) *Histogram {
	return LogHistogramOf(w, 0, lo, top, n)
}

// LogHistogramOf is LogHistogram for the quantity at a row of the
// workspace.
func LogHistogramOf(w *variate.Workspace, row int, lo, top float64, n int) *Histogram {
	if !(lo > 0) || !(top > lo) || n < 1 || !w.HasHistogram(row) {
		return nil
	}
	edges := make([]float64, n+2)
	ratio := math.Log(top / lo)
	for i := 0; i <= n; i++ {
		edges[i] = lo * math.Exp(ratio*float64(i)/float64(n))
	}
	edges[n+1] = math.Inf(1)
	mass := w.BandsOf(row, edges, make([]float64, n+1))
	first := 0
	for first < n && mass[first] < minMass {
		first++
	}
	return &Histogram{Edges: edges[first : n+1], Mass: mass[first:]}
}

// minMass is the share below which a leading log band is left off.
const minMass = 1e-4

// jsonSeries writes the per-period columns when the run kept them:
// "total", "category:<name>" and "source:<name>".
func (rep *Report) jsonSeries(j *jw) {
	s := rep.R.Series
	if s == nil {
		return
	}
	n := int(s.Periods())
	col := make([]float64, n)
	j.key("series")
	j.open('{')
	j.nums("total", s.TotalColumn(col))
	for c, name := range rep.C.Categories() {
		j.nums("category:"+name, s.CategoryColumn(c, col))
	}
	for k, name := range rep.C.SourceIDs() {
		for p := range col {
			col[p] = s.Source(k, int64(p))
		}
		j.nums("source:"+name, col)
	}
	for i, y := range rep.C.Layers() {
		j.nums("layer:"+y.ID, s.LayerColumn(i, col))
	}
	j.closeC('}')
}

// BlockFigure is a reliability block's stationary figures, for the plan.
type BlockFigure struct {
	Source                     string
	Rate, MTBF, Unavailability float64
}

// PlanJSON writes the convergence plan as JSON: the expected work, the
// exact mean, the warnings, the criteria rows, the periods the plan
// runs and what binds them, an estimate of the time, and the blocks.
func PlanJSON(w *bufio.Writer, c *variate.Compiled, e *variate.Expectation, cfg variate.Config, warnings []variate.Warning, estimate string, blocks []BlockFigure) {
	j := &jw{w: w}
	j.open('{')
	j.field("rounds_per_period", e.RoundsPerPeriod)
	j.field("steps_per_period", e.StepsPerPeriod)
	j.field("arrivals_per_period", e.ArrivalsPerPeriod)
	j.field("mean", e.Mean)
	j.key("warnings")
	j.open('[')
	for _, wn := range warnings {
		j.sep()
		j.str(wn.Code + " " + wn.Subject + ": " + wn.Message)
	}
	j.closeC(']')
	j.key("plan")
	j.open('[')
	var need int64
	binding := ""
	for _, row := range c.Plan() {
		j.sep()
		j.open('{')
		j.fieldStr("criterion", row.Criterion)
		j.fieldStr("driver", row.Driver)
		j.field("per_period", row.PlannedPerPeriod)
		j.fieldInt("periods", row.PeriodsNeeded)
		j.closeC('}')
		if row.PeriodsNeeded > need {
			need, binding = row.PeriodsNeeded, row.Criterion+" "+row.Driver
		}
	}
	j.closeC(']')
	j.fieldStr("binding", binding)
	j.fieldInt("periods", c.PlanPeriods(int64(max(cfg.BlockPeriods, 1))))
	j.fieldStr("estimated_time", estimate)
	j.key("blocks")
	j.open('[')
	for _, b := range blocks {
		j.sep()
		j.open('{')
		j.fieldStr("source", b.Source)
		j.field("rate", b.Rate)
		j.field("mtbf", b.MTBF)
		j.field("unavailability", b.Unavailability)
		j.closeC('}')
	}
	j.closeC(']')
	j.closeC('}')
	j.w.WriteByte('\n')
}

// jsonHistogram writes a drawing histogram under key, when there is one.
func jsonHistogram(j *jw, key string, h *Histogram) {
	if h == nil {
		return
	}
	j.key(key)
	j.open('{')
	j.nums("edges", h.Edges)
	j.nums("mass", h.Mass)
	j.closeC('}')
}

// ---- table ----

// WriteTable writes the human-readable report.
func (rep *Report) WriteTable(w *bufio.Writer) {
	r := rep.R
	fmt.Fprintf(w, "seed %d  periods %d  converged %v  binding %s", r.Seed, r.Periods, r.Converged, r.Binding)
	if r.Stopped {
		fmt.Fprint(w, "  stopped")
	}
	if len(r.Unmet) > 0 {
		fmt.Fprintf(w, "  unmet %s", strings.Join(r.Unmet, ","))
	}
	fmt.Fprintf(w, "\nsteps %d  beyond horizon %.6g  tail periods %d\n\n", r.Steps, r.BeyondHorizon, r.TailPeriods)
	fmt.Fprintf(w, "%-28s %12s %12s %12s %12s %12s %12s %12s %12s %8s\n", "quantity", "mean", "stderr", "sd", "p50", "p90", "p95", "p99", "p999", "share")
	statRow(w, "total", rep.Sum.Total, false, true)
	for _, name := range sortedKeys(rep.Sum.ByCategory) {
		statRow(w, "category "+name, rep.Sum.ByCategory[name], true, true)
	}
	for _, name := range sortedKeys(rep.Sum.BySource) {
		statRow(w, "source "+name, rep.Sum.BySource[name], true, true)
	}
	for _, name := range sortedKeys(rep.Sum.EntriesPer) {
		statRow(w, "entries "+name, rep.Sum.EntriesPer[name], false, rep.NodeHist)
	}
	for _, y := range rep.C.Layers() {
		statRow(w, "layer "+y.ID, rep.Sum.Layers[y.ID], true, true)
	}
	rep.writeReachTable(w)
	fmt.Fprintf(w, "\n%-12s %-32s %12s %12s %12s %5s\n", "criterion", "driver", "per period", "periods", "observed", "met")
	for _, row := range rep.Sum.Plan {
		fmt.Fprintf(w, "%-12s %-32s %12.6g %12s %12s %5v\n", row.Criterion, row.Driver, row.PlannedPerPeriod, planned(row.PeriodsNeeded), observed(row.Observed), row.Met)
	}
	rep.writeExceedanceTable(w)
	if rep.Analytic {
		rep.writeAnalyticTable(w)
	}
}

// planned is the periods a criterion needs, or "unknown" when the plan
// could not size it (the variance of a model with delays).
func planned(n int64) string {
	if n == 0 {
		return "unknown"
	}
	return strconv.FormatInt(n, 10)
}

// observed is a criterion's observed value, or "not yet" while the run
// cannot estimate it (a model with delays needs 16 full blocks).
func observed(x float64) string {
	if math.IsInf(x, 1) || math.IsNaN(x) {
		return "not yet"
	}
	return strconv.FormatFloat(x, 'g', 6, 64)
}

// statRow prints one quantity; quantiles only where a histogram exists.
func statRow(w *bufio.Writer, name string, s variate.Stat, share, quantiles bool) {
	fmt.Fprintf(w, "%-28s %12.6g %12.4g %12.6g", name, s.Mean, s.StdErr, s.StdDev)
	if quantiles {
		fmt.Fprintf(w, " %12.6g %12.6g %12.6g %12.6g %12.6g", s.P50, s.P90, s.P95, s.P99, s.P999)
	} else {
		fmt.Fprintf(w, " %12s %12s %12s %12s %12s", "-", "-", "-", "-", "-")
	}
	if share {
		fmt.Fprintf(w, " %8.4f", s.Share)
	}
	if s.Clipped {
		fmt.Fprint(w, "  (quantile clipped)")
	}
	fmt.Fprintln(w)
}

// writeReachTable lists, for each source in model order, the nodes its
// rounds reach in topological order with the share that get there.
func (rep *Report) writeReachTable(w *bufio.Writer) {
	fmt.Fprintf(w, "\n%-40s %12s %12s\n", "rounds from source reaching node", "share", "stderr")
	for _, src := range rep.C.SourceIDs() {
		nodes := rep.Sum.ReachProb[src]
		for _, node := range rep.C.NodeOrder() {
			p, ok := nodes[node]
			if !ok {
				continue
			}
			if math.IsNaN(p.P) || p.P == 0 {
				continue
			}
			fmt.Fprintf(w, "%-40s %12.6g %12.4g\n", src+" → "+node, p.P, p.StdErr)
		}
	}
	fmt.Fprintf(w, "\n%-40s %12s %12s\n", "outcome", "share", "stderr")
	for _, name := range sortedKeys(rep.Sum.OutcomeP) {
		p := rep.Sum.OutcomeP[name]
		if math.IsNaN(p.P) {
			continue
		}
		fmt.Fprintf(w, "%-40s %12.6g %12.4g\n", name, p.P, p.StdErr)
	}
}

func (rep *Report) writeExceedanceTable(w *bufio.Writer) {
	pts := rep.Sum.Exceedance
	if len(pts) == 0 {
		return
	}
	fmt.Fprintf(w, "\n%-16s %12s %12s", "x", "P(T ≥ x)", "stderr")
	if rep.bounds() {
		fmt.Fprintf(w, " %12s %12s", "lower", "upper")
	}
	fmt.Fprintln(w)
	stride := max(1, len(pts)/16)
	for i := 0; i < len(pts); i += stride {
		rep.exceedanceRow(w, pts[i], " ")
	}
}

func (rep *Report) exceedanceRow(w *bufio.Writer, p variate.ECPoint, sep string) {
	fmt.Fprintf(w, "%-16.6g%s%12.6g%s%12.4g", p.X, sep, p.P, sep, p.StdErr)
	if rep.bounds() {
		lo, hi := rep.boundsAt(p.X)
		fmt.Fprintf(w, "%s%12.6g%s%12.6g", sep, lo, sep, hi)
	}
	fmt.Fprintln(w)
}

func (rep *Report) bounds() bool { return rep.Tr != nil && rep.Tr.Applicable }

// boundsAt reads the transform's bracket on the total at x.
func (rep *Report) boundsAt(x float64) (lo, hi float64) { return BoundsAt(rep.Tr.Total, x) }

// BoundsAt reads a transform bracket at x from the lattice point at or
// above x, for both curves: a total of at least x rounds up to at least
// that point, and a total that rounds down to at least that point is
// at least x. Below the first point the upper bound is then the
// probability of a non-empty period, not one.
func BoundsAt(b variate.Bounds, x float64) (lo, hi float64) {
	k := min(int(math.Ceil(x/b.H)), len(b.Lower)-1)
	return b.Lower[k], b.Upper[k]
}

func (rep *Report) writeAnalyticTable(w *bufio.Writer) {
	e := rep.E
	fmt.Fprintf(w, "\nanalytic: mean %.6g  sd %.6g  P(T = 0) %.6g  draws per period %.6g\n", e.Mean, math.Sqrt(e.Var), e.PZero(), e.DrawsMean)
	cats := rep.C.Categories()
	for i, name := range cats {
		fmt.Fprintf(w, "  category %-20s mean %.6g\n", name, e.MeanByCategory[i])
	}
	for i, name := range rep.C.SourceIDs() {
		fmt.Fprintf(w, "  source %-22s mean %.6g\n", name, e.MeanBySource[i])
	}
	for i, name := range rep.C.NodeIDs() {
		fmt.Fprintf(w, "  entries %-21s %.6g\n", name, e.EntriesPer[i])
	}
	if rep.Tr == nil {
		return
	}
	if rep.Tr.Applicable {
		for _, y := range rep.C.Layers() {
			lo, hi := rep.Tr.Total.LayerMean(y)
			fmt.Fprintf(w, "  layer %-24s mean in [%.6g, %.6g]\n", y.ID, lo, hi)
		}
	}
	if !rep.Tr.Applicable {
		fmt.Fprintf(w, "transform: does not apply: %s\n", rep.Tr.Reason)
		return
	}
	fmt.Fprintf(w, "transform: span %.6g  points %d  tilt %.4g  P(T = 0) %.6g  quadrature nodes %d  passes %d\n", rep.Tr.H, rep.Tr.M, rep.Tr.Theta, rep.Tr.PZero, rep.Tr.QuadNodes, rep.Tr.Passes)
	for _, p := range []float64{0.5, 0.9, 0.95, 0.99, 0.999} {
		lo, hi := rep.Tr.Total.Quantile(p)
		fmt.Fprintf(w, "  p%-6g [%.6g, %.6g]\n", p*100, lo, hi)
	}
}

// ---- JSON ----

// jw writes JSON by hand from typed values.
type jw struct {
	w     *bufio.Writer
	first []bool // per open container, whether the next item is the first
}

func (j *jw) open(c byte) {
	j.w.WriteByte(c)
	j.first = append(j.first, true)
}

func (j *jw) closeC(c byte) {
	j.w.WriteByte(c)
	j.first = j.first[:len(j.first)-1]
}

func (j *jw) sep() {
	n := len(j.first) - 1
	if !j.first[n] {
		j.w.WriteByte(',')
	}
	j.first[n] = false
}

func (j *jw) key(k string) {
	j.sep()
	j.str(k)
	j.w.WriteByte(':')
}

func (j *jw) str(s string) {
	j.w.WriteByte('"')
	for _, r := range s {
		switch {
		case r == '"' || r == '\\':
			j.w.WriteByte('\\')
			j.w.WriteRune(r)
		case r < 0x20:
			fmt.Fprintf(j.w, "\\u%04x", r)
		default:
			j.w.WriteRune(r)
		}
	}
	j.w.WriteByte('"')
}

func (j *jw) num(x float64) {
	switch {
	case math.IsNaN(x) || math.IsInf(x, 0):
		j.w.WriteString("null")
	default:
		j.w.WriteString(strconv.FormatFloat(x, 'g', -1, 64))
	}
}

func (j *jw) field(k string, x float64) { j.key(k); j.num(x) }

func (j *jw) fieldInt(k string, x int64) { j.key(k); j.w.WriteString(strconv.FormatInt(x, 10)) }

func (j *jw) fieldUint(k string, x uint64) { j.key(k); j.w.WriteString(strconv.FormatUint(x, 10)) }

func (j *jw) fieldBool(k string, b bool) { j.key(k); j.w.WriteString(strconv.FormatBool(b)) }

func (j *jw) fieldStr(k, s string) { j.key(k); j.str(s) }

func (j *jw) stat(k string, s variate.Stat) {
	j.key(k)
	j.open('{')
	j.fieldInt("n", s.N)
	j.field("mean", s.Mean)
	j.field("stderr", s.StdErr)
	j.field("sd", s.StdDev)
	j.field("min", s.Min)
	j.field("max", s.Max)
	j.field("p50", s.P50)
	j.field("p90", s.P90)
	j.field("p95", s.P95)
	j.field("p99", s.P99)
	j.field("p999", s.P999)
	j.fieldBool("clipped", s.Clipped)
	j.field("share", s.Share)
	j.closeC('}')
}

func (j *jw) stats(k string, m map[string]variate.Stat) {
	j.key(k)
	j.open('{')
	for _, name := range sortedKeys(m) {
		j.stat(name, m[name])
	}
	j.closeC('}')
}

func (j *jw) props(k string, m map[string]variate.Prop) { j.propsIn(k, m, sortedKeys(m)) }

// propsIn writes the probabilities of m present in order, in that order.
func (j *jw) propsIn(k string, m map[string]variate.Prop, order []string) {
	j.key(k)
	j.open('{')
	for _, name := range order {
		p, ok := m[name]
		if !ok {
			continue
		}
		j.key(name)
		j.open('{')
		j.field("p", p.P)
		j.field("stderr", p.StdErr)
		j.closeC('}')
	}
	j.closeC('}')
}

// WriteJSON writes the report as one JSON object.
func (rep *Report) WriteJSON(w *bufio.Writer) {
	j := &jw{w: w}
	r := rep.R
	j.open('{')
	j.fieldUint("seed", r.Seed)
	j.fieldInt("periods", r.Periods)
	j.fieldBool("converged", r.Converged)
	j.fieldBool("stopped", r.Stopped)
	j.fieldStr("binding", r.Binding)
	j.key("unmet")
	j.open('[')
	for _, u := range r.Unmet {
		j.sep()
		j.str(u)
	}
	j.closeC(']')
	j.fieldUint("steps", r.Steps)
	j.field("beyond_horizon", r.BeyondHorizon)
	j.fieldInt("tail_periods", r.TailPeriods)
	j.stat("total", rep.Sum.Total)
	j.stats("by_category", rep.Sum.ByCategory)
	j.stats("by_source", rep.Sum.BySource)
	j.stats("entries_per", rep.Sum.EntriesPer)
	j.stats("contrib_per", rep.Sum.ContribPer)
	rep.jsonLayers(j)
	rep.jsonTailShares(j)
	j.key("reach")
	j.open('{')
	for _, src := range rep.C.SourceIDs() {
		j.propsIn(src, rep.Sum.ReachProb[src], rep.C.NodeOrder())
	}
	j.closeC('}')
	j.key("terminal")
	j.open('{')
	for _, src := range sortedKeys(rep.Sum.Terminal) {
		j.props(src, rep.Sum.Terminal[src])
	}
	j.closeC('}')
	j.props("outcomes", rep.Sum.OutcomeP)
	rep.jsonPlan(j)
	rep.jsonExceedance(j)
	if rep.Analytic {
		rep.jsonAnalytic(j)
	}
	jsonHistogram(j, "histogram", rep.Hist)
	jsonHistogram(j, "log_histogram", rep.LogHist)
	rep.jsonSeries(j)
	if rep.Warnings != nil {
		j.key("warnings")
		j.open('[')
		for _, wn := range rep.Warnings {
			j.sep()
			j.str(wn.Code + " " + wn.Subject + ": " + wn.Message)
		}
		j.closeC(']')
	}
	j.closeC('}')
	w.WriteByte('\n')
}

// nums writes a key with an array of numbers.
func (j *jw) nums(k string, xs []float64) {
	j.key(k)
	j.open('[')
	for _, x := range xs {
		j.sep()
		j.num(x)
	}
	j.closeC(']')
}

func (rep *Report) jsonPlan(j *jw) {
	j.key("plan")
	j.open('[')
	for _, row := range rep.Sum.Plan {
		j.sep()
		j.open('{')
		j.fieldStr("criterion", row.Criterion)
		j.fieldStr("driver", row.Driver)
		j.field("per_period", row.PlannedPerPeriod)
		j.fieldInt("periods", row.PeriodsNeeded)
		j.field("observed", row.Observed)
		j.fieldBool("met", row.Met)
		j.closeC('}')
	}
	j.closeC(']')
}

// jsonLayers writes each layer's statistics and, when the transform
// applies, the exact bounds on its mean.
func (rep *Report) jsonLayers(j *jw) {
	layers := rep.C.Layers()
	if len(layers) == 0 {
		return
	}
	j.stats("layers", rep.Sum.Layers)
	if !rep.bounds() {
		return
	}
	j.key("layer_bounds")
	j.open('{')
	for _, y := range layers {
		lo, hi := rep.Tr.Total.LayerMean(y)
		j.key(y.ID)
		j.open('{')
		j.field("mean_lower", lo)
		j.field("mean_upper", hi)
		j.closeC('}')
	}
	j.closeC('}')
}

// jsonTailShares writes who the tail belongs to at each threshold.
func (rep *Report) jsonTailShares(j *jw) {
	j.key("tail_shares")
	j.open('[')
	for _, ts := range rep.Sum.TailShares {
		j.sep()
		j.open('{')
		j.field("x", ts.X)
		j.fieldInt("periods", ts.Periods)
		j.key("by_source")
		j.open('{')
		for _, name := range sortedKeys(ts.BySource) {
			j.field(name, ts.BySource[name])
		}
		j.closeC('}')
		j.key("by_category")
		j.open('{')
		for _, name := range sortedKeys(ts.ByCategory) {
			j.field(name, ts.ByCategory[name])
		}
		j.closeC('}')
		j.closeC('}')
	}
	j.closeC(']')
}

func (rep *Report) jsonExceedance(j *jw) {
	j.key("exceedance")
	j.open('[')
	for _, p := range rep.Sum.Exceedance {
		j.sep()
		j.open('{')
		j.field("x", p.X)
		j.field("p", p.P)
		j.field("stderr", p.StdErr)
		if rep.bounds() {
			lo, hi := rep.boundsAt(p.X)
			j.field("lower", lo)
			j.field("upper", hi)
		}
		j.closeC('}')
	}
	j.closeC(']')
}

func (rep *Report) jsonAnalytic(j *jw) {
	e := rep.E
	j.key("analytic")
	j.open('{')
	j.field("mean", e.Mean)
	j.field("var", e.Var)
	j.field("pzero", e.PZero())
	j.field("draws_mean", e.DrawsMean)
	j.key("by_category")
	j.open('{')
	for i, name := range rep.C.Categories() {
		j.field(name, e.MeanByCategory[i])
	}
	j.closeC('}')
	j.key("by_source")
	j.open('{')
	for i, name := range rep.C.SourceIDs() {
		j.field(name, e.MeanBySource[i])
	}
	j.closeC('}')
	j.key("entries_per")
	j.open('{')
	for i, name := range rep.C.NodeIDs() {
		j.field(name, e.EntriesPer[i])
	}
	j.closeC('}')
	if rep.Tr != nil {
		j.key("transform")
		j.open('{')
		j.fieldBool("applicable", rep.Tr.Applicable)
		j.fieldStr("reason", rep.Tr.Reason)
		if rep.Tr.Applicable {
			j.field("h", rep.Tr.H)
			j.fieldInt("m", int64(rep.Tr.M))
			j.field("theta", rep.Tr.Theta)
			j.field("pzero", rep.Tr.PZero)
			j.fieldInt("quad_nodes", int64(rep.Tr.QuadNodes))
			j.fieldInt("passes", int64(rep.Tr.Passes))
			j.key("quantiles")
			j.open('{')
			for _, p := range []float64{0.5, 0.9, 0.95, 0.99, 0.999} {
				lo, hi := rep.Tr.Total.Quantile(p)
				j.key("p" + strconv.FormatFloat(p*100, 'g', -1, 64))
				j.open('[')
				j.sep()
				j.num(lo)
				j.sep()
				j.num(hi)
				j.closeC(']')
			}
			j.closeC('}')
		}
		j.closeC('}')
	}
	j.closeC('}')
}

// ---- CSV ----

// CSVField quotes a field when it needs it.
func CSVField(s string) string {
	if strings.ContainsAny(s, ",\"\n") {
		return `"` + strings.ReplaceAll(s, `"`, `""`) + `"`
	}
	return s
}

func csvNum(x float64) string { return strconv.FormatFloat(x, 'g', -1, 64) }

// WriteCSV writes one row per quantity.
func (rep *Report) WriteCSV(w *bufio.Writer) {
	fmt.Fprintln(w, "quantity,name,n,mean,stderr,sd,min,max,p50,p90,p95,p99,p999,clipped,share")
	csvStat(w, "total", "", rep.Sum.Total)
	for _, name := range sortedKeys(rep.Sum.ByCategory) {
		csvStat(w, "category", name, rep.Sum.ByCategory[name])
	}
	for _, name := range sortedKeys(rep.Sum.BySource) {
		csvStat(w, "source", name, rep.Sum.BySource[name])
	}
	for _, name := range sortedKeys(rep.Sum.EntriesPer) {
		csvStat(w, "entries", name, rep.Sum.EntriesPer[name])
	}
	if rep.Analytic {
		e := rep.E
		fmt.Fprintf(w, "analytic,total,,%s,,%s,,,,,,,,,\n", csvNum(e.Mean), csvNum(math.Sqrt(e.Var)))
		for i, name := range rep.C.Categories() {
			fmt.Fprintf(w, "analytic category,%s,,%s,,,,,,,,,,,\n", CSVField(name), csvNum(e.MeanByCategory[i]))
		}
		for i, name := range rep.C.SourceIDs() {
			fmt.Fprintf(w, "analytic source,%s,,%s,,,,,,,,,,,\n", CSVField(name), csvNum(e.MeanBySource[i]))
		}
	}
}

func csvStat(w *bufio.Writer, kind, name string, s variate.Stat) {
	fmt.Fprintf(w, "%s,%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%v,%s\n", kind, CSVField(name), s.N,
		csvNum(s.Mean), csvNum(s.StdErr), csvNum(s.StdDev), csvNum(s.Min), csvNum(s.Max),
		csvNum(s.P50), csvNum(s.P90), csvNum(s.P95), csvNum(s.P99), csvNum(s.P999), s.Clipped, csvNum(s.Share))
}

// WriteExceedanceFile writes the curve as CSV: x, p, stderr, and the
// transform's lower and upper bounds when they exist.
func (rep *Report) WriteExceedanceFile(path string) error {
	f, err := os.Create(path) //nolint:gosec // the output file the user named
	if err != nil {
		return fmt.Errorf("exceedance file: %w", err)
	}
	w := bufio.NewWriter(f)
	fmt.Fprint(w, "x,p,stderr")
	if rep.bounds() {
		fmt.Fprint(w, ",lower,upper")
	}
	fmt.Fprintln(w)
	for _, p := range rep.Sum.Exceedance {
		fmt.Fprintf(w, "%s,%s,%s", csvNum(p.X), csvNum(p.P), csvNum(p.StdErr))
		if rep.bounds() {
			lo, hi := rep.boundsAt(p.X)
			fmt.Fprintf(w, ",%s,%s", csvNum(lo), csvNum(hi))
		}
		fmt.Fprintln(w)
	}
	if err := flush(w); err != nil {
		return err
	}
	return closeFile(f)
}

func flush(w *bufio.Writer) error {
	if err := w.Flush(); err != nil {
		return fmt.Errorf("write output: %w", err)
	}
	return nil
}

func closeFile(f *os.File) error {
	if err := f.Close(); err != nil {
		return fmt.Errorf("close %s: %w", f.Name(), err)
	}
	return nil
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	// unordered: the keys are sorted before use
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
