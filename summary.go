package variate

import (
	"math"

	"code.eitanity.com/variate/stats"
)

// Stat summarises one per-period quantity.
type Stat struct {
	N                        int64
	Mean, StdErr, StdDev     float64
	Min, Max                 float64
	P50, P90, P95, P99, P999 float64
	Quantiles                bool // the percentiles exist: the quantity has a histogram
	Clipped                  bool
	Share                    float64

	// Per level of QuantileLevels(): the quantile's 95% interval, from the
	// order statistics, and its standard error, the interval's width over
	// 2·1.96. The interval is asymmetric in a skewed tail; quote it rather
	// than the quantile ± 1.96 standard errors. NaN in a model with
	// delays, whose correlated periods widen the interval past what
	// independent periods give.
	QLo, QHi, QSE [5]float64
}

// QuantileLevels returns the levels of P50, P90, P95, P99 and P999.
func QuantileLevels() [5]float64 { return [5]float64{0.5, 0.9, 0.95, 0.99, 0.999} }

// quantileZ is the normal quantile of the two-sided 95% interval.
const quantileZ = 1.959963984540054

// Prop is a proportion with its binomial standard error.
type Prop struct{ P, StdErr float64 }

// ECPoint is one point of the exceedance curve: P(T ≥ X) with its
// binomial standard error.
type ECPoint struct{ X, P, StdErr float64 }

// PlanRow is one convergence criterion: planned against observed.
type PlanRow struct {
	Criterion        string
	Driver           string
	PlannedPerPeriod float64
	PeriodsNeeded    int64
	Observed         float64
	Met              bool
}

// Summary is the report of a run, with names as keys.
type Summary struct {
	Total      Stat
	Exceedance []ECPoint
	ByCategory map[string]Stat // empty, as are BySource, ContribPer and TailShares, with Config.NoBreakdown
	BySource   map[string]Stat

	ReachProb  map[string]map[string]Prop
	Terminal   map[string]map[string]Prop
	OutcomeP   map[string]Prop
	EntriesPer map[string]Stat
	ContribPer map[string]Stat // value booked per period at each node
	Layers     map[string]Stat // each layer's payout per period, its share that of the total's mean
	Triggers   map[string]Stat // by the trigger's source name: the sum it tests per period
	TailShares []TailShare     // who the tail belongs to, at each threshold with a period
	Plan       []PlanRow
}

// exceedancePoints is the number of log-spaced points of the curve.
const exceedancePoints = 64

// Summarize fills a caller-owned Summary from the workspace's result,
// reusing its slices and maps.
func (w *Workspace) Summarize(dst *Summary) {
	r := &w.result
	dst.Total = w.stat(&r.Total, 1)
	dst.Exceedance = w.exceedance(dst.Exceedance[:0])
	breakdown := w.m.crit.breakdown
	dst.ByCategory = ensureStats(dst.ByCategory)
	dst.BySource = ensureStats(dst.BySource)
	dst.EntriesPer = ensureStats(dst.EntriesPer)
	dst.ContribPer = ensureStats(dst.ContribPer)
	clear(dst.ByCategory)
	clear(dst.BySource)
	clear(dst.ContribPer)
	for c, name := range r.Categories {
		if breakdown {
			dst.ByCategory[name] = w.stat(&r.ByCategory[c], r.Total.Mean())
		}
	}
	for k, name := range r.Sources {
		if breakdown {
			dst.BySource[name] = w.stat(&r.BySource[k], r.Total.Mean())
		}
	}
	for v, name := range r.Nodes {
		dst.EntriesPer[name] = w.stat(&r.EntriesPer[v], 1)
		if breakdown {
			dst.ContribPer[name] = w.stat(&r.Contrib[v], r.Total.Mean())
		}
	}
	dst.Layers = ensureStats(dst.Layers)
	for j, y := range w.m.layers {
		dst.Layers[y.ID] = w.stat(&r.Layers[j], r.Total.Mean())
	}
	dst.Triggers = ensureStats(dst.Triggers)
	for i, tr := range w.m.trigs {
		dst.Triggers[r.Sources[tr.src]] = w.stat(&r.Triggers[i], 1)
	}
	dst.TailShares = w.tailShares(dst.TailShares)
	w.summarizeReach(dst)
	dst.Plan = append(dst.Plan[:0], w.rows...)
}

func ensureStats(m map[string]Stat) map[string]Stat {
	if m == nil {
		return map[string]Stat{}
	}
	return m
}

// stat converts an Accum; share is Mean/total for breakdowns.
func (w *Workspace) stat(a *Accum, total float64) Stat {
	s := Stat{N: a.N(), Mean: a.Mean(), StdErr: a.StdErr(w.delays), StdDev: a.StdDev(), Min: a.Min, Max: a.Max}
	if total != 0 && total != 1 {
		s.Share = a.Mean() / total
	}
	if a.Hist != nil {
		s.Quantiles = true
		var q [5]float64
		for i, p := range QuantileLevels() {
			var c, ci bool
			q[i], c = a.Hist.Quantile(p)
			s.Clipped = s.Clipped || c
			if w.delays {
				s.QLo[i], s.QHi[i], s.QSE[i] = math.NaN(), math.NaN(), math.NaN()
				continue
			}
			s.QLo[i], s.QHi[i], ci = a.Hist.QuantileInterval(p, quantileZ)
			s.QSE[i] = (s.QHi[i] - s.QLo[i]) / (2 * quantileZ)
			s.Clipped = s.Clipped || ci
		}
		s.P50, s.P90, s.P95, s.P99, s.P999 = q[0], q[1], q[2], q[3], q[4]
	}
	return s
}

// exceedance returns log-spaced points of P(T ≥ x) from the total's
// histogram, from the smallest non-zero total to the largest.
func (w *Workspace) exceedance(dst []ECPoint) []ECPoint {
	return w.exceedanceOf(&w.result.Total, dst)
}

// exceedanceOf is the exceedance curve of any accumulated quantity.
func (w *Workspace) exceedanceOf(a *Accum, dst []ECPoint) []ECPoint {
	h := a.Hist
	n := float64(h.N)
	if n == 0 || !(a.Max > 0) {
		return dst
	}
	// The histogram resolves nothing below its own start, so the curve
	// begins there or at the smallest value seen, whichever is higher.
	hl, _ := h.Range()
	lo := math.Max(a.Min, hl)
	if lo > a.Max {
		// An integer histogram's log range starts above its counts; the
		// curve then starts at one.
		lo = math.Min(1, a.Max)
	}
	hi := a.Max
	if hi <= lo {
		return append(dst, ecPoint(h, lo))
	}
	step := math.Log(hi/lo) / float64(exceedancePoints-1)
	// bound: exceedancePoints
	for i := 0; i < exceedancePoints; i++ {
		dst = append(dst, ecPoint(h, lo*math.Exp(float64(i)*step)))
	}
	return dst
}

func ecPoint(h *stats.LogHist, x float64) ECPoint {
	n := float64(h.N)
	p := h.Exceedance(x)
	return ECPoint{X: x, P: p, StdErr: math.Sqrt(p * (1 - p) / n)}
}

// summarizeReach fills the proportions: reach and terminal per source and
// node, and the share of a node's entries taking each outcome.
func (w *Workspace) summarizeReach(dst *Summary) {
	r := &w.result
	m := w.m
	if dst.ReachProb == nil {
		dst.ReachProb = map[string]map[string]Prop{}
		dst.Terminal = map[string]map[string]Prop{}
	}
	if dst.OutcomeP == nil {
		dst.OutcomeP = map[string]Prop{}
	}
	for k, src := range r.Sources {
		reach, ok := dst.ReachProb[src]
		if !ok {
			reach = map[string]Prop{}
			dst.ReachProb[src] = reach
		}
		term, ok := dst.Terminal[src]
		if !ok {
			term = map[string]Prop{}
			dst.Terminal[src] = term
		}
		for v, node := range r.Nodes {
			reach[node] = prop(r.Reach[k][v], r.Events[k])
			if m.nodes[v].outStart == m.nodes[v].outEnd {
				term[node] = prop(r.Reach[k][v], r.Events[k])
			}
		}
	}
	for j, name := range r.Outcomes {
		v := m.outcomeNode[j]
		var taken, entries uint64
		for k := range r.Sources {
			taken += r.OutcomeCount[k][j]
			entries += r.Reach[k][v]
		}
		dst.OutcomeP[name] = prop(taken, entries)
	}
}

func prop(count, n uint64) Prop {
	if n == 0 {
		return Prop{P: math.NaN(), StdErr: math.NaN()}
	}
	p := float64(count) / float64(n)
	return Prop{P: p, StdErr: math.Sqrt(p * (1 - p) / float64(n))}
}
