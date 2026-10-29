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
	Clipped                  bool
	Share                    float64
}

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
	ByCategory map[string]Stat
	BySource   map[string]Stat

	ReachProb  map[string]map[string]Prop
	Terminal   map[string]map[string]Prop
	OutcomeP   map[string]Prop
	EntriesPer map[string]Stat
	ContribPer map[string]Stat // value booked per period at each node
	Layers     map[string]Stat // each layer's payout per period, its share that of the total's mean
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
	dst.ByCategory = ensureStats(dst.ByCategory)
	for c, name := range r.Categories {
		dst.ByCategory[name] = w.stat(&r.ByCategory[c], r.Total.Mean())
	}
	dst.BySource = ensureStats(dst.BySource)
	for k, name := range r.Sources {
		dst.BySource[name] = w.stat(&r.BySource[k], r.Total.Mean())
	}
	dst.EntriesPer = ensureStats(dst.EntriesPer)
	dst.ContribPer = ensureStats(dst.ContribPer)
	for v, name := range r.Nodes {
		dst.EntriesPer[name] = w.stat(&r.EntriesPer[v], 1)
		dst.ContribPer[name] = w.stat(&r.Contrib[v], r.Total.Mean())
	}
	dst.Layers = ensureStats(dst.Layers)
	for j, y := range w.m.layers {
		dst.Layers[y.ID] = w.stat(&r.Layers[j], r.Total.Mean())
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
		var c bool
		s.P50, c = a.Hist.Quantile(0.5)
		s.Clipped = s.Clipped || c
		s.P90, c = a.Hist.Quantile(0.9)
		s.Clipped = s.Clipped || c
		s.P95, c = a.Hist.Quantile(0.95)
		s.Clipped = s.Clipped || c
		s.P99, c = a.Hist.Quantile(0.99)
		s.Clipped = s.Clipped || c
		s.P999, c = a.Hist.Quantile(0.999)
		s.Clipped = s.Clipped || c
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
