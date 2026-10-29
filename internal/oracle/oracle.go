// Package oracle is a deliberately naive reference walker, used only by
// tests to check the engine's counts and totals on random graphs. It
// shares no code with the engine: no pools, no flattening, no workers. It
// draws one value at a time from each distribution with its own generator.
package oracle

import (
	"math"
	"math/rand/v2"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// Result holds per-period counts and totals.
type Result struct {
	Nodes      []string
	Sources    []string
	Categories []string
	Events     []uint64    // [source]
	Reach      [][]uint64  // [source][node]
	Entries    [][]float64 // [node][period]
	Total      []float64   // [period]
	ByCategory [][]float64 // [category][period]
	Beyond     float64     // value booked at or after the horizon
}

type walker struct {
	g       variate.Graph
	r       *rand.Rand
	periods int
	nodeIx  map[string]int
	catIx   map[string]int
	period  map[dist.Distribution]float64 // this period's draws of period-scoped pdfs
	res     *Result
	buf     [1]float64
}

// Run simulates periods [−burnIn, periods) and books into [0, periods).
func Run(g variate.Graph, periods, burnIn int, seed uint64) *Result {
	w := &walker{g: g, r: rand.New(rand.NewPCG(seed, 0x6f72616365)), periods: periods, nodeIx: map[string]int{}, catIx: map[string]int{}}
	w.res = &Result{Nodes: make([]string, len(g.Nodes)), Sources: make([]string, len(g.Sources))}
	for i, n := range g.Nodes {
		w.nodeIx[n.ID] = i
		w.res.Nodes[i] = n.ID
		for _, f := range n.Functions {
			if _, ok := w.catIx[f.Category]; !ok {
				w.catIx[f.Category] = len(w.res.Categories)
				w.res.Categories = append(w.res.Categories, f.Category)
			}
		}
	}
	w.res.Events = make([]uint64, len(g.Sources))
	w.res.Reach = make([][]uint64, len(g.Sources))
	w.res.Entries = make([][]float64, len(g.Nodes))
	for i := range w.res.Entries {
		w.res.Entries[i] = make([]float64, periods)
	}
	w.res.Total = make([]float64, periods)
	w.res.ByCategory = make([][]float64, len(w.res.Categories))
	for i := range w.res.ByCategory {
		w.res.ByCategory[i] = make([]float64, periods)
	}
	for k, s := range g.Sources {
		w.res.Sources[k] = s.ID
		w.res.Reach[k] = make([]uint64, len(g.Nodes))
	}
	// bound: periods + burnIn
	for p := -burnIn; p < periods; p++ {
		w.runPeriod(p)
	}
	return w.res
}

// draw takes one value, from this period's draw for a period-scoped pdf.
func (w *walker) draw(d dist.Distribution, scope variate.Scope) float64 {
	if scope == variate.PerPeriod {
		if v, ok := w.period[d]; ok {
			return v
		}
		d.Fill(w.r, w.buf[:])
		w.period[d] = w.buf[0]
		return w.buf[0]
	}
	d.Fill(w.r, w.buf[:])
	return w.buf[0]
}

func (w *walker) runPeriod(p int) {
	w.period = map[dist.Distribution]float64{}
	for k, s := range w.g.Sources {
		lam := w.draw(s.Frequency, variate.PerPeriod)
		n := dist.PoissonCount(w.r, lam)
		// bound: the Poisson count
		for i := uint64(0); i < n; i++ {
			if p >= 0 {
				w.res.Events[k]++
			}
			w.walk(k, w.nodeIx[s.To], float64(p)+w.r.Float64())
		}
	}
}

// walk follows one event from node v at time t to a leaf.
func (w *walker) walk(src, v int, t float64) {
	// bound: a path visits each node at most once
	for step := 0; step <= len(w.g.Nodes); step++ {
		n := w.g.Nodes[v]
		if t >= 0 {
			w.res.Reach[src][v]++
		}
		w.count(v, t)
		for _, f := range n.Functions {
			if f.Basis == variate.OnEntry {
				w.book(f, w.draw(f.Model, f.Scope), t)
			}
		}
		next, hold, ok := w.choose(n)
		if !ok {
			return
		}
		for _, f := range n.Functions {
			if f.Basis == variate.PerTime {
				w.bookRate(f, w.draw(f.Model, f.Scope), t, t+hold)
			}
		}
		v, t = next, t+hold
	}
}

// choose selects the next node and its holding time; ok is false at a leaf.
func (w *walker) choose(n variate.Node) (next int, hold float64, ok bool) {
	if n.Next != "" {
		return w.nodeIx[n.Next], w.holding(n.NextHolding, n.NextScope), true
	}
	if len(n.Outcomes) == 0 {
		return 0, 0, false
	}
	var o variate.Outcome
	if n.Mode == variate.Sequential {
		o = n.Outcomes[len(n.Outcomes)-1]
		for _, cand := range n.Outcomes[:len(n.Outcomes)-1] {
			if w.r.Float64() < w.draw(cand.P, variate.PerPeriod) {
				o = cand
				break
			}
		}
	} else {
		o = w.chooseAbsolute(n)
	}
	return w.nodeIx[o.To], w.holding(o.Holding, o.HoldingScope), true
}

func (w *walker) chooseAbsolute(n variate.Node) variate.Outcome {
	u := w.r.Float64()
	cum := 0.0
	var remainder *variate.Outcome
	for i := range n.Outcomes {
		o := &n.Outcomes[i]
		if o.P == nil {
			remainder = o
			continue
		}
		cum += w.draw(o.P, variate.PerPeriod)
		if u < cum {
			return *o
		}
	}
	if remainder != nil {
		return *remainder
	}
	return n.Outcomes[len(n.Outcomes)-1]
}

func (w *walker) holding(d dist.Continuous, scope variate.Scope) float64 {
	if d == nil {
		return 0
	}
	return w.draw(d, scope)
}

func (w *walker) count(v int, t float64) {
	p := int(math.Floor(t))
	if p >= 0 && p < w.periods {
		w.res.Entries[v][p]++
	}
}

func (w *walker) book(f variate.Function, x, t float64) {
	p := int(math.Floor(t))
	switch {
	case p < 0:
	case p >= w.periods:
		w.res.Beyond += x
	default:
		w.res.Total[p] += x
		w.res.ByCategory[w.catIx[f.Category]][p] += x
	}
}

// bookRate spreads rate × duration over the periods [a, b) overlaps.
func (w *walker) bookRate(f variate.Function, rate, a, b float64) {
	// bound: the periods a holding time spans
	for p := int(math.Floor(a)); float64(p) < b; p++ {
		lo := math.Max(a, float64(p))
		hi := math.Min(b, float64(p+1))
		if hi > lo {
			w.book(f, rate*(hi-lo), float64(p))
		}
	}
}
