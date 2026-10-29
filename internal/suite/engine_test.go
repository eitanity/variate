package suite

import (
	"context"
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"

	"code.eitanity.com/variate/dist"
)

func run(t *testing.T, g variate.Graph, cfg variate.Config, seed uint64) (*variate.Compiled, *variate.Result) {
	t.Helper()
	m, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(w.Close)
	r, err := w.Run(context.Background(), seed)
	if err != nil {
		t.Fatal(err)
	}
	return m, r
}

// zMean compares a mean of n periods with its expectation.
func zMean(mean, want, variance float64, n int64) float64 {
	return (mean - want) / math.Sqrt(variance/float64(n))
}

func TestEngineDiamondAgainstAnalytic(t *testing.T) {
	const h = 50000
	m, r := run(t, Diamond(), variate.Config{Periods: h, BlockPeriods: 1000}, 1)
	e, _ := m.Analytic()
	if z := zMean(r.Total.Mean(), e.Mean, e.Var, h); math.Abs(z) > 4.5 {
		t.Errorf("total mean %g analytic %g z=%.2f", r.Total.Mean(), e.Mean, z)
	}
	if s := r.Total.Sum.Value(); math.Abs(s-r.Total.Mean()*h) > 1e-6*s {
		t.Errorf("sum %g vs mean×n %g", s, r.Total.Mean()*h)
	}
	v := r.Total.StdDev() * r.Total.StdDev()
	if math.Abs(v-e.Var) > 0.05*e.Var {
		t.Errorf("variance %g analytic %g", v, e.Var)
	}
	for c := range r.ByCategory {
		if z := zMean(r.ByCategory[c].Mean(), e.MeanByCategory[c], e.Var, h); math.Abs(z) > 4.5 {
			t.Errorf("category %s mean %g analytic %g", r.Categories[c], r.ByCategory[c].Mean(), e.MeanByCategory[c])
		}
	}
	if z := zMean(r.BySource[0].Mean(), e.MeanBySource[0], e.Var, h); math.Abs(z) > 4.5 {
		t.Errorf("source mean %g analytic %g", r.BySource[0].Mean(), e.MeanBySource[0])
	}
	for v := range r.Nodes {
		// Entries per period are Poisson-like: variance about the mean.
		if z := zMean(r.EntriesPer[v].Mean(), e.EntriesPer[v], e.EntriesPer[v]*1.5, h); math.Abs(z) > 4.5 {
			t.Errorf("entries at %s: %g analytic %g z=%.2f", r.Nodes[v], r.EntriesPer[v].Mean(), e.EntriesPer[v], z)
		}
		if z := zMean(r.Contrib[v].Mean(), e.Contrib[v], e.Var, h); math.Abs(z) > 4.5 {
			t.Errorf("contrib at %s: %g analytic %g", r.Nodes[v], r.Contrib[v].Mean(), e.Contrib[v])
		}
	}
	ev := float64(r.Events[0])
	for v := range r.Nodes {
		p := e.Reach[0][v]
		if p > 0 && p < 1 {
			if z := (float64(r.Reach[0][v]) - ev*p) / math.Sqrt(ev*p*(1-p)); math.Abs(z) > 4.5 {
				t.Errorf("reach %s: %d of %d, p=%g z=%.2f", r.Nodes[v], r.Reach[0][v], r.Events[0], p, z)
			}
		}
	}
	sh := shapeOf(m)
	for j := range r.Outcomes {
		p := e.OutcomeRate[j] / (e.EntriesPer[sh.outNode[j]])
		n := float64(r.Reach[0][sh.outNode[j]])
		if z := (float64(r.OutcomeCount[0][j]) - n*p) / math.Sqrt(n*p*(1-p)+1e-300); p < 1 && math.Abs(z) > 4.5 {
			t.Errorf("outcome %s: %d of %g, p=%g z=%.2f", r.Outcomes[j], r.OutcomeCount[0][j], n, p, z)
		}
	}
	checkInvariants(t, m, r)
	if r.Steps == 0 || r.BeyondHorizon != 0 || r.Periods != h {
		t.Errorf("steps %d beyond %g periods %d", r.Steps, r.BeyondHorizon, r.Periods)
	}
	// Histogram quantiles sit inside the support of T.
	if q, clipped := r.Total.Hist.Quantile(0.5); clipped || q <= 0 || q > r.Total.Max {
		t.Errorf("median %g clipped %v max %g", q, clipped, r.Total.Max)
	}
}

// checkInvariants: the exact count identities.
func checkInvariants(t *testing.T, m *variate.Compiled, r *variate.Result) {
	t.Helper()
	sh := shapeOf(m)
	for k := range r.Sources {
		leaves := uint64(0)
		for v := range r.Nodes {
			if sh.leaf[v] {
				leaves += r.Reach[k][v]
				continue
			}
			from, to := sh.outFrom[v], sh.outTo[v]
			out := uint64(0)
			for j := from; j < to; j++ {
				out += r.OutcomeCount[k][j]
			}
			if out != r.Reach[k][v] {
				t.Errorf("source %s node %s: reach %d, outcomes %d", r.Sources[k], r.Nodes[v], r.Reach[k][v], out)
			}
		}
		if leaves != r.Events[k] {
			t.Errorf("source %s: events %d, leaves %d", r.Sources[k], r.Events[k], leaves)
		}
		for v := range r.Nodes {
			if v == sh.srcTarget[k] {
				continue
			}
			in := uint64(0)
			for j := range r.Outcomes {
				if sh.outTarget[j] == v {
					in += r.OutcomeCount[k][j]
				}
			}
			if in != r.Reach[k][v] {
				t.Errorf("source %s node %s: reach %d, inbound %d", r.Sources[k], r.Nodes[v], r.Reach[k][v], in)
			}
		}
	}
	// Breakdowns sum to the total.
	var cats, srcs float64
	for c := range r.ByCategory {
		cats += r.ByCategory[c].Sum.Value()
	}
	for k := range r.BySource {
		srcs += r.BySource[k].Sum.Value()
	}
	total := r.Total.Sum.Value()
	if math.Abs(cats-total) > 1e-9*total || math.Abs(srcs-total) > 1e-9*total {
		t.Errorf("breakdowns %g %g total %g", cats, srcs, total)
	}
}

func TestEngineDeterministic(t *testing.T) {
	_, a := run(t, Diamond(), variate.Config{Periods: 3000, BlockPeriods: 500}, 9)
	_, b := run(t, Diamond(), variate.Config{Periods: 3000, BlockPeriods: 500}, 9)
	if a.Total.Sum != b.Total.Sum || a.Total.W != b.Total.W || a.Events[0] != b.Events[0] || a.Steps != b.Steps {
		t.Error("same seed, different results")
	}
	_, c := run(t, Diamond(), variate.Config{Periods: 3000, BlockPeriods: 500}, 10)
	if a.Total.Sum == c.Total.Sum {
		t.Error("different seeds, same sum")
	}
}

// TestEngineTimeAccounting: a variate.PerTime rate over a holding time books
// exactly rate × duration, whichever periods it spans. A model booking
// rate 2 over 1.5 periods is compared with one booking 3 on entry from the
// same draws. They differ only by the burn-in rounds, whose on-entry
// booking at t < 0 is dropped while the per-time interval partly lands at
// or after 0; that difference lies in [0, 3 × burn-in rounds].
func TestEngineTimeAccounting(t *testing.T) {
	perTime := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(2), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Next: "b", NextHolding: C(1.5), Functions: []variate.Function{{Name: "r", Category: "k", Model: C(2), Basis: variate.PerTime}}},
			{ID: "b"},
		},
	}
	onEntry := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(2), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Next: "b", NextHolding: C(1.5), Functions: []variate.Function{Fn("v", "k", C(3))}},
			{ID: "b"},
		},
	}
	cfg := variate.Config{Periods: 2000, BlockPeriods: 100}
	_, rp := run(t, perTime, cfg, 5)
	_, ro := run(t, onEntry, cfg, 5)
	if rp.Events[0] != ro.Events[0] || rp.Steps != ro.Steps {
		t.Fatalf("events %d %d steps %d %d", rp.Events[0], ro.Events[0], rp.Steps, ro.Steps)
	}
	burnIn := float64(rp.Steps/2 - rp.Events[0]) // two steps per round
	sp := rp.Total.Sum.Value() + rp.BeyondHorizon
	so := ro.Total.Sum.Value() + ro.BeyondHorizon
	if d := sp - so; d < -1e-9*so || d > 3*burnIn+1e-9*so || rp.BeyondHorizon == 0 {
		t.Errorf("per-time %g (beyond %g) vs on-entry %g: difference %g, burn-in rounds %g", sp, rp.BeyondHorizon, so, d, burnIn)
	}
	// Every on-entry round books exactly 3.
	if so != 3*float64(ro.Events[0]) {
		t.Errorf("on-entry sum %g, events %d", so, ro.Events[0])
	}
}

func TestEngineWindowError(t *testing.T) {
	g := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(2), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Next: "b", NextHolding: C(5)},
			{ID: "b", Functions: []variate.Function{Fn("v", "k", C(1))}},
		},
	}
	m, _, err := g.Compile(variate.Config{Spill: 2})
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(variate.Config{Periods: 100, Spill: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	_, err = w.Run(context.Background(), 1)
	if err == nil || !strings.Contains(err.Error(), "beyond the window") {
		t.Errorf("got %v", err)
	}
}

// TestEngineSuperposition: two sources into one node equal one source
// with the summed rate, in the node's entry statistics.
func TestEngineSuperposition(t *testing.T) {
	two := variate.Graph{
		Sources: []variate.Source{{ID: "s1", Frequency: C(1), To: "a"}, {ID: "s2", Frequency: C(2), To: "a"}},
		Nodes:   []variate.Node{{ID: "a", Functions: []variate.Function{Fn("v", "k", Pert(1, 2, 3))}}},
	}
	one := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(3), To: "a"}},
		Nodes:   []variate.Node{{ID: "a", Functions: []variate.Function{Fn("v", "k", Pert(1, 2, 3))}}},
	}
	const h = 40000
	_, r2 := run(t, two, variate.Config{Periods: h}, 3)
	_, r1 := run(t, one, variate.Config{Periods: h}, 4)
	v := r1.Total.StdDev() * r1.Total.StdDev()
	if z := (r2.Total.Mean() - r1.Total.Mean()) / math.Sqrt(2*v/h); math.Abs(z) > 4.5 {
		t.Errorf("means %g %g z=%.2f", r2.Total.Mean(), r1.Total.Mean(), z)
	}
	v2 := r2.Total.StdDev() * r2.Total.StdDev()
	if math.Abs(v2-v) > 0.05*v {
		t.Errorf("variances %g %g", v2, v)
	}
	if z := (r2.EntriesPer[0].Mean() - 3) / math.Sqrt(3.0/h); math.Abs(z) > 4.5 {
		t.Errorf("entries %g", r2.EntriesPer[0].Mean())
	}
}

// TestEnginePeriodSharing: the per-period filter. N through per period is
// Poisson(λ(1−p)) given the draws, so Var[N] = E[λ(1−p)] + Var[λ(1−p)].
func TestEnginePeriodSharing(t *testing.T) {
	block, _ := dist.NewPERT(0.90, 0.95, 0.99, 4)
	g := variate.Graph{
		Sources: []variate.Source{{ID: "mail", Frequency: C(600), To: "filter"}},
		Nodes: []variate.Node{
			{ID: "filter", Outcomes: []variate.Outcome{{To: "blocked", P: block}, {To: "through"}}},
			{ID: "blocked"},
			{ID: "through"},
		},
	}
	const h = 20000
	_, r := run(t, g, variate.Config{Periods: h, NodeHist: true}, 6)
	through := r.EntriesPer[2]
	meanP := block.Mean()
	wantMean := 600 * (1 - meanP)
	wantVar := wantMean + 600*600*block.Var()
	if z := zMean(through.Mean(), wantMean, wantVar, h); math.Abs(z) > 4.5 {
		t.Errorf("mean %g want %g z=%.2f", through.Mean(), wantMean, z)
	}
	v := through.StdDev() * through.StdDev()
	if math.Abs(v-wantVar) > 0.1*wantVar || v < 2*wantMean {
		t.Errorf("variance %g want %g (poisson would be %g)", v, wantVar, wantMean)
	}
}

// TestEngineBurnInAndSpill: with a holding time of exactly 1, every round
// arriving in [−1, 0) books into [0, 1), every round arriving in [H−1, H)
// books beyond the horizon, and every booking crosses a period boundary.
// With blocks of 3 periods the spill crosses a block boundary every third
// period. The sum of all bookings is then exactly 10 per round arriving in
// [−1, H), burn-in rounds included.
func TestEngineBurnInAndSpill(t *testing.T) {
	g := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(4), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Next: "b", NextHolding: C(1)},
			{ID: "b", Functions: []variate.Function{Fn("v", "k", C(10))}},
		},
	}
	for _, block := range []int{3, 7, 1000} {
		m, r := run(t, g, variate.Config{Periods: 1000, BlockPeriods: block, Series: true}, 21)
		if m.Spill() != 1 {
			t.Fatalf("spill %d", m.Spill())
		}
		burnIn := r.Steps/2 - r.Events[0]
		want := 10 * float64(r.Events[0]+burnIn)
		got := r.Total.Sum.Value() + r.BeyondHorizon
		if got != want || r.BeyondHorizon == 0 || burnIn == 0 {
			t.Errorf("block %d: booked %g want %g (beyond %g, burn-in rounds %d)", block, got, want, r.BeyondHorizon, burnIn)
		}
		// Period 0 holds the burn-in rounds' bookings: at least 10 × their number
		// is impossible to pin down exactly, but the series must agree with the sum.
		sum := 0.0
		for p := int64(0); p < 1000; p++ {
			sum += r.Series.Total(p)
		}
		if math.Abs(sum-r.Total.Sum.Value()) > 1e-9*sum {
			t.Errorf("block %d: series sum %g vs accumulated %g", block, sum, r.Total.Sum.Value())
		}
	}
}
