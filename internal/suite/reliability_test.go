package suite

import (
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// chainModel is a source standing on chains, booking 1 per failure.
func chainModel(chains ...variate.Chain) variate.Graph {
	return variate.Graph{
		Sources: []variate.Source{{ID: "s", To: "n", Chains: chains}},
		Nodes:   []variate.Node{{ID: "n", Functions: []variate.Function{Fn("one", "c", C(1))}}},
	}
}

// stationary solves πQ = 0 for a small generator by iterating the
// uniformised chain until it settles.
func stationary(q [][]float64) []float64 {
	n := len(q)
	dt := 0.0
	for i := range q {
		dt = math.Max(dt, -q[i][i])
	}
	dt = 0.5 / dt
	pi := make([]float64, n)
	pi[0] = 1
	for iter := 0; iter < 2_000_000; iter++ {
		next := make([]float64, n)
		for i := range q {
			for j := range q[i] {
				step := q[i][j] * dt
				if i == j {
					step++
				}
				next[j] += pi[i] * step
			}
		}
		diff := 0.0
		for i := range pi {
			diff = math.Max(diff, math.Abs(next[i]-pi[i]))
		}
		pi = next
		if diff < 1e-16 {
			break
		}
	}
	return pi
}

// TestChainFiguresAgainstMarkov: the closed forms equal the chain on
// the number of failed units, each failing while up and repaired on its
// own, and a two-unit series solved as a four-state chain.
func TestChainFiguresAgainstMarkov(t *testing.T) {
	for _, c := range []variate.Chain{{K: 1, N: 2, Fail: 1 / 12000.0, Repair: 1 / 24.0}, {K: 2, N: 3, Fail: 1 / 6000.0, Repair: 1 / 36.0}, {K: 3, N: 5, Fail: 1e-3, Repair: 1e-2}, {K: 1, N: 1, Fail: 2e-3, Repair: 2e-2}, {K: 2, N: 2, Fail: 1 / 800.0, Repair: 1 / 20.0}} {
		n := c.N
		q := make([][]float64, n+1)
		for j := range q {
			q[j] = make([]float64, n+1)
			if j < n {
				q[j][j+1] = float64(n-j) * c.Fail
				q[j][j] -= q[j][j+1]
			}
			if j > 0 {
				q[j][j-1] = float64(j) * c.Repair
				q[j][j] -= q[j][j-1]
			}
		}
		pi := stationary(q)
		f := pi[n-c.K] * float64(c.K) * c.Fail
		u := 0.0
		for j := n - c.K + 1; j <= n; j++ {
			u += pi[j]
		}
		gotF, gotU := variate.ChainFigures([]variate.Chain{c})
		if math.Abs(gotF-f) > 1e-9*f || math.Abs(gotU-u) > 1e-12 {
			t.Errorf("%d of %d: closed form f %g u %g, chain f %g u %g", c.K, c.N, gotF, gotU, f, u)
		}
	}
	l1, m1, l2, m2 := 1/5000.0, 1/8.0, 1/20000.0, 1/48.0
	q := [][]float64{
		{-(l1 + l2), l1, l2, 0},
		{m1, -(m1 + l2), 0, l2},
		{m2, 0, -(m2 + l1), l1},
		{0, m2, m1, -(m1 + m2)},
	}
	pi := stationary(q)
	gotF, gotU := variate.ChainFigures([]variate.Chain{{K: 1, N: 1, Fail: l1, Repair: m1}, {K: 1, N: 1, Fail: l2, Repair: m2}})
	if f, u := pi[0]*(l1+l2), 1-pi[0]; math.Abs(gotF-f) > 1e-9*f || math.Abs(gotU-u) > 1e-12 {
		t.Errorf("series: closed form f %g u %g, chain f %g u %g", gotF, gotU, f, u)
	}
}

// TestChainSimulation: the simulated failure rate of a diagram matches
// its stationary frequency whether a block is one period or many, the
// exact mean is that frequency, the variance and P(T = 0) are not
// claimed, the transform declines, and a seed repeats.
func TestChainSimulation(t *testing.T) {
	cases := []struct {
		name   string
		chains []variate.Chain
	}{
		{"one of two", []variate.Chain{{K: 1, N: 2, Fail: 3, Repair: 40}}},
		{"two of three", []variate.Chain{{K: 2, N: 3, Fail: 2, Repair: 30}}},
		{"series", []variate.Chain{{K: 1, N: 2, Fail: 3, Repair: 40}, {K: 1, N: 1, Fail: 0.5, Repair: 20}, {K: 2, N: 2, Fail: 0.2}}},
	}
	for _, tc := range cases {
		g := chainModel(tc.chains...)
		want, _ := variate.ChainFigures(tc.chains)
		for _, block := range []int{1, 0} {
			c, r := run(t, g, variate.Config{Periods: 100000, Workers: 4, BlockPeriods: block}, 9)
			e, _ := c.Analytic()
			if math.Abs(e.Mean-want) > 1e-12*want {
				t.Errorf("%s: exact mean %g, want %g", tc.name, e.Mean, want)
			}
			if !math.IsNaN(e.Var) || !math.IsNaN(e.PZero()) {
				t.Errorf("%s: variance %g and P(T = 0) %g should be unknown", tc.name, e.Var, e.PZero())
			}
			se := r.Total.StdErr(true)
			if z := (r.Total.Mean() - want) / se; math.Abs(z) > 5 || math.IsNaN(z) {
				t.Errorf("%s, block %d: simulated rate %g ± %g vs stationary %g: z %g", tc.name, block, r.Total.Mean(), se, want, z)
			}
		}
		tr, err := mustCompile(t, g).Transform(variate.TransformConfig{})
		if err != nil || tr.Applicable || !strings.Contains(tr.Reason, "reliability") {
			t.Errorf("%s: transform applicable %v reason %q err %v", tc.name, tr.Applicable, tr.Reason, err)
		}
	}
	g := chainModel(cases[2].chains...)
	_, a := run(t, g, variate.Config{Periods: 20000, Workers: 2}, 4)
	_, b := run(t, g, variate.Config{Periods: 20000, Workers: 2}, 4)
	if a.Events[0] != b.Events[0] || a.Total.Mean() != b.Total.Mean() {
		t.Error("the same seed did not repeat the run")
	}
}

// TestChainUnderDispersed: a unit that is down as long as it is up
// alternates, so its failures come far more evenly than a Poisson
// source at its rate: the count's index of dispersion tends to one
// half, which the Poisson stand-in would put at one.
func TestChainUnderDispersed(t *testing.T) {
	chains := []variate.Chain{{K: 1, N: 1, Fail: 10, Repair: 10}}
	_, r := run(t, chainModel(chains...), variate.Config{Periods: 200000, Workers: 4}, 2)
	mean := r.Total.Mean()
	v := r.Total.StdDev() * r.Total.StdDev()
	if math.Abs(mean-5) > 0.1 || v > 0.8*mean {
		t.Errorf("mean %g variance %g: the counts should be under-dispersed, near half the mean", mean, v)
	}
}

// TestChainValidation: chains exclude the other laws and need sane
// shapes and rates.
func TestChainValidation(t *testing.T) {
	g := chainModel(variate.Chain{K: 1, N: 2, Fail: 1, Repair: 10})
	g.Sources[0].Frequency = C(1)
	if _, _, err := g.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), "stand in") {
		t.Errorf("chains with frequency: %v", err)
	}
	for _, bad := range []variate.Chain{{K: 3, N: 2, Fail: 1, Repair: 1}, {K: 0, N: 1, Fail: 1}, {K: 1, N: 1, Fail: math.Inf(1)}, {K: 1, N: 1, Fail: -1}} {
		if _, _, err := chainModel(bad).Compile(variate.Config{}); err == nil {
			t.Errorf("chain %+v compiled", bad)
		}
	}
}

// TestOutageHolding: a per-time cost over the outage of the block that
// fired the round books, per period, exactly the block's unavailability
// times the rate; the exact mean says so and the simulation agrees,
// whether the outage lands inside a period or runs past its end.
func TestOutageHolding(t *testing.T) {
	for _, tc := range []struct {
		name   string
		chains []variate.Chain
	}{
		{"one unit, down as long as up", []variate.Chain{{K: 1, N: 1, Fail: 10, Repair: 10}}},
		{"one of two", []variate.Chain{{K: 1, N: 2, Fail: 3, Repair: 6}}},
		{"series", []variate.Chain{{K: 1, N: 2, Fail: 3, Repair: 40}, {K: 1, N: 1, Fail: 1, Repair: 5}}},
	} {
		_, u := variate.ChainFigures(tc.chains)
		g := variate.Graph{
			Sources: []variate.Source{{ID: "s", To: "outage", Chains: tc.chains}},
			Nodes: []variate.Node{
				{ID: "outage", Functions: []variate.Function{{Name: "down", Category: "c", Model: C(1), Basis: variate.PerTime}},
					Outcomes: []variate.Outcome{{To: "back", OutageHolding: true}}},
				{ID: "back"},
			},
		}
		for _, block := range []int{1, 0} {
			c, r := run(t, g, variate.Config{Periods: 100000, Workers: 4, BlockPeriods: block}, 13)
			e, _ := c.Analytic()
			if math.Abs(e.Mean-u) > 1e-9*u {
				t.Errorf("%s: exact mean %g, want the unavailability %g", tc.name, e.Mean, u)
			}
			se := r.Total.StdErr(true)
			if z := (r.Total.Mean() - u) / se; math.Abs(z) > 5 || math.IsNaN(z) {
				t.Errorf("%s, block %d: booked %g ± %g per period, want %g: z %g", tc.name, block, r.Total.Mean(), se, u, z)
			}
		}
	}
}

// TestOutageValidation: an outage holding needs one reliability block
// reaching the node, and stands alone.
func TestOutageValidation(t *testing.T) {
	poisson := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(1), To: "n"}},
		Nodes:   []variate.Node{{ID: "n", Outcomes: []variate.Outcome{{To: "m", OutageHolding: true}}}, {ID: "m"}},
	}
	if _, _, err := poisson.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), "reliability block") {
		t.Errorf("outage from a Poisson source: %v", err)
	}
	two := variate.Graph{
		Sources: []variate.Source{{ID: "a", To: "n", Chains: []variate.Chain{{K: 1, N: 1, Fail: 1, Repair: 10}}}, {ID: "b", To: "n", Chains: []variate.Chain{{K: 1, N: 1, Fail: 1, Repair: 10}}}},
		Nodes:   []variate.Node{{ID: "n", Outcomes: []variate.Outcome{{To: "m", OutageHolding: true}}}, {ID: "m"}},
	}
	if _, _, err := two.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), "exactly one source") {
		t.Errorf("outage from two sources: %v", err)
	}
	both := chainModel(variate.Chain{K: 1, N: 1, Fail: 1, Repair: 10})
	hold, ok := Pert(1, 2, 3).(dist.Continuous)
	if !ok {
		t.Fatal("a PERT is continuous")
	}
	both.Nodes[0].Outcomes = []variate.Outcome{{To: "n2", OutageHolding: true, Holding: hold}}
	both.Nodes = append(both.Nodes, variate.Node{ID: "n2"})
	if _, _, err := both.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), "not both") {
		t.Errorf("outage beside a distribution: %v", err)
	}
}
