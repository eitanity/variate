package suite

import (
	"math"
	"slices"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
)

// seasonal is the seasonal example: storms on a twelve-value cycle,
// burst pipes on another, theft on none.
func seasonal(t *testing.T) variate.Graph {
	t.Helper()
	m, err := config.Load("../../examples/seasonal.yaml")
	if err != nil {
		t.Fatal(err)
	}
	return m.Graph
}

func phaseOf(t *testing.T, g variate.Graph, i int) *variate.Expectation {
	t.Helper()
	p, err := g.Phase(i)
	if err != nil {
		t.Fatal(err)
	}
	c, _, err := p.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if ph := c.Phase(); ph.Index != i || ph.Cycle != 12 {
		t.Fatalf("phase %+v, want %d of 12", ph, i)
	}
	e, err := c.Analytic()
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// TestPhaseExact: at each phase the storm source's mean is the
// unphased one scaled by that month's multiplier over the cycle's mean,
// P(T = 0) is exp of minus the month's rates, and over the cycle the
// phases' means average to the unphased mean.
func TestPhaseExact(t *testing.T) {
	g := seasonal(t)
	c, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	whole, _ := c.Analytic()
	storms, freeze := g.Schedules[0].Values, g.Schedules[1].Values
	cycleMean := 0.0
	for _, v := range storms {
		cycleMean += v / 12
	}
	meanSum := 0.0
	for i := 1; i <= 12; i++ {
		e := phaseOf(t, g, i)
		want := whole.MeanBySource[0] * storms[i-1] / cycleMean
		if math.Abs(e.MeanBySource[0]-want) > 1e-9*want {
			t.Errorf("phase %d: storm mean %g, want %g", i, e.MeanBySource[0], want)
		}
		p0 := math.Exp(-(4*storms[i-1] + 1.5*freeze[i-1] + 2))
		if math.Abs(e.PZero()-p0) > 1e-12 {
			t.Errorf("phase %d: P(T = 0) %g, want %g", i, e.PZero(), p0)
		}
		meanSum += e.Mean
	}
	if mean := meanSum / 12; math.Abs(mean-whole.Mean) > 1e-9*whole.Mean {
		t.Errorf("phase means average to %g, unphased %g", mean, whole.Mean)
	}
}

// TestPhaseTotalVariance: the phases' variances and means give the
// unphased variance by the law of total variance, and their P(T = 0)
// average to the unphased one: the two schedules move together.
func TestPhaseTotalVariance(t *testing.T) {
	g := seasonal(t)
	c, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	whole, _ := c.Analytic()
	var varSum, meanSum, meanSq, p0 float64
	for i := 1; i <= 12; i++ {
		e := phaseOf(t, g, i)
		varSum += e.Var
		meanSum += e.Mean
		meanSq += e.Mean * e.Mean
		p0 += e.PZero() / 12
	}
	mean := meanSum / 12
	if total := varSum/12 + meanSq/12 - mean*mean; math.Abs(total-whole.Var) > 1e-9*whole.Var {
		t.Errorf("phase variances give %g, unphased %g", total, whole.Var)
	}
	if math.Abs(p0-whole.PZero()) > 1e-12 {
		t.Errorf("phase P(T = 0) average to %g, unphased %g", p0, whole.PZero())
	}
}

// TestPhaseRun: a run at a phase draws every period at that phase, so
// its mean is the phase's exact mean, and the transform's bracket of the
// 90th percentile holds the simulated one. The losses are PERTs, whose
// light tails keep the transform's grid small.
func TestPhaseRun(t *testing.T) {
	g := seasonal(t)
	g.Nodes = slices.Clone(g.Nodes)
	for i := range g.Nodes {
		g.Nodes[i].Functions = []variate.Function{Fn("loss", "property", Pert(1e3, 5e3, 2e4))}
	}
	p, err := g.Phase(8)
	if err != nil {
		t.Fatal(err)
	}
	c, r := run(t, p, variate.Config{Periods: 50000, Workers: 4, Transform: true, TransformTol: 1e-2}, 5)
	e, _ := c.Analytic()
	if z := (r.Total.Mean() - e.Mean) / r.Total.StdErr(false); math.Abs(z) > 5 {
		t.Errorf("simulated mean %g, exact %g (z %.2f)", r.Total.Mean(), e.Mean, z)
	}
	tr := c.Transformed()
	if tr == nil || !tr.Applicable {
		t.Fatalf("no transform: %+v", tr)
	}
	lo, hi := tr.Total.Quantile(0.9)
	q, _ := r.Total.Hist.Quantile(0.9)
	if q < 0.98*lo || q > 1.02*hi {
		t.Errorf("simulated P90 %g outside the bracket [%g, %g]", q, lo, hi)
	}
}

// TestPhaseCycle: schedules of 4 and 6 values have a cycle of 12, and
// phase 8 takes the fourth of one and the second of the other; a phase
// outside the cycle names it, and a model without schedules, or one
// already at a phase, has none to take.
func TestPhaseCycle(t *testing.T) {
	g := variate.Graph{
		Schedules: []variate.Schedule{{ID: "storms", Values: []float64{1, 2, 3, 4}}, {ID: "freeze", Values: []float64{10, 20, 30, 40, 50, 60}}},
		Sources:   []variate.Source{{ID: "a", Frequency: C(1), To: "n", Schedule: "storms"}, {ID: "b", Frequency: C(1), To: "n", Schedule: "freeze"}},
		Nodes:     []variate.Node{{ID: "n", Functions: []variate.Function{Fn("one", "c", C(1))}}},
	}
	if n, err := g.Cycle(); err != nil || n != 12 {
		t.Fatalf("cycle %d, %v", n, err)
	}
	p, err := g.Phase(8)
	if err != nil {
		t.Fatal(err)
	}
	if s, f := p.Schedules[0].Values, p.Schedules[1].Values; len(s) != 1 || s[0] != 4 || len(f) != 1 || f[0] != 20 {
		t.Errorf("phase 8 took %v and %v, want [4] and [20]", s, f)
	}
	if g.Schedules[0].Values[0] != 1 || len(g.Schedules[1].Values) != 6 {
		t.Error("Phase changed the graph it was called on")
	}
	for _, i := range []int{0, 13, -1} {
		if _, err := g.Phase(i); err == nil || !strings.Contains(err.Error(), "1 to 12") {
			t.Errorf("phase %d: %v", i, err)
		}
	}
	if _, err := p.Phase(1); err == nil || !strings.Contains(err.Error(), "already") {
		t.Errorf("phase of a phased graph: %v", err)
	}
	g.Schedules = nil
	g.Sources[0].Schedule, g.Sources[1].Schedule = "", ""
	if _, err := g.Phase(1); err == nil || !strings.Contains(err.Error(), "no schedules") {
		t.Errorf("phase without schedules: %v", err)
	}
}
