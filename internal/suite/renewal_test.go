package suite

import (
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// counting is a model whose period total is the source's count: one
// node booking a constant 1 per round.
func counting(freq dist.Distribution, every int, repair dist.Distribution) variate.Graph {
	return variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: freq, To: "n", Every: every, Repair: repair}},
		Nodes:   []variate.Node{{ID: "n", Functions: []variate.Function{Fn("one", "c", C(1))}}},
	}
}

// TestErlangCountLaw: the count of every third arrival of a Poisson
// stream has the exact mean, variance and zero probability the mass
// function gives, the simulation agrees within its error, and a seed
// repeats exactly.
func TestErlangCountLaw(t *testing.T) {
	g := counting(C(6), 3, nil)
	cfg := variate.Config{Periods: 200000, Workers: 4}
	c, r := run(t, g, cfg, 11)
	e, _ := c.Analytic()
	if math.Abs(e.Mean-2) > 1e-12 {
		t.Errorf("exact mean %g, want 2", e.Mean)
	}
	// With n = 3 the count is under-dispersed against Poisson at 2.
	if !(e.Var > 0.3 && e.Var < 2) {
		t.Errorf("exact variance %g, want under the Poisson 2", e.Var)
	}
	se := r.Total.StdErr(true)
	if z := (r.Total.Mean() - e.Mean) / se; math.Abs(z) > 5 || math.IsNaN(z) {
		t.Errorf("simulated mean %g ± %g vs exact %g: z %g", r.Total.Mean(), se, e.Mean, z)
	}
	if v := r.Total.StdDev() * r.Total.StdDev(); math.Abs(v-e.Var) > 0.05*e.Var {
		t.Errorf("simulated variance %g vs exact %g", v, e.Var)
	}
	zero := float64(r.Total.Zero) / float64(r.Periods)
	if math.Abs(zero-e.PZero()) > 5*math.Sqrt(e.PZero()*(1-e.PZero())/float64(r.Periods)) {
		t.Errorf("empty share %g vs exact %g", zero, e.PZero())
	}
	_, r2 := run(t, g, cfg, 11)
	if r2.Events[0] != r.Events[0] || r2.Total.Mean() != r.Total.Mean() {
		t.Error("the same seed did not repeat the run")
	}
}

// TestErlangMixedRate: an uncertain rate carries through: the mean is
// E[λ]/n and the simulation agrees.
func TestErlangMixedRate(t *testing.T) {
	g := counting(Pert(2, 5, 12), 2, nil)
	c, r := run(t, g, variate.Config{Periods: 100000, Workers: 4}, 5)
	e, _ := c.Analytic()
	if want := Pert(2, 5, 12).Mean() / 2; math.Abs(e.Mean-want) > 1e-12 {
		t.Errorf("exact mean %g, want %g", e.Mean, want)
	}
	se := r.Total.StdErr(true)
	if z := (r.Total.Mean() - e.Mean) / se; math.Abs(z) > 5 || math.IsNaN(z) {
		t.Errorf("simulated mean %g ± %g vs exact %g", r.Total.Mean(), se, e.Mean)
	}
}

// TestErlangTransformBrackets: the exact distribution applies to the
// Erlang law and brackets the simulated curve and P(T = 0).
func TestErlangTransformBrackets(t *testing.T) {
	ln, _ := dist.NewLognormal(6, 0.8)
	g := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(5), To: "n", Every: 2}},
		Nodes:   []variate.Node{{ID: "n", Functions: []variate.Function{Fn("loss", "c", ln)}}},
	}
	m, tr := transformOf(t, g, variate.TransformConfig{Tol: 1e-2})
	if !tr.Applicable {
		t.Fatalf("not applicable: %s", tr.Reason)
	}
	const h = 40000
	_, r := run(t, g, variate.Config{Periods: h, Series: true}, 3)
	e, _ := m.Analytic()
	totals := r.Series.TotalColumn(make([]float64, h))
	checkBracketsSeries(t, "total", tr.Total, totals, 100, 4.5)
	checkQuantilesSeries(t, "total", tr.Total, totals, 4.5)
	if math.Abs(tr.PZero-e.PZero()) > 1e-9 {
		t.Errorf("pzero %g exact %g", tr.PZero, e.PZero())
	}
}

// TestRepairLaw: a source silent for a repair after each firing fires
// at 1/(1/λ + E[H]) exactly in the mean, the simulation agrees whether
// a block is one period or many (every block starts in equilibrium),
// the variance and P(T = 0) are not claimed, and the transform declines.
func TestRepairLaw(t *testing.T) {
	for _, tc := range []struct {
		name   string
		repair dist.Distribution
	}{{"constant", C(0.1)}, {"pert", Pert(0.02, 0.1, 0.3)}, {"lognormal", mustLognormal(t, math.Log(0.1), 0.5)}} {
		g := counting(C(4), 0, tc.repair)
		want := 1 / (1/4.0 + tc.repair.Mean())
		for _, block := range []int{1, 0} {
			c, r := run(t, g, variate.Config{Periods: 100000, Workers: 4, BlockPeriods: block}, 7)
			e, _ := c.Analytic()
			if math.Abs(e.Mean-want) > 1e-12 {
				t.Errorf("%s: exact mean %g, want %g", tc.name, e.Mean, want)
			}
			if !math.IsNaN(e.Var) || !math.IsNaN(e.PZero()) {
				t.Errorf("%s: variance %g and P(T = 0) %g should be unknown", tc.name, e.Var, e.PZero())
			}
			se := r.Total.StdErr(true)
			if z := (r.Total.Mean() - want) / se; math.Abs(z) > 5 || math.IsNaN(z) {
				t.Errorf("%s, block %d: simulated mean %g ± %g vs exact %g: z %g", tc.name, block, r.Total.Mean(), se, want, z)
			}
		}
		tr, err := mustCompile(t, g).Transform(variate.TransformConfig{})
		if err != nil || tr.Applicable || !strings.Contains(tr.Reason, "repair") {
			t.Errorf("%s: transform applicable %v reason %q err %v", tc.name, tr.Applicable, tr.Reason, err)
		}
	}
}

// TestRenewalValidation: the forms are exclusive and repair needs a
// constant rate.
func TestRenewalValidation(t *testing.T) {
	both := counting(C(4), 2, C(0.1))
	if _, _, err := both.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), "exclusive") {
		t.Errorf("both forms: %v", err)
	}
	uncertain := counting(Pert(1, 4, 8), 0, C(0.1))
	if _, _, err := uncertain.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), "constant frequency") {
		t.Errorf("uncertain rate with repair: %v", err)
	}
}

func mustCompile(t *testing.T, g variate.Graph) *variate.Compiled {
	t.Helper()
	c, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	return c
}
