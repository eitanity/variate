package suite

import (
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// factorModel is two Poisson sources booking a constant 1 per round,
// so the period total is the sum of their counts, both scaled by one
// shared factor when named.
func factorModel(factor dist.Distribution, share bool) variate.Graph {
	g := variate.Graph{
		Sources: []variate.Source{{ID: "a", Frequency: C(2), To: "n"}, {ID: "b", Frequency: C(3), To: "n"}},
		Nodes:   []variate.Node{{ID: "n", Functions: []variate.Function{Fn("one", "c", C(1))}}},
	}
	if factor != nil {
		g.Factors = []variate.Factor{{ID: "climate", Model: factor}}
		g.Sources[0].Factor = "climate"
		if share {
			g.Sources[1].Factor = "climate"
		}
	}
	return g
}

// TestSharedFactorMoments: with one factor of mean 1 and E[F²] = 1.25
// on both rates, E[T] = 5 and Var[T] = 5 + 25 × 0.25; on one rate only,
// Var[T] = 5 + 4 × 0.25; and the simulation agrees with each.
func TestSharedFactorMoments(t *testing.T) {
	f := mustLognormal(t, -math.Log(1.25)/2, math.Sqrt(math.Log(1.25)))
	if math.Abs(f.Mean()-1) > 1e-12 || math.Abs(f.Var()-0.25) > 1e-12 {
		t.Fatalf("factor mean %g var %g", f.Mean(), f.Var())
	}
	for _, tc := range []struct {
		name  string
		share bool
		want  float64
	}{{"shared", true, 5 + 25*0.25}, {"one source", false, 5 + 4*0.25}} {
		g := factorModel(f, tc.share)
		c, r := run(t, g, variate.Config{Periods: 200000, Workers: 4}, 21)
		e, _ := c.Analytic()
		if math.Abs(e.Mean-5) > 1e-12 || math.Abs(e.Var-tc.want) > 1e-9 {
			t.Errorf("%s: exact mean %g var %g, want 5 and %g", tc.name, e.Mean, e.Var, tc.want)
		}
		se := r.Total.StdErr(false)
		if z := (r.Total.Mean() - 5) / se; math.Abs(z) > 5 {
			t.Errorf("%s: simulated mean %g ± %g", tc.name, r.Total.Mean(), se)
		}
		if v := r.Total.StdDev() * r.Total.StdDev(); math.Abs(v-tc.want) > 0.05*tc.want {
			t.Errorf("%s: simulated variance %g vs exact %g", tc.name, v, tc.want)
		}
	}
	plain := factorModel(nil, false)
	c, _, _ := plain.Compile(variate.Config{})
	e, _ := c.Analytic()
	if math.Abs(e.Var-5) > 1e-12 {
		t.Errorf("without a factor Var %g, want 5", e.Var)
	}
}

// TestSharedFactorTransform: the exact distribution mixes over the
// shared factor, brackets the simulated curve and gives P(T = 0)
// exactly: E[exp(−5F)].
func TestSharedFactorTransform(t *testing.T) {
	f := Pert(0.5, 1, 2)
	g := factorModel(f, true)
	ln, _ := dist.NewLognormal(5, 0.6)
	g.Nodes[0].Functions = []variate.Function{Fn("loss", "c", ln)}
	m, tr := transformOf(t, g, variate.TransformConfig{Tol: 1e-2})
	if !tr.Applicable {
		t.Fatalf("not applicable: %s", tr.Reason)
	}
	if tr.Passes < 2 {
		t.Errorf("the factor should make the transform mix: %d passes", tr.Passes)
	}
	const h = 40000
	_, r := run(t, g, variate.Config{Periods: h, Series: true}, 8)
	totals := r.Series.TotalColumn(make([]float64, h))
	checkBracketsSeries(t, "total", tr.Total, totals, 100, 4.5)
	checkQuantilesSeries(t, "total", tr.Total, totals, 4.5)
	e, _ := m.Analytic()
	rule := make([]float64, 64)
	w := make([]float64, 64)
	f.Nodes(rule, w)
	want := 0.0
	for i := range rule {
		want += w[i] * math.Exp(-5*rule[i])
	}
	if math.Abs(e.PZero()-want) > 1e-9 || math.Abs(tr.PZero-want) > 1e-6 {
		t.Errorf("P(T = 0): exact %g transform %g, want %g", e.PZero(), tr.PZero, want)
	}
	zero := float64(r.Total.Zero) / float64(r.Periods)
	if math.Abs(zero-want) > 5*math.Sqrt(want*(1-want)/h) {
		t.Errorf("empty share %g vs %g", zero, want)
	}
}

// TestFactorValidation: a factor must exist, scale a Poisson rate only,
// and be a distribution on [0, ∞).
func TestFactorValidation(t *testing.T) {
	g := factorModel(Pert(0.5, 1, 2), true)
	g.Sources[0].Factor = "weather"
	if _, _, err := g.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("unknown factor: %v", err)
	}
	g = factorModel(Pert(0.5, 1, 2), true)
	g.Sources[0].Every = 3
	if _, _, err := g.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), "Poisson rate only") {
		t.Errorf("factor with every: %v", err)
	}
	g = factorModel(Pert(-1, 1, 2), true)
	if _, _, err := g.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), "does not fit") {
		t.Errorf("negative factor: %v", err)
	}
}
