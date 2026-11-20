package suite

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// lognormalMoment is E[G^k] of a lognormal of the given mean and sd.
func lognormalMoment(mean, sd float64, k int) float64 {
	s2 := math.Log(1 + sd*sd/(mean*mean))
	mu := math.Log(mean) - s2/2
	fk := float64(k)
	return math.Exp(fk*mu + fk*fk*s2/2)
}

// inflationGraph: 4 rounds a period book X, gamma of mean 2 and sd 1,
// times the factor, and 1 unscaled; with onRate the factor scales the
// rate too. The factor is a lognormal of mean 1.1 and sd 0.4.
func inflationGraph(onRate bool) variate.Graph {
	x, _ := dist.NewGammaFromMeanSD(2, 1)
	g, _ := dist.NewLognormalFromMeanSD(1.1, 0.4)
	src := variate.Source{ID: "s", Frequency: C(4), To: "a"}
	if onRate {
		src.Factor = "inflation"
	}
	scaled := Fn("x", "cost", x)
	scaled.Factor = "inflation"
	return variate.Graph{ID: "inflation",
		Factors: []variate.Factor{{ID: "inflation", Model: g}},
		Sources: []variate.Source{src},
		Nodes:   []variate.Node{{ID: "a", Functions: []variate.Function{scaled, Fn("y", "fee", C(1))}}},
	}
}

// inflationExact is inflationGraph's exact mean and variance: given G,
// a compound Poisson of rate λG^r, r = 1 with the factor on the rate,
// of the value GX + 1; then E[Var | G] + Var(E | G).
func inflationExact(onRate bool) (mean, vr float64) {
	const lam, mx, m2x = 4.0, 2.0, 5.0
	g := func(k int) float64 { return lognormalMoment(1.1, 0.4, k) }
	if !onRate {
		mean = lam * (g(1)*mx + 1)
		vr = lam*(g(2)*m2x+2*g(1)*mx+1) + lam*lam*mx*mx*(g(2)-g(1)*g(1))
		return mean, vr
	}
	mean = lam * (g(2)*mx + g(1))
	within := lam * (g(3)*m2x + 2*g(2)*mx + g(1))
	between := lam * lam * (mx*mx*(g(4)-g(2)*g(2)) + 2*mx*(g(3)-g(2)*g(1)) + g(2) - g(1)*g(1))
	return mean, within + between
}

// TestValueFactorExact: the exact mean and variance are the closed
// forms, with the factor on the values alone and on the rate too, and
// the breakdown by category moves with the factor.
func TestValueFactorExact(t *testing.T) {
	for _, onRate := range []bool{false, true} {
		m, _, err := inflationGraph(onRate).Compile(variate.Config{})
		if err != nil {
			t.Fatal(err)
		}
		e, err := m.Analytic()
		if err != nil {
			t.Fatal(err)
		}
		mean, vr := inflationExact(onRate)
		if math.Abs(e.Mean-mean) > 1e-9*mean || math.Abs(e.Var-vr) > 1e-9*vr {
			t.Errorf("on the rate %v: mean %v var %v, want %v and %v", onRate, e.Mean, e.Var, mean, vr)
		}
		cost := e.MeanByCategory[indexOf(m.Categories(), "cost")]
		if want := 4 * 2 * lognormalMoment(1.1, 0.4, 1); onRate {
			want = 4 * 2 * lognormalMoment(1.1, 0.4, 2)
			if math.Abs(cost-want) > 1e-9*want {
				t.Errorf("cost %v, want %v", cost, want)
			}
		} else if math.Abs(cost-want) > 1e-9*want {
			t.Errorf("cost %v, want %v", cost, want)
		}
	}
}

// TestValueFactorSimulated: over 200,000 periods the simulated mean
// and standard deviation are within their errors of the exact ones.
func TestValueFactorSimulated(t *testing.T) {
	const periods = 200000
	for _, onRate := range []bool{false, true} {
		_, _, s := runSummary(t, inflationGraph(onRate), periods, 21)
		mean, vr := inflationExact(onRate)
		sd := math.Sqrt(vr)
		if math.Abs(s.Total.Mean-mean) > 4*s.Total.StdErr || math.Abs(s.Total.StdDev-sd) > 0.02*sd {
			t.Errorf("on the rate %v: %v ± %v sd %v, want %v and %v", onRate, s.Total.Mean, s.Total.StdErr, s.Total.StdDev, mean, sd)
		}
	}
}

// TestValueFactorOne: a function naming a constant factor of 1 gives
// the run of one that does not to the bit, and a constant 2 doubles the scaled values exactly.
func TestValueFactorOne(t *testing.T) {
	plain := inflationGraph(false)
	plain.Nodes[0].Functions[0].Factor = ""
	plain.Factors[0].Model = C(1)
	_, _, want := runSummary(t, plain, 20000, 5)
	one := inflationGraph(false)
	one.Factors[0].Model = C(1)
	_, _, got := runSummary(t, one, 20000, 5)
	if got.Total != want.Total || got.ByCategory["cost"] != want.ByCategory["cost"] {
		t.Errorf("factor 1: %v, without %v", got.Total, want.Total)
	}
	two := inflationGraph(false)
	two.Factors[0].Model = C(2)
	m, _, err := two.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := m.Analytic()
	if want := 4 * (2*2 + 1.0); math.Abs(e.Mean-want) > 1e-12 {
		t.Errorf("factor 2: mean %v, want %v", e.Mean, want)
	}
}

// TestValueFactorThreshold: a threshold on a scaled value routes by the
// scaled value: the rounds above 3 are 4·E[P(X > 3/G)], which the rule
// doubles to, simulated within their error.
func TestValueFactorThreshold(t *testing.T) {
	x, _ := dist.NewGammaFromMeanSD(2, 1)
	g, _ := dist.NewLognormalFromMeanSD(1.1, 0.4)
	fn := Fn("x", "cost", x)
	fn.Factor = "inflation"
	graph := variate.Graph{ID: "deductible",
		Factors: []variate.Factor{{ID: "inflation", Model: g}},
		Sources: []variate.Source{{ID: "s", Frequency: C(4), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{fn}, Threshold: &variate.Threshold{Function: "x", Over: 3, Above: "big", Below: "small"}},
			{ID: "big"}, {ID: "small"},
		},
	}
	// E[P(X > 3/G)] over G's quantiles, 20,000 midpoints.
	want := 0.0
	const n = 20000
	for i := range n {
		gi := g.Quantile((float64(i) + 0.5) / n)
		want += (1 - x.CDF(3/gi)) / n
	}
	want *= 4
	m, _, err := graph.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := m.Analytic()
	big := indexOf(m.NodeIDs(), "big")
	if math.Abs(e.EntriesPer[big]-want) > 1e-6*want {
		t.Errorf("entries above 3: %v, want %v", e.EntriesPer[big], want)
	}
	_, _, s := runSummary(t, graph, 100000, 3)
	if got := s.EntriesPer["big"]; math.Abs(got.Mean-want) > 4*got.StdErr {
		t.Errorf("simulated above 3: %v ± %v, want %v", got.Mean, got.StdErr, want)
	}
}

// TestValueFactorStarted: a started run's exact mean per period is the
// steady state's when the factor is on the rate and the values: 4·(2E[G²]
// + E[G]), every period of an empty start without holdings.
func TestValueFactorStarted(t *testing.T) {
	g := inflationGraph(true)
	g.Start = variate.Start{Horizon: 3}
	m, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := m.Analytic()
	want, _ := inflationExact(true)
	for k, row := range e.ByPeriod {
		if math.Abs(row[0]-want) > 1e-9*want {
			t.Errorf("period %d: %v, want %v", k, row[0], want)
		}
	}
	if math.Abs(e.Mean-3*want) > 1e-9*want {
		t.Errorf("horizon %v, want %v", e.Mean, 3*want)
	}
}

// TestValueFactorRefusals: a function names a factor the model has.
func TestValueFactorRefusals(t *testing.T) {
	g := inflationGraph(false)
	g.Nodes[0].Functions[0].Factor = "wages"
	if _, _, err := g.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), "factor wages not found") {
		t.Errorf("unknown factor: %v", err)
	}
}

// TestValueFactorTransform: the exact distribution mixes over a factor
// on a value as over one on a rate: inflation (lognormal mean 1.1, sd
// 0.4) on a cost, and on the cost and the rate, brackets 200,000
// simulated periods' curve and quantiles, its mean the conditioned
// one; a constant factor gives the transform of the model with the
// value scaled, bit for bit.
func TestValueFactorTransform(t *testing.T) {
	for _, onRate := range []bool{false, true} {
		g := inflationGraph(onRate)
		m, tr := transformOf(t, g, variate.TransformConfig{Tol: 1e-2})
		if !tr.Applicable {
			t.Fatalf("on rate %v: not applicable: %s", onRate, tr.Reason)
		}
		if tr.Passes < 2 {
			t.Errorf("on rate %v: the factor should make the transform mix: %d passes", onRate, tr.Passes)
		}
		e, _ := m.Analytic()
		mean, _ := inflationExact(onRate)
		if math.Abs(e.Mean-mean) > 1e-9*mean {
			t.Errorf("on rate %v: mean %v, want %v", onRate, e.Mean, mean)
		}
		t.Logf("on rate %v: %d passes, grid %d, work %d", onRate, tr.Passes, tr.M, tr.Work)
		const h = 200000
		_, r := run(t, g, variate.Config{Periods: h, Series: true}, 8)
		totals := r.Series.TotalColumn(make([]float64, h))
		name := fmt.Sprintf("on rate %v", onRate)
		checkBracketsSeries(t, name, tr.Total, totals, 100, 4.5)
		checkQuantilesSeries(t, name, tr.Total, totals, 4.5)
	}
	fixed := inflationGraph(false)
	c, _ := dist.NewConstant(1.3)
	fixed.Factors[0].Model = c
	_, a := transformOf(t, fixed, variate.TransformConfig{})
	plain := inflationGraph(false)
	x, _ := dist.NewGammaFromMeanSD(2, 1)
	sx, _ := dist.NewScaled(x, 1.3)
	plain.Factors, plain.Nodes[0].Functions[0].Factor, plain.Nodes[0].Functions[0].Model = nil, "", sx
	_, b := transformOf(t, plain, variate.TransformConfig{})
	if !slices.Equal(a.Total.Upper, b.Total.Upper) || !slices.Equal(a.Total.Lower, b.Total.Lower) {
		t.Errorf("a constant factor's transform differs from the scaled value's")
	}
}

// TestValueFactorLattice: a constant value scaled by a drawn factor
// takes any value, so the totals are on no lattice; without the factor
// they are multiples of it.
func TestValueFactorLattice(t *testing.T) {
	g, _ := dist.NewLognormalFromMeanSD(1.1, 0.4)
	fee := Fn("fee", "fee", C(100))
	fee.Factor = "inflation"
	graph := variate.Graph{ID: "fees",
		Factors: []variate.Factor{{ID: "inflation", Model: g}},
		Sources: []variate.Source{{ID: "s", Frequency: C(4), To: "a"}},
		Nodes:   []variate.Node{{ID: "a", Functions: []variate.Function{fee}}},
	}
	for _, scaled := range []bool{true, false} {
		if !scaled {
			graph.Factors, graph.Nodes[0].Functions[0].Factor = nil, ""
		}
		m, _, err := graph.Compile(variate.Config{})
		if err != nil {
			t.Fatal(err)
		}
		want := 100.0
		if scaled {
			want = 0
		}
		if u := m.LatticeUnit(); u != want {
			t.Errorf("scaled %v: lattice unit %v, want %v", scaled, u, want)
		}
	}
}
