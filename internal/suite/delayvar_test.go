package suite

import (
	"context"
	"math"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// heldPath: 8 rounds a period book X, gamma of mean 3 and sd 1, at a,
// and a rate of 4 a period, category lost, over the holding τ to b,
// where Y, gamma of mean 5 and sd 2, is booked.
func heldPath(tau dist.Distribution) variate.Graph {
	x, _ := dist.NewGammaFromMeanSD(3, 1)
	y, _ := dist.NewGammaFromMeanSD(5, 2)
	r := Fn("r", "lost", C(4))
	r.Basis = variate.PerTime
	return variate.Graph{ID: "held",
		Sources: []variate.Source{{ID: "s", Frequency: C(8), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{Fn("x", "x", x), r}, Next: "b", NextHolding: tau},
			{ID: "b", Functions: []variate.Function{Fn("y", "y", y)}},
		},
	}
}

// heldConstExact is heldPath's variance with a constant holding c: Q is
// X² + Y² + 16·G₂(c) + 2XY·K(c) + 8X·H(c) + 8Y·H(c), K(c) = (1 − c)⁺,
// H(c) = ∫₀^c K and G₂(c) = ∫∫_{[0,c]²} K, times λ = 8.
func heldConstExact(c float64) float64 {
	k := math.Max(0, 1-c)
	u := math.Min(c, 1)
	hc := u - u*u/2
	g2 := c*c - c*c*c/3
	if c > 1 {
		g2 = c - 1.0/3
	}
	return 8 * (10 + 29 + 16*g2 + 2*3*5*k + 8*3*hc + 8*5*hc)
}

// TestDelayConstant: with a constant holding the variance is the closed
// form, on the lag grid's points or between them, and past a period.
func TestDelayConstant(t *testing.T) {
	for _, c := range []float64{0.3, 0.5, 0.71234, 1.5} {
		_, e := depExact(t, heldPath(C(c)))
		if want := heldConstExact(c); !within(e.Var, want, 1e-9) {
			t.Errorf("holding %v: var %v, want %v", c, e.Var, want)
		}
	}
}

// TestDelayZero: a holding of 0 books as no holding does: the variance
// of the model without it.
func TestDelayZero(t *testing.T) {
	_, held := depExact(t, heldPath(C(0)))
	g := heldPath(C(0))
	g.Nodes[0].NextHolding = nil
	g.Nodes[0].Functions = g.Nodes[0].Functions[:1]
	_, none := depExact(t, g)
	if !within(held.Var, none.Var, 1e-12) {
		t.Errorf("holding 0: var %v, without %v", held.Var, none.Var)
	}
}

// TestDelaySimulated: with a lognormal holding of median 2 days, 3
// months or a year in a year's period, the exact standard deviation of
// the total, each category, the source and each node is within its
// error of the simulated over 2^20 periods.
func TestDelaySimulated(t *testing.T) {
	for _, med := range []float64{2.0 / 365, 0.25, 1} {
		tau, _ := dist.NewLognormalFromMedianP90(med, 3*med)
		g := heldPath(tau)
		m, e := depExact(t, g)
		_, _, s := runSummary(t, g, 1<<20, 9)
		check := func(what string, vr float64, st variate.Stat) {
			sd := math.Sqrt(vr)
			if se := st.StdDev / math.Sqrt(2*float64(st.N)); !(math.Abs(st.StdDev-sd) <= 4*se) {
				t.Errorf("median %v, %s: exact sd %v, simulated %v ± %v", med, what, sd, st.StdDev, se)
			}
		}
		check("total", e.Var, s.Total)
		for _, c := range []string{"x", "y", "lost"} {
			check("category "+c, e.VarByCategory[indexOf(m.Categories(), c)], s.ByCategory[c])
		}
		check("source", e.VarBySource[0], s.BySource["s"])
		if math.IsNaN(e.ContribVar[0]) || math.IsNaN(e.ContribVar[1]) {
			t.Errorf("median %v: node variances %v", med, e.ContribVar)
		}
	}
}

// TestDelayTotals: a named total's variance with holding times is its
// categories', the lost production and X, against the simulation.
func TestDelayTotals(t *testing.T) {
	tau, _ := dist.NewLognormalFromMedianP90(0.25, 0.75)
	g := heldPath(tau)
	g.Totals = []variate.Total{{ID: "net", Categories: []string{"x"}, Less: []string{"lost"}}}
	_, e := depExact(t, g)
	_, _, s := runSummary(t, g, 1<<20, 4)
	sd, st := math.Sqrt(e.VarByTotal[0]), s.Totals["net"]
	if se := st.StdDev / math.Sqrt(2*float64(st.N)); !(math.Abs(st.StdDev-sd) <= 4*se) {
		t.Errorf("net: exact sd %v, simulated %v ± %v", sd, st.StdDev, se)
	}
}

// TestDelayDependence: values and a holding that move together at rank
// 0.7: the exact standard deviation, conditioned on the group's normal,
// is within its error of the simulated, at holdings of 2 days and of 3
// months.
func TestDelayDependence(t *testing.T) {
	for _, med := range []float64{2.0 / 365, 0.25} {
		tau, _ := dist.NewLognormalFromMedianP90(med, 3*med)
		g := heldPath(tau)
		g.Dependence = []variate.Dependence{{Members: []string{"a.x", "a->b", "b.y"}, Rank: 0.7}}
		_, e := depExact(t, g)
		g.Dependence = nil
		_, base := depExact(t, g)
		g.Dependence = []variate.Dependence{{Members: []string{"a.x", "a->b", "b.y"}, Rank: 0.7}}
		_, _, s := runSummary(t, g, 1<<20, 3)
		sd, se := math.Sqrt(e.Var), s.Total.StdDev/math.Sqrt(2*float64(s.Total.N))
		if !(math.Abs(s.Total.StdDev-sd) <= 4*se) || e.Mean != base.Mean {
			t.Errorf("median %v: exact sd %v (mean %v, apart %v), simulated %v ± %v", med, sd, e.Mean, base.Mean, s.Total.StdDev, se)
		}
	}
}

// TestDelayNotComputed: a mixed frequency, a value drawn once a period
// and a schedule leave the variance with holding times NaN.
func TestDelayNotComputed(t *testing.T) {
	tau, _ := dist.NewLognormalFromMedianP90(0.25, 0.75)
	mixed := heldPath(tau)
	f, _ := dist.NewGammaFromMeanSD(8, 2)
	mixed.Sources[0].Frequency = f
	period := heldPath(tau)
	period.Nodes[1].Functions[0].Scope = variate.PerPeriod
	for name, g := range map[string]variate.Graph{"mixed": mixed, "period": period} {
		if _, e := depExact(t, g); !math.IsNaN(e.Var) {
			t.Errorf("%s: var %v, want NaN", name, e.Var)
		}
	}
}

// TestDelayPlan: with holdings of a year, neighbouring periods move
// together, and the plan sizes the total's error by the long-run
// variance per period, λ·E[B²] of a round's whole booking B: the
// variance of sums over 1,000 periods, per period, over 2^23 periods
// is it within 6%, four of its standard errors, where the per-period
// variance is 60% below.
func TestDelayPlan(t *testing.T) {
	tau, _ := dist.NewLognormalFromMedianP90(1, 3)
	g := heldPath(tau)
	cfg := variate.Config{Periods: 1 << 23, AllowUnconverged: true, TargetRSE: 0.001, Series: true}
	m, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 2)
	if err != nil {
		t.Fatal(err)
	}
	var s variate.Summary
	w.Summarize(&s)
	e, _ := m.Analytic()
	const block = 1000
	var s1, s2, k float64
	for start := int64(0); start+block <= r.Series.Periods(); start += block {
		sum := 0.0
		for p := start; p < start+block; p++ {
			sum += r.Series.Total(p)
		}
		s1, s2, k = s1+sum, s2+sum*sum, k+1
	}
	observed := (s2/k - s1*s1/k/k) / block
	for _, row := range s.Plan {
		if row.Criterion != "target-rse" || row.Driver != "total" {
			continue
		}
		planned := float64(row.PeriodsNeeded) * 1e-6 * e.Mean * e.Mean
		if !within(planned, observed, 0.06) {
			t.Errorf("planned long-run variance %v, blocks' %v (per-period %v)", planned, observed, e.Var)
		}
		return
	}
	t.Error("no target-rse row for the total")
}
