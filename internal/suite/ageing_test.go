package suite

import (
	"math"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
)

// ageing is the ageing example: a pump fleet over a twenty-year life
// whose yearly schedule is the power law of shape 2.
func ageing(t *testing.T) variate.Graph {
	t.Helper()
	m, err := config.Load("../../examples/ageing.yaml")
	if err != nil {
		t.Fatal(err)
	}
	return m.Graph
}

// TestAgeingSchedule: the schedule is L·(H(i) − H(i − 1))/H(L) with
// H(t) = t², so phase i's mean is the whole life's mean times that
// value, and the phases' means average to the whole life's.
func TestAgeingSchedule(t *testing.T) {
	g := ageing(t)
	c, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	whole, _ := c.Analytic()
	const life = 20
	h := func(x float64) float64 { return x * x }
	meanSum := 0.0
	for i := 1; i <= life; i++ {
		v := life * (h(float64(i)) - h(float64(i-1))) / h(life)
		if got := g.Schedules[0].Values[i-1]; math.Abs(got-v) > 1e-12 {
			t.Errorf("year %d: schedule %g, power law %g", i, got, v)
		}
		p, err := g.Phase(i)
		if err != nil {
			t.Fatal(err)
		}
		pc, _, err := p.Compile(variate.Config{})
		if err != nil {
			t.Fatal(err)
		}
		e, _ := pc.Analytic()
		if want := whole.Mean * v; math.Abs(e.Mean-want) > 1e-9*want {
			t.Errorf("year %d: mean %g, want %g", i, e.Mean, want)
		}
		meanSum += e.Mean
	}
	if mean := meanSum / life; math.Abs(mean-whole.Mean) > 1e-9*whole.Mean {
		t.Errorf("years' means average to %g, whole life %g", mean, whole.Mean)
	}
}

// TestAgeingLastYear: a run of the last year agrees with its exact mean,
// and the transform applies at the default tolerance, its bracket of
// the 90th percentile holding the simulated one.
func TestAgeingLastYear(t *testing.T) {
	p, err := ageing(t).Phase(20)
	if err != nil {
		t.Fatal(err)
	}
	c, r := run(t, p, variate.Config{Periods: 50000, Workers: 4, Transform: true}, 9)
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
