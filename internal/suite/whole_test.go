package suite

import (
	"math"
	"testing"

	"code.eitanity.com/variate"
)

// budget is the Panjer model's lattice total with a trigger on the whole
// of it: a period whose total is above 6 sets off a review that books 2
// or 5 with even chances.
func budget() variate.Graph {
	g := latticeSeverity(C(2.5))
	g.Nodes = append(g.Nodes,
		variate.Node{ID: "review", Trigger: &variate.Trigger{Over: 6, Inputs: []variate.TriggerInput{{Node: "b"}, {Node: "c"}}},
			Outcomes: []variate.Outcome{{To: "minor", P: C(0.5)}, {To: "major"}}},
		variate.Node{ID: "minor", Functions: []variate.Function{Fn("two", "k", C(2))}},
		variate.Node{ID: "major", Functions: []variate.Function{Fn("five", "k", C(5))}},
	)
	return g
}

// budgetMasses is the exact law of the budget model's total on the
// integers up to n: Panjer's for the sum, with each mass above 6 moved
// up by 2 or 5.
func budgetMasses(n int) []float64 {
	s := panjer(0, 2.5, math.Exp(-2.5), []float64{0, 0.6, 0, 0.4}, n)
	t := make([]float64, n+6)
	for k, x := range s {
		if k <= 6 {
			t[k] += x
			continue
		}
		t[k+2] += x / 2
		t[k+5] += x / 2
	}
	return t
}

// TestWholeTotalExact: a trigger summing the whole total has the
// variance of S + 1{S > t}·W within the transform's Tol (one term of it
// is read from the sum's bracket), the mean exact, and the transform's
// bracket on the total holds the exact law at every point up to its
// 0.999 quantile, within Tol up to the 0.99 one.
func TestWholeTotalExact(t *testing.T) {
	c, tr := transformOf(t, budget(), variate.TransformConfig{})
	if !tr.Applicable {
		t.Fatal(tr.Reason)
	}
	masses := budgetMasses(400)
	mean, second := 0.0, 0.0
	for k, x := range masses {
		mean += float64(k) * x
		second += float64(k*k) * x
	}
	e, _ := c.Analytic()
	if !near(e.Mean, mean, 1e-9) || !near(e.Var, second-mean*mean, 1e-3) {
		t.Errorf("mean %g, variance %g; exact %g, %g", e.Mean, e.Var, mean, second-mean*mean)
	}
	step := math.Round(1 / tr.H)
	if math.Abs(step*tr.H-1) > 1e-12 {
		t.Fatalf("span %g does not divide 1", tr.H)
	}
	_, q999 := tr.Total.Quantile(0.999)
	_, q99 := tr.Total.Quantile(0.99)
	cdf := 0.0
	for k := 0; float64(k) <= q999; k++ {
		ec := 1 - cdf
		cdf += masses[k]
		x := int(float64(k) * step)
		lo, hi := tr.Total.Lower[x], tr.Total.Upper[x]
		if lo > ec || hi < ec {
			t.Errorf("P(T ≥ %d) [%.12g, %.12g] excludes the exact %.12g", k, lo, hi, ec)
		}
		if float64(k) <= q99 && hi-lo > 1e-3*hi {
			t.Errorf("P(T ≥ %d) [%.12g, %.12g] wider than Tol", k, lo, hi)
		}
	}
}

// TestWholeTotalRun: a run of a lognormal sum with the trigger agrees
// with the exact standard deviation and with the transform's quantiles;
// a second trigger, or a sum of part of the total, leaves the variance
// NaN and the transform refused.
func TestWholeTotalRun(t *testing.T) {
	g := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(6), To: "repair"}},
		Nodes: []variate.Node{
			{ID: "repair", Functions: []variate.Function{Fn("parts", "cost", lognormal(8000, 30000)), Fn("callout", "cost", Pert(500, 1200, 4000))}},
			{ID: "review", Functions: []variate.Function{Fn("fee", "cost", Pert(5000, 1e4, 3e4))},
				Trigger: &variate.Trigger{Over: 1.5e5, Inputs: []variate.TriggerInput{{Node: "repair"}}}},
		},
	}
	c, r := run(t, g, variate.Config{Periods: 200000, Workers: 4, Transform: true}, 11)
	e, _ := c.Analytic()
	sd := math.Sqrt(e.Var)
	if got := r.Total.StdDev(); math.Abs(got-sd) > 0.02*sd {
		t.Errorf("simulated sd %g, exact %g", got, sd)
	}
	tr := c.Transformed()
	if tr == nil || !tr.Applicable {
		t.Fatalf("no transform: %+v", tr)
	}
	for _, p := range []float64{0.5, 0.9, 0.99} {
		lo, hi := tr.Total.Quantile(p)
		q, _ := r.Total.Hist.Quantile(p)
		if q < 0.98*lo || q > 1.02*hi {
			t.Errorf("simulated P%g %g outside [%g, %g]", 100*p, q, lo, hi)
		}
	}
	part := g
	part.Nodes = append([]variate.Node(nil), g.Nodes...)
	part.Nodes[1].Trigger = &variate.Trigger{Over: 1.5e5, Inputs: []variate.TriggerInput{{Node: "repair", Function: "parts"}}}
	two := g
	two.Nodes = append(append([]variate.Node(nil), g.Nodes...), variate.Node{ID: "audit", Trigger: &variate.Trigger{Over: 3e5, Inputs: []variate.TriggerInput{{Node: "repair"}}}})
	for name, h := range map[string]variate.Graph{"part": part, "two": two} {
		m, tr := transformOf(t, h, variate.TransformConfig{})
		e, _ := m.Analytic()
		if !math.IsNaN(e.Var) || tr.Applicable {
			t.Errorf("%s: variance %g, transform applies %v", name, e.Var, tr.Applicable)
		}
	}
}
