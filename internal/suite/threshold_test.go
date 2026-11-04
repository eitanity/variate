package suite

import (
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// thresholded is repairs at an uncertain rate, each booking a fee and a
// lognormal cost; a cost above 5000 goes on to a claim, which books an
// audit, and one at or below it ends the round.
func thresholded(cost dist.Distribution) variate.Graph {
	return variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: Pert(3, 5, 8), To: "repair"}},
		Nodes: []variate.Node{
			{ID: "repair", Functions: []variate.Function{Fn("fee", "admin", C(100)), Fn("cost", "parts", cost)},
				Threshold: &variate.Threshold{Function: "cost", Over: 5000, Above: "claim", Below: "done"}},
			{ID: "claim", Functions: []variate.Function{Fn("audit", "admin", Pert(1e3, 2e3, 5e3))}},
			{ID: "done"},
		},
	}
}

// handSplit is the same model split by hand: the repair goes to one of
// two nodes with the probabilities of the cost's two sides, each booking
// the cost's half on its side.
func handSplit(t *testing.T, cost dist.Distribution) variate.Graph {
	t.Helper()
	hi, s, err := dist.NewSplit(cost, 5000, true)
	if err != nil {
		t.Fatal(err)
	}
	lo, _, err := dist.NewSplit(cost, 5000, false)
	if err != nil {
		t.Fatal(err)
	}
	return variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: Pert(3, 5, 8), To: "repair"}},
		Nodes: []variate.Node{
			{ID: "repair", Functions: []variate.Function{Fn("fee", "admin", C(100))},
				Outcomes: []variate.Outcome{{To: "hi", P: C(s)}, {To: "lo", P: C(1 - s)}}},
			{ID: "hi", Functions: []variate.Function{Fn("cost", "parts", hi)}, Next: "claim"},
			{ID: "lo", Functions: []variate.Function{Fn("cost", "parts", lo)}, Next: "done"},
			{ID: "claim", Functions: []variate.Function{Fn("audit", "admin", Pert(1e3, 2e3, 5e3))}},
			{ID: "done"},
		},
	}
}

func lognormal(median, p90 float64) dist.Distribution {
	d, _ := dist.NewLognormalFromMedianP90(median, p90)
	return d
}

// TestThresholdExact: a threshold's exact mean, variance, P(T = 0),
// category means and the claim's entries equal those of the model split
// by hand, for a continuous cost and for one with an atom at the
// threshold, which goes below; the repair node's contribution is the
// hand split's repair and halves together.
func TestThresholdExact(t *testing.T) {
	atom, _ := dist.NewCategorical([]float64{1000, 5000, 9000}, []float64{0.5, 0.3, 0.2})
	for _, tc := range []struct {
		name  string
		cost  dist.Distribution
		above float64
	}{
		{"lognormal", lognormal(2000, 8000), lognormal(2000, 8000).Survival(5000)},
		{"atom at the threshold", atom, 0.2},
	} {
		c, _, err := thresholded(tc.cost).Compile(variate.Config{})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		h, _, err := handSplit(t, tc.cost).Compile(variate.Config{})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		e, _ := c.Analytic()
		f, _ := h.Analytic()
		for _, q := range []struct {
			what string
			a, b float64
		}{
			{"mean", e.Mean, f.Mean},
			{"variance", e.Var, f.Var},
			{"P(T = 0)", e.PZero(), f.PZero()},
			{"admin mean", e.MeanByCategory[0], f.MeanByCategory[0]},
			{"parts mean", e.MeanByCategory[1], f.MeanByCategory[1]},
			{"admin variance", e.VarByCategory[0], f.VarByCategory[0]},
			{"parts variance", e.VarByCategory[1], f.VarByCategory[1]},
			{"source variance", e.VarBySource[0], f.VarBySource[0]},
			{"claim entries", e.EntriesPer[1], f.EntriesPer[3]},
			{"repair contribution", e.Contrib[0], f.Contrib[0] + f.Contrib[1] + f.Contrib[2]},
			{"draws", e.DrawsMean, f.DrawsMean - f.EntriesPer[1] - f.EntriesPer[2] + e.EntriesPer[0]},
		} {
			if !near(q.a, q.b, 1e-9) {
				t.Errorf("%s: %s %g, hand split %g", tc.name, q.what, q.a, q.b)
			}
		}
		if got := e.OutcomeRate[0] / e.EntriesPer[0]; !near(got, tc.above, 1e-12) {
			t.Errorf("%s: share above %g, want %g", tc.name, got, tc.above)
		}
	}
}

// TestThresholdTransform: the transform applies to the thresholded
// model, and its quantile brackets overlap the hand split's.
func TestThresholdTransform(t *testing.T) {
	cost := lognormal(2000, 8000)
	_, a := transformOf(t, thresholded(cost), variate.TransformConfig{})
	_, b := transformOf(t, handSplit(t, cost), variate.TransformConfig{})
	if !a.Applicable || !b.Applicable {
		t.Fatalf("refused: %q, %q", a.Reason, b.Reason)
	}
	for _, p := range []float64{0.5, 0.9, 0.99, 0.999} {
		alo, ahi := a.Total.Quantile(p)
		blo, bhi := b.Total.Quantile(p)
		if alo-a.H > bhi || blo-b.H > ahi {
			t.Errorf("P%g: [%g, %g] and hand split [%g, %g] do not overlap", 100*p, alo, ahi, blo, bhi)
		}
	}
}

// TestThresholdRun: a run branches on the value it books, so the share
// above is P(cost > 5000), and the total's mean and P90 agree with the
// exact ones.
func TestThresholdRun(t *testing.T) {
	cost := lognormal(2000, 8000)
	c, r := run(t, thresholded(cost), variate.Config{Periods: 100000, Workers: 4, Transform: true}, 3)
	e, _ := c.Analytic()
	if z := (r.Total.Mean() - e.Mean) / r.Total.StdErr(false); math.Abs(z) > 5 {
		t.Errorf("simulated mean %g, exact %g (z %.2f)", r.Total.Mean(), e.Mean, z)
	}
	n := float64(r.Reach[0][0])
	above := float64(r.OutcomeCount[0][0]) / n
	s := cost.Survival(5000)
	if se := math.Sqrt(s * (1 - s) / n); math.Abs(above-s) > 5*se {
		t.Errorf("share above %g, want %g ± %g", above, s, se)
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

// TestThresholdErrors: a threshold beside outcomes, on a sequential
// node, on a missing or period-scoped function, beside a per-time
// function, or naming a missing node is refused with its reason.
func TestThresholdErrors(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(n *variate.Node)
		want string
	}{
		{"outcomes", func(n *variate.Node) { n.Outcomes = []variate.Outcome{{To: "done"}} }, "threshold and outcomes"},
		{"sequential", func(n *variate.Node) { n.Mode = variate.Sequential }, "sequential"},
		{"missing function", func(n *variate.Node) { n.Threshold.Function = "nope" }, "threshold function nope not found"},
		{"missing node", func(n *variate.Node) { n.Threshold.Above = "nowhere" }, "threshold node nowhere not found"},
		{"infinite", func(n *variate.Node) { n.Threshold.Over = math.Inf(1) }, "finite"},
		{"period scope", func(n *variate.Node) { n.Functions[1].Scope = variate.PerPeriod }, "drawn per event"},
		{"per time", func(n *variate.Node) {
			n.Functions = append(n.Functions, variate.Function{Name: "hire", Category: "admin", Model: C(1), Basis: variate.PerTime})
		}, "no holding time"},
	} {
		g := thresholded(lognormal(2000, 8000))
		th := *g.Nodes[0].Threshold
		g.Nodes[0].Threshold = &th
		g.Nodes[0].Functions = append([]variate.Function(nil), g.Nodes[0].Functions...)
		tc.edit(&g.Nodes[0])
		_, _, err := g.Compile(variate.Config{})
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error %v, want %q", tc.name, err, tc.want)
		}
	}
}
