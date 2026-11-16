package suite

import (
	"context"
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// levelRun runs g and returns its summary and result.
func levelRun(t *testing.T, g variate.Graph, cfg variate.Config) (variate.Summary, *variate.Result) {
	t.Helper()
	c, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(w.Close)
	r, err := w.Run(context.Background(), cfg.Seed)
	if err != nil {
		t.Fatal(err)
	}
	var s variate.Summary
	w.Summarize(&s)
	return s, r
}

// poissonAtLeast is P(N ≥ k) for N Poisson with mean lam.
func poissonAtLeast(lam float64, k int) float64 {
	term, below := math.Exp(-lam), 0.0
	for j := range k {
		below += term
		term *= lam / float64(j+1)
	}
	return 1 - below
}

// countGraph: rounds at rate lam a period reach node a, which books 1.
func countGraph(lam float64) variate.Graph {
	return variate.Graph{ID: "count", Sources: []variate.Source{{ID: "s", Frequency: C(lam), To: "a"}},
		Nodes: []variate.Node{{ID: "a", Functions: []variate.Function{Fn("v", "c", C(1))}}}}
}

// TestLevelPoisson: a period's total and entries are Poisson with mean 3,
// so the chance of reaching each value is the Poisson tail, within four
// standard errors, the error itself the binomial one.
func TestLevelPoisson(t *testing.T) {
	at := []variate.Level{{Kind: "total", X: 5}, {Kind: "entries", Name: "a", X: 1}, {Kind: "category", Name: "c", X: 8}, {Kind: "total", X: 4.5}}
	s, _ := levelRun(t, countGraph(3), variate.Config{Periods: 100000, Seed: 3, At: at})
	want := []float64{poissonAtLeast(3, 5), poissonAtLeast(3, 1), poissonAtLeast(3, 8), poissonAtLeast(3, 5)}
	if len(s.At) != len(at) {
		t.Fatalf("%d levels reported, want %d", len(s.At), len(at))
	}
	for i, a := range s.At {
		binomial := math.Sqrt(want[i] * (1 - want[i]) / float64(a.N))
		if a.Level != at[i] || math.Abs(a.P-want[i]) > 4*binomial || math.Abs(a.StdErr-binomial) > 0.05*binomial {
			t.Errorf("%+v: P %v ± %v, want %v ± %v", a.Level, a.P, a.StdErr, want[i], binomial)
		}
	}
}

// TestLevelStarted: in a started run the chance is over repeats of the
// horizon, whose total is Poisson with mean 3 a period times 4 periods.
func TestLevelStarted(t *testing.T) {
	g := countGraph(3)
	g.Start = variate.Start{Horizon: 4}
	s, _ := levelRun(t, g, variate.Config{Repeats: 40000, Seed: 5, At: []variate.Level{{Kind: "total", X: 15}}})
	a, want := s.At[0], poissonAtLeast(12, 15)
	if a.N != 40000 || math.Abs(a.P-want) > 4*a.StdErr {
		t.Errorf("repeats %d, P(T ≥ 15) %v ± %v, want %v", a.N, a.P, a.StdErr, want)
	}
}

// TestLevelSeries: with holding times the periods are counted as the
// series keeps them, exactly, and the error is the batch-means one,
// above the binomial as a period's per-time value runs into the next.
func TestLevelSeries(t *testing.T) {
	hold, _ := dist.NewExponential(3)
	g := variate.Graph{ID: "held", Sources: []variate.Source{{ID: "s", Frequency: C(2), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{{Name: "r", Category: "c", Model: C(1), Basis: variate.PerTime}}, Outcomes: []variate.Outcome{{To: "b", Holding: hold}}},
			{ID: "b"},
		}}
	const x = 8
	s, r := levelRun(t, g, variate.Config{Periods: 64000, Seed: 9, Series: true, At: []variate.Level{{Kind: "total", X: x}}})
	reached := 0
	for p := range r.Series.Periods() {
		if r.Series.Total(p) >= x {
			reached++
		}
	}
	a := s.At[0]
	if got := a.P * float64(a.N); math.Abs(got-float64(reached)) > 1e-6 || a.N != r.Series.Periods() {
		t.Errorf("%v periods of %d reach %v, the series has %d of %d", got, a.N, x, reached, r.Series.Periods())
	}
	if binomial := math.Sqrt(a.P * (1 - a.P) / float64(a.N)); !(a.StdErr > binomial) {
		t.Errorf("stderr %v, not above the binomial %v for correlated periods", a.StdErr, binomial)
	}
}

// TestLevelRefusals: a quantity the model does not have, a breakdown the
// run leaves out, and a value that is not a finite number are refused.
func TestLevelRefusals(t *testing.T) {
	for _, c := range []struct {
		name string
		cfg  variate.Config
		want string
	}{
		{"no node", variate.Config{At: []variate.Level{{Kind: "entries", Name: "z", X: 1}}}, "no quantity entries:z"},
		{"no kind", variate.Config{At: []variate.Level{{Kind: "queue", Name: "a", X: 1}}}, "no quantity queue:a"},
		{"breakdown off", variate.Config{NoBreakdown: true, At: []variate.Level{{Kind: "category", Name: "c", X: 1}}}, "no quantity category:c"},
		{"not finite", variate.Config{At: []variate.Level{{Kind: "total", X: math.Inf(1)}}}, "finite"},
	} {
		if _, _, err := countGraph(1).Compile(c.cfg); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want one naming %q", c.name, err, c.want)
		}
	}
}
