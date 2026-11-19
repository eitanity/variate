package suite

import (
	"context"
	"math"
	"sort"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// lnParams is a lognormal's μ and σ from its mean and sd.
func lnParams(mean, sd float64) (mu, sigma float64) {
	s2 := math.Log(1 + sd*sd/(mean*mean))
	return math.Log(mean) - s2/2, math.Sqrt(s2)
}

// lnCov is the covariance of two lognormals of normal correlation ρ:
// E[X]E[Y](e^(ρσ₁σ₂) − 1).
func lnCov(m1, s1, m2, s2, rho float64) float64 {
	_, a := lnParams(m1, s1)
	_, b := lnParams(m2, s2)
	return m1 * m2 * math.Expm1(rho*a*b)
}

// within reports whether got is within tol·|want| of want, NaN never.
func within(got, want, tol float64) bool {
	return math.Abs(got-want) <= tol*math.Abs(want)
}

// rhoOf is the normal correlation of a rank correlation.
func rhoOf(rank float64) float64 { return 2 * math.Sin(math.Pi*rank/6) }

// pairGraph: 4 rounds a period book X, lognormal of mean 3 and sd 2,
// and Y, lognormal of mean 5 and sd 4, at one node, moving together at
// the rank.
func pairGraph(rank float64) variate.Graph {
	x, _ := dist.NewLognormalFromMeanSD(3, 2)
	y, _ := dist.NewLognormalFromMeanSD(5, 4)
	return variate.Graph{ID: "pair",
		Sources:    []variate.Source{{ID: "s", Frequency: C(4), To: "a"}},
		Nodes:      []variate.Node{{ID: "a", Functions: []variate.Function{Fn("x", "x", x), Fn("y", "y", y)}}},
		Dependence: []variate.Dependence{{Members: []string{"a.x", "a.y"}, Rank: rank}},
	}
}

// depExact compiles g and gives its exact figures.
func depExact(t *testing.T, g variate.Graph) (*variate.Compiled, *variate.Expectation) {
	t.Helper()
	m, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, err := m.Analytic()
	if err != nil {
		t.Fatal(err)
	}
	return m, e
}

// TestDependencePair: at one node the variance is the independent one
// plus 2λ·Cov(X, Y), the lognormals' closed form at every rank, and the
// mean and each category's variance are unchanged.
func TestDependencePair(t *testing.T) {
	_, base := depExact(t, pairGraph(0))
	for _, rank := range []float64{0.25, 0.6, 1} {
		m, e := depExact(t, pairGraph(rank))
		want := base.Var + 2*4*lnCov(3, 2, 5, 4, rhoOf(rank))
		if e.Mean != base.Mean || !within(e.Var, want, 1e-9) {
			t.Errorf("rank %v: mean %v var %v, want %v and %v", rank, e.Mean, e.Var, base.Mean, want)
		}
		for _, c := range []string{"x", "y"} {
			i := indexOf(m.Categories(), c)
			if !within(e.VarByCategory[i], base.VarByCategory[i], 1e-12) {
				t.Errorf("rank %v: %s var %v, independent %v", rank, c, e.VarByCategory[i], base.VarByCategory[i])
			}
		}
	}
}

// TestDependenceGammas: two gammas of one shape in lockstep are one a
// multiple of the other, Cov = (θ₂/θ₁)·Var(X): shape 4, scales 1 and 3.
func TestDependenceGammas(t *testing.T) {
	x, _ := dist.NewGammaFromMeanSD(4, 2)
	y, _ := dist.NewGammaFromMeanSD(12, 6)
	g := pairGraph(1)
	g.Nodes[0].Functions = []variate.Function{Fn("x", "x", x), Fn("y", "y", y)}
	_, e := depExact(t, g)
	g.Dependence = nil
	_, base := depExact(t, g)
	if want := base.Var + 2*4*3*4; !within(e.Var, want, 1e-9) {
		t.Errorf("var %v, want %v", e.Var, want)
	}
}

// pathGraph: at a, X lognormal of mean 3 and sd 2; then b with
// probability 0.6, booking Y lognormal of mean 5 and sd 4, or c,
// booking Z gamma of mean 2 and sd 1. X, Y and Z move together.
func pathGraph(rank float64) variate.Graph {
	x, _ := dist.NewLognormalFromMeanSD(3, 2)
	y, _ := dist.NewLognormalFromMeanSD(5, 4)
	z, _ := dist.NewLognormalFromMeanSD(2, 1)
	return variate.Graph{ID: "path",
		Sources: []variate.Source{{ID: "s", Frequency: C(4), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{Fn("x", "x", x)}, Outcomes: []variate.Outcome{{To: "b", P: C(0.6)}, {To: "c"}}},
			{ID: "b", Functions: []variate.Function{Fn("y", "y", y)}},
			{ID: "c", Functions: []variate.Function{Fn("z", "z", z)}},
		},
		Dependence: []variate.Dependence{{Members: []string{"a.x", "b.y", "c.z"}, Rank: rank}},
	}
}

// TestDependencePath: along a path the variance gains each pair's
// covariance times the chance the round books both: 2λ·(0.6·Cov(X, Y)
// + 0.4·Cov(X, Z)), Y and Z never in one round; the mean is unchanged.
func TestDependencePath(t *testing.T) {
	_, base := depExact(t, pathGraph(0))
	for _, rank := range []float64{0.5, 1} {
		_, e := depExact(t, pathGraph(rank))
		rho := rhoOf(rank)
		want := base.Var + 2*4*(0.6*lnCov(3, 2, 5, 4, rho)+0.4*lnCov(3, 2, 2, 1, rho))
		if !within(e.Mean, base.Mean, 1e-12) || !within(e.Var, want, 1e-9) {
			t.Errorf("rank %v: mean %v var %v, want %v and %v", rank, e.Mean, e.Var, base.Mean, want)
		}
	}
}

// TestDependenceSimulated: over 400,000 periods the simulated mean and
// standard deviation are within their errors of the exact ones, at one
// node and along the path.
func TestDependenceSimulated(t *testing.T) {
	for name, g := range map[string]variate.Graph{"pair": pairGraph(0.8), "path": pathGraph(0.8)} {
		_, e := depExact(t, g)
		_, _, s := runSummary(t, g, 400000, 13)
		sd := math.Sqrt(e.Var)
		if !(math.Abs(s.Total.Mean-e.Mean) <= 4*s.Total.StdErr) || !within(s.Total.StdDev, sd, 0.02) {
			t.Errorf("%s: %v ± %v sd %v, want %v and %v", name, s.Total.Mean, s.Total.StdErr, s.Total.StdDev, e.Mean, sd)
		}
	}
}

// rankCorr is Spearman's rank correlation of two samples.
func rankCorr(x, y []float64) float64 {
	rank := func(v []float64) []float64 {
		idx := make([]int, len(v))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return v[idx[a]] < v[idx[b]] })
		out := make([]float64, len(v))
		for r, i := range idx {
			out[i] = float64(r)
		}
		return out
	}
	rx, ry := rank(x), rank(y)
	n := float64(len(x))
	var sx, sy, sxx, syy, sxy float64
	for i := range rx {
		sx, sy, sxx, syy, sxy = sx+rx[i], sy+ry[i], sxx+rx[i]*rx[i], syy+ry[i]*ry[i], sxy+rx[i]*ry[i]
	}
	return (sxy/n - sx*sy/n/n) / math.Sqrt((sxx/n-sx*sx/n/n)*(syy/n-sy*sy/n/n))
}

// togetherGraph: at a, a value X and a rate over the holding τ to b, where
// Y is booked; rate the rate's law, members the group.
func togetherGraph(rate dist.Distribution, members []string) variate.Graph {
	x, _ := dist.NewLognormalFromMeanSD(3, 2)
	tau, _ := dist.NewGammaFromMeanSD(0.2, 0.1)
	y, _ := dist.NewLognormalFromMeanSD(5, 4)
	r := Fn("r", "lost", rate)
	r.Basis = variate.PerTime
	return variate.Graph{ID: "held",
		Sources: []variate.Source{{ID: "s", Frequency: C(4), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{Fn("x", "x", x), r}, Next: "b", NextHolding: tau},
			{ID: "b", Functions: []variate.Function{Fn("y", "y", y)}},
		},
		Dependence: []variate.Dependence{{Members: members, Rank: 0.7}},
	}
}

// togetherRounds runs g watching b, giving each round's value at a, its
// holding and its value at b.
func togetherRounds(t *testing.T, g variate.Graph) (va, tau, vb []float64) {
	t.Helper()
	obs := observerFunc(func(r *variate.Round) error {
		if len(r.Steps) == 2 {
			va = append(va, r.Steps[0].Value)
			tau = append(tau, r.Steps[1].T-r.Steps[0].T)
			vb = append(vb, r.Steps[1].Value)
		}
		return nil
	})
	cfg := variate.Config{Periods: 20000, AllowUnconverged: true, Observer: obs}
	m, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if _, err := w.Run(context.Background(), 5); err != nil {
		t.Fatal(err)
	}
	return va, tau, vb
}

// TestDependenceHeld: a holding and a rate as members move with the
// values: over 80,000 rounds the rank correlations of the value at a,
// the holding and the value at b, and of the rate and the value at b,
// are 0.7 within their errors.
func TestDependenceHeld(t *testing.T) {
	va, tau, vb := togetherRounds(t, togetherGraph(C(2), []string{"a.x", "a->b", "b.y"}))
	x := make([]float64, len(va))
	for i := range va {
		x[i] = va[i] - 2*tau[i]
	}
	se := (1 - 0.49) / math.Sqrt(float64(len(x)))
	for name, pair := range map[string][2][]float64{"x, τ": {x, tau}, "τ, y": {tau, vb}, "x, y": {x, vb}} {
		if got := rankCorr(pair[0], pair[1]); math.Abs(got-0.7) > 4*se {
			t.Errorf("%s: rank correlation %v, want 0.7 ± %v", name, got, se)
		}
	}
	r, _ := dist.NewLognormalFromMeanSD(10, 5)
	g := togetherGraph(r, []string{"a.r", "b.y"})
	g.Nodes[0].Functions = g.Nodes[0].Functions[1:]
	va, tau, vb = togetherRounds(t, g)
	rate := make([]float64, len(va))
	for i := range va {
		rate[i] = va[i] / tau[i]
	}
	if got := rankCorr(rate, vb); math.Abs(got-0.7) > 4*se {
		t.Errorf("rate, y: rank correlation %v, want 0.7 ± %v", got, se)
	}
}

// TestDependenceZero: a group of rank 0 is none, the run the same to
// the bit.
func TestDependenceZero(t *testing.T) {
	g := pathGraph(0)
	_, _, with := runSummary(t, g, 20000, 3)
	g.Dependence = nil
	_, _, without := runSummary(t, g, 20000, 3)
	if with.Total != without.Total || with.ByCategory["y"] != without.ByCategory["y"] {
		t.Errorf("rank 0 %v, without %v", with.Total, without.Total)
	}
}

// TestDependenceRefusals: a group is refused that names too few
// members, a rank outside [0, 1], a member twice or one a round does
// not draw once of its own; a node's rate and holding together; and the
// transform says why it does not apply.
func TestDependenceRefusals(t *testing.T) {
	cases := map[string]struct {
		edit func(*variate.Graph)
		want string
	}{
		"one":      {func(g *variate.Graph) { g.Dependence[0].Members = []string{"a.x"} }, "two members"},
		"rank":     {func(g *variate.Graph) { g.Dependence[0].Rank = 1.5 }, "rank must be in [0, 1]"},
		"twice":    {func(g *variate.Graph) { g.Dependence[0].Members = []string{"a.x", "a.x"} }, "named twice"},
		"node":     {func(g *variate.Graph) { g.Dependence[0].Members[2] = "q.y" }, "node q not found"},
		"function": {func(g *variate.Graph) { g.Dependence[0].Members[2] = "b.w" }, "function w not found"},
		"shape":    {func(g *variate.Graph) { g.Dependence[0].Members[2] = "by" }, "not node.function"},
		"no hold":  {func(g *variate.Graph) { g.Dependence[0].Members[1] = "a->q" }, "no holding to q"},
		"untimed":  {func(g *variate.Graph) { g.Dependence[0].Members[1] = "a->b" }, "has no holding time"},
		"period": {func(g *variate.Graph) {
			g.Nodes[1].Functions[0].Scope = variate.PerPeriod
		}, "drawn once a period"},
		"operand": {func(g *variate.Graph) {
			g.Nodes[1].Functions = append(g.Nodes[1].Functions, variate.Function{Name: "z", Category: "z", Expr: "y * 2"})
		}, "an expression's operand"},
		"trigger": {func(g *variate.Graph) {
			g.Nodes = append(g.Nodes, variate.Node{ID: "audit", Trigger: &variate.Trigger{Over: 1, Inputs: []variate.TriggerInput{{Node: "b"}}}})
		}, "a trigger reads"},
	}
	for name, c := range cases {
		g := pathGraph(0.5)
		c.edit(&g)
		if _, _, err := g.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want %q", name, err, c.want)
		}
	}
	held := togetherGraph(C(2), []string{"a.r", "a->b"})
	if _, _, err := held.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), "a rate and its holding") {
		t.Errorf("rate and holding: %v", err)
	}
	_, tr := transformOf(t, pairGraph(0.5), variate.TransformConfig{})
	if tr.Applicable || !strings.Contains(tr.Reason, "move together") {
		t.Errorf("applicable %v: %s", tr.Applicable, tr.Reason)
	}
}

// TestDependenceZeroChance: P(T = 0) is not given when a member can be
// 0, its chance of 0 with the others no longer theirs apart; it is with
// members never 0.
func TestDependenceZeroChance(t *testing.T) {
	if _, e := depExact(t, pairGraph(0.5)); math.IsNaN(e.PZero()) {
		t.Errorf("members never 0: P(T = 0) %v", e.PZero())
	}
	g := pairGraph(0.5)
	g.Nodes[0].Functions[0].Model = C(0)
	if _, e := depExact(t, g); !math.IsNaN(e.PZero()) {
		t.Errorf("a member always 0: P(T = 0) %v, want NaN", e.PZero())
	}
}
