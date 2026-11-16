package suite

import (
	"context"
	"math"
	"slices"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// splitGraph: rounds at rate l1 a period book 1 to category c1 at node a,
// and at rate l2 book 1 to c2 at node b, so a period's total is Poisson
// with mean l1 + l2, and given the total each round is c1's with chance
// l1/(l1 + l2).
func splitGraph(l1, l2 float64) variate.Graph {
	return variate.Graph{ID: "split",
		Sources: []variate.Source{{ID: "s1", Frequency: C(l1), To: "a"}, {ID: "s2", Frequency: C(l2), To: "b"}},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{Fn("v", "c1", C(1))}},
			{ID: "b", Functions: []variate.Function{Fn("v", "c2", C(1))}},
		}}
}

// part finds a condition's figures for a quantity.
func part(t *testing.T, s variate.WhenStat, kind, name string) variate.WhenPart {
	t.Helper()
	i := slices.IndexFunc(s.Parts, func(p variate.WhenPart) bool { return p.Kind == kind && p.Name == name })
	if i < 0 {
		t.Fatalf("%s: no %s:%s among %v", variate.WhenName(s.When), kind, name, s.Parts)
	}
	return s.Parts[i]
}

// poissonBand is, for N Poisson with mean lam and k ≤ N < k2 (k2 < 0 for
// no upper bound), the chance, E[N] and E[N(N−1)] in it: λ·P(k−1 ≤ N <
// k2−1) and λ²·P(k−2 ≤ N < k2−2) over the chance.
func poissonBand(lam float64, k, k2 int) (p, mean, fact2 float64) {
	at := func(j int) float64 {
		if j <= 0 {
			return 1
		}
		return poissonAtLeast(lam, j)
	}
	band := func(lo, hi int) float64 {
		if k2 < 0 {
			return at(lo)
		}
		return at(lo) - at(hi)
	}
	p = band(k, k2)
	return p, lam * band(k-1, k2-1) / p, lam * lam * band(k-2, k2-2) / p
}

// checkSplit checks a condition on splitGraph's total against the
// Poisson figures: the chance, the total's and c1's means within four
// standard errors, the total's error within 10% of the conditional
// spread over √Met, and c1's share, l1/(l1 + l2) in every period, exact
// with an error of 0 to rounding.
func checkSplit(t *testing.T, s variate.WhenStat, l1, lam float64, k, k2 int) {
	t.Helper()
	p, mean, fact2 := poissonBand(lam, k, k2)
	name := variate.WhenName(s.When)
	if s.Reason != "" || math.Abs(s.P-p) > 4*s.StdErr {
		t.Errorf("%s: P %v ± %v, want %v (%q)", name, s.P, s.StdErr, p, s.Reason)
	}
	tot := part(t, s, "total", "")
	sd := math.Sqrt(fact2 + mean - mean*mean)
	if se := sd / math.Sqrt(float64(s.Met)); math.Abs(tot.Mean-mean) > 4*se || math.Abs(tot.MeanStdErr-se) > 0.1*se {
		t.Errorf("%s: mean total %v ± %v, want %v ± %v", name, tot.Mean, tot.MeanStdErr, mean, se)
	}
	c1 := part(t, s, "category", "c1")
	if want := l1 / lam * mean; math.Abs(c1.Mean-want) > 4*c1.MeanStdErr {
		t.Errorf("%s: mean c1 %v ± %v, want %v", name, c1.Mean, c1.MeanStdErr, want)
	}
	if want := l1 / lam; math.Abs(c1.Share-want) > 4*c1.ShareStdErr {
		t.Errorf("%s: c1's share %v ± %v, want %v", name, c1.Share, c1.ShareStdErr, want)
	}
	if !math.IsNaN(tot.Share) {
		t.Errorf("%s: the total has a share %v", name, tot.Share)
	}
}

// TestWhenPoisson: over the periods whose Poisson total of mean 3 is at
// least 6, in [2, 5), and at its 1-in-100 level, which for this law is 8
// (P(N ≥ 8) = 0.0119, P(N ≥ 9) = 0.0038), the run's figures match the
// conditional Poisson ones.
func TestWhenPoisson(t *testing.T) {
	const l1, l2 = 1, 2
	when := []variate.When{
		{Kind: "total", From: variate.Bound{X: 6}},
		{Kind: "total", From: variate.Bound{X: 2}, To: &variate.Bound{X: 5}},
		{Kind: "total", From: variate.Bound{OneIn: 100}},
	}
	s, _ := levelRun(t, splitGraph(l1, l2), variate.Config{Periods: 400000, Seed: 3, When: when})
	if len(s.When) != len(when) {
		t.Fatalf("%d conditions reported, want %d", len(s.When), len(when))
	}
	checkSplit(t, s.When[0], l1, l1+l2, 6, -1)
	checkSplit(t, s.When[1], l1, l1+l2, 2, 5)
	if lv := s.When[2].Lower; lv.X != 8 || lv.StdErr != 0 {
		t.Errorf("1-in-100 level %v ± %v, want 8 exactly", lv.X, lv.StdErr)
	}
	checkSplit(t, s.When[2], l1, l1+l2, 8, -1)
}

// TestWhenStarted: in a started run the condition is on the horizon's
// total, Poisson with mean 3 a period over 4 periods.
func TestWhenStarted(t *testing.T) {
	g := splitGraph(1, 2)
	g.Start = variate.Start{Horizon: 4}
	when := []variate.When{{Kind: "total", From: variate.Bound{X: 15}}, {Kind: "total", From: variate.Bound{OneIn: 20}}}
	s, _ := levelRun(t, g, variate.Config{Repeats: 100000, Seed: 5, When: when})
	if s.When[0].N != 100000 {
		t.Errorf("repeats %d, want 100000", s.When[0].N)
	}
	checkSplit(t, s.When[0], 4, 12, 15, -1)
	// P(N ≥ 18) = 0.0630 and P(N ≥ 19) = 0.0374 for Poisson(12).
	if lv := s.When[1].Lower; lv.X != 18 {
		t.Errorf("1-in-20 level %v, want 18", lv.X)
	}
	checkSplit(t, s.When[1], 4, 12, 18, -1)
}

// burstGraph: rounds at rate 2 book an exponential value at a, and at a
// per-time rate of 1 at b over an exponential holding, so periods are
// continuous and, through the holding, correlated.
func burstGraph() variate.Graph {
	v, _ := dist.NewExponential(1)
	hold, _ := dist.NewExponential(3)
	return variate.Graph{ID: "held", Sources: []variate.Source{{ID: "s", Frequency: C(2), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{Fn("v", "lump", v)}, Outcomes: []variate.Outcome{{To: "b"}}},
			{ID: "b", Functions: []variate.Function{{Name: "r", Category: "rate", Model: C(1), Basis: variate.PerTime}}, Outcomes: []variate.Outcome{{To: "c", Holding: hold}}},
			{ID: "c"},
		}}
}

// TestWhenSeries: the run's figures are those of the periods it kept,
// exactly: a return period's level is the ⌈n/N⌉-th largest total, and the
// periods at or above it, and those of a value and a band, have the
// series' count and means. With the holding the errors are the batch
// groups', and finite.
func TestWhenSeries(t *testing.T) {
	when := []variate.When{
		{Kind: "total", From: variate.Bound{OneIn: 50}},
		{Kind: "category", Name: "lump", From: variate.Bound{X: 4}},
		{Kind: "total", From: variate.Bound{OneIn: 10}, To: &variate.Bound{OneIn: 100}},
	}
	s, r := levelRun(t, burstGraph(), variate.Config{Periods: 64000, Seed: 9, Series: true, When: when})
	n := r.Series.Periods()
	totals := make([]float64, n)
	for p := range n {
		totals[p] = r.Series.Total(p)
	}
	sorted := slices.Clone(totals)
	slices.SortFunc(sorted, func(a, b float64) int { return -cmpFloat(a, b) })
	level := func(oneIn float64) float64 { return sorted[int(math.Ceil(float64(n)/oneIn))-1] }
	lump := slices.Index(r.Categories, "lump")
	check := func(i int, in func(p int64) bool) {
		var met, sum, sumLump float64
		for p := range n {
			if in(p) {
				met++
				sum += totals[p]
				sumLump += r.Series.Category(lump, p)
			}
		}
		st := s.When[i]
		tot, lu := part(t, st, "total", ""), part(t, st, "category", "lump")
		if float64(st.Met) != met || math.Abs(tot.Mean-sum/met) > 1e-9*sum/met || math.Abs(lu.Mean-sumLump/met) > 1e-9*sum/met {
			t.Errorf("%s: %d periods, means %v and %v; the series has %v, %v and %v", variate.WhenName(st.When), st.Met, tot.Mean, lu.Mean, met, sum/met, sumLump/met)
		}
		// At return periods alone the count is fixed, and so the chance.
		fixed := st.From.OneIn > 0 && (st.To == nil || st.To.OneIn > 0)
		if !(tot.MeanStdErr > 0) || !(lu.ShareStdErr > 0) || fixed != (st.StdErr == 0) || !(st.StdErr >= 0) {
			t.Errorf("%s: errors %v, %v, %v, want finite, positive, the chance's 0 at return periods alone", variate.WhenName(st.When), tot.MeanStdErr, lu.ShareStdErr, st.StdErr)
		}
	}
	x50, x10, x100 := level(50), level(10), level(100)
	if st := s.When[0]; st.Lower.X != x50 || !(st.Lower.StdErr > 0) {
		t.Errorf("1-in-50 level %v ± %v, the series' %v", st.Lower.X, st.Lower.StdErr, x50)
	}
	check(0, func(p int64) bool { return totals[p] >= x50 })
	check(1, func(p int64) bool { return r.Series.Category(lump, p) >= 4 })
	check(2, func(p int64) bool { return totals[p] >= x10 && totals[p] < x100 })
}

// twoExpGraph: rounds at rate 2 book Exponential(1) to category lump,
// and at rate 1 Exponential(0.5) to big, so the share of a bad period
// changes with how bad it is.
func twoExpGraph() variate.Graph {
	v, _ := dist.NewExponential(1)
	u, _ := dist.NewExponential(0.5)
	return variate.Graph{ID: "two", Sources: []variate.Source{{ID: "s1", Frequency: C(2), To: "a"}, {ID: "s2", Frequency: C(1), To: "b"}},
		Nodes: []variate.Node{{ID: "a", Functions: []variate.Function{Fn("v", "lump", v)}}, {ID: "b", Functions: []variate.Function{Fn("v", "big", u)}}}}
}

// TestWhenErrors: over 80 seeds, the errors stated for a value, a return
// period, whose level moves the periods it holds, and a band of both,
// are within the spread of the figures they describe, which over 80
// seeds is itself known to about 8%: in the two-source model, whose
// periods are independent, and with a holding, through the batch groups.
func TestWhenErrors(t *testing.T) {
	when := []variate.When{
		{Kind: "total", From: variate.Bound{X: 8}},
		{Kind: "total", From: variate.Bound{OneIn: 30}},
		{Kind: "total", From: variate.Bound{X: 6}, To: &variate.Bound{OneIn: 30}},
	}
	for _, g := range []variate.Graph{twoExpGraph(), burstGraph()} {
		figs := make([][6][]float64, len(when)) // mean, its error, share, its error, chance, its error
		for seed := uint64(1); seed <= 80; seed++ {
			s := whenSummary(t, g, variate.Config{Periods: 20000, Seed: seed, When: when})
			for i, st := range s.When {
				lu := part(t, st, "category", "lump")
				for k, v := range [6]float64{lu.Mean, lu.MeanStdErr, lu.Share, lu.ShareStdErr, st.P, st.StdErr} {
					figs[i][k] = append(figs[i][k], v)
				}
			}
		}
		for i, f := range figs {
			for k, name := range [3]string{"mean", "share", "chance"} {
				spread, stated := spreadOf(f[2*k]), meanOf(f[2*k+1])
				if spread == 0 && stated == 0 {
					continue
				}
				if r := stated / spread; !(r > 0.75 && r < 1.33) {
					t.Errorf("%s %s: lump's %s stated error %.4g, spread %.4g", g.ID, variate.WhenName(when[i]), name, stated, spread)
				}
			}
		}
	}
}

// whenSummary runs g and returns its summary, closing the workspace at
// once, as a test of many runs would otherwise hold every one.
func whenSummary(t *testing.T, g variate.Graph, cfg variate.Config) variate.Summary {
	t.Helper()
	c, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if _, err := w.Run(context.Background(), cfg.Seed); err != nil {
		t.Fatal(err)
	}
	var s variate.Summary
	w.Summarize(&s)
	return s
}

func meanOf(v []float64) float64 {
	m := 0.0
	for _, x := range v {
		m += x
	}
	return m / float64(len(v))
}

func spreadOf(v []float64) float64 {
	m, q := meanOf(v), 0.0
	for _, x := range v {
		q += (x - m) * (x - m)
	}
	return math.Sqrt(q / float64(len(v)-1))
}

func cmpFloat(a, b float64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

// TestWhenRefusals: a quantity the model does not have, a bound that is
// not a finite value or a return period of at least 1, and an upper bound
// not above the lower are refused.
func TestWhenRefusals(t *testing.T) {
	for _, c := range []struct {
		name string
		when variate.When
		want string
	}{
		{"no node", variate.When{Kind: "entries", Name: "z"}, "no quantity entries:z"},
		{"not finite", variate.When{Kind: "total", From: variate.Bound{X: math.NaN()}}, "finite"},
		{"short period", variate.When{Kind: "total", From: variate.Bound{OneIn: 0.5}}, "1/N with N at least 1"},
		{"values reversed", variate.When{Kind: "total", From: variate.Bound{X: 5}, To: &variate.Bound{X: 5}}, "above the lower"},
		{"periods reversed", variate.When{Kind: "total", From: variate.Bound{OneIn: 100}, To: &variate.Bound{OneIn: 10}}, "above the lower"},
	} {
		if _, _, err := countGraph(1).Compile(variate.Config{When: []variate.When{c.when}}); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want one naming %q", c.name, err, c.want)
		}
	}
}
