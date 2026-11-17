package suite

import (
	"context"
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// factoryGraph: 4 stops a period, each booking labour (gamma, mean 2, sd 1),
// parts (lognormal, mean 3, sd 2) and hours (gamma, mean 10, sd 4), and
// with chance 1/2 lost production (gamma, mean 5, sd 3). Hours are kept
// outside the total; maintenance sums labour and parts, cost the three
// in money, and worked the hours.
func factoryGraph() variate.Graph {
	gamma := func(mean, sd float64) dist.Distribution {
		d, _ := dist.NewGammaFromMeanSD(mean, sd)
		return d
	}
	parts, _ := dist.NewLognormalFromMeanSD(3, 2)
	return variate.Graph{ID: "line",
		Sources: []variate.Source{{ID: "stop", Frequency: C(4), To: "repair"}},
		Nodes: []variate.Node{
			{ID: "repair",
				Functions: []variate.Function{Fn("crew", "labour", gamma(2, 1)), Fn("spares", "parts", parts), Fn("worked", "hours", gamma(10, 4))},
				Outcomes:  []variate.Outcome{{To: "lost", P: C(0.5)}, {To: "done"}}},
			{ID: "lost", Functions: []variate.Function{Fn("output", "production", gamma(5, 3))}},
			{ID: "done"},
		},
		Totals: []variate.Total{
			{ID: "maintenance", Categories: []string{"labour", "parts"}},
			{ID: "cost", Categories: []string{"labour", "parts", "production"}},
			{ID: "worked", Categories: []string{"hours"}},
		},
		OutsideTotal: []string{"hours"},
	}
}

// factoryExact is factoryGraph's named totals' means and variances, compound
// Poisson at rate 4: λ·E[X] and λ·E[X²] of what one stop books, with
// E[X²] = sd² + mean² for each value and the values independent.
var factoryExact = map[string][2]float64{
	"maintenance": {4 * (2 + 3), 4 * (5 + 13 + 2*2*3)},
	"cost":        {4 * (2 + 3 + 0.5*5), 4 * (5 + 13 + 2*2*3 + 0.5*34 + 2*0.5*5*(2+3))},
	"worked":      {4 * 10, 4 * 116},
}

// TestTotalsExact: the named totals' exact means and variances are the
// closed forms, the total is cost's, and the hours stay out of the
// total, the source and P(T = 0).
func TestTotalsExact(t *testing.T) {
	c, _, err := factoryGraph().Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, err := c.Analytic()
	if err != nil {
		t.Fatal(err)
	}
	for i, id := range c.TotalIDs() {
		want := factoryExact[id]
		if math.Abs(e.MeanByTotal[i]-want[0]) > 1e-9*want[0] || math.Abs(e.VarByTotal[i]-want[1]) > 1e-9*want[1] {
			t.Errorf("%s: mean %v var %v, want %v", id, e.MeanByTotal[i], e.VarByTotal[i], want)
		}
	}
	cost := factoryExact["cost"]
	if math.Abs(e.Mean-cost[0]) > 1e-9*cost[0] || math.Abs(e.Var-cost[1]) > 1e-9*cost[1] || math.Abs(e.MeanBySource[0]-cost[0]) > 1e-9*cost[0] {
		t.Errorf("total mean %v var %v source %v, want cost's %v", e.Mean, e.Var, e.MeanBySource[0], cost)
	}
	if want := math.Exp(-4); math.Abs(e.PZero()-want) > 1e-12 {
		t.Errorf("P(T = 0) %v, want %v", e.PZero(), want)
	}
}

// TestTotalsSimulated: in every period a named total is the sum of its
// categories and the total the sum of those in it, which is cost; over
// 200,000 periods each named total's mean and standard deviation are
// within four standard errors of the closed forms, and the summary and
// the quantity rows carry them.
func TestTotalsSimulated(t *testing.T) {
	const periods = 200000
	c, _, err := factoryGraph().Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	cfg := variate.Config{Periods: periods, Series: true, Workers: 4, AllowUnconverged: true}
	w, err := c.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 11)
	if err != nil {
		t.Fatal(err)
	}
	cat := map[string]int{}
	for i, name := range c.Categories() {
		cat[name] = i
	}
	sum := func(p int64, names ...string) float64 {
		s := 0.0
		for _, n := range names {
			s += r.Series.Category(cat[n], p)
		}
		return s
	}
	named := make([][]float64, len(c.TotalIDs()))
	for i := range named {
		named[i] = r.Series.NamedColumn(i, make([]float64, periods))
	}
	for p := range int64(periods) {
		tot := r.Series.Total(p)
		money := sum(p, "labour", "parts", "production")
		if math.Abs(tot-money) > 1e-9*math.Max(1, tot) || named[1][p] != tot && math.Abs(named[1][p]-tot) > 1e-9*tot {
			t.Fatalf("period %d: total %v, money %v, cost %v", p, tot, money, named[1][p])
		}
		if math.Abs(named[0][p]-sum(p, "labour", "parts")) > 1e-9*math.Max(1, tot) || named[2][p] != sum(p, "hours") {
			t.Fatalf("period %d: maintenance %v, worked %v", p, named[0][p], named[2][p])
		}
	}
	var s variate.Summary
	w.Summarize(&s)
	for _, id := range c.TotalIDs() {
		st, want := s.Totals[id], factoryExact[id]
		sd := math.Sqrt(want[1])
		if math.Abs(st.Mean-want[0]) > 4*st.StdErr || math.Abs(st.StdDev-sd) > 4*sd/math.Sqrt(2*periods)*2 {
			t.Errorf("%s: mean %v ± %v sd %v, want %v and %v", id, st.Mean, st.StdErr, st.StdDev, want[0], sd)
		}
		row, ok := w.QuantityRow("total", id)
		if !ok || w.QuantityStat(row).Mean != st.Mean || !w.HasHistogram(row) {
			t.Errorf("%s: row %d ok %v", id, row, ok)
		}
	}
	if sh := s.Totals["maintenance"].Share; math.Abs(sh-s.Totals["maintenance"].Mean/s.Total.Mean) > 1e-12 {
		t.Errorf("maintenance's share %v", sh)
	}
	if !math.IsNaN(s.Totals["worked"].Share) || !math.IsNaN(s.ByCategory["hours"].Share) {
		t.Errorf("hours shares %v and %v, want NaN outside the total", s.Totals["worked"].Share, s.ByCategory["hours"].Share)
	}
}

// TestTotalsNoBreakdown: without the breakdowns the named totals are
// still kept, the same as with them at the same seed.
func TestTotalsNoBreakdown(t *testing.T) {
	with, without := func() variate.Summary {
		_, _, s := runSummary(t, factoryGraph(), 20000, 5)
		return s
	}(), func() variate.Summary {
		cfg := variate.Config{Periods: 20000, AllowUnconverged: true, Workers: 4, NoBreakdown: true}
		c, _, err := factoryGraph().Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		w, err := c.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
		if _, err := w.Run(context.Background(), 5); err != nil {
			t.Fatal(err)
		}
		var s variate.Summary
		w.Summarize(&s)
		return s
	}()
	if len(without.ByCategory) != 0 {
		t.Errorf("categories kept without the breakdowns: %v", without.ByCategory)
	}
	for id, st := range with.Totals {
		if without.Totals[id].Mean != st.Mean || without.Totals[id].P99 != st.P99 {
			t.Errorf("%s: %v without the breakdowns, %v with", id, without.Totals[id], st)
		}
	}
}

// TestTotalsTransform: a named total of one category has that
// category's exact bracket, and one of every category in the total has
// the total's.
func TestTotalsTransform(t *testing.T) {
	g := factoryGraph()
	g.Totals = []variate.Total{{ID: "labour_only", Categories: []string{"labour"}}, {ID: "money", Categories: []string{"labour", "parts", "production"}}}
	m, tr := transformOf(t, g, variate.TransformConfig{})
	if !tr.Applicable {
		t.Fatal(tr.Reason)
	}
	labour := -1
	for i, name := range m.Categories() {
		if name == "labour" {
			labour = i
		}
	}
	same := func(what string, a, b variate.Bounds) {
		t.Helper()
		if len(a.Lower) != len(b.Lower) {
			t.Fatalf("%s: %d and %d points", what, len(a.Lower), len(b.Lower))
		}
		for k := range a.Lower {
			if math.Abs(a.Lower[k]-b.Lower[k]) > 1e-12 || math.Abs(a.Upper[k]-b.Upper[k]) > 1e-12 {
				t.Fatalf("%s: point %d [%v, %v] against [%v, %v]", what, k, a.Lower[k], a.Upper[k], b.Lower[k], b.Upper[k])
			}
		}
	}
	same("labour", tr.ByTotal[0], tr.ByCategory[labour])
	same("money", tr.ByTotal[1], tr.Total)
}

// TestTotalsStarted: in a started run with a holding, a named total's
// exact mean in each period is the sum of its categories', and the
// simulated means are within four standard errors of it.
func TestTotalsStarted(t *testing.T) {
	g := factoryGraph()
	hold, _ := dist.NewExponential(1.5)
	g.Nodes[0].Outcomes[0].Holding = hold
	g.Start = variate.Start{Horizon: 4}
	m, r := startedRun(t, g, variate.Config{Repeats: 40000, Seed: 9})
	e, err := m.Analytic()
	if err != nil {
		t.Fatal(err)
	}
	cats := m.Categories()
	l := len(cats) + len(m.SourceIDs()) + 2*len(m.NodeIDs()) + 1
	for k := range 4 {
		for i, id := range m.TotalIDs() {
			want := 0.0
			for c, name := range cats {
				if strings.Contains(strings.Join(factoryGraph().Totals[i].Categories, ","), name) {
					want += e.ByPeriod[k][1+c]
				}
			}
			if got := e.ByPeriod[k][l+i]; math.IsNaN(want) || math.Abs(got-want) > 1e-9*math.Max(1, want) {
				t.Errorf("period %d %s: exact %v, sum of categories %v", k+1, id, got, want)
			}
			nearMean(t, id, r.ByPeriod.Named(i, k), want)
		}
	}
}

// TestTotalsRefusals: each mistake in naming totals is refused with the
// total or category named.
func TestTotalsRefusals(t *testing.T) {
	for _, tc := range []struct {
		name    string
		totals  []variate.Total
		outside []string
		want    string
	}{
		{"unknown category", []variate.Total{{ID: "x", Categories: []string{"fuel"}}}, nil, `total "x": no function books category fuel`},
		{"duplicate id", []variate.Total{{ID: "x", Categories: []string{"labour"}}, {ID: "x", Categories: []string{"parts"}}}, nil, `total "1": empty or duplicate id`},
		{"empty id", []variate.Total{{Categories: []string{"labour"}}}, nil, `total "0": empty or duplicate id`},
		{"no categories", []variate.Total{{ID: "x"}}, nil, `total "x": no categories`},
		{"category twice", []variate.Total{{ID: "x", Categories: []string{"labour", "labour"}}}, nil, `total "x": category labour is listed twice`},
		{"unknown outside", nil, []string{"fuel"}, `outside_total "fuel": no function books this category`},
		{"all outside", nil, []string{"labour", "parts", "hours", "production"}, `outside_total "labour": every category is outside the total`},
	} {
		g := factoryGraph()
		g.Totals, g.OutsideTotal = tc.totals, tc.outside
		if _, _, err := g.Compile(variate.Config{}); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want %q", tc.name, err, tc.want)
		}
	}
}

// TestTotalsConditions: a condition on a named total and a level of one
// are counted on its row, and a condition lists the named totals among
// its parts, maintenance's share within its error of its mean over the
// total's.
func TestTotalsConditions(t *testing.T) {
	cfg := variate.Config{Periods: 50000, AllowUnconverged: true, Workers: 4,
		When: []variate.When{{Kind: "total", Name: "maintenance", From: variate.Bound{OneIn: 10}}},
		At:   []variate.Level{{Kind: "total", Name: "cost", X: 40}}}
	cfg.Seed = 3
	s := whenSummary(t, factoryGraph(), cfg)
	if len(s.When) != 1 || s.When[0].Reason != "" {
		t.Fatalf("when %+v", s.When)
	}
	w := s.When[0]
	if math.Abs(w.P-0.1) > 0.002 {
		t.Errorf("P %v, want 0.1", w.P)
	}
	m := part(t, w, "total", "maintenance")
	tot := part(t, w, "total", "")
	if m.Mean < w.Lower.X || math.Abs(m.Share-m.Mean/tot.Mean) > 4*m.ShareStdErr+1e-9 {
		t.Errorf("maintenance %+v, total %+v, level %v", m, tot, w.Lower.X)
	}
	if hours := part(t, w, "total", "worked"); !math.IsNaN(hours.Share) {
		t.Errorf("worked's share %v, want NaN", hours.Share)
	}
	if len(s.At) != 1 || s.At[0].N != 50000 || !(s.At[0].P > 0 && s.At[0].P < 1) {
		t.Errorf("at %+v", s.At)
	}
}
