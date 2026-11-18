package suite

import (
	"context"
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
)

// discountMix is startedMix with a named total over both categories and
// a trigger fed by b's values, which books 1 in y when it fires.
func discountMix(horizon int, d variate.Discount) variate.Graph {
	g := startedMix(horizon)
	g.Totals = []variate.Total{{ID: "both", Categories: []string{"x", "y"}}}
	g.Nodes = append(g.Nodes, variate.Node{ID: "audit",
		Trigger:   &variate.Trigger{Over: 25, Inputs: []variate.TriggerInput{{Node: "b"}}},
		Functions: []variate.Function{Fn("fee", "y", C(1))}})
	g.Discount = d
	return g
}

// endOfPeriod is period k's factor at rate r, k from 0: (1+r)^-(k+1).
func endOfPeriod(r float64, k int) float64 { return math.Pow(1+r, -float64(k+1)) }

// agrees reports whether a and b agree to rounding, NaN with NaN.
func agrees(a, b float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.IsNaN(a) && math.IsNaN(b)
	}
	return math.Abs(a-b) <= 1e-12*math.Max(1, math.Abs(b))
}

// TestDiscountExact: discounting every category makes every exact value
// in every period its factor times the undiscounted one, the named
// total's included, and leaves the entries; the horizon's mean is the
// sum of the discounted periods. Discounting x alone scales x, leaves y,
// and gives a total of y plus the discounted x.
func TestDiscountExact(t *testing.T) {
	const h, r = 6, 0.1
	exact := func(d variate.Discount) (*variate.Compiled, *variate.Expectation) {
		m, _, err := discountMix(h, d).Compile(variate.Config{})
		if err != nil {
			t.Fatal(err)
		}
		e, err := m.Analytic()
		if err != nil {
			t.Fatal(err)
		}
		return m, e
	}
	m, plain := exact(variate.Discount{})
	_, all := exact(variate.Discount{Rate: r, Categories: []string{"x", "y"}})
	_, onlyX := exact(variate.Discount{Rate: r, Categories: []string{"x"}})
	ncat, nsrc, nnode := len(m.Categories()), len(m.SourceIDs()), len(m.NodeIDs())
	entries := map[int]bool{}
	for v := range nnode {
		entries[entryColumn(m, v)] = true
	}
	horizon, finite := 0.0, 0
	for k := range h {
		f := endOfPeriod(r, k)
		for q, x := range plain.ByPeriod[k] {
			want := f * x
			if entries[q] {
				want = x
			}
			if !agrees(all.ByPeriod[k][q], want) {
				t.Errorf("period %d column %d: %v, want %v", k, q, all.ByPeriod[k][q], want)
			}
			if !math.IsNaN(x) {
				finite++
			}
		}
		x, y := plain.ByPeriod[k][1], plain.ByPeriod[k][2]
		row := onlyX.ByPeriod[k]
		if !agrees(row[1], f*x) || !agrees(row[2], y) {
			t.Errorf("period %d: x %v y %v, want %v and %v", k, row[1], row[2], f*x, y)
		}
		horizon += all.ByPeriod[k][0]
	}
	if finite < h*(1+ncat+nsrc) {
		t.Errorf("%d finite exact values, too few to test", finite)
	}
	if !agrees(all.Mean, horizon) || !agrees(all.MeanByTotal[0], all.MeanByCategory[0]+all.MeanByCategory[1]) {
		t.Errorf("horizon mean %v, want %v; both %v", all.Mean, horizon, all.MeanByTotal[0])
	}
}

// TestDiscountSimulated: at the same seed a discounted run simulates the
// same rounds, so every period's mean of every value is its factor times
// the undiscounted one; nothing is discounted twice, as the entries, the
// trigger's undiscounted sum and its firings are unchanged, and the
// horizon's total is the sum of the discounted periods.
func TestDiscountSimulated(t *testing.T) {
	const h, r, repeats = 6, 0.25, 4000
	run := func(d variate.Discount) (*variate.Compiled, *variate.Result, variate.Summary) {
		cfg := variate.Config{Repeats: repeats, Workers: 4, AllowUnconverged: true}
		c, _, err := discountMix(h, d).Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		w, err := c.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
		res, err := w.Run(context.Background(), 3)
		if err != nil {
			t.Fatal(err)
		}
		var s variate.Summary
		w.Summarize(&s)
		return c, res, s
	}
	m, plain, ps := run(variate.Discount{})
	_, disc, ds := run(variate.Discount{Rate: r, Categories: []string{"x", "y"}})
	horizon := 0.0
	for k := range h {
		f := endOfPeriod(r, k)
		a, b := plain.ByPeriod, disc.ByPeriod
		same := func(what string, got, want float64) {
			t.Helper()
			if math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
				t.Errorf("period %d %s: %v, want %v", k, what, got, want)
			}
		}
		same("total", b.Total(k).Mean, f*a.Total(k).Mean)
		same("both", b.Named(0, k).Mean, f*a.Named(0, k).Mean)
		for c := range m.Categories() {
			same("category", b.Category(c, k).Mean, f*a.Category(c, k).Mean)
		}
		for s := range m.SourceIDs() {
			same("source", b.Source(s, k).Mean, f*a.Source(s, k).Mean)
		}
		for v := range m.NodeIDs() {
			same("contribution", b.Contrib(v, k).Mean, f*a.Contrib(v, k).Mean)
			same("entries", b.Entries(v, k).Mean, a.Entries(v, k).Mean)
		}
		same("trigger sum", b.Trigger(0, k).Mean, a.Trigger(0, k).Mean)
		horizon += b.Total(k).Mean
	}
	if a, b := ps.EntriesPer["audit"].Mean, ds.EntriesPer["audit"].Mean; a == 0 || a != b {
		t.Errorf("audit fired %v times a repeat discounted, %v not", b, a)
	}
	if math.Abs(ds.Total.Mean-horizon) > 1e-9*horizon || math.Abs(ds.Totals["both"].Mean-ds.Total.Mean) > 1e-9*horizon {
		t.Errorf("horizon mean %v, both %v, want %v", ds.Total.Mean, ds.Totals["both"].Mean, horizon)
	}
}

// TestDiscountPoisson: a Poisson source of rate 3 booking 2 over a
// horizon of 10 at 8% a period has a discounted horizon of mean
// 3·2·Σv^p and variance 3·4·Σv^(2p), v = 1/1.08; over 40,000 repeats the
// simulated figures lie within four standard errors.
func TestDiscountPoisson(t *testing.T) {
	const h, r, repeats = 10, 0.08, 40000
	g := variate.Graph{ID: "flat", Start: variate.Start{Horizon: h},
		Sources:  []variate.Source{{ID: "s", Frequency: C(3), To: "a"}},
		Nodes:    []variate.Node{{ID: "a", Functions: []variate.Function{Fn("f", "c", C(2))}}},
		Discount: variate.Discount{Rate: r, Categories: []string{"c"}}}
	var s1, s2 float64
	for k := range h {
		f := endOfPeriod(r, k)
		s1 += f
		s2 += f * f
	}
	mean, sd := 6*s1, math.Sqrt(12*s2)
	m, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := m.Analytic()
	if !agrees(e.Mean, mean) {
		t.Errorf("exact mean %v, want %v", e.Mean, mean)
	}
	_, _, s := runStarted(t, g, repeats, 9)
	if math.Abs(s.Total.Mean-mean) > 4*s.Total.StdErr || math.Abs(s.Total.StdDev-sd) > 4*sd/math.Sqrt(2*repeats) {
		t.Errorf("simulated mean %v ± %v sd %v, want %v and %v", s.Total.Mean, s.Total.StdErr, s.Total.StdDev, mean, sd)
	}
}

// runStarted runs a started graph for the given repeats.
func runStarted(t *testing.T, g variate.Graph, repeats int64, seed uint64) (*variate.Compiled, *variate.Result, variate.Summary) {
	t.Helper()
	cfg := variate.Config{Repeats: repeats, AllowUnconverged: true, Workers: 4}
	c, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	res, err := w.Run(context.Background(), seed)
	if err != nil {
		t.Fatal(err)
	}
	var s variate.Summary
	w.Summarize(&s)
	return c, res, s
}

// TestDiscountFactors: factors give any curve, and a discount whose every
// factor is 1, at rate 0 or given, is none: the run is the same to the
// bit as one without it.
func TestDiscountFactors(t *testing.T) {
	half := []float64{0.5, 0.5, 0.5, 0.5}
	m, _, err := discountMix(4, variate.Discount{Factors: half, Categories: []string{"y"}}).Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if !m.Discounted() || strings.Join(m.DiscountedCategories(), ",") != "y" {
		t.Errorf("discounted %v %v", m.Discounted(), m.DiscountedCategories())
	}
	_, _, plain := runStarted(t, discountMix(4, variate.Discount{}), 2000, 4)
	for _, d := range []variate.Discount{{Categories: []string{"x"}}, {Factors: []float64{1, 1, 1, 1}, Categories: []string{"x", "y"}}} {
		c, _, s := runStarted(t, discountMix(4, d), 2000, 4)
		if c.Discounted() || s.Total != plain.Total || s.ByCategory["y"] != plain.ByCategory["y"] {
			t.Errorf("%v: %v, want %v", d, s.Total, plain.Total)
		}
	}
	_, _, s := runStarted(t, discountMix(4, variate.Discount{Factors: half, Categories: []string{"x", "y"}}), 2000, 4)
	if math.Abs(s.Total.Mean-plain.Total.Mean/2) > 1e-9*plain.Total.Mean {
		t.Errorf("half: %v, want %v", s.Total.Mean, plain.Total.Mean/2)
	}
}

// TestDiscountRefusals: a discount needs a started run and categories a
// function books, each once; a total is discounted through its
// categories, not named; and a rate or factors that make sense.
func TestDiscountRefusals(t *testing.T) {
	steady := discountMix(4, variate.Discount{Rate: 0.1, Categories: []string{"x"}})
	steady.Start = variate.Start{}
	cases := map[string]variate.Graph{
		"steady-state":         steady,
		"names no categories":  discountMix(4, variate.Discount{Rate: 0.1}),
		"both a rate":          discountMix(4, variate.Discount{Rate: 0.1, Factors: []float64{1, 1, 1, 1}, Categories: []string{"x"}}),
		"has 3 factors":        discountMix(4, variate.Discount{Factors: []float64{1, 1, 1}, Categories: []string{"x"}}),
		"period 2":             discountMix(4, variate.Discount{Factors: []float64{1, -1, 1, 1}, Categories: []string{"x"}}),
		"greater than -1":      discountMix(4, variate.Discount{Rate: -1, Categories: []string{"x"}}),
		"listed twice":         discountMix(4, variate.Discount{Rate: 0.1, Categories: []string{"x", "x"}}),
		"through its":          discountMix(4, variate.Discount{Rate: 0.1, Categories: []string{"both"}}),
		"no function books":    discountMix(4, variate.Discount{Rate: 0.1, Categories: []string{"z"}}),
		"factor must be >= 0":  discountMix(4, variate.Discount{Factors: []float64{1, math.NaN(), 1, 1}, Categories: []string{"x"}}),
		"must be greater than": discountMix(4, variate.Discount{Rate: math.Inf(1), Categories: []string{"x"}}),
	}
	for want, g := range cases { // unordered: each case stands alone
		_, _, err := g.Compile(variate.Config{})
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: %v", want, err)
		}
	}
}
