package suite

import (
	"context"
	"math"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// optionBase: 5 rounds a period branch 0.3/0.7 to a gamma value at b or
// a lognormal at c, then a gamma at d.
func optionBase() variate.Graph {
	gb, _ := dist.NewGammaFromMeanSD(4, 2)
	lc, _ := dist.NewLognormalFromMedianP90(3, 8)
	gd, _ := dist.NewGammaFromMeanSD(10, 3)
	return variate.Graph{ID: "option",
		Sources: []variate.Source{{ID: "s", Frequency: C(5), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Outcomes: []variate.Outcome{{To: "b", P: C(0.3)}, {To: "c", P: C(0.7)}}},
			{ID: "b", Next: "d", Functions: []variate.Function{Fn("fb", "x", gb)}},
			{ID: "c", Next: "d", Functions: []variate.Function{Fn("fc", "x", lc)}},
			{ID: "d", Functions: []variate.Function{Fn("fd", "y", gd)}},
		}}
}

// optionTotals runs g at seed and returns its period totals.
func optionTotals(t *testing.T, g variate.Graph, periods int64, seed uint64) []float64 {
	t.Helper()
	cfg := variate.Config{Periods: periods, AllowUnconverged: true, Workers: 2, Series: true}
	m, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), seed)
	if err != nil {
		t.Fatal(err)
	}
	return r.Series.TotalColumn(make([]float64, periods))
}

// pairStats is the correlation of a and b, and the standard deviations
// of a, b and b − a.
func pairStats(a, b []float64) (rho, sa, sb, sd float64) {
	n := float64(len(a))
	var ma, mb float64
	for i := range a {
		ma, mb = ma+a[i], mb+b[i]
	}
	ma, mb = ma/n, mb/n
	var va, vb, c float64
	for i := range a {
		va += (a[i] - ma) * (a[i] - ma)
		vb += (b[i] - mb) * (b[i] - mb)
		c += (a[i] - ma) * (b[i] - mb)
	}
	va, vb, c = va/(n-1), vb/(n-1), c/(n-1)
	return c / math.Sqrt(va*vb), math.Sqrt(va), math.Sqrt(vb), math.Sqrt(math.Max(0, va+vb-2*c))
}

// TestKeyedOptions: two models at one seed draw alike wherever an edit
// does not reach, so their period totals move together, for an edit of
// every kind: a probability, a value's law of the same or another kind,
// a source's rate, an added function, node or source; a model at
// another seed does not.
func TestKeyedOptions(t *testing.T) {
	gb, _ := dist.NewGammaFromMeanSD(4.5, 2)
	lb, _ := dist.NewLognormalFromMedianP90(4, 8)
	gx, _ := dist.NewGammaFromMeanSD(2, 1)
	edit := func(f func(*variate.Graph)) variate.Graph { g := optionBase(); f(&g); return g }
	cases := []struct {
		name string
		g    variate.Graph
		min  float64 // the least correlation; measured in the comment
	}{
		{"probability 0.3 to 0.35", edit(func(g *variate.Graph) { // 0.998
			g.Nodes[0].Outcomes = []variate.Outcome{{To: "b", P: C(0.35)}, {To: "c", P: C(0.65)}}
		}), 0.99},
		{"law, same kind", edit(func(g *variate.Graph) { g.Nodes[1].Functions[0].Model = gb }), 0.99},  // 1.000
		{"law, other kind", edit(func(g *variate.Graph) { g.Nodes[1].Functions[0].Model = lb }), 0.99}, // 0.999
		{"rate 5 to 5.5", edit(func(g *variate.Graph) { g.Sources[0].Frequency = C(5.5) }), 0.95},      // 0.975
		{"function added at d", edit(func(g *variate.Graph) { // 0.998
			g.Nodes[3].Functions = append(g.Nodes[3].Functions, Fn("fx", "y", gx))
		}), 0.99},
		{"function added at b, first", edit(func(g *variate.Graph) { // 0.997
			g.Nodes[1].Functions = append([]variate.Function{Fn("fx", "y", gx)}, g.Nodes[1].Functions...)
		}), 0.99},
		{"node added between c and d", edit(func(g *variate.Graph) { // 0.997
			g.Nodes[2].Next = "e"
			g.Nodes = append(g.Nodes[:3], variate.Node{ID: "e", Next: "d", Functions: []variate.Function{Fn("fe", "y", gx)}}, g.Nodes[3])
		}), 0.99},
		{"source added into c", edit(func(g *variate.Graph) { // 0.912
			g.Sources = append([]variate.Source{{ID: "t", Frequency: C(1), To: "c"}}, g.Sources...)
		}), 0.85},
	}
	const n = 1 << 15
	base := optionTotals(t, optionBase(), n, 7)
	for _, c := range cases {
		rho, _, _, _ := pairStats(base, optionTotals(t, c.g, n, 7))
		if !(rho >= c.min) {
			t.Errorf("%s: correlation %.3f, want at least %v", c.name, rho, c.min)
		}
	}
	if rho, _, _, _ := pairStats(base, optionTotals(t, optionBase(), n, 8)); math.Abs(rho) > 4/math.Sqrt(n) {
		t.Errorf("another seed: correlation %.3f", rho)
	}
}

// TestKeyedPairedError: the difference of two options' means at one
// seed has the error of the paired periods, sd(B − A)/√n, a twentieth of
// the independent runs' for a probability moved by 0.05: over 200 seeds
// the spread of the difference is within 20% of it.
func TestKeyedPairedError(t *testing.T) {
	moved := optionBase()
	moved.Nodes[0].Outcomes = []variate.Outcome{{To: "b", P: C(0.35)}, {To: "c", P: C(0.65)}}
	const n, seeds = 2048, 200
	var s1, s2, paired, indep float64
	for s := uint64(1); s <= seeds; s++ {
		a, b := optionTotals(t, optionBase(), n, s), optionTotals(t, moved, n, s)
		_, sa, sb, sd := pairStats(a, b)
		d := 0.0
		for i := range a {
			d += b[i] - a[i]
		}
		d /= n
		s1, s2 = s1+d, s2+d*d
		paired += sd / math.Sqrt(n)
		indep += math.Sqrt((sa*sa + sb*sb) / n)
	}
	spread := math.Sqrt((s2 - s1*s1/seeds) / (seeds - 1))
	paired, indep = paired/seeds, indep/seeds
	if r := spread / paired; r < 0.8 || r > 1.2 {
		t.Errorf("spread %v, paired error %v: ratio %.3f", spread, paired, r)
	}
	if r := paired / indep; r > 0.1 {
		t.Errorf("paired error %v, independent %v: ratio %.3f", paired, indep, r)
	}
}
