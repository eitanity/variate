package suite

import (
	"context"
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// compareOptions: the option base with its probability moved, and with
// a node added between c and d.
func compareOptions() []variate.Graph {
	moved := optionBase()
	moved.Nodes[0].Outcomes = []variate.Outcome{{To: "b", P: C(0.35)}, {To: "c", P: C(0.65)}}
	added := optionBase()
	gx, _ := dist.NewGammaFromMeanSD(2, 1)
	added.Nodes[2].Next = "e"
	added.Nodes = append(added.Nodes[:3], variate.Node{ID: "e", Next: "d", Functions: []variate.Function{Fn("fe", "y", gx)}}, added.Nodes[3])
	return []variate.Graph{moved, added}
}

// TestCompareExact: where the analytic expectations give the figure, a
// row is their exact difference with no error.
func TestCompareExact(t *testing.T) {
	cfg := variate.Config{Seed: 7}
	c, err := optionBase().Compare(context.Background(), compareOptions(), variate.CompareConfig{Run: cfg})
	if err != nil {
		t.Fatal(err)
	}
	base := exactMean(t, optionBase(), cfg)
	for i, o := range compareOptions() {
		r := c.Rows[i]
		want := exactMean(t, o, cfg) - base
		if r.Method != variate.MethodExact || r.SE != 0 || math.Abs(r.Diff.Lo-want) > 1e-9*math.Abs(want) || r.Diff.Lo != r.Diff.Hi {
			t.Errorf("option %d: %+v, want an exact difference of %v", i+1, r, want)
		}
	}
	if c.Figure != "mean" || c.Periods != 0 {
		t.Errorf("figure %q, periods %d", c.Figure, c.Periods)
	}
}

func exactMean(t *testing.T, g variate.Graph, cfg variate.Config) float64 {
	t.Helper()
	m, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	e, err := m.Analytic()
	if err != nil {
		t.Fatal(err)
	}
	return e.Mean
}

// TestComparePairedError: a simulated row's standard error is the
// spread of its difference over seeds, for a moved probability and an
// added node, and that spread is a fraction of what two independent runs
// would give, under 0.4: about a third for the added node, whose
// independent runs spread 0.84. The 90th percentile over groups of 256
// periods: over groups of 64 the error is 11–17% short.
func TestComparePairedError(t *testing.T) {
	const seeds, n = 100, 8192
	opts := compareOptions()
	var s1, s2, se, b1, b2, v1, v2 [2]float64
	for s := uint64(1); s <= seeds; s++ {
		// A work budget of one grid point keeps the transform off, so
		// the 90th percentile is simulated.
		run := variate.Config{Seed: s, Periods: n, AllowUnconverged: true, Workers: 2, TransformWork: 1}
		c, err := optionBase().Compare(context.Background(), opts, variate.CompareConfig{Figure: "p90", Run: run})
		if err != nil {
			t.Fatal(err)
		}
		if c.Periods != n || c.Seed != s {
			t.Fatalf("periods %d, seed %d", c.Periods, c.Seed)
		}
		for i, r := range c.Rows {
			if r.Method != variate.MethodSimulation {
				t.Fatalf("option %d: method %s", i+1, r.Method)
			}
			d := r.Diff.Lo
			s1[i], s2[i], se[i] = s1[i]+d, s2[i]+d*d, se[i]+r.SE
			b1[i], b2[i] = b1[i]+r.Base.Lo, b2[i]+r.Base.Lo*r.Base.Lo
			v1[i], v2[i] = v1[i]+r.Value.Lo, v2[i]+r.Value.Lo*r.Value.Lo
		}
	}
	sd := func(a, b float64) float64 { return math.Sqrt((b - a*a/seeds) / (seeds - 1)) }
	for i, name := range []string{"probability moved", "node added"} {
		spread, paired := sd(s1[i], s2[i]), se[i]/seeds
		indep := math.Hypot(sd(b1[i], b2[i]), sd(v1[i], v2[i]))
		if r := spread / paired; r < 0.75 || r > 1.33 {
			t.Errorf("%s: spread %v, paired error %v: ratio %.3f", name, spread, paired, r)
		}
		if r := spread / indep; r > 0.4 {
			t.Errorf("%s: spread %v, independent runs' %v: ratio %.3f", name, spread, indep, r)
		}
		t.Logf("%s: spread %.4g, paired error %.4g, independent %.4g", name, spread, paired, indep)
	}
}

// TestCompareErrors: no option, a bad figure, and an option that does
// not compile, named by its place.
func TestCompareErrors(t *testing.T) {
	ctx := context.Background()
	broken := optionBase()
	broken.Nodes[1].Next = "nowhere"
	for _, c := range []struct {
		opts []variate.Graph
		fig  string
		want string
	}{
		{nil, "", "compare: no option"},
		{compareOptions(), "p200", "compare: figure p200"},
		{[]variate.Graph{optionBase(), broken}, "p90", "compare: option 2: "},
	} {
		run := variate.Config{Seed: 1, Periods: 256, AllowUnconverged: true, TransformWork: 1}
		_, err := optionBase().Compare(ctx, c.opts, variate.CompareConfig{Figure: c.fig, Run: run})
		if err == nil || !strings.HasPrefix(err.Error(), c.want) {
			t.Errorf("%q: error %v, want %q", c.want, err, c.want)
		}
	}
}
