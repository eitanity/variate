package suite

import (
	"bytes"
	"context"
	"math"
	"math/rand/v2"
	"strconv"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
	"code.eitanity.com/variate/report"
)

// Metamorphic tests: a transformation of a model with a known effect on
// its results, checked on the random DAG generator's models. The exact
// expectations and the transform must agree to round-off; a simulation
// whose random streams the transformation reorders is compared by a
// score test on the mean, and one whose streams it leaves alone must be
// byte-identical.

const metaSeeds = 6

// exactOf compiles a graph and returns its model and expectations.
func exactOf(t *testing.T, g variate.Graph) (*variate.Compiled, *variate.Expectation) {
	t.Helper()
	c, _, err := g.Compile(variate.Config{AllowUnconverged: true})
	if err != nil {
		t.Fatalf("%s: %v", g.ID, err)
	}
	e, err := c.Analytic()
	if err != nil {
		t.Fatal(err)
	}
	return c, e
}

func near(a, b, rel float64) bool {
	if math.IsInf(a, 0) || math.IsInf(b, 0) || math.IsNaN(a) || math.IsNaN(b) {
		return (math.IsInf(a, 1) && math.IsInf(b, 1)) || (math.IsNaN(a) && math.IsNaN(b))
	}
	return math.Abs(a-b) <= rel*math.Max(1e-300, math.Max(math.Abs(a), math.Abs(b)))
}

// sameExpectations checks the whole-model expectations and the per-node
// entries by node name.
func sameExpectations(t *testing.T, what string, ca, cb *variate.Compiled, ea, eb *variate.Expectation) {
	t.Helper()
	if !near(ea.Mean, eb.Mean, 1e-9) || !near(ea.Var, eb.Var, 1e-9) || !near(ea.PZero(), eb.PZero(), 1e-9) || !near(ea.StepsPerPeriod, eb.StepsPerPeriod, 1e-9) {
		t.Errorf("%s: mean %g/%g var %g/%g pzero %g/%g steps %g/%g", what, ea.Mean, eb.Mean, ea.Var, eb.Var, ea.PZero(), eb.PZero(), ea.StepsPerPeriod, eb.StepsPerPeriod)
	}
	entries := map[string]float64{}
	for v, name := range ca.NodeIDs() {
		entries[name] = ea.EntriesPer[v]
	}
	for v, name := range cb.NodeIDs() {
		if want, ok := entries[name]; ok && !near(want, eb.EntriesPer[v], 1e-9) {
			t.Errorf("%s: entries per period at %s %g vs %g", what, name, want, eb.EntriesPer[v])
		}
	}
}

// sameTransform compares the exact brackets of two models that must be
// the same distribution: identical curves when the grids agree, else
// overlapping brackets at common points.
func sameTransform(t *testing.T, what string, ca, cb *variate.Compiled) {
	t.Helper()
	cfg := variate.TransformConfig{Tol: 1e-2, MaxGrid: 1 << 17}
	ta, _ := ca.Transform(cfg)
	tb, _ := cb.Transform(cfg)
	if ta.Applicable != tb.Applicable {
		t.Errorf("%s: transform applicable %v (%s) vs %v (%s)", what, ta.Applicable, ta.Reason, tb.Applicable, tb.Reason)
		return
	}
	if !ta.Applicable {
		return
	}
	if !near(ta.PZero, tb.PZero, 1e-9) {
		t.Errorf("%s: transform P(T=0) %g vs %g", what, ta.PZero, tb.PZero)
	}
	a, b := ta.Total, tb.Total
	if a.H == b.H && len(a.Lower) == len(b.Lower) {
		for k := range a.Lower {
			if !near(a.Lower[k], b.Lower[k], 1e-9) || !near(a.Upper[k], b.Upper[k], 1e-9) {
				t.Errorf("%s: transform curves differ at k=%d: [%g, %g] vs [%g, %g]", what, k, a.Lower[k], a.Upper[k], b.Lower[k], b.Upper[k])
				return
			}
		}
		return
	}
	for k := 0; k < len(a.Lower); k += max(1, len(a.Lower)/100) {
		x := float64(k) * a.H
		j := int(x / b.H)
		if j >= len(b.Lower) {
			break
		}
		lo, hi := b.Lower[min(j+1, len(b.Lower)-1)], b.Upper[j]
		if a.Lower[k] > hi+1e-9 || lo > a.Upper[k]+1e-9 {
			t.Errorf("%s: brackets do not overlap at x=%g: [%g, %g] vs [%g, %g]", what, x, a.Lower[k], a.Upper[k], lo, hi)
			return
		}
	}
}

// simulate runs a fixed length and returns the report and the result.
func simulate(t *testing.T, g variate.Graph, seed uint64) ([]byte, *variate.Result) {
	t.Helper()
	cfg := variate.Config{Periods: 20000, BlockPeriods: 100, Workers: 2, AllowUnconverged: true}
	c, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatalf("%s: %v", g.ID, err)
	}
	w, err := c.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), seed)
	if err != nil {
		t.Fatal(err)
	}
	rep, _ := report.New(c, w, r, false)
	return rep.JSON(), r
}

// sameMean compares two simulated means by a score test with the pooled
// standard error; |z| < 4.5.
func sameMean(t *testing.T, what string, ra, rb *variate.Result) {
	t.Helper()
	se := func(r *variate.Result) float64 {
		s := r.Total.StdErr(true)
		if math.IsNaN(s) {
			s = r.Total.StdErr(false)
		}
		return s
	}
	pooled := math.Sqrt(se(ra)*se(ra) + se(rb)*se(rb))
	if pooled == 0 {
		if ra.Total.Mean() != rb.Total.Mean() {
			t.Errorf("%s: means %g vs %g with no spread", what, ra.Total.Mean(), rb.Total.Mean())
		}
		return
	}
	if z := (ra.Total.Mean() - rb.Total.Mean()) / pooled; math.Abs(z) > 4.5 {
		t.Errorf("%s: simulated means %g vs %g, z=%.2f", what, ra.Total.Mean(), rb.Total.Mean(), z)
	}
}

func sameBytes(t *testing.T, what string, a, b []byte) {
	t.Helper()
	if !bytes.Equal(a, b) {
		t.Errorf("%s: results differ:\n%s", what, firstDiff(a, b))
	}
}

// ---- transformations --------------------------------------------------------

// withSplitSource replaces the first constant-rate source by two sources
// with rates summing to it, feeding the same node.
func withSplitSource(g variate.Graph) (variate.Graph, bool) {
	for i, s := range g.Sources {
		c, ok := s.Frequency.(dist.Constant)
		if !ok {
			continue
		}
		rate := c.Mean()
		out := g
		out.Sources = append([]variate.Source{}, g.Sources...)
		out.Sources[i] = variate.Source{ID: s.ID + "_a", Frequency: C(rate * 0.3), To: s.To}
		out.Sources = append(out.Sources, variate.Source{ID: s.ID + "_b", Frequency: C(rate * 0.7), To: s.To})
		return out, true
	}
	return g, false
}

// withPassThrough routes the first source through a new node with no
// functions and a plain next.
func withPassThrough(g variate.Graph) variate.Graph {
	out := g
	out.Sources = append([]variate.Source{}, g.Sources...)
	out.Nodes = append([]variate.Node{{ID: "pass", Next: g.Sources[0].To}}, g.Nodes...)
	out.Sources[0].To = "pass"
	return out
}

// withShuffledOrder reorders nodes and, within absolute-mode nodes,
// outcomes.
func withShuffledOrder(g variate.Graph, seed uint64) variate.Graph {
	r := rand.New(rand.NewPCG(seed, 1))
	out := g
	out.Nodes = append([]variate.Node{}, g.Nodes...)
	r.Shuffle(len(out.Nodes), func(i, j int) { out.Nodes[i], out.Nodes[j] = out.Nodes[j], out.Nodes[i] })
	for i := range out.Nodes {
		n := &out.Nodes[i]
		if n.Mode == variate.Absolute && len(n.Outcomes) > 1 && allFixed(n.Outcomes) {
			n.Outcomes = append([]variate.Outcome{}, n.Outcomes...)
			r.Shuffle(len(n.Outcomes), func(a, b int) { n.Outcomes[a], n.Outcomes[b] = n.Outcomes[b], n.Outcomes[a] })
		}
	}
	return out
}

func allFixed(outs []variate.Outcome) bool {
	for _, o := range outs {
		if o.P == nil {
			return false // a remainder must stay last
		}
		if _, ok := o.P.(dist.Constant); !ok {
			return false
		}
	}
	return true
}

// withCategoricalConstants writes every constant function value as a
// one-atom categorical.
func withCategoricalConstants(t *testing.T, g variate.Graph) (variate.Graph, bool) {
	t.Helper()
	changed := false
	out := g
	out.Nodes = append([]variate.Node{}, g.Nodes...)
	for i := range out.Nodes {
		fns := append([]variate.Function{}, out.Nodes[i].Functions...)
		for f := range fns {
			if c, ok := fns[f].Model.(dist.Constant); ok {
				cat, err := dist.NewCategorical([]float64{c.Mean()}, []float64{1})
				if err != nil {
					t.Fatal(err)
				}
				fns[f].Model = cat
				changed = true
			}
		}
		out.Nodes[i].Functions = fns
	}
	return out, changed
}

// withNextAsOutcome writes every plain next as a one-outcome selection
// of probability 1.
func withNextAsOutcome(g variate.Graph) (variate.Graph, bool) {
	changed := false
	out := g
	out.Nodes = append([]variate.Node{}, g.Nodes...)
	for i := range out.Nodes {
		n := &out.Nodes[i]
		if n.Next != "" && n.NextHolding == nil {
			n.Outcomes = []variate.Outcome{{To: n.Next, P: C(1)}}
			n.Next = ""
			changed = true
		}
	}
	return out, changed
}

// withSplitConstant splits every constant function into two halves.
func withSplitConstant(g variate.Graph) (variate.Graph, bool) {
	changed := false
	out := g
	out.Nodes = append([]variate.Node{}, g.Nodes...)
	for i := range out.Nodes {
		var fns []variate.Function
		for _, f := range out.Nodes[i].Functions {
			if c, ok := f.Model.(dist.Constant); ok && f.Basis == variate.OnEntry {
				half := C(c.Mean() / 2)
				fns = append(fns, variate.Function{Name: f.Name + "_1", Category: f.Category, Model: half, Scope: f.Scope},
					variate.Function{Name: f.Name + "_2", Category: f.Category, Model: half, Scope: f.Scope})
				changed = true
			} else {
				fns = append(fns, f)
			}
		}
		out.Nodes[i].Functions = fns
	}
	return out, changed
}

// merged prefixes two graphs and joins them into one model.
func merged(a, b variate.Graph) variate.Graph {
	out := variate.Graph{ID: "merged"}
	for _, g := range []variate.Graph{a, b} {
		p := g.ID + "."
		for _, s := range g.Sources {
			s.ID, s.To = p+s.ID, p+s.To
			out.Sources = append(out.Sources, s)
		}
		for _, n := range g.Nodes {
			n.ID = p + n.ID
			if n.Next != "" {
				n.Next = p + n.Next
			}
			n.Outcomes = append([]variate.Outcome{}, n.Outcomes...)
			for j := range n.Outcomes {
				n.Outcomes[j].To = p + n.Outcomes[j].To
			}
			out.Nodes = append(out.Nodes, n)
		}
	}
	return out
}

// ---- tests ------------------------------------------------------------------

func TestMetamorphicSplitSource(t *testing.T) {
	for seed := uint64(1); seed <= metaSeeds; seed++ {
		g := genGraph(t, seed, false)
		h, ok := withSplitSource(g)
		if !ok {
			continue
		}
		ca, ea := exactOf(t, g)
		cb, eb := exactOf(t, h)
		what := "split source seed " + strconv.FormatUint(seed, 10)
		sameExpectations(t, what, ca, cb, ea, eb)
		sameTransform(t, what, ca, cb)
		_, ra := simulate(t, g, seed)
		_, rb := simulate(t, h, seed)
		sameMean(t, what, ra, rb)
	}
}

func TestMetamorphicPassThrough(t *testing.T) {
	for seed := uint64(1); seed <= metaSeeds; seed++ {
		for _, delays := range []bool{false, true} {
			g := genGraph(t, seed, delays)
			h := withPassThrough(g)
			ca, ea := exactOf(t, g)
			cb, eb := exactOf(t, h)
			what := "pass-through seed " + strconv.FormatUint(seed, 10)
			if !near(ea.Mean, eb.Mean, 1e-9) || !near(ea.Var, eb.Var, 1e-9) || !near(ea.PZero(), eb.PZero(), 1e-9) {
				t.Errorf("%s: mean %g/%g var %g/%g", what, ea.Mean, eb.Mean, ea.Var, eb.Var)
			}
			if !delays {
				sameTransform(t, what, ca, cb)
			}
			_, ra := simulate(t, g, seed)
			_, rb := simulate(t, h, seed)
			sameMean(t, what, ra, rb)
		}
	}
}

func TestMetamorphicShuffledOrder(t *testing.T) {
	for seed := uint64(1); seed <= metaSeeds; seed++ {
		g := genGraph(t, seed, false)
		h := withShuffledOrder(g, seed)
		ca, ea := exactOf(t, g)
		cb, eb := exactOf(t, h)
		what := "shuffled seed " + strconv.FormatUint(seed, 10)
		sameExpectations(t, what, ca, cb, ea, eb)
		sameTransform(t, what, ca, cb)
		_, ra := simulate(t, g, seed)
		_, rb := simulate(t, h, seed)
		sameMean(t, what, ra, rb)
	}
}

// Constants written as one-atom categoricals draw from their own
// streams only, so the simulation must be byte-identical.
func TestMetamorphicConstantForms(t *testing.T) {
	for seed := uint64(1); seed <= metaSeeds; seed++ {
		g := genGraph(t, seed, seed%2 == 0)
		h, ok := withCategoricalConstants(t, g)
		if !ok {
			continue
		}
		ca, ea := exactOf(t, g)
		cb, eb := exactOf(t, h)
		what := "constant forms seed " + strconv.FormatUint(seed, 10)
		sameExpectations(t, what, ca, cb, ea, eb)
		if seed%2 == 1 {
			sameTransform(t, what, ca, cb)
		}
		ja, _ := simulate(t, g, seed)
		jb, _ := simulate(t, h, seed)
		sameBytes(t, what, ja, jb)
	}
}

// A plain next and a one-outcome selection of probability 1 compile to
// the same thing, so everything must be byte-identical.
func TestMetamorphicNextAsOutcome(t *testing.T) {
	for seed := uint64(1); seed <= metaSeeds; seed++ {
		g := genGraph(t, seed, false)
		h, ok := withNextAsOutcome(g)
		if !ok {
			continue
		}
		ca, ea := exactOf(t, g)
		cb, eb := exactOf(t, h)
		what := "next as outcome seed " + strconv.FormatUint(seed, 10)
		sameExpectations(t, what, ca, cb, ea, eb)
		sameTransform(t, what, ca, cb)
		ja, _ := simulate(t, g, seed)
		jb, _ := simulate(t, h, seed)
		sameBytes(t, what, ja, jb)
	}
}

// Splitting a constant function in two halves changes nothing about the
// period total; the draws count doubles, which only the grid sizing sees.
func TestMetamorphicSplitConstant(t *testing.T) {
	for seed := uint64(1); seed <= metaSeeds; seed++ {
		g := genGraph(t, seed, false)
		h, ok := withSplitConstant(g)
		if !ok {
			continue
		}
		ca, ea := exactOf(t, g)
		cb, eb := exactOf(t, h)
		what := "split constant seed " + strconv.FormatUint(seed, 10)
		if !near(ea.Mean, eb.Mean, 1e-9) || !near(ea.Var, eb.Var, 1e-9) || !near(ea.PZero(), eb.PZero(), 1e-9) {
			t.Errorf("%s: mean %g/%g var %g/%g", what, ea.Mean, eb.Mean, ea.Var, eb.Var)
		}
		sameTransform(t, what, ca, cb)
		// The extra pools shift every later stream, so compare statistically.
		_, ra := simulate(t, g, seed)
		_, rb := simulate(t, h, seed)
		sameMean(t, what, ra, rb)
	}
}

// Two graphs merged into one model: the mean adds, P(T = 0) multiplies,
// and each source's expectation is unchanged.
func TestMetamorphicMergedGraphs(t *testing.T) {
	for seed := uint64(1); seed+1 <= metaSeeds; seed += 2 {
		a := genGraph(t, seed, false)
		b := genGraph(t, seed+1, false)
		a.ID, b.ID = "a", "b"
		m := merged(a, b)
		_, ea := exactOf(t, a)
		_, eb := exactOf(t, b)
		cm, em := exactOf(t, m)
		what := "merged seeds " + strconv.FormatUint(seed, 10)
		if !near(em.Mean, ea.Mean+eb.Mean, 1e-9) || !near(em.Var, ea.Var+eb.Var, 1e-9) || !near(em.PZero(), ea.PZero()*eb.PZero(), 1e-9) {
			t.Errorf("%s: mean %g vs %g+%g, var %g vs %g+%g, pzero %g vs %g·%g", what, em.Mean, ea.Mean, eb.Mean, em.Var, ea.Var, eb.Var, em.PZero(), ea.PZero(), eb.PZero())
		}
		for k, name := range cm.SourceIDs() {
			var want float64
			for i, src := range a.Sources {
				if "a."+src.ID == name {
					want = ea.MeanBySource[i]
				}
			}
			for i, src := range b.Sources {
				if "b."+src.ID == name {
					want = eb.MeanBySource[i]
				}
			}
			if !near(em.MeanBySource[k], want, 1e-9) {
				t.Errorf("%s: source %s mean %g vs %g", what, name, em.MeanBySource[k], want)
			}
		}
		_, ra := simulate(t, a, seed)
		_, rb := simulate(t, b, seed)
		_, rm := simulate(t, m, seed)
		se := math.Sqrt(ra.Total.StdErr(false)*ra.Total.StdErr(false) + rb.Total.StdErr(false)*rb.Total.StdErr(false) + rm.Total.StdErr(false)*rm.Total.StdErr(false))
		if z := (rm.Total.Mean() - ra.Total.Mean() - rb.Total.Mean()) / se; math.Abs(z) > 4.5 {
			t.Errorf("%s: merged mean %g vs %g + %g, z=%.2f", what, rm.Total.Mean(), ra.Total.Mean(), rb.Total.Mean(), z)
		}
	}
}
