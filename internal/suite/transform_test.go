package suite

import (
	"context"
	"math"
	"strconv"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
	"gonum.org/v1/gonum/mathext"
)

// compoundExponential: Poisson(λ) rounds each booking Exponential(mean).
func compoundExponential(lam, mean float64) variate.Graph {
	e, _ := dist.NewExponential(mean)
	return variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(lam), To: "a"}},
		Nodes:   []variate.Node{{ID: "a", Functions: []variate.Function{Fn("x", "k", e)}}},
	}
}

// exactCompoundExpCDF is P(T ≤ x) for Poisson(λ) and Exponential(mean):
// e^(−λ) + Σ_n P(N = n)·Gamma(n, mean).CDF(x).
func exactCompoundExpCDF(lam, mean, x float64) float64 {
	cdf := math.Exp(-lam)
	pn := math.Exp(-lam)
	for n := 1; n < 200; n++ {
		pn *= lam / float64(n)
		cdf += pn * mathext.GammaIncReg(float64(n), x/mean)
	}
	return cdf
}

func transformOf(t *testing.T, g variate.Graph, cfg variate.TransformConfig) (*variate.Compiled, *variate.Transformed) {
	t.Helper()
	m, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	tr, err := m.Transform(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return m, tr
}

// curveMean is Σ_k P(T ≥ kh)·h over the grid.
func curveMean(ec []float64, h float64) float64 {
	sum := 0.0
	for k := 1; k < len(ec); k++ {
		sum += ec[k] * h
	}
	return sum
}

// checkBracketsSeries tests, at every 1/steps of the grid where at least 20
// periods are expected beyond the point, that the observed exceedance of
// the series is not significantly outside the bounds: a one-sided score
// test at the bound, |z| < zMax.
func checkBracketsSeries(t *testing.T, name string, b variate.Bounds, series []float64, steps int, zMax float64) {
	t.Helper()
	n := float64(len(series))
	stride := max(1, len(b.Lower)/steps)
	for k := 0; k < len(b.Lower); k += stride {
		x := float64(k) * b.H
		if b.Upper[k]*n < 20 {
			break
		}
		obs := 0
		for _, v := range series {
			if v >= x {
				obs++
			}
		}
		p := float64(obs) / n
		if p > b.Upper[k] {
			if z := (p - b.Upper[k]) / math.Sqrt(b.Upper[k]*(1-b.Upper[k])/n); z > zMax {
				t.Errorf("%s x=%g: observed %g above upper %g, z=%.2f", name, x, p, b.Upper[k], z)
			}
		}
		if p < b.Lower[k] && b.Lower[k] < 1 {
			if z := (b.Lower[k] - p) / math.Sqrt(b.Lower[k]*(1-b.Lower[k])/n); z > zMax {
				t.Errorf("%s x=%g: observed %g below lower %g, z=%.2f", name, x, p, b.Lower[k], z)
			}
		}
	}
}

// checkQuantilesSeries inverts the series at a few levels and checks
// each sample quantile against the bracket, by the same score test on the
// exceedance at the bracket's ends.
func checkQuantilesSeries(t *testing.T, name string, b variate.Bounds, series []float64, zMax float64) {
	t.Helper()
	n := float64(len(series))
	for _, p := range []float64{0.5, 0.9, 0.99} {
		if (1-p)*n < 50 {
			continue
		}
		lo, hi := b.Quantile(p)
		above := func(x float64) float64 {
			c := 0
			for _, v := range series {
				if v >= x {
					c++
				}
			}
			return float64(c) / n
		}
		// At the upper end the true exceedance is at most 1 − p.
		if obs := above(hi); obs > 1-p {
			if z := (obs - (1 - p)) / math.Sqrt(p*(1-p)/n); z > zMax {
				t.Errorf("%s q%g: %g above upper end %g, z=%.2f", name, p, obs, hi, z)
			}
		}
		// Just below the lower end the true exceedance exceeds 1 − p.
		if obs := above(lo - b.H); obs < 1-p {
			if z := ((1 - p) - obs) / math.Sqrt(p*(1-p)/n); z > zMax {
				t.Errorf("%s q%g: %g below lower end %g, z=%.2f", name, p, obs, lo, z)
			}
		}
	}
}

func TestTransformCompoundExponential(t *testing.T) {
	const lam, mean = 3, 2
	_, tr := transformOf(t, compoundExponential(lam, mean), variate.TransformConfig{})
	if !tr.Applicable {
		t.Fatalf("not applicable: %s", tr.Reason)
	}
	b := tr.Total
	worst := 0.0
	for k := 0; k < len(b.Lower); k += len(b.Lower) / 200 {
		x := float64(k) * b.H
		exact := 1 - exactCompoundExpCDF(lam, mean, x)
		if k == 0 {
			exact = 1
		}
		if exact < b.Lower[k]-1e-12 || exact > b.Upper[k]+1e-12 {
			t.Errorf("x=%g: exact %.6g outside [%.6g, %.6g]", x, exact, b.Lower[k], b.Upper[k])
		}
		worst = math.Max(worst, b.Upper[k]-b.Lower[k])
	}
	if worst > 1e-3 {
		t.Errorf("bracket width %g", worst)
	}
	if math.Abs(tr.PZero-math.Exp(-lam)) > 1e-10 {
		t.Errorf("P(T=0) %g exact %g", tr.PZero, math.Exp(-lam))
	}
	lo, hi := b.Quantile(0.99)
	// Invert the exact CDF by bisection for the check.
	a, c := 0.0, 1000.0
	for i := 0; i < 100; i++ {
		mid := (a + c) / 2
		if exactCompoundExpCDF(lam, mean, mid) < 0.99 {
			a = mid
		} else {
			c = mid
		}
	}
	if a < lo-b.H || a > hi+b.H {
		t.Errorf("P99 %g outside [%g, %g]", a, lo, hi)
	}
	if tr.QuadNodes != 0 || tr.Passes != 1 {
		t.Errorf("quad nodes %d passes %d", tr.QuadNodes, tr.Passes)
	}
}

func TestTransformRefusals(t *testing.T) {
	type refusal struct {
		name string
		g    variate.Graph
		want string
	}
	cases := make([]refusal, 0, 5)
	g := Diamond()
	g.Nodes[0].Outcomes[0].Holding = C(1)
	cases = append(cases, refusal{"delays", g, "delays"})
	g = Diamond()
	for i := range g.Nodes {
		g.Nodes[i].Functions = append(g.Nodes[i].Functions, variate.Function{Name: "p" + g.Nodes[i].ID, Category: "k", Model: C(1), Scope: variate.PerPeriod})
	}
	cases = append(cases, refusal{"four period-scoped", g, "more than 3 period-scoped"})
	g = Diamond()
	g.Sources[0].Frequency = C(2)
	ln, _ := dist.NewLognormal(0, 0.5)
	g.Nodes[1].Functions[0].Model = ln
	g.Nodes[1].Functions[0].Scope = variate.PerPeriod
	cases = append(cases, refusal{"unbounded period value", g, "span too many lattice points"})
	g = Diamond()
	g.Nodes[1].Functions[0].Model = Pert(1, 2, 30)
	g.Nodes[1].Functions[0].Scope = variate.PerPeriod
	cases = append(cases, refusal{"continuous period value with quadrature", g, "above the work cap"})
	g = compoundExponential(1<<21, 1)
	cases = append(cases, refusal{"too many draws", g, "too many draws per period"})
	for _, c := range cases {
		m, _, err := c.g.Compile(variate.Config{AllowUnconverged: true})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		tr, err := m.Transform(variate.TransformConfig{})
		if err != nil {
			t.Fatal(err)
		}
		if tr.Applicable || !strings.Contains(tr.Reason, c.want) {
			t.Errorf("%s: applicable=%v reason %q, want %q", c.name, tr.Applicable, tr.Reason, c.want)
		}
	}
}

// TestTransformBracketsSimulation: on the diamond, the simulated
// exceedance and quantiles lie within the bounds.
func TestTransformBracketsSimulation(t *testing.T) {
	g := Diamond()
	m, tr := transformOf(t, g, variate.TransformConfig{})
	if !tr.Applicable {
		t.Fatalf("not applicable: %s", tr.Reason)
	}
	const h = 40000
	_, r := run(t, g, variate.Config{Periods: h, Series: true}, 3)
	e, _ := m.Analytic()
	totals := r.Series.TotalColumn(make([]float64, h))
	checkBracketsSeries(t, "total", tr.Total, totals, 100, 4.5)
	checkQuantilesSeries(t, "total", tr.Total, totals, 4.5)
	for c := range tr.ByCategory {
		checkBracketsSeries(t, "category "+m.Categories()[c], tr.ByCategory[c], r.Series.CategoryColumn(c, make([]float64, h)), 50, 4.5)
	}
	if tr.PZero <= 0 || math.Abs(tr.PZero-e.PZero()) > 1e-9 {
		t.Errorf("pzero %g exact %g", tr.PZero, e.PZero())
	}
	if len(tr.ByCategory) != 2 || len(tr.BySource) != 1 {
		t.Errorf("breakdowns %d %d", len(tr.ByCategory), len(tr.BySource))
	}
}

// TestTransformPERTFrequency: a PERT frequency goes through the quadrature.
func TestTransformPERTFrequency(t *testing.T) {
	g := Diamond() // frequency PERT(1, 2, 4)
	m, tr := transformOf(t, g, variate.TransformConfig{})
	if !tr.Applicable {
		t.Fatalf("not applicable: %s", tr.Reason)
	}
	if tr.QuadNodes == 0 {
		t.Error("expected quadrature nodes")
	}
	e, _ := m.Analytic()
	meanLo, meanHi := curveMean(tr.Total.Lower, tr.H), curveMean(tr.Total.Upper, tr.H)
	if meanLo > e.Mean || meanHi < e.Mean {
		t.Errorf("curve means %g %g do not bracket %g", meanLo, meanHi, e.Mean)
	}
}

// periodProbabilityDiamond is the diamond with a constant frequency and a
// Beta(2, 5) probability drawn per period at the first node.
func periodProbabilityDiamond() variate.Graph {
	g := Diamond()
	g.Sources[0].Frequency = C(13.0 / 6)
	g.Nodes[0].Outcomes = []variate.Outcome{{To: "b", P: Beta(2, 5)}, {To: "c"}}
	return g
}

// TestTransformPeriodProbability: the mixture over a period-scoped
// probability brackets the simulation and the exact mean and P(T = 0).
func TestTransformPeriodProbability(t *testing.T) {
	g := periodProbabilityDiamond()
	m, tr := transformOf(t, g, variate.TransformConfig{Tol: 1e-2})
	if !tr.Applicable {
		t.Fatalf("not applicable: %s", tr.Reason)
	}
	if tr.Passes < 2 {
		t.Errorf("passes %d", tr.Passes)
	}
	e, _ := m.Analytic()
	meanLo, meanHi := curveMean(tr.Total.Lower, tr.H), curveMean(tr.Total.Upper, tr.H)
	if meanLo > e.Mean || meanHi < e.Mean {
		t.Errorf("curve means %g %g do not bracket %g", meanLo, meanHi, e.Mean)
	}
	if math.Abs(tr.PZero-e.PZero()) > 1e-9 {
		t.Errorf("pzero %g exact %g", tr.PZero, e.PZero())
	}
	const h = 40000
	_, r := run(t, g, variate.Config{Periods: h, Series: true}, 5)
	totals := r.Series.TotalColumn(make([]float64, h))
	checkBracketsSeries(t, "total", tr.Total, totals, 100, 4.5)
	checkQuantilesSeries(t, "total", tr.Total, totals, 4.5)
}

// TestTransformCategoricalProbabilityExact: a probability with three atoms
// mixes exactly: P(T = 0) is the weighted sum of the fixed-probability
// values to 1e-14, and the bracket overlaps the weighted sum of the
// fixed-probability brackets.
func TestTransformCategoricalProbabilityExact(t *testing.T) {
	values := []float64{0.2, 0.5, 0.8}
	probs := []float64{0.5, 0.3, 0.2}
	cat, err := dist.NewCategorical(values, probs)
	if err != nil {
		t.Fatal(err)
	}
	g := periodProbabilityDiamond()
	g.Nodes[0].Outcomes[0].P = cat
	cfg := variate.TransformConfig{Tol: 1e-2}
	_, mixed := transformOf(t, g, cfg)
	if !mixed.Applicable {
		t.Fatalf("not applicable: %s", mixed.Reason)
	}
	if mixed.Passes != 3 {
		t.Errorf("passes %d, want 3", mixed.Passes)
	}
	pzero := 0.0
	fixed := make([]*variate.Transformed, len(values))
	for i, p := range values {
		gi := periodProbabilityDiamond()
		gi.Nodes[0].Outcomes[0].P = C(p)
		_, fixed[i] = transformOf(t, gi, cfg)
		if !fixed[i].Applicable {
			t.Fatalf("fixed %g: %s", p, fixed[i].Reason)
		}
		pzero += probs[i] * fixed[i].PZero
	}
	if math.Abs(pzero-mixed.PZero) > 1e-14 {
		t.Errorf("P(T=0) mixed %.17g, from fixed %.17g", mixed.PZero, pzero)
	}
	b := mixed.Total
	for k := 0; k < len(b.Lower); k += len(b.Lower) / 50 {
		x := float64(k) * b.H
		lo, hi := 0.0, 0.0
		for i := range values {
			bi := fixed[i].Total
			kl := min(int(math.Ceil(x/bi.H)), len(bi.Lower)-1)
			ku := min(int(math.Floor(x/bi.H)), len(bi.Upper)-1)
			lo += probs[i] * bi.Lower[kl]
			hi += probs[i] * bi.Upper[ku]
		}
		if b.Lower[k] > hi+1e-12 || lo > b.Upper[k]+1e-12 {
			t.Errorf("x=%g: mixed [%g, %g] and summed [%g, %g] do not overlap", x, b.Lower[k], b.Upper[k], lo, hi)
		}
	}
}

// TestTransformPeriodValue: a period-scoped value with three atoms on the
// lattice is summed exactly over its atoms and brackets the simulation.
func TestTransformPeriodValue(t *testing.T) {
	cat, err := dist.NewCategorical([]float64{1, 2, 4}, []float64{0.5, 0.3, 0.2})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := dist.NewExponential(3)
	g := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(2), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{{Name: "fee", Category: "k", Model: cat, Scope: variate.PerPeriod}, Fn("fix", "k", C(1))}, Next: "b"},
			{ID: "b", Functions: []variate.Function{Fn("x", "m", e)}},
		},
	}
	m, tr := transformOf(t, g, variate.TransformConfig{})
	if !tr.Applicable {
		t.Fatalf("not applicable: %s", tr.Reason)
	}
	if tr.Passes != 3 {
		t.Errorf("passes %d, want 3", tr.Passes)
	}
	ex, _ := m.Analytic()
	meanLo, meanHi := curveMean(tr.Total.Lower, tr.H), curveMean(tr.Total.Upper, tr.H)
	if meanLo > ex.Mean || meanHi < ex.Mean {
		t.Errorf("curve means %g %g do not bracket %g", meanLo, meanHi, ex.Mean)
	}
	if math.Abs(tr.PZero-ex.PZero()) > 1e-9 {
		t.Errorf("pzero %g exact %g", tr.PZero, ex.PZero())
	}
	const h = 40000
	_, r := run(t, g, variate.Config{Periods: h, Series: true}, 7)
	totals := r.Series.TotalColumn(make([]float64, h))
	checkBracketsSeries(t, "total", tr.Total, totals, 100, 4.5)
}

// TestTransformEntries: the count distributions bracket the simulated
// entries per node and their exact means.
func TestTransformEntries(t *testing.T) {
	for _, tc := range []struct {
		name string
		g    variate.Graph
	}{{"pert frequency", Diamond()}, {"period probability", periodProbabilityDiamond()}} {
		m, _, err := tc.g.Compile(variate.Config{})
		if err != nil {
			t.Fatal(err)
		}
		cs, err := m.TransformEntries(variate.TransformConfig{})
		if err != nil {
			t.Fatal(err)
		}
		if !cs.Applicable {
			t.Fatalf("%s: %s", tc.name, cs.Reason)
		}
		e, _ := m.Analytic()
		const h = 40000
		_, r := run(t, tc.g, variate.Config{Periods: h, Series: true}, 11)
		for v := range cs.Entries {
			b := cs.Entries[v]
			if b.H != 1 || len(b.Lower) != cs.M || b.DiscGap != nil {
				t.Errorf("%s node %d: shape H=%g len=%d", tc.name, v, b.H, len(b.Lower))
			}
			lo, hi := curveMean(b.Lower, 1), curveMean(b.Upper, 1)
			if lo > e.EntriesPer[v] || hi < e.EntriesPer[v] {
				t.Errorf("%s node %d: means %g %g do not bracket %g", tc.name, v, lo, hi, e.EntriesPer[v])
			}
			checkBracketsSeries(t, tc.name+" node "+m.NodeIDs()[v], b, r.Series.EntriesColumn(v, make([]float64, h)), cs.M, 4.5)
		}
	}
}

// TestTransformTailCriterion: with Config.Transform, the tail criterion
// counts periods at or beyond the transform's quantile bound.
func TestTransformTailCriterion(t *testing.T) {
	g := compoundExponential(3, 2)
	cfg := variate.Config{Transform: true, TailP: 0.99, TailHits: 50, MinHits: 1, TargetRSE: 0.5, ContribRSE: 0.5, Batch: 4}
	m, ws, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range ws {
		if w.Code == "no-transform" || w.Code == "tail-beyond-range" {
			t.Errorf("unexpected warning %+v", w)
		}
	}
	tr := m.Transformed()
	if tr == nil || !tr.Applicable {
		t.Fatal("no transform at compile")
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(w.Close)
	r, err := w.Run(context.Background(), 13)
	if err != nil {
		t.Fatal(err)
	}
	if r.TailPeriods < 50 || !r.Converged {
		t.Errorf("tail periods %d converged %v binding %s", r.TailPeriods, r.Converged, r.Binding)
	}
	_, hi := tr.Total.Quantile(0.99)
	// About 1% of periods lie beyond the bound; the count is binomial.
	if z := zBinomial(uint64(max(r.TailPeriods, 0)), float64(r.Periods), 0.01); math.Abs(z) > 4.5 {
		t.Errorf("tail periods %d of %d beyond %g, z=%.2f", r.TailPeriods, r.Periods, hi, z)
	}
	var sum variate.Summary
	w.Summarize(&sum)
	found := false
	for _, row := range sum.Plan {
		if row.Criterion == "tail" {
			found = true
			if row.Observed != float64(r.TailPeriods) || !row.Met {
				t.Errorf("tail row %+v", row)
			}
		}
	}
	if !found {
		t.Error("no tail row")
	}
}

// transformGraph adapts a random DAG to the transform: no delays, no
// period-scoped values, constants on a half-unit lattice, and at most
// maxScoped period-scoped probabilities; it returns the number kept.
func transformGraph(t *testing.T, seed uint64, maxScoped int) (variate.Graph, int) {
	t.Helper()
	g := genGraph(t, seed, false)
	scoped := 0
	for i := range g.Nodes {
		for f := range g.Nodes[i].Functions {
			fn := &g.Nodes[i].Functions[f]
			fn.Scope = variate.PerEvent
			if c, ok := fn.Model.(dist.Constant); ok {
				fn.Model = C(math.Ceil(c.Mean()*2) / 2)
			}
		}
		for j := range g.Nodes[i].Outcomes {
			o := &g.Nodes[i].Outcomes[j]
			if _, ok := o.P.(dist.Beta); ok {
				scoped++
				if scoped > maxScoped {
					o.P = C(o.P.Mean())
				}
			}
		}
	}
	return g, min(scoped, maxScoped)
}

// TestTransformRandomDAGs: on random DAGs the curve means bracket the
// exact mean without sampling, and the simulated exceedance lies within
// the bounds (Bonferroni over the points tested). The fast run keeps one
// period-scoped probability per model; the stat run allows three.
func TestTransformRandomDAGs(t *testing.T) {
	SkipUnderRace(t)
	checkRandomDAGs(t, []uint64{1, 6, 7}, 1, 1<<19)
}

func checkRandomDAGs(t *testing.T, seeds []uint64, maxScoped, maxGrid int) {
	t.Helper()
	const h = 20000
	for _, seed := range seeds {
		g, _ := transformGraph(t, seed, maxScoped)
		m, tr := transformOf(t, g, variate.TransformConfig{Tol: 1e-2, MaxGrid: maxGrid})
		if !tr.Applicable {
			t.Logf("seed %d: %s", seed, tr.Reason)
			continue
		}
		e, _ := m.Analytic()
		meanLo, meanHi := curveMean(tr.Total.Lower, tr.H), curveMean(tr.Total.Upper, tr.H)
		if meanLo > e.Mean || meanHi < e.Mean {
			t.Errorf("seed %d: curve means %g %g do not bracket %g", seed, meanLo, meanHi, e.Mean)
		}
		if math.Abs(tr.PZero-e.PZero()) > 1e-9 {
			t.Errorf("seed %d: pzero %g exact %g", seed, tr.PZero, e.PZero())
		}
		_, r := run(t, g, variate.Config{Periods: h, Series: true}, seed)
		totals := r.Series.TotalColumn(make([]float64, h))
		checkBracketsSeries(t, "seed "+strconv.FormatUint(seed, 10), tr.Total, totals, 50, 5)
	}
}

// TestTransformHeavySeverityTail: a lognormal severity whose tail is
// folded away still passes the lattice-mean self-check, since the
// dropped tail's mean is allowed for.
func TestTransformHeavySeverityTail(t *testing.T) {
	SkipUnderRace(t)
	ln, _ := dist.NewLognormalFromMedianP90(5e4, 2e5)
	g := variate.Graph{
		Sources: []variate.Source{{ID: "trip", Frequency: C(5), To: "repair"}},
		Nodes: []variate.Node{
			{ID: "repair", Functions: []variate.Function{Fn("callout", "labour", Pert(500, 1200, 4000))},
				Outcomes: []variate.Outcome{{To: "replace", P: Beta(2, 8)}, {To: "done"}}},
			{ID: "replace", Functions: []variate.Function{Fn("pump", "capital", ln)}},
			{ID: "done"},
		},
	}
	_, tr := transformOf(t, g, variate.TransformConfig{Tol: 1e-2})
	if !tr.Applicable {
		t.Fatalf("not applicable: %s", tr.Reason)
	}
}

// TestTransformWorkBudget: a budget too small for the first grid refuses
// with the budget named, the default budget counts the work spent, and a
// MaxGrid below the smallest grid is raised to it.
func TestTransformWorkBudget(t *testing.T) {
	g := Diamond() // frequency PERT(1, 2, 4), so the rate is a quadrature
	_, tr := transformOf(t, g, variate.TransformConfig{Tol: 1e-2, MaxWork: 1000})
	if tr.Applicable || !strings.Contains(tr.Reason, "work budget of 1000") {
		t.Fatalf("applicable %v reason %q", tr.Applicable, tr.Reason)
	}
	_, tr = transformOf(t, g, variate.TransformConfig{Tol: 1e-2, MaxGrid: 1})
	if !tr.Applicable || tr.M < 1<<16 || tr.Work == 0 {
		t.Fatalf("applicable %v m %d work %d reason %q", tr.Applicable, tr.M, tr.Work, tr.Reason)
	}
	_, capped := transformOf(t, g, variate.TransformConfig{Tol: 1e-2, MaxWork: tr.Work})
	if !capped.Applicable || capped.Work != tr.Work {
		t.Errorf("exact budget: applicable %v work %d vs %d", capped.Applicable, capped.Work, tr.Work)
	}
	_, short := transformOf(t, g, variate.TransformConfig{Tol: 1e-2, MaxWork: tr.Work - 1})
	if short.Applicable || !strings.Contains(short.Reason, "work budget") {
		t.Errorf("one short: applicable %v reason %q", short.Applicable, short.Reason)
	}
}
