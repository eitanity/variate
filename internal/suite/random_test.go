package suite

import (
	"math"
	"math/rand/v2"
	"sort"
	"strconv"
	"testing"

	"code.eitanity.com/variate"

	"code.eitanity.com/variate/dist"
	"code.eitanity.com/variate/internal/oracle"
	"code.eitanity.com/variate/stats"
)

// genGraph builds a random DAG covering the engine features: 2–5 sources,
// 5–30 nodes, 2–4 outcomes per node, fixed and uncertain probabilities in
// both modes, period-scoped values and holding times, shared nodes, both
// function bases, mixed families, and a share of function-less nodes.
func genGraph(t testing.TB, seed uint64, delays bool) variate.Graph {
	t.Helper()
	r := rand.New(rand.NewPCG(seed, 77))
	n := 5 + r.IntN(26)
	g := variate.Graph{ID: "rand" + strconv.FormatUint(seed, 10)}
	cats := []string{"labour", "parts", "downtime"}
	for i := 0; i < n; i++ {
		nd := variate.Node{ID: "n" + strconv.Itoa(i)}
		leaf := i >= n-2 || r.Float64() < 0.15
		if !leaf {
			if r.Float64() < 0.2 {
				nd.Next = "n" + strconv.Itoa(i+1+r.IntN(n-i-1))
				if delays && r.Float64() < 0.4 {
					nd.NextHolding, nd.NextScope = genHolding(t, r)
				}
			} else {
				nd.Mode, nd.Outcomes = genOutcomes(t, r, i, n, delays)
			}
		}
		if r.Float64() < 0.5 {
			for f := 0; f < 1+r.IntN(2); f++ {
				fn := variate.Function{Name: "f" + strconv.Itoa(i) + "_" + strconv.Itoa(f), Category: cats[r.IntN(3)], Model: genValue(t, r)}
				if !leaf && r.Float64() < 0.3 {
					fn.Basis = variate.PerTime
				}
				if r.Float64() < 0.25 {
					fn.Scope = variate.PerPeriod
				}
				nd.Functions = append(nd.Functions, fn)
			}
		}
		g.Nodes = append(g.Nodes, nd)
	}
	for k := 0; k < 2+r.IntN(4); k++ {
		g.Sources = append(g.Sources, variate.Source{ID: "s" + strconv.Itoa(k), Frequency: genFrequency(t, r), To: "n" + strconv.Itoa(r.IntN(max(1, n/2)))})
	}
	return g
}

func genOutcomes(t testing.TB, r *rand.Rand, i, n int, delays bool) (variate.Mode, []variate.Outcome) {
	t.Helper()
	count := 2 + r.IntN(3)
	if n-i-1 < count {
		count = n - i - 1
	}
	outs := make([]variate.Outcome, count)
	for j := range outs {
		outs[j].To = "n" + strconv.Itoa(i+1+r.IntN(n-i-1))
		if delays && r.Float64() < 0.3 {
			outs[j].Holding, outs[j].HoldingScope = genHolding(t, r)
		}
	}
	switch u := r.Float64(); {
	case u < 0.2 && count == 2:
		a := 1 + 4*r.Float64()
		b := 1 + 4*r.Float64()
		outs[0].P = mustBeta(t, a, b)
		return variate.Absolute, outs
	case u < 0.4:
		for j := 0; j < count-1; j++ {
			if r.Float64() < 0.5 {
				outs[j].P = mustBeta(t, 2, 2+3*r.Float64())
			} else {
				outs[j].P = C(0.2 + 0.5*r.Float64())
			}
		}
		return variate.Sequential, outs
	default:
		w := make([]float64, count)
		sum := 0.0
		for j := range w {
			w[j] = 0.1 + r.Float64()
			sum += w[j]
		}
		for j := range w {
			outs[j].P = C(w[j] / sum)
		}
		if r.Float64() < 0.3 {
			outs[count-1].P = nil
		}
		return variate.Absolute, outs
	}
}

// The constructors below fail the test on an error, so a bad parameter
// never becomes a zero-valued distribution.
func mustBeta(t testing.TB, a, b float64) dist.Beta {
	t.Helper()
	d, err := dist.NewBeta(a, b)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func mustPERT(t testing.TB, l, p, h float64) dist.PERT {
	t.Helper()
	d, err := dist.NewPERT(l, p, h, 4)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func mustLognormal(t testing.TB, mu, sigma float64) dist.Lognormal {
	t.Helper()
	d, err := dist.NewLognormal(mu, sigma)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func mustGamma(t testing.TB, k, th float64) dist.Gamma {
	t.Helper()
	d, err := dist.NewGamma(k, th)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func mustUniform(t testing.TB, a, b float64) dist.Uniform {
	t.Helper()
	d, err := dist.NewUniform(a, b)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func mustExponential(t testing.TB, m float64) dist.Exponential {
	t.Helper()
	d, err := dist.NewExponential(m)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func genValue(t testing.TB, r *rand.Rand) dist.Distribution {
	t.Helper()
	switch r.IntN(5) {
	case 0:
		return C(1 + 10*r.Float64())
	case 1:
		lo := math.Pow(10, 3*r.Float64())
		return mustPERT(t, lo, lo*(1+r.Float64()), lo*(2+3*r.Float64()))
	case 2:
		return mustLognormal(t, r.Float64()*3, 0.2+0.6*r.Float64())
	case 3:
		return mustGamma(t, 1+4*r.Float64(), 1+r.Float64())
	default:
		return mustUniform(t, 1, 2+5*r.Float64())
	}
}

func genHolding(t testing.TB, r *rand.Rand) (dist.Continuous, variate.Scope) {
	t.Helper()
	var d dist.Continuous
	switch r.IntN(4) {
	case 0:
		d = C(0.1 + 2*r.Float64())
	case 1:
		peak := 0.5 + r.Float64()
		d = mustPERT(t, 0, peak, peak+0.5+2*r.Float64())
	case 2:
		d = mustExponential(t, 0.2+r.Float64())
	default:
		d = mustUniform(t, 0, 1+r.Float64())
	}
	if r.Float64() < 0.25 {
		return d, variate.PerPeriod
	}
	return d, variate.PerEvent
}

func genFrequency(t testing.TB, r *rand.Rand) dist.Distribution {
	t.Helper()
	switch r.IntN(3) {
	case 0:
		return C(0.5 + 4*r.Float64())
	case 1:
		lo := 0.2 + r.Float64()
		return mustPERT(t, lo, lo+r.Float64(), lo+1+3*r.Float64())
	default:
		return mustGamma(t, 2+3*r.Float64(), 0.3+0.5*r.Float64())
	}
}

// zBinomial compares a count with its binomial expectation.
func zBinomial(count uint64, n float64, p float64) float64 {
	if p <= 0 || p >= 1 || n == 0 {
		return 0
	}
	return (float64(count) - n*p) / math.Sqrt(n*p*(1-p))
}

func TestRandomDAGsAgainstAnalytic(t *testing.T) {
	for seed := uint64(1); seed <= 16; seed++ {
		delays := seed%2 == 0
		g := genGraph(t, seed, delays)
		name := g.ID
		const h = 20000
		cfg := variate.Config{Periods: h, BlockPeriods: 250, Series: true}
		m, r := run(t, g, cfg, seed)
		e, _ := m.Analytic()
		checkInvariants(t, m, r)
		sh := shapeOf(m)
		for k := range r.Sources {
			ev := float64(r.Events[k])
			for v := range r.Nodes {
				if z := zBinomial(r.Reach[k][v], ev, e.Reach[k][v]); math.Abs(z) > 4.5 {
					t.Errorf("%s reach %s/%s z=%.2f", name, r.Sources[k], r.Nodes[v], z)
				}
				if z := zBinomial(r.Reach[k][v], ev, e.Terminal[k][v]); e.Terminal[k][v] > 0 && math.Abs(z) > 4.5 {
					t.Errorf("%s terminal %s/%s z=%.2f", name, r.Sources[k], r.Nodes[v], z)
				}
			}
			for j := range r.Outcomes {
				v := sh.outNode[j]
				p := e.OutcomeRate[j] / e.EntriesPer[v]
				if z := zBinomial(r.OutcomeCount[k][j], float64(r.Reach[k][v]), p); e.EntriesPer[v] > 0 && math.Abs(z) > 4.5 {
					t.Errorf("%s outcome %s from %s z=%.2f", name, r.Outcomes[j], r.Sources[k], z)
				}
			}
		}
		for v := range r.Nodes {
			if z := zAccum(&r.EntriesPer[v], e.EntriesPer[v], delays); math.Abs(z) > 4.5 {
				t.Errorf("%s entries %s: %g analytic %g z=%.2f", name, r.Nodes[v], r.EntriesPer[v].Mean(), e.EntriesPer[v], z)
			}
		}
		if z := zAccum(&r.Total, e.Mean, delays); math.Abs(z) > 4.5 {
			t.Errorf("%s total %g analytic %g z=%.2f", name, r.Total.Mean(), e.Mean, z)
		}
		for c := range r.ByCategory {
			if z := zAccum(&r.ByCategory[c], e.MeanByCategory[c], delays); math.Abs(z) > 4.5 {
				t.Errorf("%s category %s: %g analytic %g z=%.2f", name, r.Categories[c], r.ByCategory[c].Mean(), e.MeanByCategory[c], z)
			}
		}
		for k := range r.BySource {
			if z := zAccum(&r.BySource[k], e.MeanBySource[k], delays); math.Abs(z) > 4.5 {
				t.Errorf("%s source %s: %g analytic %g z=%.2f", name, r.Sources[k], r.BySource[k].Mean(), e.MeanBySource[k], z)
			}
		}
		if !delays {
			totals := r.Series.TotalColumn(make([]float64, h))
			if z := zVariance(totals, e.Var); math.Abs(z) > 4.5 {
				t.Errorf("%s variance %g analytic %g z=%.2f", name, r.Total.StdDev()*r.Total.StdDev(), e.Var, z)
			}
		}
		histogramAccuracy(t, name, &r.Total, r.Series.TotalColumn(make([]float64, h)), 1e-3)
		if delays {
			steadyState(t, name, r.Series)
		}
	}
}

// zAccum is the z-score of an accumulated mean against its expectation,
// with the batch-means error when the model has delays.
func zAccum(a *variate.Accum, want float64, delays bool) float64 {
	se := a.StdErr(delays)
	if se == 0 || math.IsNaN(se) {
		return 0
	}
	return (a.Mean() - want) / se
}

// zVariance compares the sample variance with an exact one using the
// sample fourth moment for its standard error.
func zVariance(xs []float64, want float64) float64 {
	n := float64(len(xs))
	m := 0.0
	for _, x := range xs {
		m += x
	}
	m /= n
	var m2, m4 float64
	for _, x := range xs {
		d := (x - m) * (x - m)
		m2 += d
		m4 += d * d
	}
	m2 /= n
	m4 /= n
	return (m2 - want) / math.Sqrt((m4-m2*m2)/n)
}

// histogramAccuracy: every histogram quantile within res of the type-1
// quantile of the raw series, exceedance exact at bin edges and within a
// bin's mass between them.
func histogramAccuracy(t *testing.T, name string, a *variate.Accum, raw []float64, res float64) {
	t.Helper()
	sorted := append([]float64(nil), raw...)
	sort.Float64s(sorted)
	for _, p := range []float64{0.5, 0.9, 0.99, 0.999} {
		got, clipped := a.Hist.Quantile(p)
		want := stats.QuantileSorted(sorted, p)
		if clipped || math.Abs(got-want) > res*want+1e-12 {
			t.Errorf("%s quantile %g: hist %g raw %g clipped %v", name, p, got, want, clipped)
		}
	}
	for _, x := range []float64{sorted[len(sorted)/2], sorted[len(sorted)*9/10]} {
		if x <= 0 {
			continue
		}
		k := int(math.Floor(math.Log(x/firstEdge(a.Hist)) / math.Log1p(res)))
		e := a.Hist.Edge(k)
		rawAbove := int64(len(sorted) - sort.SearchFloat64s(sorted, e))
		if got := a.Hist.AtOrAbove(e); got != rawAbove {
			t.Errorf("%s edge %g: hist %d raw %d", name, e, got, rawAbove)
		}
	}
}

func firstEdge(h *stats.LogHist) float64 {
	lo, _ := h.Range()
	return lo
}

// steadyState: with delays, the mean of the first 1% of periods agrees
// with the overall mean within sampling error.
func steadyState(t *testing.T, name string, s *variate.Series) {
	t.Helper()
	n := s.Periods()
	head := n / 100
	var sumHead, sum, sq float64
	for p := int64(0); p < n; p++ {
		x := s.Total(p)
		sum += x
		sq += x * x
		if p < head {
			sumHead += x
		}
	}
	mean := sum / float64(n)
	v := sq/float64(n) - mean*mean
	if z := (sumHead/float64(head) - mean) / math.Sqrt(v/float64(head)); math.Abs(z) > 4.5 {
		t.Errorf("%s steady state: first 1%% mean %g overall %g z=%.2f", name, sumHead/float64(head), mean, z)
	}
}

// ---- oracle equivalence ----

func ksPValue(d float64, n float64) float64 {
	sn := math.Sqrt(n)
	z := d * (sn + 0.12 + 0.11/sn)
	if z < 0.3 {
		return 1
	}
	sum := 0.0
	for k := 1; k <= 200; k++ {
		term := 2 * math.Exp(-2*float64(k*k)*z*z)
		if k%2 == 0 {
			term = -term
		}
		sum += term
		if math.Abs(term) < 1e-16 {
			break
		}
	}
	return sum
}

// ks2 is the two-sample Kolmogorov–Smirnov p-value; with ties it is
// conservative.
func ks2(a, b []float64) (d, p float64) {
	sa := append([]float64(nil), a...)
	sb := append([]float64(nil), b...)
	sort.Float64s(sa)
	sort.Float64s(sb)
	i, j := 0, 0
	for i < len(sa) && j < len(sb) {
		x := math.Min(sa[i], sb[j])
		for i < len(sa) && sa[i] <= x {
			i++
		}
		for j < len(sb) && sb[j] <= x {
			j++
		}
		d = math.Max(d, math.Abs(float64(i)/float64(len(sa))-float64(j)/float64(len(sb))))
	}
	n := float64(len(sa)) * float64(len(sb)) / float64(len(sa)+len(sb))
	return d, ksPValue(d, n)
}

func corr(a, b []float64) float64 {
	n := float64(len(a))
	var ma, mb float64
	for i := range a {
		ma += a[i]
		mb += b[i]
	}
	ma /= n
	mb /= n
	var sab, saa, sbb float64
	for i := range a {
		sab += (a[i] - ma) * (b[i] - mb)
		saa += (a[i] - ma) * (a[i] - ma)
		sbb += (b[i] - mb) * (b[i] - mb)
	}
	if saa == 0 || sbb == 0 {
		return 0
	}
	return sab / math.Sqrt(saa*sbb)
}

func TestRandomDAGsAgainstOracle(t *testing.T) {
	for seed := uint64(1); seed <= 12; seed++ {
		delays := seed%2 == 0
		g := genGraph(t, seed, delays)
		name := g.ID
		const h = 6000
		m, r := run(t, g, variate.Config{Periods: h, BlockPeriods: 500, Series: true}, seed+100)
		o := oracle.Run(g, h, m.Spill(), seed+200)
		buf := make([]float64, h)
		if _, p := ks2(r.Series.TotalColumn(buf), o.Total); p < 1e-4 {
			t.Errorf("%s total distribution differs from the oracle, p=%.3g", name, p)
		}
		for c, cat := range r.Categories {
			oc := -1
			for i, oc2 := range o.Categories {
				if oc2 == cat {
					oc = i
				}
			}
			if oc < 0 {
				t.Fatalf("%s category %s missing in the oracle", name, cat)
			}
			if _, p := ks2(r.Series.CategoryColumn(c, buf), o.ByCategory[oc]); p < 1e-4 {
				t.Errorf("%s category %s differs from the oracle, p=%.3g", name, cat, p)
			}
		}
		for v := range r.Nodes {
			if _, p := ks2(r.Series.EntriesColumn(v, buf), o.Entries[v]); p < 1e-4 {
				t.Errorf("%s entries at %s differ from the oracle, p=%.3g", name, r.Nodes[v], p)
			}
		}
		// Parent–child correlation of entries, compared by Fisher's z.
		sh := shapeOf(m)
		for j := range r.Outcomes {
			parent, child := sh.outNode[j], sh.outTarget[j]
			ec := corr(r.Series.EntriesColumn(parent, make([]float64, h)), r.Series.EntriesColumn(child, buf))
			oc := corr(o.Entries[parent], o.Entries[child])
			if math.Abs(ec) >= 1 || math.Abs(oc) >= 1 {
				continue
			}
			if z := (math.Atanh(ec) - math.Atanh(oc)) / math.Sqrt(2.0/(h-3)); math.Abs(z) > 4.5 {
				t.Errorf("%s correlation %s→%s: engine %.3f oracle %.3f z=%.2f", name, r.Nodes[parent], r.Nodes[child], ec, oc, z)
			}
		}
	}
}
