package dist

import (
	"math"
	"math/rand/v2"
	"sort"
	"testing"

	"gonum.org/v1/gonum/mathext"
)

// Test policy: fixed seeds; means by |z| < 4.5; distributions by
// Kolmogorov–Smirnov (continuous) or χ² (discrete) with p > 1e-4.
const (
	zLimit = 4.5
	pLimit = 1e-4
)

func newRand(seed uint64) *rand.Rand {
	return rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
}

func draws(d Distribution, seed uint64, n int) []float64 {
	buf := make([]float64, n)
	d.Fill(newRand(seed), buf)
	return buf
}

// ksPValue is the asymptotic Kolmogorov p-value for statistic d at sample
// size n, with Stephens' small-sample correction.
func ksPValue(d float64, n int) float64 {
	sn := math.Sqrt(float64(n))
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

// ksTest checks sorted draws against an exact CDF.
func ksTest(t *testing.T, name string, xs []float64, cdf func(float64) float64) {
	t.Helper()
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	n := float64(len(s))
	dmax := 0.0
	for i, x := range s {
		f := cdf(x)
		d1 := f - float64(i)/n
		d2 := float64(i+1)/n - f
		dmax = math.Max(dmax, math.Max(d1, d2))
	}
	if p := ksPValue(dmax, len(s)); p < pLimit {
		t.Errorf("%s: KS D=%.3g p=%.3g", name, dmax, p)
	}
}

// momentsTest compares the sample mean and variance with the exact ones.
// The variance comparison needs a sample that reaches the mass the
// variance lives in: when the top 0.01% of squared deviations carries more
// than a fifth of their sum, the variance is tail-dominated and no
// feasible sample settles it (a Lognormal with σ = 2.5 keeps 42% of its
// second moment beyond the 1 − 1e-7 quantile), so that test is skipped.
func momentsTest(t *testing.T, name string, xs []float64, mean, variance float64) {
	t.Helper()
	n := float64(len(xs))
	sum := 0.0
	for _, x := range xs {
		sum += x
	}
	m := sum / n
	d2 := make([]float64, len(xs))
	var m2, m4 float64
	for i, x := range xs {
		dlt := x - m
		d2[i] = dlt * dlt
		m2 += d2[i]
		m4 += d2[i] * d2[i]
	}
	if z := (m - mean) / math.Sqrt(variance/n); math.Abs(z) > zLimit {
		t.Errorf("%s: mean %.6g vs %.6g, z=%.2f", name, m, mean, z)
	}
	if share := topShare(d2, max(1, len(xs)/10000)); share > 0.2 {
		t.Logf("%s: top 0.01%% of draws carries %.0f%% of the variance; variance not tested", name, 100*share)
		return
	}
	m2 /= n
	m4 /= n
	se := math.Sqrt((m4 - m2*m2) / n)
	if se == 0 {
		return
	}
	if z := (m2 - variance) / se; math.Abs(z) > zLimit {
		t.Errorf("%s: var %.6g vs %.6g, z=%.2f", name, m2, variance, z)
	}
}

// topShare returns the share of the sum of vs held by its k largest
// entries, found with a min-heap of size k.
func topShare(vs []float64, k int) float64 {
	h := make([]float64, 0, k)
	total := 0.0
	for _, v := range vs {
		total += v
		if len(h) < k {
			h = append(h, v)
			siftUp(h, len(h)-1)
		} else if v > h[0] {
			h[0] = v
			siftDown(h, 0)
		}
	}
	top := 0.0
	for _, v := range h {
		top += v
	}
	if total == 0 {
		return 0
	}
	return top / total
}

func siftUp(h []float64, i int) {
	for i > 0 {
		p := (i - 1) / 2
		if h[p] <= h[i] {
			return
		}
		h[p], h[i] = h[i], h[p]
		i = p
	}
}

func siftDown(h []float64, i int) {
	for {
		l, r := 2*i+1, 2*i+2
		s := i
		if l < len(h) && h[l] < h[s] {
			s = l
		}
		if r < len(h) && h[r] < h[s] {
			s = r
		}
		if s == i {
			return
		}
		h[s], h[i] = h[i], h[s]
		i = s
	}
}

// boundsTest checks every draw lies in [lo, hi] and is finite.
func boundsTest(t *testing.T, name string, xs []float64, lo, hi float64) {
	t.Helper()
	for i, x := range xs {
		if math.IsNaN(x) || math.IsInf(x, 0) || x < lo || x > hi {
			t.Fatalf("%s: draw %d = %g outside [%g, %g]", name, i, x, lo, hi)
		}
	}
}

// ulps is the relative width the inverse test allows around a quantile.
const ulps = 32 * 0x1p-52

var (
	lowerPs = []float64{1e-12, 1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99}
	upperQs = []float64{1e-12, 1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.3, 0.5}
)

// inverseTest checks CDF∘Quantile and Survival∘QuantileUpper across both
// tails. The agreement must be within tol relative, or within the spread of
// the function over a few ulps of the quantile, whichever is larger: near a
// bound of the support float64 cannot represent the quantile finely enough
// for the probability to round-trip, and that limit is the representation's,
// not the inverter's. A few ulps rather than one, because a composed map
// such as log10 can be flat over a single ulp of its argument, and a chain
// of three maps costs a few dozen. absTol is an absolute allowance in
// probability for wrappers that form their argument on the probability
// line, where F(lo) + p·mass cannot represent p finer than an ulp of F(lo).
func inverseTest(t *testing.T, name string, d Distribution, q Quantiler, tol, absTol float64) {
	t.Helper()
	for _, p := range lowerPs {
		x := q.Quantile(p)
		got := d.CDF(x)
		spread := math.Abs(d.CDF(x*(1+ulps)) - d.CDF(x*(1-ulps)))
		if math.Abs(got-p) > tol*p+spread+absTol {
			t.Errorf("%s: CDF(Quantile(%g)) = %.17g (ulp spread %.3g)", name, p, got, spread)
		}
	}
	for _, qq := range upperQs {
		x := q.QuantileUpper(qq)
		got := q.Survival(x)
		spread := math.Abs(q.Survival(x*(1+ulps)) - q.Survival(x*(1-ulps)))
		if math.Abs(got-qq) > tol*qq+spread+absTol {
			t.Errorf("%s: Survival(QuantileUpper(%g)) = %.17g (ulp spread %.3g)", name, qq, got, spread)
		}
	}
}

// chiSquareTest bins integer draws over [kmin, kmax], pooling neighbours
// until each bin's expectation is at least 5, with the tails folded into
// the end bins. The tail masses come from the CDF.
func chiSquareTest(t *testing.T, name string, xs []float64, d Distribution, kmin, kmax int) {
	t.Helper()
	pm, ok := d.(PMFer)
	if !ok {
		t.Fatalf("%s: no PMF", name)
	}
	pmf := pm.PMF
	t.Helper()
	n := float64(len(xs))
	width := kmax - kmin + 1
	obs := make([]float64, width)
	for _, x := range xs {
		k := int(x) - kmin
		if k < 0 {
			k = 0
		}
		if k >= width {
			k = width - 1
		}
		obs[k]++
	}
	exp := make([]float64, width)
	for k := range exp {
		exp[k] = n * pmf(float64(k+kmin))
	}
	// Mass outside the range goes to the end bins.
	exp[0] += n * d.CDF(float64(kmin-1))
	exp[width-1] += n * (1 - d.CDF(float64(kmax)))
	var chi2 float64
	bins := 0
	accO, accE := 0.0, 0.0
	for k := 0; k < width; k++ {
		accO += obs[k]
		accE += exp[k]
		if accE >= 5 || k == width-1 {
			if accE > 0 {
				chi2 += (accO - accE) * (accO - accE) / accE
				bins++
			}
			accO, accE = 0, 0
		}
	}
	if bins < 2 {
		return
	}
	df := float64(bins - 1)
	if p := mathext.GammaIncRegComp(df/2, chi2/2); p < pLimit {
		t.Errorf("%s: chi2=%.3g df=%d p=%.3g", name, chi2, bins-1, p)
	}
}
