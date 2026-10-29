package dist

import (
	"fmt"
	"math"
	"testing"

	"gonum.org/v1/gonum/mathext"
)

// discreteInverseTest checks the integer quantile semantics: Quantile(p) is
// the smallest k with CDF(k) ≥ p and QuantileUpper(q) the smallest k with
// Survival(k) ≤ q.
func discreteInverseTest(t *testing.T, name string, d Distribution, q Quantiler) {
	t.Helper()
	for _, p := range lowerPs {
		k := q.Quantile(p)
		if d.CDF(k) < p || (k > 0 && d.CDF(k-1) >= p) {
			t.Errorf("%s: Quantile(%g) = %g, CDF %g / %g", name, p, k, d.CDF(k-1), d.CDF(k))
		}
	}
	for _, qq := range upperQs {
		k := q.QuantileUpper(qq)
		if q.Survival(k) > qq || (k > 0 && q.Survival(k-1) <= qq) {
			t.Errorf("%s: QuantileUpper(%g) = %g", name, qq, k)
		}
	}
}

func integerDrawsTest(t *testing.T, name string, xs []float64, lo, hi float64) {
	t.Helper()
	boundsTest(t, name, xs, lo, hi)
	for i, x := range xs {
		if x != math.Floor(x) {
			t.Fatalf("%s: draw %d = %g not an integer", name, i, x)
		}
	}
}

func countRange(mean, sd, lo, hi float64) (kmin, kmax int) {
	kmin = int(math.Max(lo, math.Floor(mean-8*sd-3)))
	kmax = int(math.Min(hi, math.Ceil(mean+8*sd+3)))
	return kmin, kmax
}

func TestBinomialSuite(t *testing.T) {
	cases := [][2]float64{
		{5, 0.3}, {20, 0.3}, {40, 0.3}, {100, 0.5}, {100, 0.95}, {1e4, 1e-3}, {1e4, 0.999},
		{1e9, 0.5}, {1e9, 1e-7}, {1e9, 1e-9}, {33, 0.49}, {33, 0.51}, {10, 0}, {10, 1}, {0, 0.3},
	}
	seed := uint64(6000)
	for _, c := range cases {
		name := fmt.Sprintf("binomial(%g,%g)", c[0], c[1])
		d, err := NewBinomial(c[0], c[1])
		if err != nil {
			t.Fatal(err)
		}
		xs := draws(d, seed, nDraws)
		seed++
		integerDrawsTest(t, name, xs, 0, c[0])
		if d.Var() > 0 {
			momentsTest(t, name, xs, d.Mean(), d.Var())
			kmin, kmax := countRange(d.Mean(), math.Sqrt(d.Var()), 0, c[0])
			chiSquareTest(t, name, xs, d, kmin, kmax)
			discreteInverseTest(t, name, d, d)
		}
	}
	for _, c := range [][2]float64{{-1, 0.5}, {2.5, 0.5}, {10, -0.1}, {10, 1.5}, {math.Inf(1), 0.5}} {
		if _, err := NewBinomial(c[0], c[1]); err == nil {
			t.Errorf("NewBinomial(%g, %g) accepted", c[0], c[1])
		}
	}
}

func TestBernoulli(t *testing.T) {
	for i, p := range []float64{0, 0.3, 1} {
		d, err := NewBernoulli(p)
		if err != nil {
			t.Fatal(err)
		}
		xs := draws(d, uint64(6100+i), nDraws)
		integerDrawsTest(t, "bernoulli", xs, 0, 1)
		if p > 0 && p < 1 {
			momentsTest(t, "bernoulli", xs, p, p*(1-p))
		}
	}
	if _, err := NewBernoulli(1.1); err == nil {
		t.Error("accepted p > 1")
	}
}

func TestGeometricSuite(t *testing.T) {
	seed := uint64(6200)
	for _, p := range []float64{1e-3, 0.1, 0.5, 0.9, 1} {
		name := fmt.Sprintf("geometric(%g)", p)
		d, err := NewGeometric(p)
		if err != nil {
			t.Fatal(err)
		}
		xs := draws(d, seed, nDraws)
		seed++
		integerDrawsTest(t, name, xs, 0, math.Inf(1))
		if p < 1 {
			momentsTest(t, name, xs, d.Mean(), d.Var())
			kmin, kmax := countRange(d.Mean(), math.Sqrt(d.Var()), 0, math.Inf(1))
			chiSquareTest(t, name, xs, d, kmin, kmax)
			discreteInverseTest(t, name, d, d)
		}
	}
	for _, p := range []float64{0, -0.1, 1.5} {
		if _, err := NewGeometric(p); err == nil {
			t.Errorf("NewGeometric(%g) accepted", p)
		}
	}
}

func TestNegativeBinomialSuite(t *testing.T) {
	seed := uint64(6300)
	for _, c := range [][2]float64{{0.5, 0.3}, {2.5, 0.5}, {10, 0.9}, {1, 0.01}, {100, 0.2}, {0.3, 0.999}} {
		name := fmt.Sprintf("negbin(%g,%g)", c[0], c[1])
		d, err := NewNegativeBinomial(c[0], c[1])
		if err != nil {
			t.Fatal(err)
		}
		xs := draws(d, seed, nDraws)
		seed++
		integerDrawsTest(t, name, xs, 0, math.Inf(1))
		momentsTest(t, name, xs, d.Mean(), d.Var())
		kmin, kmax := countRange(d.Mean(), math.Sqrt(d.Var()), 0, math.Inf(1))
		chiSquareTest(t, name, xs, d, kmin, kmax)
		discreteInverseTest(t, name, d, d)
	}
	d, err := NewNegativeBinomialFromMeanSize(12, 3)
	if err != nil || !near(d.Mean(), 12, 1e-12) || !near(d.Var(), 12+144.0/3, 1e-12) {
		t.Error("negbin from mean, size", err)
	}
	for _, c := range [][2]float64{{0, 0.5}, {1, 0}, {1, 1}, {-1, 0.5}} {
		if _, err := NewNegativeBinomial(c[0], c[1]); err == nil {
			t.Errorf("NewNegativeBinomial(%g, %g) accepted", c[0], c[1])
		}
	}
}

func TestCategoricalSuite(t *testing.T) {
	values := []float64{1, 2, 5, 10, 2}
	probs := []float64{0.5, 0.2, 0.15, 0.05, 0.1}
	d, err := NewCategorical(values, probs)
	if err != nil {
		t.Fatal(err)
	}
	xs := draws(d, 6400, nDraws)
	boundsTest(t, "categorical", xs, 1, 10)
	momentsTest(t, "categorical", xs, d.Mean(), d.Var())
	// χ² on the distinct values.
	counts := map[float64]float64{}
	for _, x := range xs {
		counts[x]++
	}
	n := float64(len(xs))
	chi2 := 0.0
	for _, v := range []float64{1, 2, 5, 10} {
		e := n * d.PMF(v)
		chi2 += (counts[v] - e) * (counts[v] - e) / e
	}
	if p := mathext.GammaIncRegComp(1.5, chi2/2); p < pLimit {
		t.Errorf("categorical: chi2=%.3g p=%.3g", chi2, p)
	}
	if !near(d.PMF(2), 0.3, 1e-12) || !near(d.CDF(2), 0.8, 1e-12) || d.CDF(1.5) != 0.5 || d.CDF(0) != 0 || d.CDF(10) != 1 || d.PMF(3) != 0 {
		t.Error("categorical PMF/CDF with a tie")
	}
}

// TestAliasTableExact reconstructs every probability from the alias table,
// which checks entries far below what a sample can see.
func TestAliasTableExact(t *testing.T) {
	probs := []float64{1 - 1e-9 - 2e-7 - 0.3, 1e-9, 2e-7, 0.3}
	values := []float64{0, 1, 2, 3}
	d, err := NewCategorical(values, probs)
	if err != nil {
		t.Fatal(err)
	}
	n := float64(len(probs))
	for i, want := range probs {
		got := d.prob[i]
		for j := range d.prob {
			if d.alias[j] == int32(i) && d.prob[j] < 1 {
				got += 1 - d.prob[j]
			}
		}
		got /= n
		if math.Abs(got-want) > 1e-15 {
			t.Errorf("alias table: p[%d] = %.17g want %.17g", i, got, want)
		}
	}
	if _, err := NewCategorical([]float64{1, 2}, []float64{0.6, 0.5}); err == nil {
		t.Error("accepted p summing to 1.1")
	}
	if _, err := NewCategorical([]float64{1}, []float64{1, 0}); err == nil {
		t.Error("accepted mismatched lengths")
	}
}

func TestEmpirical(t *testing.T) {
	d, err := NewEmpirical([]float64{3, 1, 4, 1, 5})
	if err != nil {
		t.Fatal(err)
	}
	if !near(d.PMF(1), 0.4, 1e-12) || !near(d.PMF(3), 0.2, 1e-12) || !near(d.CDF(3), 0.6, 1e-12) || !near(d.Mean(), 2.8, 1e-12) {
		t.Errorf("empirical pmf %g %g cdf %g mean %g", d.PMF(1), d.PMF(3), d.CDF(3), d.Mean())
	}
	xs := draws(d, 6500, nDraws)
	momentsTest(t, "empirical", xs, d.Mean(), d.Var())
	if _, err := NewEmpirical(nil); err == nil {
		t.Error("accepted empty")
	}
}
