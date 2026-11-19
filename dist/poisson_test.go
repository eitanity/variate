package dist

import (
	"fmt"
	"math"
	"testing"
)

var poissonRates = []float64{1e-6, 1e-2, 0.5, 9.999, 21.999, 22, 37.5, 1e4, 1e7}

func poissonRange(lambda float64) (kmin, kmax int) {
	sd := math.Sqrt(lambda)
	kmin = int(math.Max(0, math.Floor(lambda-8*sd-3)))
	kmax = int(math.Ceil(lambda + 8*sd + 3))
	return kmin, kmax
}

func TestPoissonSuite(t *testing.T) {
	seed := uint64(3000)
	for _, lambda := range poissonRates {
		name := fmt.Sprintf("poisson(%g)", lambda)
		d, err := NewPoisson(lambda)
		if err != nil {
			t.Fatal(err)
		}
		xs := draws(d, seed, nDraws)
		seed++
		boundsTest(t, name, xs, 0, math.Inf(1))
		for i, x := range xs {
			if x != math.Floor(x) {
				t.Fatalf("%s: draw %d = %g not an integer", name, i, x)
			}
		}
		momentsTest(t, name, xs, d.Mean(), d.Var())
		kmin, kmax := poissonRange(lambda)
		chiSquareTest(t, name, xs, d, kmin, kmax)
		poissonInverseTest(t, name, d)
	}
}

// poissonInverseTest checks the discrete quantile semantics: Quantile(p) is
// the smallest k with CDF(k) ≥ p, and QuantileUpper(q) the smallest k with
// Survival(k) ≤ q.
func poissonInverseTest(t *testing.T, name string, d Poisson) {
	t.Helper()
	for _, p := range lowerPs {
		k := d.Quantile(p)
		if d.CDF(k) < p || (k > 0 && d.CDF(k-1) >= p) {
			t.Errorf("%s: Quantile(%g) = %g, CDF %g / %g", name, p, k, d.CDF(k-1), d.CDF(k))
		}
	}
	for _, q := range upperQs {
		k := d.QuantileUpper(q)
		if d.Survival(k) > q || (k > 0 && d.Survival(k-1) <= q) {
			t.Errorf("%s: QuantileUpper(%g) = %g", name, q, k)
		}
	}
}

func TestPoissonZero(t *testing.T) {
	d, err := NewPoisson(0)
	if err != nil {
		t.Fatal(err)
	}
	xs := draws(d, 1, 1000)
	for _, x := range xs {
		if x != 0 {
			t.Fatalf("draw %g", x)
		}
	}
	if d.PMF(0) != 1 || d.PMF(1) != 0 || d.CDF(0) != 1 || d.Quantile(0.5) != 0 {
		t.Error("degenerate Poisson")
	}
}

func TestPoissonRejectsParams(t *testing.T) {
	for _, l := range []float64{-1, math.NaN(), math.Inf(1)} {
		if _, err := NewPoisson(l); err == nil {
			t.Errorf("NewPoisson(%g) accepted", l)
		}
	}
}

func TestPoissonCount(t *testing.T) {
	seed := uint64(4000)
	for _, lambda := range []float64{0.5, 9.999, 37.5, 1e4} {
		name := fmt.Sprintf("count(%g)", lambda)
		d, err := NewPoisson(lambda)
		if err != nil {
			t.Fatal(err)
		}
		r := newRand(seed)
		seed++
		xs := make([]float64, nDraws)
		for i := range xs {
			xs[i] = float64(PoissonCount(r, lambda))
		}
		momentsTest(t, name, xs, lambda, lambda)
		kmin, kmax := poissonRange(lambda)
		chiSquareTest(t, name, xs, d, kmin, kmax)
	}
	if PoissonCount(newRand(1), 0) != 0 || PoissonCount(newRand(1), -1) != 0 {
		t.Error("non-positive rate must give 0")
	}
}

// TestPoissonCountAt: the count at u is the quantile, for rates either
// side of the threshold, and rises with the rate at one u.
func TestPoissonCountAt(t *testing.T) {
	for _, lam := range []float64{0.3, 5, 21.9, 22, 40, 1000, 1e6} {
		for _, u := range []float64{1e-12, 0.001, 0.2, 0.5, 0.77, 0.999, 1 - 1e-12} {
			if got, want := float64(PoissonCountAt(lam, u)), poissonQuantile(lam, u); got != want {
				t.Errorf("λ %v, u %v: count %v, quantile %v", lam, u, got, want)
			}
			if PoissonCountAt(lam*1.1, u) < PoissonCountAt(lam, u) {
				t.Errorf("λ %v, u %v: falls with the rate", lam, u)
			}
		}
	}
}
