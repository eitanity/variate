package dist

import (
	"math"
	"testing"
)

// quantileFractionTest checks, for distributions without a finite variance,
// that the fraction of draws at or below Quantile(p) is p within binomial
// error.
func quantileFractionTest(t *testing.T, name string, xs []float64, q Quantiler) {
	t.Helper()
	n := float64(len(xs))
	for _, p := range []float64{0.5, 0.9, 0.99, 0.999} {
		x := q.Quantile(p)
		c := 0
		for _, v := range xs {
			if v <= x {
				c++
			}
		}
		if z := (float64(c) - n*p) / math.Sqrt(n*p*(1-p)); math.Abs(z) > zLimit {
			t.Errorf("%s: fraction below Quantile(%g) = %.5f, z=%.2f", name, p, float64(c)/n, z)
		}
	}
}

// meanTest is the mean half of momentsTest, for a finite variance whose
// fourth moment is infinite: there the sample variance has no usable
// standard error.
func meanTest(t *testing.T, name string, xs []float64, mean, variance float64) {
	t.Helper()
	n := float64(len(xs))
	sum := 0.0
	for _, x := range xs {
		sum += x
	}
	if z := (sum/n - mean) / math.Sqrt(variance/n); math.Abs(z) > zLimit {
		t.Errorf("%s: mean %.6g vs %.6g, z=%.2f", name, sum/n, mean, z)
	}
}

// runSuite is the table-driven check every continuous family goes through.
// heavy marks a finite variance with an infinite fourth moment.
func runSuite(t *testing.T, name string, d Distribution, q Quantiler, seed uint64, heavy bool) {
	t.Helper()
	runSuiteAbs(t, name, d, q, seed, heavy, 0)
}

// runSuiteAbs is runSuite with an absolute probability allowance for the
// inverse test (see inverseTest).
func runSuiteAbs(t *testing.T, name string, d Distribution, q Quantiler, seed uint64, heavy bool, absTol float64) {
	t.Helper()
	xs := draws(d, seed, nDraws)
	lo, hi := d.Support()
	boundsTest(t, name, xs, lo, hi)
	ksTest(t, name, xs, d.CDF)
	switch {
	case math.IsInf(d.Var(), 0):
		quantileFractionTest(t, name, xs, q)
	case heavy:
		meanTest(t, name, xs, d.Mean(), d.Var())
		quantileFractionTest(t, name, xs, q)
	case d.Var() > 0:
		momentsTest(t, name, xs, d.Mean(), d.Var())
	}
	inverseTest(t, name, d, q, 1e-12, absTol)
}

// zeroSource makes r.Float64() return exactly 0, the value that would give
// an infinite draw to any sampler taking log U or a negative power of U.
type zeroSource struct{}

func (zeroSource) Uint64() uint64 { return 0 }
