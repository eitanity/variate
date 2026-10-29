package dist

import (
	"math"
	"math/rand/v2"
	"testing"
)

// constSource returns the same word forever, so a rejection sampler whose
// test fails on that word fails on every try and must fall through to the
// quantile.
type constSource struct{ v uint64 }

func (c constSource) Uint64() uint64 { return c.v }

// wordFor returns the source word whose Float64 is f: rand/v2 reads the
// low 53 bits.
func wordFor(f float64) uint64 { return uint64(f * (1 << 53)) }

// TestRejectionFallbackGamma: NormFloat64 reads the low 32 bits as a
// signed index, so bit 31 set with the low 7 bits clear selects the
// ziggurat's tail segment with a negative sign, and bit 52 makes Float64
// about 0.5. NormFloat64 is then about −3.6 on every call and
// Marsaglia–Tsang rejects every candidate for shape 1. The draw must be
// the quantile at unit(r).
func TestRejectionFallbackGamma(t *testing.T) {
	r := rand.New(constSource{1<<52 | 1<<31})
	if f := r.Float64(); f < 0.5 || f > 0.5001 {
		t.Fatalf("Float64 = %g", f)
	}
	if z := r.NormFloat64(); z > -3 {
		t.Fatalf("NormFloat64 = %g, not in the rejecting range", z)
	}
	g, _ := NewGamma(1, 1)
	buf := make([]float64, 3)
	g.Fill(r, buf)
	want := g.Quantile(1 - r.Float64())
	for _, x := range buf {
		if !near(x, want, 1e-12) {
			t.Errorf("fallback draw %g want quantile %g", x, want)
		}
	}
}

// TestRejectionFallbackPoisson: Float64 = 0.999 puts the PTRS proposal in
// the region us < 0.013 with v > us, which is rejected before any test.
func TestRejectionFallbackPoisson(t *testing.T) {
	r := rand.New(constSource{wordFor(0.999)})
	d, _ := NewPoisson(50)
	buf := make([]float64, 3)
	d.Fill(r, buf)
	want := poissonQuantile(50, 1-r.Float64())
	for _, x := range buf {
		if x != want {
			t.Errorf("fallback draw %g want quantile %g", x, want)
		}
	}
}

// TestRejectionFallbackRobert: a uniform small enough that every
// exponential proposal overshoots the upper bound, so the draw comes from
// the inverse of the limiting exponential.
func TestRejectionFallbackRobert(t *testing.T) {
	d, _ := NewTruncatedNormal(0, 1, 6, 6.5)
	u := math.Exp(-d.alpha*0.5) / 2
	r := rand.New(constSource{wordFor(1 - u)})
	buf := make([]float64, 3)
	d.Fill(r, buf)
	a, b := 6.0, 6.5
	span := -math.Expm1(-a * (b - a))
	want := a - math.Log1p(-(1-u)*span)/a
	for _, x := range buf {
		if !near(x, want, 1e-12) {
			t.Errorf("fallback draw %.15g want %.15g", x, want)
		}
	}
}
