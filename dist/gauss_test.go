package dist

import (
	"math"
	"testing"
)

// TestGaussRule: the 4-point rule built from a 200-point Gauss–Legendre
// measure on [2, 5] is the 4-point Gauss–Legendre rule, and from a
// lognormal's quantile measure it integrates g^k, k < 8, to the
// lognormal's moments; a measure with too few points is refused.
func TestGaussRule(t *testing.T) {
	u, _ := NewUniform(2, 5)
	px, pw := make([]float64, 200), make([]float64, 200)
	u.Nodes(px, pw)
	x, w := make([]float64, 4), make([]float64, 4)
	if !GaussRule(px, pw, x, w) {
		t.Fatal("no rule")
	}
	wx, ww := make([]float64, 4), make([]float64, 4)
	u.Nodes(wx, ww)
	for i := range x {
		j := 3 - i // the Legendre nodes come descending
		if math.Abs(x[i]-wx[j]) > 1e-12 || math.Abs(w[i]-ww[j]) > 1e-12 {
			t.Errorf("node %d: %v %v, want %v %v", i, x[i], w[i], wx[j], ww[j])
		}
	}
	// Lognormal moments E[G^k] = exp(kμ + k²σ²/2), from its exact atoms
	// as a stand-in measure: 2000 equal-weight quantile points.
	ln, _ := NewLognormal(0, 0.4)
	n := 4000
	px, pw = make([]float64, n), make([]float64, n)
	for i := range px {
		px[i] = ln.Quantile((float64(i) + 0.5) / float64(n))
		pw[i] = 1 / float64(n)
	}
	if !GaussRule(px, pw, x, w) {
		t.Fatal("no lognormal rule")
	}
	for k := range 8 {
		got, want := 0.0, 0.0
		for i := range x {
			got += w[i] * math.Pow(x[i], float64(k))
		}
		for i := range px {
			want += pw[i] * math.Pow(px[i], float64(k))
		}
		if math.Abs(got-want) > 1e-9*want {
			t.Errorf("E[G^%d]: %v, the measure's %v", k, got, want)
		}
	}
	if GaussRule([]float64{1, 2, 3}, []float64{0.5, 0.5, 0}, x[:3], w[:3]) {
		t.Error("a rule of 3 nodes from 2 points")
	}
}
