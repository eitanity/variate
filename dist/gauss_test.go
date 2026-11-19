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

// TestHermiteRule: the n-point rule integrates the standard normal's
// moments to degree 2n-1: E[Z^2k] = (2k-1)!!, odd moments 0.
func TestHermiteRule(t *testing.T) {
	for _, n := range []int{1, 4, 16, 64, 256} {
		x, w := make([]float64, n), make([]float64, n)
		if !HermiteRule(x, w) {
			t.Fatalf("n %d: no rule", n)
		}
		for k := 0; k < 2*n && k <= 12; k++ {
			got, want := 0.0, 0.0
			for i := range x {
				got += w[i] * math.Pow(x[i], float64(k))
			}
			if k%2 == 0 {
				want = 1
				for j := k - 1; j > 0; j -= 2 {
					want *= float64(j)
				}
			}
			if math.Abs(got-want) > 1e-11*math.Max(1, want) {
				t.Errorf("n %d: E[Z^%d] %v, want %v", n, k, got, want)
			}
		}
	}
}

// TestLegendreRule: the n-point rule integrates u^k on [0, 1] to
// 1/(k+1) for k below 2n.
func TestLegendreRule(t *testing.T) {
	for _, n := range []int{1, 4, 8, 32} {
		x, w := make([]float64, n), make([]float64, n)
		if !LegendreRule(x, w) {
			t.Fatalf("n %d: no rule", n)
		}
		for k := 0; k < 2*n && k <= 20; k++ {
			got := 0.0
			for i := range x {
				got += w[i] * math.Pow(x[i], float64(k))
			}
			if want := 1 / float64(k+1); math.Abs(got-want) > 1e-14 {
				t.Errorf("n %d: ∫u^%d %v, want %v", n, k, got, want)
			}
		}
	}
}
