package dist

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// TestScaledSuite: f·X passes the family suite for continuous and
// heavy-tailed X, at factors below and above one.
func TestScaledSuite(t *testing.T) {
	pe, _ := NewPERT(1, 3, 10, 4)
	g, _ := NewGamma(2.5, 3)
	ln, _ := NewLognormal(0, 1)
	pa, _ := NewPareto(1, 1.5)
	cases := []struct {
		name  string
		d     Distribution
		heavy bool
	}{{"pert", pe, false}, {"gamma", g, false}, {"lognormal", ln, false}, {"pareto", pa, true}}
	seed := uint64(9100)
	for _, c := range cases {
		for _, f := range []float64{0.9, 1.1, 250} {
			s, err := NewScaled(c.d, f)
			if err != nil {
				t.Fatal(err)
			}
			runSuite(t, "scaled "+c.name, s, s, seed, c.heavy)
			seed++
		}
	}
}

// TestScaledIsTheDrawTimesF: from the same stream, every draw of f·X is
// f times the draw of X, and the moments, support, quantiles, transform
// and nodes are X's carried through x ↦ f·x.
func TestScaledIsTheDrawTimesF(t *testing.T) {
	be, _ := NewBetaFromMeanSD(0.3, 0.1)
	f := 1.1
	s, _ := NewScaled(be, f)
	a, b := make([]float64, 64), make([]float64, 64)
	be.Fill(rand.New(rand.NewPCG(1, 2)), a)
	s.Fill(rand.New(rand.NewPCG(1, 2)), b)
	for i := range a {
		if b[i] != f*a[i] {
			t.Fatalf("draw %d: %g, want %g", i, b[i], f*a[i])
		}
	}
	lo, hi := be.Support()
	slo, shi := s.Support()
	if s.Mean() != f*be.Mean() || s.Var() != f*f*be.Var() || slo != f*lo || shi != f*hi {
		t.Errorf("moments %g %g support [%g, %g]", s.Mean(), s.Var(), slo, shi)
	}
	for _, p := range []float64{1e-6, 0.3, 0.999} {
		if x := s.Quantile(p); math.Abs(s.CDF(x)-p) > 1e-12 || x != f*be.Quantile(p) {
			t.Errorf("quantile %g: %g", p, x)
		}
		if x := s.QuantileUpper(p); x != f*be.QuantileUpper(p) || math.Abs(s.Survival(x)-p) > 1e-12 {
			t.Errorf("upper quantile %g: %g", p, x)
		}
	}
	g, _ := NewGamma(2.5, 3)
	sg, _ := NewScaled(g, f)
	for _, z := range []complex128{0.5, complex(1, 2)} {
		got, ok := sg.Laplace(z)
		want, _ := g.Laplace(z * complex(f, 0))
		if !ok || got != want || cmplx.Abs(got) == 0 {
			t.Errorf("transform at %v: %v, want %v", z, got, want)
		}
	}
	x, w := make([]float64, 8), make([]float64, 8)
	bx, bw := make([]float64, 8), make([]float64, 8)
	if !s.Nodes(x, w) || !be.Nodes(bx, bw) {
		t.Fatal("no nodes")
	}
	for i := range x {
		if x[i] != f*bx[i] || w[i] != bw[i] {
			t.Errorf("node %d: %g %g", i, x[i], w[i])
		}
	}
}

// TestScaledValidation: the factor is finite and positive.
func TestScaledValidation(t *testing.T) {
	c, _ := NewConstant(2)
	for _, f := range []float64{0, -1, math.Inf(1), math.NaN()} {
		if _, err := NewScaled(c, f); err == nil {
			t.Errorf("factor %g accepted", f)
		}
	}
	if s, err := NewScaled(c, 3); err != nil || s.Mean() != 6 || s.Factor() != 3 || s.Base() != Distribution(c) {
		t.Errorf("constant scaled by 3: %v %v", s, err)
	}
}
