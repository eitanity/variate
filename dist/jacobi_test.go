package dist

import (
	"math"
	"testing"
)

// TestBetaNodesMoments: the Gauss–Jacobi rule reproduces the Beta moments
// E[X^k] = Π_{i<k} (α+i)/(α+β+i) exactly for k < 2n.
func TestBetaNodesMoments(t *testing.T) {
	for _, c := range [][2]float64{{1, 1}, {2, 5}, {1.5, 3.7}, {0.6, 0.8}, {5, 1}, {30, 12}} {
		for _, n := range []int{1, 2, 3, 8, 30} {
			x := make([]float64, n)
			w := make([]float64, n)
			if !betaNodes(c[0], c[1], x, w) {
				t.Fatalf("beta(%g,%g) n=%d did not converge", c[0], c[1], n)
			}
			for k := 0; k < 2*n; k++ {
				want := 1.0
				for i := 0; i < k; i++ {
					want *= (c[0] + float64(i)) / (c[0] + c[1] + float64(i))
				}
				got := 0.0
				for i := range x {
					got += w[i] * math.Pow(x[i], float64(k))
				}
				if math.Abs(got-want) > 1e-12*want+1e-15 {
					t.Errorf("beta(%g,%g) n=%d moment %d: %.15g want %.15g", c[0], c[1], n, k, got, want)
				}
			}
		}
	}
}

func TestNodesFamilies(t *testing.T) {
	p, _ := NewPERT(10, 20, 40, 4)
	x := make([]float64, 6)
	w := make([]float64, 6)
	if !p.Nodes(x, w) {
		t.Fatal("pert nodes")
	}
	mean := 0.0
	for i := range x {
		mean += w[i] * x[i]
	}
	if math.Abs(mean-p.Mean()) > 1e-12*p.Mean() {
		t.Errorf("pert mean from nodes %g want %g", mean, p.Mean())
	}
	u, _ := NewUniform(2, 5)
	if !u.Nodes(x, w) {
		t.Fatal("uniform nodes")
	}
	m2 := 0.0
	for i := range x {
		m2 += w[i] * x[i] * x[i]
	}
	if math.Abs(m2-(u.Var()+u.Mean()*u.Mean())) > 1e-12*m2 {
		t.Errorf("uniform second moment %g", m2)
	}
	c, _ := NewConstant(3)
	c.Nodes(x, w)
	if x[0] != 3 || w[0] != 1 || w[1] != 0 {
		t.Error("constant nodes")
	}
	g, _ := NewGamma(2, 1)
	if g.Nodes(x, w) {
		t.Error("gamma should have no rule")
	}
}
