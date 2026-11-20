package variate

import (
	"math"
	"testing"

	"code.eitanity.com/variate/dist"
)

// TestSubMomentsBounded: the cells' moments of a law whose support starts
// and ends inside a cell, a PERT over days in a year's period whose
// density is a power of the distance to its ends, or a uniform, sum to
// the law's mean and second moment within 1e-11 on every grid, the cells
// split at the ends; on one Gauss–Legendre rule across them they are off
// by up to 6e-9, unevenly from grid to grid.
func TestSubMomentsBounded(t *testing.T) {
	p, _ := dist.NewPERT(1.0/365, 3.0/365, 10.0/365, dist.DefaultPERTShape)
	u, _ := dist.NewUniform(0.2, 0.7)
	for name, d := range map[string]dist.Distribution{"pert": p, "uniform": u} {
		for _, n := range []int{128, 256, 1000, 4096} {
			var mu [4][]float64
			for k := range mu {
				mu[k] = make([]float64, n)
			}
			subMoments(d, n, &mu)
			var s1, s2 float64
			for j := range n {
				s1, s2 = s1+mu[1][j], s2+mu[2][j]
			}
			m1, m2 := d.Mean(), d.Var()+d.Mean()*d.Mean()
			if math.Abs(s1/m1-1) > 1e-11 || math.Abs(s2/m2-1) > 1e-11 {
				t.Errorf("%s on %d cells: mean off by %.3g, second moment by %.3g", name, n, s1/m1-1, s2/m2-1)
			}
		}
	}
}
