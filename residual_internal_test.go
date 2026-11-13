package variate

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"code.eitanity.com/variate/dist"
)

// TestResidualTable: the drawn residuals follow the equilibrium law
// G(r) = 1 − E[(h − r)⁺]/E[h], by a Kolmogorov–Smirnov test at the 0.001
// level, for light, bounded and heavy repair times.
func TestResidualTable(t *testing.T) {
	exp, _ := dist.NewExponential(3)
	uni, _ := dist.NewUniform(1, 4)
	pert, _ := dist.NewPERT(1, 3, 10, 4)
	par, _ := dist.NewPareto(2, 1.5)
	ll, _ := dist.NewLogLogistic(2, 5)
	for _, c := range []struct {
		name string
		d    dist.Distribution
	}{{"exponential", exp}, {"uniform", uni}, {"pert", pert}, {"pareto 1.5", par}, {"loglogistic 2", ll}} {
		tab := newResidTable(c.d)
		r := rand.New(rand.NewPCG(7, 11))
		const n = 20000
		xs := make([]float64, n)
		for i := range xs {
			xs[i] = tab.draw(r)
		}
		slices.Sort(xs)
		mean := c.d.Mean()
		d := 0.0
		for i, x := range xs {
			g := 1 - dist.ExcessMean(c.d, x)/mean
			d = math.Max(d, math.Max(math.Abs(g-float64(i)/n), math.Abs(g-float64(i+1)/n)))
		}
		if d*math.Sqrt(n) > 1.95 {
			t.Errorf("%s: KS distance %.4g, √n·D %.3g", c.name, d, d*math.Sqrt(n))
		}
	}
}
