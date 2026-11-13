package dist

import (
	"math"
	"testing"
)

// TestExcessMean checks E[(X − c)⁺] against closed forms: light and
// heavy tails, bounded supports, an atom, and c below the support.
func TestExcessMean(t *testing.T) {
	exp, _ := NewExponential(4)
	ll2, _ := NewLogLogistic(2, 5)
	ll12, _ := NewLogLogistic(1.2, 5)
	par, _ := NewPareto(2, 1.5)
	con, _ := NewConstant(3)
	uni, _ := NewUniform(0, 10)
	// The log-logistic's excess is a·z^(1−b)/(b−1)·₂F₁(1, (b−1)/b; (2b−1)/b; −z^−b)
	// with z = c/a; two terms of the series suffice where z^−b is tiny.
	llTail := func(b, a, c float64) float64 {
		z, x := c/a, math.Pow(c/a, -b)
		return a * math.Pow(z, 1-b) / (b - 1) * (1 - (b-1)/(2*b-1)*x)
	}
	cases := []struct {
		name    string
		d       Distribution
		c, want float64
		tol     float64
	}{
		{"exponential 1", exp, 1, 4 * math.Exp(-0.25), 1e-10},
		{"exponential 200", exp, 200, 4 * math.Exp(-50), 1e-10},
		{"loglogistic(2,5) 7", ll2, 7, 5 * math.Atan(5.0/7), 1e-9},
		{"loglogistic(2,5) 3e8", ll2, 3e8, 5 * math.Atan(5/3e8), 1e-9},
		{"loglogistic(1.2,5) 1e6", ll12, 1e6, llTail(1.2, 5, 1e6), 1e-6},
		{"pareto below", par, 1, par.Mean() - 1, 1e-15},
		{"pareto 50", par, 50, math.Pow(2, 1.5) * math.Pow(50, -0.5) / 0.5, 1e-6},
		{"constant below", con, 1, 2, 0},
		{"constant above", con, 4, 0, 0},
		{"uniform 4", uni, 4, 36.0 / 20, 1e-12},
	}
	for _, c := range cases {
		got := ExcessMean(c.d, c.c)
		if err := math.Abs(got-c.want) / math.Max(c.want, 1e-300); err > c.tol {
			t.Errorf("%s: %.15g want %.15g (rel %.2g)", c.name, got, c.want, err)
		}
	}
}
