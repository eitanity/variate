package dist

import (
	"math"
	"math/cmplx"
	"testing"
)

// TestLogLaplace: exp(LogLaplace) is Laplace, and for the gamma the
// principal (1 + θs)^(−k), to 1e-14 relative at s = f·(1 − w) over the
// unit disc, the negative axis included, for non-integer shapes; the
// families without a closed logarithm say so.
func TestLogLaplace(t *testing.T) {
	ws := make([]complex128, 0, 256)
	for j := range 256 {
		ws = append(ws, cmplx.Rect(float64(j%16)/15, 2*math.Pi*float64(j)/256))
	}
	gammaRef := func(k, theta float64) func(complex128) complex128 {
		return func(s complex128) complex128 {
			return cmplx.Exp(-complex(k, 0) * cmplx.Log(1+complex(theta, 0)*s))
		}
	}
	c, _ := NewConstant(3.7)
	type tc struct {
		d   Distribution
		ref func(complex128) complex128
	}
	cases := make([]tc, 0, 7)
	cases = append(cases, tc{c, func(s complex128) complex128 { return cmplx.Exp(-3.7 * s) }})
	for _, k := range []float64{0.3, 2.5, 7.7} {
		g, _ := NewGamma(k, 1.9)
		sg, _ := NewScaled(g, 0.4)
		cases = append(cases, tc{g, gammaRef(k, 1.9)}, tc{sg, gammaRef(k, 0.4*1.9)})
	}
	for _, c := range cases {
		for _, f := range []float64{0.2, 1, 6} {
			for _, w := range ws {
				s := complex(f, 0) * (1 - w)
				l, ok := c.d.LogLaplace(s)
				if !ok {
					t.Fatalf("%T: no closed log", c.d)
				}
				lap, _ := c.d.Laplace(s)
				want := c.ref(s)
				for what, got := range map[string]complex128{"exp(LogLaplace)": cmplx.Exp(l), "Laplace": lap} {
					if e := cmplx.Abs(got-want) / cmplx.Abs(want); e > 1e-14 {
						t.Errorf("%T at s=%v: %s %v, want %v, relative %g", c.d, s, what, got, want, e)
					}
				}
			}
		}
	}
	e, _ := NewExponential(2)
	ln, _ := NewLognormal(0, 1)
	for _, d := range []Distribution{e, ln} {
		if _, ok := d.LogLaplace(1); ok {
			t.Errorf("%T: a closed log, want none", d)
		}
	}
}
