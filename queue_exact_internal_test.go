package variate

import (
	"testing"

	"code.eitanity.com/variate/dist"
)

// TestPollaczekKhinchineBoundsErlang: for exponential work at one server
// the Pollaczek–Khinchine quantile intervals hold Erlang C's exact
// quantiles, at loads from light to heavy; they widen with the residuals
// a wait sums, ρ/(1−ρ) on average, to 8% at load 0.97, plus a fiftieth
// of a mean service.
func TestPollaczekKhinchineBoundsErlang(t *testing.T) {
	for _, c := range []struct{ rho, width float64 }{{0.2, 0.01}, {0.6, 0.01}, {0.9, 0.02}, {0.97, 0.08}} {
		rho := c.rho
		work, _ := dist.NewExponential(1.233)
		lam := rho / 1.233
		pk := pollaczekKhinchine(lam, rho, 1, work)
		ec := erlangC(lam, 1/1.233, 1)
		if d := pk.Wait - ec.Wait; d > 1e-12*ec.Wait || d < -1e-12*ec.Wait {
			t.Errorf("load %g: mean wait %g, Erlang C %g", rho, pk.Wait, ec.Wait)
		}
		for i := range ec.WaitLo {
			x := ec.WaitLo[i]
			if !(pk.WaitLo[i] <= x && x <= pk.WaitHi[i] && pk.WaitHi[i]-pk.WaitLo[i] <= c.width*x+0.02*1.233) {
				t.Errorf("load %g level %d: Erlang C %g, Pollaczek–Khinchine [%g, %g]", rho, i, x, pk.WaitLo[i], pk.WaitHi[i])
			}
		}
	}
}

// TestIsExponential: a law is taken as exponential by its survival, so a
// gamma or Weibull of shape 1 is one, and a gamma of shape 2, a
// lognormal and a constant are not.
func TestIsExponential(t *testing.T) {
	ex, _ := dist.NewExponential(2)
	g1, _ := dist.NewGamma(1, 2)
	w1, _ := dist.NewWeibull(1, 2)
	g2, _ := dist.NewGamma(2, 1)
	ln, _ := dist.NewLognormalFromMeanSD(2, 2)
	c, _ := dist.NewConstant(2)
	for _, x := range []struct {
		name string
		d    dist.Distribution
		want bool
	}{{"exponential", ex, true}, {"gamma 1", g1, true}, {"weibull 1", w1, true}, {"gamma 2", g2, false}, {"lognormal", ln, false}, {"constant", c, false}} {
		if got := isExponential(x.d); got != x.want {
			t.Errorf("%s: exponential %v, want %v", x.name, got, x.want)
		}
	}
}
