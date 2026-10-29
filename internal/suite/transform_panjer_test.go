package suite

import (
	"math"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// latticeSeverity is a model whose severity is 1 with probability 0.6
// and 3 otherwise, so T lives on the integers and both roundings are
// exact once the span divides 1.
func latticeSeverity(freq dist.Distribution) variate.Graph {
	return variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: freq, To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Outcomes: []variate.Outcome{{To: "b", P: C(0.6)}, {To: "c", P: C(0.4)}}},
			{ID: "b", Functions: []variate.Function{Fn("one", "k", C(1))}},
			{ID: "c", Functions: []variate.Function{Fn("three", "k", C(3))}},
		},
	}
}

// panjer runs the (a, b, 0) recursion for severity masses f on the
// integers, returning the masses of T up to n.
func panjer(a, b, g0 float64, f []float64, n int) []float64 {
	g := make([]float64, n+1)
	g[0] = g0
	for k := 1; k <= n; k++ {
		sum := 0.0
		for j := 1; j <= k && j < len(f); j++ {
			sum += (a + b*float64(j)/float64(k)) * f[j] * g[k-j]
		}
		g[k] = sum / (1 - a*f[0])
	}
	return g
}

// TestTransformPanjer: FFT and Panjer agree to 1e-10 for a Poisson
// frequency and for a Gamma-mixed (negative binomial) one.
func TestTransformPanjer(t *testing.T) {
	f := []float64{0, 0.6, 0, 0.4}
	gamma, _ := dist.NewGamma(2.5, 1.2)
	for _, tc := range []struct {
		name string
		freq dist.Distribution
		a, b float64
		g0   float64
	}{
		{"poisson", C(2.5), 0, 2.5, math.Exp(-2.5)},
		// Gamma(shape r, scale θ) mixing gives NegBin(r, q = θ/(1+θ)):
		// a = q, b = (r − 1)q, g_0 = (1 + θ)^(−r).
		{"negative binomial", gamma, 1.2 / 2.2, 1.5 * 1.2 / 2.2, math.Pow(2.2, -2.5)},
	} {
		_, tr := transformOf(t, latticeSeverity(tc.freq), variate.TransformConfig{})
		if !tr.Applicable {
			t.Fatalf("%s: %s", tc.name, tr.Reason)
		}
		step := math.Round(1 / tr.H)
		if math.Abs(step*tr.H-1) > 1e-12 || step < 1 {
			t.Fatalf("%s: span %g does not divide 1", tc.name, tr.H)
		}
		_, q999 := tr.Total.Quantile(0.999)
		_, q99 := tr.Total.Quantile(0.99)
		n := min(int(q999), int(float64(len(tr.Total.Lower))/step)-1)
		g := panjer(tc.a, tc.b, tc.g0, f, n)
		cdf := 0.0
		for k := 0; k <= n; k++ {
			x := int(float64(k) * step)
			ec := 1 - cdf
			cdf += g[k]
			lo, hi := tr.Total.Lower[x], tr.Total.Upper[x]
			if lo > ec || hi < ec {
				t.Errorf("%s: P(T ≥ %d) FFT [%.12g, %.12g] excludes Panjer %.12g", tc.name, k, lo, hi, ec)
			}
			// Both roundings are exact here, so the computed curve is the
			// lower bound plus the round-off allowance.
			if curve := lo + tr.Total.Roundoff[x]; float64(k) <= q99 && lo > 0 && math.Abs(curve-ec) > 1e-10 {
				t.Errorf("%s: P(T ≥ %d) FFT %.12g Panjer %.12g", tc.name, k, curve, ec)
			}
		}
	}
}
