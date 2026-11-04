package suite

import (
	"math"
	"math/bits"
	"strconv"
	"strings"
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

// fleet is Poisson(rate) failures each costing a lognormal repair with
// p90 1.5 times its median, and with branch a 2% chance of a major loss
// on top: a steep distribution over a range set by the rare loss.
func fleet(rate float64, repair, major dist.Lognormal, branch bool) variate.Graph {
	fail := variate.Node{ID: "fail", Functions: []variate.Function{Fn("repair", "k", repair)}}
	g := variate.Graph{Sources: []variate.Source{{ID: "s", Frequency: C(rate), To: "fail"}}}
	if branch {
		fail.Outcomes = []variate.Outcome{{To: "major", P: C(0.02)}, {To: "ok", P: C(0.98)}}
		g.Nodes = append(g.Nodes, variate.Node{ID: "major", Functions: []variate.Function{Fn("loss", "k", major)}}, variate.Node{ID: "ok"})
	}
	g.Nodes = append(g.Nodes, fail)
	return g
}

// rounded is a lognormal's masses on the lattice of step h up to n,
// rounded down (the mass of [kh, (k+1)h) at k) or up (of ((k−1)h, kh] at
// k), with the mass beyond n at n: exact for P(X ≥ x) up to nh.
func rounded(d dist.Lognormal, h float64, n int, up bool) []float64 {
	f := make([]float64, n+1)
	shift := 1.0
	if up {
		shift = 0
	}
	for k := range n {
		f[k] = d.CDF((float64(k)+shift)*h) - d.CDF((float64(k)+shift-1)*h)
	}
	f[n] = d.Survival((float64(n) + shift - 1) * h)
	return f
}

// lattice quantile: the smallest k·h with P(T ≥ k·h) ≤ 1 − p.
func latticeQuantile(g []float64, h, p float64) float64 {
	cdf := 0.0
	for k := range g {
		if 1-cdf <= 1-p {
			return float64(k) * h
		}
		cdf += g[k]
	}
	return float64(len(g)) * h
}

// TestTransformQuantileTol: a steep total over a wide range, which a
// tolerance on P(T ≥ x) alone refuses at every grid, applies with every
// quantile from P50 to P99.9 within Tol, and each bracket overlaps the
// one Panjer gives with the repair and the loss rounded down and up on a
// lattice of its own.
func TestTransformQuantileTol(t *testing.T) {
	repair, _ := dist.NewLognormalFromMedianP90(2e3, 3e3)
	major, _ := dist.NewLognormalFromMedianP90(2e5, 3e5)
	ps := []float64{0.5, 0.9, 0.99, 0.999}
	for _, tc := range []struct {
		rate   float64
		branch bool
	}{{30, false}, {100, false}, {10, true}} {
		what := "rate " + strconv.FormatFloat(tc.rate, 'g', -1, 64) + " branch " + strconv.FormatBool(tc.branch)
		cfg := variate.TransformConfig{}
		_, tr := transformOf(t, fleet(tc.rate, repair, major, tc.branch), cfg)
		if !tr.Applicable {
			// With a 32-bit int the memory budget is 1.5 GB, and the case
			// that needs 2^22 points is refused by it.
			if bits.UintSize == 32 && strings.Contains(tr.Reason, "MB at") {
				continue
			}
			t.Errorf("%s: refused: %s", what, tr.Reason)
			continue
		}
		_, top := tr.Total.Quantile(0.999)
		n := 20000
		h := 1.2 * top / float64(n)
		var q [2][]float64
		for i, up := range []bool{false, true} {
			f := rounded(repair, h, n, up)
			if tc.branch {
				m := rounded(major, h, n, up)
				sum := make([]float64, n+1)
				for a := range f {
					sum[a] += 0.98 * f[a]
					for b := 0; a+b <= n; b++ {
						sum[min(a+b, n)] += 0.02 * f[a] * m[b]
					}
				}
				// The pairs beyond n land at n.
				left := 1.0
				for k := range n {
					left -= sum[k]
				}
				sum[n] = left
				f = sum
			}
			g := panjer(0, tc.rate, math.Exp(-tc.rate*(1-f[0])), f, n)
			for _, p := range ps {
				q[i] = append(q[i], latticeQuantile(g, h, p))
			}
		}
		for i, p := range ps {
			lo, hi := tr.Total.Quantile(p)
			if hi-lo > 1e-3*math.Max(hi, tr.H) {
				t.Errorf("%s: P%g bracket [%g, %g] wider than Tol", what, 100*p, lo, hi)
			}
			// The true quantile lies in [lo − H, hi] and in [down − h, up].
			if lo-tr.H > q[1][i] || hi < q[0][i]-h {
				t.Errorf("%s: P%g transform [%g, %g] misses Panjer [%g, %g]", what, 100*p, lo, hi, q[0][i], q[1][i])
			}
		}
	}
}
