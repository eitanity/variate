package stats

import (
	"math"
	"math/rand/v2"
	"testing"
)

// compoundPoissonCDF is P(S ≤ x) for S a Poisson(lam) count of unit
// exponentials: P(N = 0) plus each Erlang(k) CDF weighted by P(N = k).
func compoundPoissonCDF(lam, x float64) float64 {
	if x < 0 {
		return 0
	}
	pk := math.Exp(-lam)
	cdf := pk
	term, tail := 1.0, 0.0 // x^j/j! and Σ_{j<k} x^j/j!
	// bound: P(N ≥ 200) is far below double precision for lam 3
	for k := 1; k < 200; k++ {
		pk *= lam / float64(k)
		tail += term
		term *= x / float64(k)
		cdf += pk * (1 - math.Exp(-x)*tail)
	}
	return cdf
}

// bisect returns the smallest x with f(x) ≥ p, f nondecreasing on x ≥ 0.
func bisect(f func(float64) float64, p float64) float64 {
	lo, hi := 0.0, 1.0
	// bound: doubling to past the quantile
	for i := 0; i < 64 && f(hi) < p; i++ {
		hi *= 2
	}
	// bound: 200 halvings exhaust a double
	for range 200 {
		m := (lo + hi) / 2
		if f(m) < p {
			lo = m
		} else {
			hi = m
		}
	}
	return hi
}

// TestQuantileIntervalCoverage checks the 95% order-statistic interval
// against known quantiles over 1000 seeds: it must hold the true
// quantile 93–97% of the time at each reported level, on a lognormal, a
// Pareto with α 1.5 and a compound Poisson with an atom at zero. At
// 10,000 periods p999 rests on 10 values and the interval over-covers,
// 97.3% on the lognormal; 100,000 gives it 100.
func TestQuantileIntervalCoverage(t *testing.T) {
	const (
		n     = 100_000
		seeds = 1000
		z     = 1.959963984540054
		lam   = 3.0
	)
	levels := []float64{0.5, 0.9, 0.95, 0.99, 0.999}
	cases := []struct {
		name   string
		stream uint64
		mean   float64
		draw   func(r *rand.Rand) float64
		quant  func(p float64) float64
	}{
		{"lognormal", 0, math.Exp(0.5),
			func(r *rand.Rand) float64 { return math.Exp(r.NormFloat64()) },
			func(p float64) float64 { return math.Exp(math.Sqrt2 * math.Erfinv(2*p-1)) }},
		{"pareto 1.5", 1, 3,
			func(r *rand.Rand) float64 { return math.Pow(1-r.Float64(), -1/1.5) },
			func(p float64) float64 { return math.Pow(1-p, -1/1.5) }},
		{"compound poisson", 2, lam,
			func(r *rand.Rand) float64 {
				s := 0.0
				// bound: a Poisson(3) count, past 30 with probability below 1e-20
				for at := r.ExpFloat64(); at < lam; at += r.ExpFloat64() {
					s += r.ExpFloat64()
				}
				return s
			},
			func(p float64) float64 {
				if p <= math.Exp(-lam) {
					return 0
				}
				return bisect(func(x float64) float64 { return compoundPoissonCDF(lam, x) }, p)
			}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			intervalCoverage(t, n, seeds, z, c.stream, levels, c.mean, c.draw, c.quant)
		})
	}
}

// intervalCoverage counts, over seeds samples of n draws, how often the
// interval holds the true quantile at each level.
func intervalCoverage(t *testing.T, n, seeds int, z float64, stream uint64, levels []float64,
	mean float64, draw func(*rand.Rand) float64, quant func(float64) float64) {
	truth := make([]float64, len(levels))
	for i, p := range levels {
		truth[i] = quant(p)
	}
	covered := make([]int, len(levels))
	h := NewLogHist(mean, 1e-3)
	for s := range seeds {
		h.Reset()
		r := rand.New(rand.NewPCG(uint64(s), stream))
		for range n {
			h.Add(draw(r))
		}
		for i, p := range levels {
			lo, hi, clipped := h.QuantileInterval(p, z)
			if clipped {
				t.Fatalf("p%g: interval clipped", p)
			}
			if lo <= truth[i] && truth[i] <= hi {
				covered[i]++
			}
		}
	}
	for i, p := range levels {
		if got := float64(covered[i]) / float64(seeds); got < 0.93 || got > 0.97 {
			t.Errorf("p%g: coverage %.3f, want 0.93–0.97", p, got)
		} else {
			t.Logf("p%g: coverage %.3f", p, got)
		}
	}
}

func TestQuantileIntervalEdges(t *testing.T) {
	h := NewLogHist(1, 1e-3)
	if lo, hi, _ := h.QuantileInterval(0.5, 2); !math.IsNaN(lo) || !math.IsNaN(hi) {
		t.Fatalf("empty histogram: got [%g, %g], want NaN", lo, hi)
	}
	for range 10 {
		h.Add(1)
	}
	// Every value equal: the interval collapses onto it, clamped at p 1.
	lo, hi, clipped := h.QuantileInterval(0.999, 2)
	q, _ := h.Quantile(0.999)
	if lo != q || hi != q || clipped {
		t.Fatalf("constant sample: got [%g, %g] clipped %v, want [%g, %g]", lo, hi, clipped, q, q)
	}
	h.Add(1e12) // past hi
	if _, _, clipped := h.QuantileInterval(0.99, 2); !clipped {
		t.Fatal("interval reaching Over: clipped not set")
	}
}
