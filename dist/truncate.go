package dist

import (
	"math"
	"math/rand/v2"
)

// Truncated restricts a continuous distribution to [lo, hi] and
// renormalises. Draws are by inversion on the side of the range that keeps
// its precision, so it costs one quantile evaluation per draw.
type Truncated struct {
	d              Continuous
	lo, hi         float64
	pr             probRange
	mass           float64
	upperForm      bool // the range lies above the median: use survival units
	mean, variance float64
}

func (d Truncated) Fill(r *rand.Rand, dst []float64) {
	if d.upperForm {
		for i := range dst {
			dst[i] = clamp(d.d.QuantileUpper(d.pr.sLo+r.Float64()*d.mass), d.lo, d.hi)
		}
		return
	}
	for i := range dst {
		dst[i] = clamp(d.d.Quantile(d.pr.pLo+r.Float64()*d.mass), d.lo, d.hi)
	}
}

func (d Truncated) Mean() float64 { return d.mean }

func (d Truncated) Var() float64 { return d.variance }

func (d Truncated) Support() (lo, hi float64) { return d.lo, d.hi }

func (d Truncated) CDF(x float64) float64 {
	if x <= d.lo {
		return 0
	}
	if x >= d.hi {
		return 1
	}
	if d.upperForm {
		return clamp((d.pr.sHi-d.d.Survival(x))/d.mass, 0, 1)
	}
	return clamp((d.d.CDF(x)-d.pr.pLo)/d.mass, 0, 1)
}

func (d Truncated) Survival(x float64) float64 {
	if x <= d.lo {
		return 1
	}
	if x >= d.hi {
		return 0
	}
	if d.upperForm {
		return clamp((d.d.Survival(x)-d.pr.sLo)/d.mass, 0, 1)
	}
	return clamp((d.pr.pHi-d.d.CDF(x))/d.mass, 0, 1)
}

func (d Truncated) Quantile(p float64) float64 {
	if d.upperForm {
		return clamp(d.d.QuantileUpper(d.pr.sHi-p*d.mass), d.lo, d.hi)
	}
	return clamp(d.d.Quantile(d.pr.pLo+p*d.mass), d.lo, d.hi)
}

func (d Truncated) QuantileUpper(q float64) float64 {
	if d.upperForm {
		return clamp(d.d.QuantileUpper(d.pr.sLo+q*d.mass), d.lo, d.hi)
	}
	return clamp(d.d.Quantile(d.pr.pHi-q*d.mass), d.lo, d.hi)
}

// Capped is min(X, c): the distribution of X with its upper tail collapsed
// into an atom at c. Draws cost the same as X.
type Capped struct {
	d              Continuous
	c              float64
	lo             float64
	atom           float64 // P(X ≥ c)
	mean, variance float64
}

func (d Capped) Fill(r *rand.Rand, dst []float64) {
	d.d.Fill(r, dst)
	for i, x := range dst {
		if x > d.c {
			dst[i] = d.c
		}
	}
}

func (d Capped) Mean() float64 { return d.mean }

func (d Capped) Var() float64 { return d.variance }

func (d Capped) Support() (lo, hi float64) { return d.lo, d.c }

func (d Capped) CDF(x float64) float64 {
	if x >= d.c {
		return 1
	}
	return d.d.CDF(x)
}

func (d Capped) Survival(x float64) float64 {
	if x >= d.c {
		return 0
	}
	return d.d.Survival(x)
}

func (d Capped) Quantile(p float64) float64 { return math.Min(d.d.Quantile(p), d.c) }

func (d Capped) QuantileUpper(q float64) float64 { return math.Min(d.d.QuantileUpper(q), d.c) }

// Atom is the probability mass at the cap.
func (d Capped) Atom() float64 { return d.atom }
