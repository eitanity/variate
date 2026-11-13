package dist

import (
	"math"

	"gonum.org/v1/gonum/integrate/quad"
)

const (
	// excessNodes is the Gauss–Legendre rule on each octave.
	excessNodes = 32
	// excessOctaves bounds the octaves [c·2^i, c·2^(i+1)) summed, beyond
	// which the tail is extrapolated.
	excessOctaves = 1100
	// excessTol is the share of the sum below which an extrapolated tail
	// ends the sum.
	excessTol = 1e-10
)

// ExcessMean returns E[(X − c)⁺] = ∫_c^∞ P(X > u) du for a distribution on
// [0, ∞) with a finite mean. Above the support's lower end it sums
// Gauss–Legendre rules over the octaves [c·2^i, c·2^(i+1)) and, once
// successive octaves shrink by a steady ratio r, adds the geometric rest
// r/(1 − r) of the last one, which is the tail of a power law exactly. A
// step in the survival function within an octave costs accuracy, not
// convergence.
func ExcessMean(d Distribution, c float64) float64 {
	lo, hi := d.Support()
	switch {
	case c <= lo:
		return d.Mean() - c
	case c >= hi:
		return 0
	}
	x := make([]float64, excessNodes)
	w := make([]float64, excessNodes)
	quad.Legendre{}.FixedLocations(x, w, 0, 1)
	sum, prev, r := 0.0, 0.0, 1.0
	a := c
	// bound: excessOctaves
	for i := 0; i < excessOctaves; i++ {
		b := math.Min(2*a, hi)
		part := 0.0
		for j := range x {
			part += w[j] * d.Survival(a+(b-a)*x[j])
		}
		part *= b - a
		sum += part
		if part == 0 || b >= hi {
			return sum
		}
		if i > 0 {
			r = part / prev
		}
		if r < 1 && part*r/(1-r) <= excessTol*sum {
			return sum + part*r/(1-r)
		}
		prev, a = part, b
	}
	if r >= 1 {
		return math.Inf(1)
	}
	// The octaves ran out: extrapolate at the last ratio.
	return sum + prev*r/(1-r)
}
