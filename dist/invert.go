package dist

import "math"

// cdfPair is a continuous distribution seen by the inverter: its CDF,
// survival function and density.
type cdfPair interface {
	cdf(x float64) float64
	sf(x float64) float64
	pdf(x float64) float64
}

const (
	// invertSteps bounds the Newton iteration of the inverters.
	invertSteps = 100
	// invertTol is the relative agreement at which the inverter stops.
	invertTol = 1e-15
)

// invert solves cdf(x) = p (upper false) or sf(x) = q (upper true) on the
// bracket [lo, hi] by Newton steps safeguarded by bisection, starting from
// x0. The bracket must contain the root; the target function is monotone on
// it. It returns the best point after invertSteps iterations if the
// tolerance is never met.
func invert[F cdfPair](f F, target float64, upper bool, lo, hi, x0 float64) float64 {
	x := clamp(x0, lo, hi)
	if x == lo || x == hi || math.IsNaN(x) {
		x = lo + (hi-lo)/2
	}
	// bound: invertSteps
	for i := 0; i < invertSteps; i++ {
		var g, dg float64
		if upper {
			g = f.sf(x) - target
			dg = -f.pdf(x)
		} else {
			g = f.cdf(x) - target
			dg = f.pdf(x)
		}
		if math.Abs(g) <= invertTol*target {
			return x
		}
		// Shrink the bracket with the sign of g. The CDF rises and the
		// survival function falls, so the two cases flip.
		if (g > 0) != upper {
			hi = x
		} else {
			lo = x
		}
		next := x - g/dg
		if !(next > lo && next < hi) || dg == 0 || math.IsNaN(next) {
			next = lo + (hi-lo)/2
		}
		if next == x || hi-lo <= invertTol*math.Abs(x) {
			return next
		}
		x = next
	}
	return x
}

// bracketUpper doubles hi until cdf(hi) ≥ p (upper false) or sf(hi) ≤ q
// (upper true), for distributions unbounded above.
func bracketUpper[F cdfPair](f F, target float64, upper bool, hi float64) float64 {
	// bound: quantileSteps doublings
	for i := 0; i < quantileSteps; i++ {
		if upper && f.sf(hi) <= target || !upper && f.cdf(hi) >= target {
			return hi
		}
		hi *= 2
	}
	return hi
}
