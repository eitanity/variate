package dist

import (
	"math"

	"gonum.org/v1/gonum/integrate/quad"
)

// Continuous is a distribution with the tail-safe inverses the wrappers
// need.
type Continuous interface {
	Distribution
	Quantiler
}

const (
	// rangeStartNodes and rangeDoublings bound the quadrature over a
	// probability range: 64 nodes doubling at most 10 times, to 65536.
	rangeStartNodes = 64
	rangeDoublings  = 10
	rangeTol        = 1e-13
)

// probRange is a slice of a distribution's probability mass, held on both
// sides: [pLo, pHi] in CDF units and [sLo, sHi] in survival units, so each
// half of the integral can be taken where it keeps its precision.
type probRange struct {
	q        Quantiler
	pLo, pHi float64
	sLo, sHi float64
}

// newProbRange describes the mass of d on (lo, hi].
func newProbRange(d Continuous, lo, hi float64) probRange {
	return probRange{q: d, pLo: d.CDF(lo), pHi: d.CDF(hi), sLo: d.Survival(hi), sHi: d.Survival(lo)}
}

// mass is the probability of the range, taken on the side that does not
// cancel.
func (pr probRange) mass() float64 {
	if pr.pLo > 0.5 {
		return pr.sHi - pr.sLo
	}
	return pr.pHi - pr.pLo
}

// integrate returns ∫ g(x) dF(x) over the range, with g(x) = x for power 1
// and (x − centre)² for power 2, by Gauss–Legendre in the probability
// domain on each half, doubling the nodes until the value settles. ok is
// false when the doubling cap was reached first.
func (pr probRange) integrate(power int, centre float64) (value float64, ok bool) {
	n := rangeStartNodes
	var prev float64
	// bound: rangeDoublings
	for i := 0; i <= rangeDoublings; i++ {
		value = pr.integrateN(power, centre, n)
		if i > 0 && math.Abs(value-prev) <= rangeTol*math.Abs(value) {
			return value, true
		}
		prev = value
		n *= 2
	}
	return value, false
}

// rangePower is the exponent of the substitution u = lo + span·t^k on each
// half: it flattens the endpoint singularity of a quantile at a tail, where
// Q′(u) → ∞, without harming a smooth interior range.
const rangePower = 8

func (pr probRange) integrateN(power int, centre float64, n int) float64 {
	x := make([]float64, n)
	w := make([]float64, n)
	quad.Legendre{}.FixedLocations(x, w, 0, 1)
	sum := 0.0
	// Lower half in CDF units: u ∈ [pLo, min(pHi, ½)], x = Q(u).
	if lo, hi := pr.pLo, math.Min(pr.pHi, 0.5); hi > lo {
		span := hi - lo
		for i := range x {
			tk, jac := substitute(x[i])
			sum += w[i] * jac * span * gMoment(pr.q.Quantile(lo+span*tk), power, centre)
		}
	}
	// Upper half in survival units: s ∈ [sLo, min(sHi, ½)], x = QU(s).
	if lo, hi := pr.sLo, math.Min(pr.sHi, 0.5); hi > lo {
		span := hi - lo
		for i := range x {
			tk, jac := substitute(x[i])
			sum += w[i] * jac * span * gMoment(pr.q.QuantileUpper(lo+span*tk), power, centre)
		}
	}
	return sum
}

// substitute returns t^k and its derivative k·t^(k−1) for k = rangePower.
func substitute(t float64) (tk, jac float64) {
	t2 := t * t
	t4 := t2 * t2
	tk = t4 * t4
	return tk, rangePower * t4 * t2 * t
}

func gMoment(x float64, power int, centre float64) float64 {
	if power == 1 {
		return x
	}
	dlt := x - centre
	return dlt * dlt
}

// rangeMoments returns the mean and variance of X conditional on the
// range, from two passes, the second centred on the mean.
func rangeMoments(pr probRange) (mean, variance float64, ok bool) {
	z := pr.mass()
	m, ok1 := pr.integrate(1, 0)
	mean = m / z
	v, ok2 := pr.integrate(2, mean)
	return mean, v / z, ok1 && ok2
}
