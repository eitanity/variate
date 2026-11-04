package dist

import (
	"math"
)

const (
	familySplit = "threshold"
	// maxSplitTerms bounds the terms summed for an integer family's half.
	maxSplitTerms = 1 << 24
	// splitTailTol is the mass left unsummed in an integer family's upper
	// half, relative to the half's own mass.
	splitTailTol = 1e-16
	// atomStep is the share of the mass beyond the last atom that the walk
	// steps past to find the next.
	atomStep = 1e-9
)

// NewSplit returns the half of d that a threshold at t keeps: X given
// X > t when above is set, else X given X ≤ t, so an atom at t goes
// below. mass is the half's probability, P(X > t) or P(X ≤ t); when it
// is 0 the half is empty and the Truncated is the zero value. The
// moments are summed for a finite or an integer family and taken by
// quadrature in probability units otherwise, which an atom does not
// disturb: the probability range holds it on the side it belongs to.
func NewSplit(d Distribution, t float64, above bool) (half Truncated, mass float64, err error) {
	if math.IsNaN(t) || math.IsInf(t, 0) {
		return Truncated{}, 0, paramError(familySplit, "over", t)
	}
	slo, shi := d.Support()
	var pr probRange
	// The open ends are set exactly, not evaluated at an infinity.
	if above {
		pr = probRange{q: d, pLo: d.CDF(t), pHi: 1, sLo: 0, sHi: d.Survival(t)}
		half = Truncated{d: d, lo: math.Max(t, slo), hi: shi, closed: t < slo}
	} else {
		// The lower half starts at the support's start, and holds an
		// atom there.
		pr = probRange{q: d, pLo: 0, pHi: d.CDF(t), sLo: d.Survival(t), sHi: 1}
		half = Truncated{d: d, lo: slo, hi: math.Min(t, shi), closed: true}
	}
	half.pr = pr
	half.mass = pr.mass()
	half.upperForm = pr.pLo > 0.5
	if !(half.mass > 0) || half.hi < half.lo {
		return Truncated{}, 0, nil
	}
	if half.mean, half.variance, err = splitMoments(d, half); err != nil {
		return Truncated{}, 0, err
	}
	return half, half.mass, nil
}

// splitMoments is the mean and variance of a half: by quadrature in
// probability units, which a smooth quantile settles, else by walking
// the half's atoms, which a step quantile has. An atom at the top of the
// half, as a cap has, bends the quantile there: the quadrature runs up to
// just below it and adds it.
func splitMoments(d Distribution, half Truncated) (mean, variance float64, err error) {
	lo, hi := d.Support()
	if lo == hi {
		return lo, 0, nil
	}
	pr, top, atom := half.pr, half.hi, 0.0
	if !math.IsInf(top, 1) {
		below := math.Nextafter(top, math.Inf(-1))
		if a := d.Survival(below) - d.Survival(top); a > 0 && below > half.lo {
			atom = a
			pr.pHi, pr.sLo = d.CDF(below), d.Survival(below)
		}
	}
	if atom == 0 && math.IsInf(top, 1) && math.IsInf(d.Mean(), 1) {
		return math.Inf(1), math.Inf(1), nil
	}
	m1, ok := pr.integrate(1, 0)
	if !ok {
		return walkAtoms(d, half)
	}
	if atom == 0 && math.IsInf(top, 1) && math.IsInf(d.Var(), 1) {
		return m1 / half.mass, math.Inf(1), nil
	}
	m2, ok := pr.integrate(2, 0)
	if !ok {
		return walkAtoms(d, half)
	}
	if atom > 0 {
		m1 += atom * top
		m2 += atom * top * top
	}
	mean = m1 / half.mass
	return mean, math.Max(0, m2/half.mass-mean*mean), nil
}

// walkAtoms sums a discrete half atom by atom: the next atom is the
// quantile just past the mass taken so far, its mass the step to it, read
// in survival units above the median and in CDF units below. It stops
// when the mass beyond is below splitTailTol of the half's, and fails on
// a half that is not a run of atoms.
func walkAtoms(d Distribution, half Truncated) (mean, variance float64, err error) {
	var s1, s2 float64
	x := half.lo
	if !half.closed {
		x = nextAtom(d, half.lo)
	}
	// bound: maxSplitTerms
	for i := 0; x <= half.hi; i++ {
		if i == maxSplitTerms || math.IsInf(x, 0) || math.IsNaN(x) {
			return 0, 0, relationError(familySplit, "moments did not converge")
		}
		p := atMass(d, x)
		if !(p > 0) {
			return 0, 0, relationError(familySplit, "moments did not converge")
		}
		s1 += p * x
		s2 += p * x * x
		if d.Survival(x) <= splitTailTol*half.mass {
			break
		}
		next := nextAtom(d, x)
		if !(next > x) {
			return 0, 0, relationError(familySplit, "moments did not converge")
		}
		x = next
	}
	mean = s1 / half.mass
	return mean, math.Max(0, s2/half.mass-mean*mean), nil
}

// nextAtom is the smallest value above x that holds more than
// atomStep of the mass beyond x: the quantile that far past the mass at
// or below x. An atom it steps over holds less than that, which the
// moments' tolerance absorbs.
func nextAtom(d Distribution, x float64) float64 {
	if c := d.CDF(x); c <= 0.5 {
		return d.Quantile(c + atomStep*(1-c))
	}
	s := d.Survival(x)
	return d.QuantileUpper(s * (1 - atomStep))
}

// atMass is P(X = x), taken on the side that keeps its precision.
func atMass(d Distribution, x float64) float64 {
	below := math.Nextafter(x, math.Inf(-1))
	if d.CDF(x) <= 0.5 {
		return d.CDF(x) - d.CDF(below)
	}
	return d.Survival(below) - d.Survival(x)
}
