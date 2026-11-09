package dist

import "math"

const (
	familyTruncated = "truncate"
	familyCapped    = "cap"
	// minRetainedMass is the smallest probability a truncation may keep.
	minRetainedMass = 1e-9
)

// NewTruncated validates lo < hi, that the range keeps at least 1e-9 of
// the mass, and computes the moments by quadrature.
func NewTruncated(d Continuous, lo, hi float64) (Truncated, error) {
	t, err := NewTruncatedDensity(d, lo, hi)
	if err != nil {
		return Truncated{}, err
	}
	pr, hi := t.pr, t.hi
	// A tail kept whole keeps its infinite moments.
	if math.IsInf(hi, 1) && math.IsInf(d.Mean(), 1) {
		t.mean, t.variance = math.Inf(1), math.Inf(1)
		return t, nil
	}
	if math.IsInf(hi, 1) && math.IsInf(d.Var(), 1) {
		m, ok := pr.integrate(1, 0)
		if !ok {
			return Truncated{}, relationError(familyTruncated, "moments did not converge")
		}
		t.mean, t.variance = m/t.mass, math.Inf(1)
		return t, nil
	}
	var ok bool
	t.mean, t.variance, ok = rangeMoments(pr)
	if !ok {
		return Truncated{}, relationError(familyTruncated, "moments did not converge")
	}
	return t, nil
}

// NewTruncatedDensity validates as NewTruncated and leaves Mean and Var
// NaN, for evaluating densities at many parameter values: it skips the
// quadrature, which is nearly all of the cost.
func NewTruncatedDensity(d Continuous, lo, hi float64) (Truncated, error) {
	if math.IsNaN(lo) || math.IsNaN(hi) || !(lo < hi) {
		return Truncated{}, relationError(familyTruncated, "need lo < hi")
	}
	slo, shi := d.Support()
	lo, hi = math.Max(lo, slo), math.Min(hi, shi)
	if !(lo < hi) {
		return Truncated{}, relationError(familyTruncated, "range lies outside the support")
	}
	pr := newProbRange(d, lo, hi)
	t := Truncated{d: d, lo: lo, hi: hi, pr: pr, mass: pr.mass(), upperForm: pr.pLo > 0.5, mean: math.NaN(), variance: math.NaN()}
	if !(t.mass >= minRetainedMass) {
		return Truncated{}, relationError(familyTruncated, "retained mass below 1e-9")
	}
	return t, nil
}

// NewCapped validates that c lies inside the support and computes the
// moments from the partial expectations below c and the atom at c.
func NewCapped(d Continuous, c float64) (Capped, error) {
	k, err := NewCappedDensity(d, c)
	if err != nil {
		return Capped{}, err
	}
	if _, shi := d.Support(); c >= shi {
		k.mean, k.variance = d.Mean(), d.Var()
		return k, nil
	}
	pr := newProbRange(d, k.lo, c)
	m, ok1 := pr.integrate(1, 0)
	k.mean = m + c*k.atom
	v, ok2 := pr.integrate(2, k.mean)
	k.variance = v + k.atom*(c-k.mean)*(c-k.mean)
	if !ok1 || !ok2 {
		return Capped{}, relationError(familyCapped, "moments did not converge")
	}
	return k, nil
}

// NewCappedDensity validates as NewCapped and leaves Mean and Var NaN,
// for evaluating densities at many parameter values.
func NewCappedDensity(d Continuous, c float64) (Capped, error) {
	if !finite(c) {
		return Capped{}, paramError(familyCapped, paramValue, c)
	}
	slo, _ := d.Support()
	if !(c > slo) {
		return Capped{}, relationError(familyCapped, "cap at or below the support")
	}
	return Capped{d: d, c: c, lo: slo, atom: d.Survival(c), mean: math.NaN(), variance: math.NaN()}, nil
}
