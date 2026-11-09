package dist

import "math"

// LogContribution is the log-likelihood of one record known to lie in
// (a, b]: ln[F(b) − F(a)]. An exact record has a == b and contributes
// LogDensity(a). A right-censored record has b at or above the top of
// the support, +Inf included, and contributes LogSurvival(a). A
// left-censored record has a below the support, −Inf included, and
// contributes ln F(b). It is NaN when a > b or either end is NaN.
func LogContribution(d Distribution, a, b float64) float64 {
	if math.IsNaN(a) || math.IsNaN(b) || a > b {
		return math.NaN()
	}
	if a == b {
		return d.LogDensity(a)
	}
	lo, hi := d.Support()
	below, above := a < lo, b >= hi
	switch {
	case below && above:
		return 0
	case above:
		return d.LogSurvival(a)
	case below:
		return math.Log(d.CDF(b))
	}
	if d.CDF(a) > 0.5 {
		return logUpperBand(d, a, b)
	}
	return math.Log(math.Max(0, d.CDF(b)-d.CDF(a)))
}

// logUpperBand is ln[S(a) − S(b)], in log space where S(a) has
// underflowed. S(a) − S(b) does not cancel where F(b) − F(a) would.
func logUpperBand(d Distribution, a, b float64) float64 {
	if diff := d.Survival(a) - d.Survival(b); diff > 0 {
		return math.Log(diff)
	}
	la, lb := d.LogSurvival(a), d.LogSurvival(b)
	if math.IsInf(la, -1) {
		return la
	}
	return la + logOneMinusExp(lb-la)
}

// logOneMinusExp is ln(1 − eᵗ) for t ≤ 0, on the form that keeps its
// digits on each side of −ln 2.
func logOneMinusExp(t float64) float64 {
	if t > -math.Ln2 {
		return math.Log(-math.Expm1(t))
	}
	return math.Log1p(-math.Exp(t))
}
