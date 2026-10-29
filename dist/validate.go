package dist

import "math"

// checkPERTParams validates the parameters shared by PERT and LogPERT.
func checkPERTParams(family string, low, peak, high, t float64) error {
	if math.IsNaN(low) || math.IsInf(low, 0) {
		return paramError(family, paramLow, low)
	}
	if math.IsNaN(peak) || math.IsInf(peak, 0) {
		return paramError(family, paramPeak, peak)
	}
	if math.IsNaN(high) || math.IsInf(high, 0) {
		return paramError(family, paramHigh, high)
	}
	if !(t >= 0) || math.IsInf(t, 0) {
		return paramError(family, paramT, t)
	}
	if low > peak || peak > high {
		return relationError(family, "need low <= peak <= high")
	}
	span := high - low
	if span == 0 {
		return relationError(family, "low == high; use a constant")
	}
	if math.IsInf(span, 0) {
		return relationError(family, "high - low overflows")
	}
	return nil
}

// NewUniform validates a < b, both finite, with a finite span.
func NewUniform(a, b float64) (Uniform, error) {
	if !finite(a) {
		return Uniform{}, paramError(familyUniform, paramA, a)
	}
	if !finite(b) {
		return Uniform{}, paramError(familyUniform, paramB, b)
	}
	if !(a < b) || math.IsInf(b-a, 0) {
		return Uniform{}, relationError(familyUniform, "need a < b with a finite span")
	}
	return Uniform{a: a, b: b, span: b - a}, nil
}

// NewLogUniform validates 0 < a < b.
func NewLogUniform(a, b float64) (LogUniform, error) {
	if !(a > 0) || !finite(a) {
		return LogUniform{}, paramError(familyLogUniform, paramA, a)
	}
	if !finite(b) {
		return LogUniform{}, paramError(familyLogUniform, paramB, b)
	}
	if !(a < b) {
		return LogUniform{}, relationError(familyLogUniform, "need a < b")
	}
	return LogUniform{a: a, b: b, logA: math.Log(a), logRatio: math.Log(b / a)}, nil
}

// NewTriangular validates low ≤ mode ≤ high with low < high, all finite.
func NewTriangular(low, mode, high float64) (Triangular, error) {
	if !finite(low) {
		return Triangular{}, paramError(familyTriangular, paramLow, low)
	}
	if !finite(mode) {
		return Triangular{}, paramError(familyTriangular, paramMode, mode)
	}
	if !finite(high) {
		return Triangular{}, paramError(familyTriangular, paramHigh, high)
	}
	if low > mode || mode > high || !(low < high) || math.IsInf(high-low, 0) {
		return Triangular{}, relationError(familyTriangular, "need low <= mode <= high, low < high, finite span")
	}
	span := high - low
	return Triangular{
		low: low, mode: mode, high: high, span: span,
		fc: (mode - low) / span, cL: span * (mode - low), cH: span * (high - mode),
	}, nil
}

// NewTruncatedNormal validates sigma > 0, lo < hi, and that the interval
// keeps a representable mass: both standardised bounds within ±37 unless
// infinite.
func NewTruncatedNormal(mu, sigma, lo, hi float64) (TruncatedNormal, error) {
	if !finite(mu) {
		return TruncatedNormal{}, paramError(familyTruncNormal, paramMu, mu)
	}
	if !(sigma > 0) || !finite(sigma) {
		return TruncatedNormal{}, paramError(familyTruncNormal, paramSigma, sigma)
	}
	if math.IsNaN(lo) || math.IsNaN(hi) || !(lo < hi) {
		return TruncatedNormal{}, relationError(familyTruncNormal, "need lo < hi")
	}
	a, b := (lo-mu)/sigma, (hi-mu)/sigma
	if a > normalTailLimit || b < -normalTailLimit {
		return TruncatedNormal{}, relationError(familyTruncNormal, "interval too far in a tail: its mass underflows")
	}
	d := TruncatedNormal{mu: mu, sigma: sigma, lo: lo, hi: hi, a: a, b: b}
	d.setup()
	return d, nil
}
