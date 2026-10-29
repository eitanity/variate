package dist

import "math"

// lognormalZ90 is the standard normal 90th percentile.
const lognormalZ90 = 1.2815515655446004

// NewLognormalFromMedianP90 converts a median and a 90th percentile.
func NewLognormalFromMedianP90(median, p90 float64) (Lognormal, error) {
	if !(median > 0) || !finite(median) {
		return Lognormal{}, paramError(familyLognormal, "median", median)
	}
	if !(p90 > median) || !finite(p90) {
		return Lognormal{}, relationError(familyLognormal, "need p90 > median > 0")
	}
	return NewLognormal(math.Log(median), math.Log(p90/median)/lognormalZ90)
}

// NewLognormalFromMeanSD converts a mean and standard deviation.
func NewLognormalFromMeanSD(mean, sd float64) (Lognormal, error) {
	if !(mean > 0) || !finite(mean) {
		return Lognormal{}, paramError(familyLognormal, paramMean, mean)
	}
	if !(sd > 0) || !finite(sd) {
		return Lognormal{}, paramError(familyLognormal, "sd", sd)
	}
	s2 := math.Log1p(sd * sd / (mean * mean))
	return NewLognormal(math.Log(mean)-s2/2, math.Sqrt(s2))
}

// NewGammaFromMeanSD converts a mean and standard deviation.
func NewGammaFromMeanSD(mean, sd float64) (Gamma, error) {
	if !(mean > 0) || !finite(mean) {
		return Gamma{}, paramError(familyGamma, paramMean, mean)
	}
	if !(sd > 0) || !finite(sd) {
		return Gamma{}, paramError(familyGamma, "sd", sd)
	}
	return NewGamma(mean*mean/(sd*sd), sd*sd/mean)
}

// NewGammaFromShapeRate converts a shape and rate.
func NewGammaFromShapeRate(shape, rate float64) (Gamma, error) {
	if !(rate > 0) || !finite(rate) {
		return Gamma{}, paramError(familyGamma, "rate", rate)
	}
	return NewGamma(shape, 1/rate)
}

// NewBetaFromMeanSD converts a mean and standard deviation; it fails when
// no Beta has them, that is when sd² ≥ mean·(1 − mean).
func NewBetaFromMeanSD(mean, sd float64) (Beta, error) {
	if !(mean > 0 && mean < 1) {
		return Beta{}, paramError(familyBeta, paramMean, mean)
	}
	v := sd * sd
	if !(v > 0) || v >= mean*(1-mean) {
		return Beta{}, relationError(familyBeta, "need 0 < sd^2 < mean*(1-mean)")
	}
	k := mean*(1-mean)/v - 1
	return NewBeta(mean*k, (1-mean)*k)
}

// NewExponentialFromRate converts a rate.
func NewExponentialFromRate(rate float64) (Exponential, error) {
	if !(rate > 0) || !finite(rate) {
		return Exponential{}, paramError(familyExponential, "rate", rate)
	}
	return NewExponential(1 / rate)
}
