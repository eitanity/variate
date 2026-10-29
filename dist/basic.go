package dist

import (
	"math"
	"math/rand/v2"
)

const (
	familyConstant    = "constant"
	familyUniform     = "uniform"
	familyLogUniform  = "loguniform"
	familyTriangular  = "triangular"
	familyExponential = "exponential"
	familyLognormal   = "lognormal"
	familyWeibull     = "weibull"
	paramValue        = "value"
	paramA            = "a"
	paramB            = "b"
	paramMode         = "mode"
	paramMean         = "mean"
	paramMu           = "mu"
	paramSigma        = "sigma"
)

func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

// Constant is a point mass.
type Constant struct{ c float64 }

// NewConstant validates that the value is finite.
func NewConstant(c float64) (Constant, error) {
	if !finite(c) {
		return Constant{}, paramError(familyConstant, paramValue, c)
	}
	return Constant{c: c}, nil
}

func (d Constant) Fill(_ *rand.Rand, dst []float64) {
	for i := range dst {
		dst[i] = d.c
	}
}

func (d Constant) Mean() float64 { return d.c }

func (d Constant) Var() float64 { return 0 }

func (d Constant) Support() (lo, hi float64) { return d.c, d.c }

func (d Constant) CDF(x float64) float64 {
	if x < d.c {
		return 0
	}
	return 1
}

func (d Constant) Survival(x float64) float64 {
	if x < d.c {
		return 1
	}
	return 0
}

func (d Constant) Quantile(float64) float64 { return d.c }

func (d Constant) QuantileUpper(float64) float64 { return d.c }

// Uniform is the continuous uniform on [a, b].
type Uniform struct{ a, b, span float64 }

func (d Uniform) Fill(r *rand.Rand, dst []float64) {
	for i := range dst {
		dst[i] = d.a + d.span*r.Float64()
	}
}

func (d Uniform) Mean() float64 { return d.a + d.span/2 }

func (d Uniform) Var() float64 { return d.span * d.span / 12 }

func (d Uniform) Support() (lo, hi float64) { return d.a, d.b }

func (d Uniform) CDF(x float64) float64 { return clamp((x-d.a)/d.span, 0, 1) }

func (d Uniform) Survival(x float64) float64 { return clamp((d.b-x)/d.span, 0, 1) }

func (d Uniform) Quantile(p float64) float64 { return clamp(d.a+d.span*p, d.a, d.b) }

func (d Uniform) QuantileUpper(q float64) float64 { return clamp(d.b-d.span*q, d.a, d.b) }

// LogUniform is uniform in the logarithm on [a, b].
type LogUniform struct{ a, b, logA, logRatio float64 }

func (d LogUniform) Fill(r *rand.Rand, dst []float64) {
	for i := range dst {
		dst[i] = clamp(math.Exp(d.logA+d.logRatio*r.Float64()), d.a, d.b)
	}
}

func (d LogUniform) Mean() float64 { return (d.b - d.a) / d.logRatio }

func (d LogUniform) Var() float64 {
	m := d.Mean()
	return (d.b*d.b-d.a*d.a)/(2*d.logRatio) - m*m
}

func (d LogUniform) Support() (lo, hi float64) { return d.a, d.b }

func (d LogUniform) CDF(x float64) float64 {
	if x <= d.a {
		return 0
	}
	return clamp(math.Log(x/d.a)/d.logRatio, 0, 1)
}

func (d LogUniform) Survival(x float64) float64 {
	if x >= d.b {
		return 0
	}
	return clamp(math.Log(d.b/x)/d.logRatio, 0, 1)
}

func (d LogUniform) Quantile(p float64) float64 {
	return clamp(math.Exp(d.logA+d.logRatio*p), d.a, d.b)
}

func (d LogUniform) QuantileUpper(q float64) float64 {
	return clamp(math.Exp(math.Log(d.b)-d.logRatio*q), d.a, d.b)
}

// Triangular is the triangular distribution on [low, high] with the given
// mode. The closed forms are written so that nothing cancels when the mode
// sits at an end of the range.
type Triangular struct {
	low, mode, high float64
	span, fc        float64 // fc = CDF at the mode
	cL, cH          float64 // span·(mode−low), span·(high−mode)
}

func (d Triangular) Fill(r *rand.Rand, dst []float64) {
	for i := range dst {
		dst[i] = d.Quantile(r.Float64())
	}
}

func (d Triangular) Mean() float64 { return (d.low + d.mode + d.high) / 3 }

func (d Triangular) Var() float64 {
	l, m, h := d.low, d.mode, d.high
	return (l*l + m*m + h*h - l*m - l*h - m*h) / 18
}

func (d Triangular) Support() (lo, hi float64) { return d.low, d.high }

func (d Triangular) CDF(x float64) float64 {
	switch {
	case x <= d.low:
		return 0
	case x >= d.high:
		return 1
	case x < d.mode:
		return (x - d.low) * (x - d.low) / d.cL
	default:
		// 1 − (high−x)²/cH, with the difference formed before dividing.
		return ((x-d.low)*(d.span+d.high-x) - d.span*(d.mode-d.low)) / d.cH
	}
}

func (d Triangular) Survival(x float64) float64 {
	switch {
	case x <= d.low:
		return 1
	case x >= d.high:
		return 0
	case x >= d.mode:
		return (d.high - x) * (d.high - x) / d.cH
	default:
		return ((d.high-x)*(d.span+x-d.low) - d.span*(d.high-d.mode)) / d.cL
	}
}

func (d Triangular) Quantile(p float64) float64 {
	if p < d.fc {
		return clamp(d.low+math.Sqrt(p*d.cL), d.low, d.high)
	}
	// high − sqrt((1−p)·cH), as a distance from low so a small p does not cancel.
	off := (d.span*(d.mode-d.low) + p*d.cH) / (d.span + math.Sqrt((1-p)*d.cH))
	return clamp(d.low+off, d.low, d.high)
}

func (d Triangular) QuantileUpper(q float64) float64 {
	if q <= 1-d.fc {
		return clamp(d.high-math.Sqrt(q*d.cH), d.low, d.high)
	}
	off := (d.span*(d.high-d.mode) + q*d.cL) / (d.span + math.Sqrt((1-q)*d.cL))
	return clamp(d.high-off, d.low, d.high)
}

// Exponential has the given mean.
type Exponential struct{ mean float64 }

// NewExponential validates mean > 0.
func NewExponential(mean float64) (Exponential, error) {
	if !(mean > 0) || !finite(mean) {
		return Exponential{}, paramError(familyExponential, paramMean, mean)
	}
	return Exponential{mean: mean}, nil
}

func (d Exponential) Fill(r *rand.Rand, dst []float64) {
	for i := range dst {
		dst[i] = d.mean * r.ExpFloat64()
	}
}

func (d Exponential) Mean() float64 { return d.mean }

func (d Exponential) Var() float64 { return d.mean * d.mean }

func (d Exponential) Support() (lo, hi float64) { return 0, math.Inf(1) }

func (d Exponential) CDF(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return -math.Expm1(-x / d.mean)
}

func (d Exponential) Survival(x float64) float64 {
	if x <= 0 {
		return 1
	}
	return math.Exp(-x / d.mean)
}

func (d Exponential) Quantile(p float64) float64 { return -d.mean * math.Log1p(-p) }

func (d Exponential) QuantileUpper(q float64) float64 { return -d.mean * math.Log(q) }

// Lognormal is exp of a Normal(mu, sigma).
type Lognormal struct{ mu, sigma float64 }

// NewLognormal validates sigma > 0 and a finite mu.
func NewLognormal(mu, sigma float64) (Lognormal, error) {
	if !finite(mu) {
		return Lognormal{}, paramError(familyLognormal, paramMu, mu)
	}
	if !(sigma > 0) || !finite(sigma) {
		return Lognormal{}, paramError(familyLognormal, paramSigma, sigma)
	}
	return Lognormal{mu: mu, sigma: sigma}, nil
}

func (d Lognormal) Fill(r *rand.Rand, dst []float64) {
	for i := range dst {
		dst[i] = math.Exp(d.mu + d.sigma*r.NormFloat64())
	}
}

func (d Lognormal) Mean() float64 { return math.Exp(d.mu + d.sigma*d.sigma/2) }

func (d Lognormal) Var() float64 {
	s2 := d.sigma * d.sigma
	return math.Expm1(s2) * math.Exp(2*d.mu+s2)
}

func (d Lognormal) Support() (lo, hi float64) { return 0, math.Inf(1) }

func (d Lognormal) CDF(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return normCDF((math.Log(x) - d.mu) / d.sigma)
}

func (d Lognormal) Survival(x float64) float64 {
	if x <= 0 {
		return 1
	}
	return normSurvival((math.Log(x) - d.mu) / d.sigma)
}

func (d Lognormal) Quantile(p float64) float64 {
	return math.Exp(d.mu + d.sigma*normQuantile(p))
}

func (d Lognormal) QuantileUpper(q float64) float64 {
	return math.Exp(d.mu + d.sigma*normQuantileUpper(q))
}

// Weibull has shape k and scale lambda.
type Weibull struct{ shape, scale, invShape float64 }

// NewWeibull validates shape > 0 and scale > 0.
func NewWeibull(shape, scale float64) (Weibull, error) {
	if !(shape > 0) || !finite(shape) {
		return Weibull{}, paramError(familyWeibull, paramShape, shape)
	}
	if !(scale > 0) || !finite(scale) {
		return Weibull{}, paramError(familyWeibull, paramScale, scale)
	}
	return Weibull{shape: shape, scale: scale, invShape: 1 / shape}, nil
}

func (d Weibull) Fill(r *rand.Rand, dst []float64) {
	for i := range dst {
		dst[i] = d.scale * math.Exp(d.invShape*math.Log(-math.Log(unit(r))))
	}
}

func (d Weibull) Mean() float64 { return d.scale * math.Gamma(1+d.invShape) }

func (d Weibull) Var() float64 {
	g1 := math.Gamma(1 + d.invShape)
	return d.scale * d.scale * (math.Gamma(1+2*d.invShape) - g1*g1)
}

func (d Weibull) Support() (lo, hi float64) { return 0, math.Inf(1) }

func (d Weibull) CDF(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return -math.Expm1(-math.Pow(x/d.scale, d.shape))
}

func (d Weibull) Survival(x float64) float64 {
	if x <= 0 {
		return 1
	}
	return math.Exp(-math.Pow(x/d.scale, d.shape))
}

func (d Weibull) Quantile(p float64) float64 {
	return d.scale * math.Pow(-math.Log1p(-p), d.invShape)
}

func (d Weibull) QuantileUpper(q float64) float64 {
	return d.scale * math.Pow(-math.Log(q), d.invShape)
}
