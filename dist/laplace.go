package dist

import (
	"math"
	"math/cmplx"
)

// Closed-form Laplace transforms E[exp(−s·X)] where they exist. For a
// mixed-Poisson count with rate X, the probability generating function at
// w is the transform at s = 1 − w. LogLaplace is a logarithm of it, for
// summing several transforms' exponents and taking one exp; it is
// closed for the constant and the gamma, and false elsewhere.

func (d Constant) Laplace(s complex128) (complex128, bool) {
	return cmplx.Exp(-s * complex(d.c, 0)), true
}

func (d Constant) LogLaplace(s complex128) (complex128, bool) {
	return -s * complex(d.c, 0), true
}

// Gamma: (1 + θs)^(−k) through the principal logarithm; with Re(s) ≥ 0
// the argument has real part at least 1, away from the branch cut.
func (d Gamma) Laplace(s complex128) (complex128, bool) {
	l, _ := d.LogLaplace(s)
	return cmplx.Exp(l), true
}

// LogLaplace is −k·ln(1 + θs), the logarithm's real part taken as half
// that of the squared modulus, which needs no hypotenuse.
func (d Gamma) LogLaplace(s complex128) (complex128, bool) {
	z := 1 + complex(d.scale, 0)*s
	x, y := real(z), imag(z)
	return complex(-d.shape*0.5*math.Log(x*x+y*y), -d.shape*math.Atan2(y, x)), true
}

func (d Exponential) Laplace(s complex128) (complex128, bool) {
	return 1 / (1 + complex(d.mean, 0)*s), true
}

// The exponential has a closed logarithm too, but its transform is one
// division, cheaper than the logarithm a sum of exponents would take.
func (Exponential) LogLaplace(complex128) (complex128, bool) { return 0, false }

// Categorical: the exact finite mixture.
func (d Categorical) Laplace(s complex128) (complex128, bool) {
	var sum complex128
	for i, v := range d.values {
		sum += complex(d.probs[i], 0) * cmplx.Exp(-s*complex(v, 0))
	}
	return sum, true
}

func (d Uniform) Laplace(s complex128) (complex128, bool) {
	if s == 0 {
		return 1, true
	}
	return (cmplx.Exp(-s*complex(d.a, 0)) - cmplx.Exp(-s*complex(d.b, 0))) / (s * complex(d.span, 0)), true
}

func (Beta) Laplace(complex128) (complex128, bool)            { return 0, false }
func (PERT) Laplace(complex128) (complex128, bool)            { return 0, false }
func (LogPERT) Laplace(complex128) (complex128, bool)         { return 0, false }
func (Poisson) Laplace(complex128) (complex128, bool)         { return 0, false }
func (LogUniform) Laplace(complex128) (complex128, bool)      { return 0, false }
func (Triangular) Laplace(complex128) (complex128, bool)      { return 0, false }
func (Lognormal) Laplace(complex128) (complex128, bool)       { return 0, false }
func (Weibull) Laplace(complex128) (complex128, bool)         { return 0, false }
func (LogLogistic) Laplace(complex128) (complex128, bool)     { return 0, false }
func (Normal) Laplace(complex128) (complex128, bool)          { return 0, false }
func (TruncatedNormal) Laplace(complex128) (complex128, bool) { return 0, false }
func (Pareto) Laplace(complex128) (complex128, bool)          { return 0, false }
func (GPD) Laplace(complex128) (complex128, bool)             { return 0, false }
func (Bernoulli) Laplace(complex128) (complex128, bool)       { return 0, false }
func (Binomial) Laplace(complex128) (complex128, bool)        { return 0, false }
func (Geometric) Laplace(complex128) (complex128, bool)       { return 0, false }
func (NegativeBinomial) Laplace(complex128) (complex128, bool) {
	return 0, false
}
func (Truncated) Laplace(complex128) (complex128, bool) { return 0, false }
func (Capped) Laplace(complex128) (complex128, bool)    { return 0, false }

func (Categorical) LogLaplace(complex128) (complex128, bool)      { return 0, false }
func (Uniform) LogLaplace(complex128) (complex128, bool)          { return 0, false }
func (Beta) LogLaplace(complex128) (complex128, bool)             { return 0, false }
func (PERT) LogLaplace(complex128) (complex128, bool)             { return 0, false }
func (LogPERT) LogLaplace(complex128) (complex128, bool)          { return 0, false }
func (Poisson) LogLaplace(complex128) (complex128, bool)          { return 0, false }
func (LogUniform) LogLaplace(complex128) (complex128, bool)       { return 0, false }
func (Triangular) LogLaplace(complex128) (complex128, bool)       { return 0, false }
func (Lognormal) LogLaplace(complex128) (complex128, bool)        { return 0, false }
func (Weibull) LogLaplace(complex128) (complex128, bool)          { return 0, false }
func (LogLogistic) LogLaplace(complex128) (complex128, bool)      { return 0, false }
func (Normal) LogLaplace(complex128) (complex128, bool)           { return 0, false }
func (TruncatedNormal) LogLaplace(complex128) (complex128, bool)  { return 0, false }
func (Pareto) LogLaplace(complex128) (complex128, bool)           { return 0, false }
func (GPD) LogLaplace(complex128) (complex128, bool)              { return 0, false }
func (Bernoulli) LogLaplace(complex128) (complex128, bool)        { return 0, false }
func (Binomial) LogLaplace(complex128) (complex128, bool)         { return 0, false }
func (Geometric) LogLaplace(complex128) (complex128, bool)        { return 0, false }
func (NegativeBinomial) LogLaplace(complex128) (complex128, bool) { return 0, false }
func (Truncated) LogLaplace(complex128) (complex128, bool)        { return 0, false }
func (Capped) LogLaplace(complex128) (complex128, bool)           { return 0, false }
