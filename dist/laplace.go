package dist

import (
	"math/cmplx"
)

// Closed-form Laplace transforms E[exp(−s·X)] where they exist. For a
// mixed-Poisson count with rate X, the probability generating function at
// w is the transform at s = 1 − w.

func (d Constant) Laplace(s complex128) (complex128, bool) {
	return cmplx.Exp(-s * complex(d.c, 0)), true
}

// Gamma: (1 + θs)^(−k) through the principal logarithm; with Re(s) ≥ 0
// the argument has real part at least 1, away from the branch cut.
func (d Gamma) Laplace(s complex128) (complex128, bool) {
	return cmplx.Exp(-complex(d.shape, 0) * cmplx.Log(1+complex(d.scale, 0)*s)), true
}

func (d Exponential) Laplace(s complex128) (complex128, bool) {
	return 1 / (1 + complex(d.mean, 0)*s), true
}

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
