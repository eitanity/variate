package dist

import (
	"errors"
	"math/rand/v2"
	"strconv"
)

// Distribution is one family with fixed parameters. Fill writes len(dst)
// independent draws and never allocates. Laplace is E[exp(−s·X)] for
// complex s with Re(s) ≥ 0 in closed form; ok is false for a family
// without one, and the caller integrates numerically instead. Nodes fills
// a Gauss rule for the family's own density (see nodes.go); false for a
// family without one.
//
// LogDensity is the natural log of the probability density or mass at x:
// ln f(x) with f = dF/dx for a continuous family, ln P(X = x) for a
// discrete family or at an atom of a mixed distribution, −Inf off the
// support. IsAtom reports P(X = x) > 0. LogSurvival is ln P(X > x), in
// closed form where one exists, so that it stays finite past the point
// where Survival underflows.
type Distribution interface {
	Fill(r *rand.Rand, dst []float64)
	Mean() float64
	Var() float64
	Support() (lo, hi float64)
	CDF(x float64) float64
	Laplace(s complex128) (value complex128, ok bool)
	Nodes(x, w []float64) bool
	LogDensity(x float64) float64
	IsAtom(x float64) bool
	LogSurvival(x float64) float64
	Quantiler
}

// Quantiler is implemented by every continuous distribution. Survival and
// QuantileUpper work on the upper tail directly so that no precision is lost
// to 1 − CDF.
type Quantiler interface {
	Quantile(p float64) float64
	Survival(x float64) float64
	QuantileUpper(q float64) float64
}

// PMFer is implemented by discrete distributions. Values are integers stored
// as float64, except for Categorical and Empirical.
type PMFer interface {
	PMF(x float64) float64
}

// maxRejections bounds every rejection loop; after it the draw comes from the
// quantile function.
const maxRejections = 1000

// unit returns a uniform in (0, 1], safe for logarithms and negative powers.
func unit(r *rand.Rand) float64 {
	return 1 - r.Float64()
}

// paramError names the parameter that failed validation.
func paramError(family, param string, v float64) error {
	return errors.New(family + ": invalid " + param + " " + strconv.FormatFloat(v, 'g', -1, 64))
}

// relationError describes a constraint between parameters that failed.
func relationError(family, msg string) error {
	return errors.New(family + ": " + msg)
}

// clamp restricts x to [lo, hi].
func clamp(x, lo, hi float64) float64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}
