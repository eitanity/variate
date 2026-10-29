package dist

import (
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/mathext"
)

const (
	familyPoisson = "poisson"
	paramLambda   = "lambda"

	// ptrsThreshold is the rate at which the sampler switches from inversion
	// to the transformed rejection method.
	ptrsThreshold = 22
	// inversionSteps bounds the sequential search; for rates below the
	// threshold the mass beyond it is far below 1e-100.
	inversionSteps = 256
	// quantileSteps bounds the bracketing and bisection of the quantile.
	quantileSteps = 64
)

// ptrs holds the per-rate constants of Hörmann's PTRS method.
type ptrs struct {
	lambda, logLambda, b, a, invAlpha, vr float64
}

func newPTRS(lambda float64) ptrs {
	slam := math.Sqrt(lambda)
	b := 0.931 + 2.53*slam
	return ptrs{
		lambda:    lambda,
		logLambda: math.Log(lambda),
		b:         b,
		a:         -0.059 + 0.02483*b,
		invAlpha:  1.1239 + 1.1328/(b-3.4),
		vr:        0.9277 - 3.6224/(b-2),
	}
}

// draw returns one Poisson count by PTRS, falling back to the quantile
// function after maxRejections tries.
func (c ptrs) draw(r *rand.Rand) float64 {
	for try := 0; try < maxRejections; try++ {
		u := r.Float64() - 0.5
		v := r.Float64()
		us := 0.5 - math.Abs(u)
		k := math.Floor((2*c.a/us+c.b)*u + c.lambda + 0.43)
		if us >= 0.07 && v <= c.vr {
			return k
		}
		if k < 0 || (us < 0.013 && v > us) {
			continue
		}
		lhs := math.Log(v) + math.Log(c.invAlpha) - math.Log(c.a/(us*us)+c.b)
		rhs := -c.lambda + k*c.logLambda - lgamma(k+1)
		if lhs <= rhs {
			return k
		}
	}
	return poissonQuantile(c.lambda, unit(r))
}

// poissonInversion draws by sequential search from P(0) = e^−λ.
func poissonInversion(r *rand.Rand, lambda, expNeg float64) float64 {
	u := r.Float64()
	p := expNeg
	s := p
	k := 0.0
	// bound: inversionSteps
	for step := 0; u > s && step < inversionSteps; step++ {
		k++
		p *= lambda / k
		s += p
	}
	return k
}

// poissonQuantile returns the smallest k with CDF(k) ≥ p, by bracketing and
// bisection on the regularised incomplete gamma function.
func poissonQuantile(lambda, p float64) float64 {
	if p <= 0 {
		return 0
	}
	if lambda == 0 {
		return 0
	}
	lo, hi := 0.0, math.Ceil(lambda+10*math.Sqrt(lambda+1)+50)
	// bound: quantileSteps doublings
	for i := 0; i < quantileSteps && poissonCDF(lambda, hi) < p; i++ {
		hi *= 2
	}
	if poissonCDF(lambda, lo) >= p {
		return lo
	}
	// Invariant: CDF(lo) < p ≤ CDF(hi).
	// bound: quantileSteps bisections
	for i := 0; i < quantileSteps && hi-lo > 1; i++ {
		mid := math.Floor((lo + hi) / 2)
		if poissonCDF(lambda, mid) >= p {
			hi = mid
		} else {
			lo = mid
		}
	}
	return hi
}

// poissonQuantileUpper returns the smallest k with Survival(k) ≤ q.
func poissonQuantileUpper(lambda, q float64) float64 {
	if q >= 1 || lambda == 0 {
		return 0
	}
	lo, hi := 0.0, math.Ceil(lambda+10*math.Sqrt(lambda+1)+50)
	// bound: quantileSteps doublings
	for i := 0; i < quantileSteps && poissonSurvival(lambda, hi) > q; i++ {
		hi *= 2
	}
	if poissonSurvival(lambda, lo) <= q {
		return lo
	}
	// Invariant: Survival(lo) > q ≥ Survival(hi).
	// bound: quantileSteps bisections
	for i := 0; i < quantileSteps && hi-lo > 1; i++ {
		mid := math.Floor((lo + hi) / 2)
		if poissonSurvival(lambda, mid) <= q {
			hi = mid
		} else {
			lo = mid
		}
	}
	return hi
}

// poissonCDF is P(K ≤ k) = Q(⌊k⌋+1, λ), the regularised upper incomplete gamma.
func poissonCDF(lambda, k float64) float64 {
	if k < 0 {
		return 0
	}
	if lambda == 0 {
		return 1
	}
	return mathext.GammaIncRegComp(math.Floor(k)+1, lambda)
}

// poissonSurvival is P(K > k) = P(⌊k⌋+1, λ).
func poissonSurvival(lambda, k float64) float64 {
	if k < 0 {
		return 1
	}
	if lambda == 0 {
		return 0
	}
	return mathext.GammaIncReg(math.Floor(k)+1, lambda)
}

// Poisson is the Poisson distribution with a fixed rate.
type Poisson struct {
	lambda float64
	expNeg float64 // e^−λ, for the inversion path
	c      ptrs    // valid when lambda ≥ ptrsThreshold
}

// NewPoisson validates lambda ≥ 0 and finite.
func NewPoisson(lambda float64) (Poisson, error) {
	if !(lambda >= 0) || math.IsInf(lambda, 0) {
		return Poisson{}, paramError(familyPoisson, paramLambda, lambda)
	}
	d := Poisson{lambda: lambda, expNeg: math.Exp(-lambda)}
	if lambda >= ptrsThreshold {
		d.c = newPTRS(lambda)
	}
	return d, nil
}

func (d Poisson) Fill(r *rand.Rand, dst []float64) {
	switch {
	case d.lambda == 0:
		for i := range dst {
			dst[i] = 0
		}
	case d.lambda < ptrsThreshold:
		for i := range dst {
			dst[i] = poissonInversion(r, d.lambda, d.expNeg)
		}
	default:
		for i := range dst {
			dst[i] = d.c.draw(r)
		}
	}
}

func (d Poisson) Mean() float64 { return d.lambda }

func (d Poisson) Var() float64 { return d.lambda }

func (d Poisson) Support() (lo, hi float64) { return 0, math.Inf(1) }

func (d Poisson) CDF(x float64) float64 { return poissonCDF(d.lambda, x) }

func (d Poisson) Survival(x float64) float64 { return poissonSurvival(d.lambda, x) }

func (d Poisson) Quantile(p float64) float64 { return poissonQuantile(d.lambda, p) }

func (d Poisson) QuantileUpper(q float64) float64 { return poissonQuantileUpper(d.lambda, q) }

// PMF returns P(K = x); zero off the integers.
func (d Poisson) PMF(x float64) float64 {
	if x < 0 || x != math.Floor(x) {
		return 0
	}
	if d.lambda == 0 {
		if x == 0 {
			return 1
		}
		return 0
	}
	return math.Exp(x*math.Log(d.lambda) - d.lambda - lgamma(x+1))
}

// PoissonCount draws one count for a rate that changes on every call, such
// as a mixed-Poisson rate drawn per period. The PTRS constants are built
// inline: a square root and a logarithm per call.
func PoissonCount(r *rand.Rand, lambda float64) uint64 {
	var k float64
	switch {
	case lambda <= 0:
		return 0
	case lambda < ptrsThreshold:
		k = poissonInversion(r, lambda, math.Exp(-lambda))
	default:
		k = newPTRS(lambda).draw(r)
	}
	return uint64(k)
}
