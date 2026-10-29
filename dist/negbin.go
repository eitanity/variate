package dist

import (
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/mathext"
)

const (
	familyGeometric = "geometric"
	familyNegBin    = "negbin"
	paramR          = "r"
	paramSize       = "size"
)

// Geometric counts failures before the first success; support {0, 1, …}.
type Geometric struct{ p, logQ float64 }

// NewGeometric validates 0 < p ≤ 1.
func NewGeometric(p float64) (Geometric, error) {
	if !(p > 0 && p <= 1) {
		return Geometric{}, paramError(familyGeometric, paramP, p)
	}
	return Geometric{p: p, logQ: math.Log1p(-p)}, nil
}

func (d Geometric) Fill(r *rand.Rand, dst []float64) {
	if d.p == 1 {
		for i := range dst {
			dst[i] = 0
		}
		return
	}
	for i := range dst {
		dst[i] = math.Floor(math.Log(unit(r)) / d.logQ)
	}
}

func (d Geometric) Mean() float64 { return (1 - d.p) / d.p }

func (d Geometric) Var() float64 { return (1 - d.p) / (d.p * d.p) }

func (d Geometric) Support() (lo, hi float64) { return 0, math.Inf(1) }

// CDF is 1 − (1−p)^(⌊k⌋+1).
func (d Geometric) CDF(x float64) float64 {
	if x < 0 {
		return 0
	}
	return -math.Expm1((math.Floor(x) + 1) * d.logQ)
}

func (d Geometric) Survival(x float64) float64 {
	if x < 0 {
		return 1
	}
	return math.Exp((math.Floor(x) + 1) * d.logQ)
}

// Quantile is the smallest k with CDF(k) ≥ p.
func (d Geometric) Quantile(p float64) float64 {
	if p <= 0 || d.p == 1 {
		return 0
	}
	return math.Max(0, math.Ceil(math.Log1p(-p)/d.logQ-1))
}

// QuantileUpper is the smallest k with Survival(k) ≤ q.
func (d Geometric) QuantileUpper(q float64) float64 {
	if q >= 1 || d.p == 1 {
		return 0
	}
	return math.Max(0, math.Ceil(math.Log(q)/d.logQ-1))
}

func (d Geometric) PMF(x float64) float64 {
	if x < 0 || x != math.Floor(x) {
		return 0
	}
	return d.p * math.Exp(x*d.logQ)
}

// NegativeBinomial counts failures before the r-th success with success
// probability p; r is real. It is drawn exactly as Poisson(Gamma(r, (1−p)/p)).
type NegativeBinomial struct {
	r, p  float64
	gamma Gamma
}

// NewNegativeBinomial validates r > 0 and 0 < p < 1.
func NewNegativeBinomial(r, p float64) (NegativeBinomial, error) {
	if !(r > 0) || !finite(r) {
		return NegativeBinomial{}, paramError(familyNegBin, paramR, r)
	}
	if !(p > 0 && p < 1) {
		return NegativeBinomial{}, paramError(familyNegBin, paramP, p)
	}
	g, err := NewGamma(r, (1-p)/p)
	if err != nil {
		return NegativeBinomial{}, err
	}
	return NegativeBinomial{r: r, p: p, gamma: g}, nil
}

// NewNegativeBinomialFromMeanSize converts a mean and size: r = size,
// p = size/(size + mean), so Var = mean + mean²/size.
func NewNegativeBinomialFromMeanSize(mean, size float64) (NegativeBinomial, error) {
	if !(mean > 0) || !finite(mean) {
		return NegativeBinomial{}, paramError(familyNegBin, paramMean, mean)
	}
	if !(size > 0) || !finite(size) {
		return NegativeBinomial{}, paramError(familyNegBin, paramSize, size)
	}
	return NewNegativeBinomial(size, size/(size+mean))
}

func (d NegativeBinomial) Fill(r *rand.Rand, dst []float64) {
	for i := range dst {
		dst[i] = float64(PoissonCount(r, d.gamma.g.draw(r)*d.gamma.scale))
	}
}

func (d NegativeBinomial) Mean() float64 { return d.r * (1 - d.p) / d.p }

func (d NegativeBinomial) Var() float64 { return d.r * (1 - d.p) / (d.p * d.p) }

func (d NegativeBinomial) Support() (lo, hi float64) { return 0, math.Inf(1) }

// CDF is P(K ≤ k) = I_p(r, ⌊k⌋ + 1).
func (d NegativeBinomial) CDF(x float64) float64 {
	if x < 0 {
		return 0
	}
	return mathext.RegIncBeta(d.r, math.Floor(x)+1, d.p)
}

func (d NegativeBinomial) Survival(x float64) float64 {
	if x < 0 {
		return 1
	}
	return mathext.RegIncBeta(math.Floor(x)+1, d.r, 1-d.p)
}

// Quantile is the smallest k with CDF(k) ≥ p, by bracketing and bisection.
func (d NegativeBinomial) Quantile(p float64) float64 {
	if p <= 0 {
		return 0
	}
	lo, hi := -1.0, math.Ceil(d.Mean()+10*math.Sqrt(d.Var())+50)
	// bound: quantileSteps doublings
	for i := 0; i < quantileSteps && d.CDF(hi) < p; i++ {
		hi *= 2
	}
	// bound: quantileSteps bisections
	for i := 0; i < quantileSteps && hi-lo > 1; i++ {
		mid := math.Floor((lo + hi) / 2)
		if d.CDF(mid) >= p {
			hi = mid
		} else {
			lo = mid
		}
	}
	return hi
}

// QuantileUpper is the smallest k with Survival(k) ≤ q.
func (d NegativeBinomial) QuantileUpper(q float64) float64 {
	if q >= 1 {
		return 0
	}
	lo, hi := -1.0, math.Ceil(d.Mean()+10*math.Sqrt(d.Var())+50)
	// bound: quantileSteps doublings
	for i := 0; i < quantileSteps && d.Survival(hi) > q; i++ {
		hi *= 2
	}
	// bound: quantileSteps bisections
	for i := 0; i < quantileSteps && hi-lo > 1; i++ {
		mid := math.Floor((lo + hi) / 2)
		if d.Survival(mid) <= q {
			hi = mid
		} else {
			lo = mid
		}
	}
	return hi
}

func (d NegativeBinomial) PMF(x float64) float64 {
	if x < 0 || x != math.Floor(x) {
		return 0
	}
	return math.Exp(lgamma(x+d.r) - lgamma(x+1) - lgamma(d.r) + d.r*math.Log(d.p) + x*math.Log1p(-d.p))
}
