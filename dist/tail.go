package dist

import (
	"math"
	"math/rand/v2"
)

const (
	familyPareto = "pareto"
	familyGPD    = "gpd"
	paramMin     = "min"
	paramLoc     = "loc"
	paramXi      = "shape"
)

// Pareto has minimum xm and tail index alpha.
type Pareto struct{ xm, alpha, invAlpha float64 }

// NewPareto validates xm > 0 and alpha > 0.
func NewPareto(xm, alpha float64) (Pareto, error) {
	if !(xm > 0) || !finite(xm) {
		return Pareto{}, paramError(familyPareto, paramMin, xm)
	}
	if !(alpha > 0) || !finite(alpha) {
		return Pareto{}, paramError(familyPareto, paramAlpha, alpha)
	}
	return Pareto{xm: xm, alpha: alpha, invAlpha: 1 / alpha}, nil
}

func (d Pareto) Fill(r *rand.Rand, dst []float64) {
	for i := range dst {
		dst[i] = d.xm * math.Exp(-d.invAlpha*math.Log(unit(r)))
	}
}

// Mean is +Inf for alpha ≤ 1.
func (d Pareto) Mean() float64 {
	if d.alpha <= 1 {
		return math.Inf(1)
	}
	return d.alpha * d.xm / (d.alpha - 1)
}

// Var is +Inf for alpha ≤ 2.
func (d Pareto) Var() float64 {
	if d.alpha <= 2 {
		return math.Inf(1)
	}
	a1 := d.alpha - 1
	return d.xm * d.xm * d.alpha / (a1 * a1 * (d.alpha - 2))
}

func (d Pareto) Support() (lo, hi float64) { return d.xm, math.Inf(1) }

func (d Pareto) CDF(x float64) float64 {
	if x <= d.xm {
		return 0
	}
	return -math.Expm1(-d.alpha * math.Log(x/d.xm))
}

func (d Pareto) Survival(x float64) float64 {
	if x <= d.xm {
		return 1
	}
	return math.Pow(d.xm/x, d.alpha)
}

func (d Pareto) Quantile(p float64) float64 {
	return d.xm * math.Exp(-d.invAlpha*math.Log1p(-p))
}

func (d Pareto) QuantileUpper(q float64) float64 {
	return d.xm * math.Pow(q, -d.invAlpha)
}

// GPD is the generalised Pareto with location mu, scale sigma and shape xi.
// Shape 0 is the exponential; a negative shape bounds the support above.
type GPD struct {
	mu, sigma, xi float64
	hi            float64 // upper support bound: +Inf unless xi < 0
}

// NewGPD validates sigma > 0 and finite mu and xi.
func NewGPD(mu, sigma, xi float64) (GPD, error) {
	if !finite(mu) {
		return GPD{}, paramError(familyGPD, paramLoc, mu)
	}
	if !(sigma > 0) || !finite(sigma) {
		return GPD{}, paramError(familyGPD, paramScale, sigma)
	}
	if !finite(xi) {
		return GPD{}, paramError(familyGPD, paramXi, xi)
	}
	d := GPD{mu: mu, sigma: sigma, xi: xi, hi: math.Inf(1)}
	if xi < 0 {
		d.hi = mu - sigma/xi
	}
	return d, nil
}

// fromUpper maps an upper-tail probability q = Survival(x) to x.
func (d GPD) fromUpper(q float64) float64 {
	var y float64
	if d.xi == 0 {
		y = -math.Log(q)
	} else {
		y = math.Expm1(-d.xi*math.Log(q)) / d.xi
	}
	return clamp(d.mu+d.sigma*y, d.mu, d.hi)
}

func (d GPD) Fill(r *rand.Rand, dst []float64) {
	if d.xi == 0 {
		for i := range dst {
			dst[i] = d.mu + d.sigma*r.ExpFloat64()
		}
		return
	}
	for i := range dst {
		dst[i] = d.fromUpper(unit(r))
	}
}

// Mean is +Inf for xi ≥ 1.
func (d GPD) Mean() float64 {
	if d.xi >= 1 {
		return math.Inf(1)
	}
	return d.mu + d.sigma/(1-d.xi)
}

// Var is +Inf for xi ≥ ½.
func (d GPD) Var() float64 {
	if d.xi >= 0.5 {
		return math.Inf(1)
	}
	x1 := 1 - d.xi
	return d.sigma * d.sigma / (x1 * x1 * (1 - 2*d.xi))
}

func (d GPD) Support() (lo, hi float64) { return d.mu, d.hi }

func (d GPD) Survival(x float64) float64 {
	if x <= d.mu {
		return 1
	}
	if x >= d.hi {
		return 0
	}
	y := (x - d.mu) / d.sigma
	if d.xi == 0 {
		return math.Exp(-y)
	}
	return math.Exp(-math.Log1p(d.xi*y) / d.xi)
}

func (d GPD) CDF(x float64) float64 {
	if x <= d.mu {
		return 0
	}
	if x >= d.hi {
		return 1
	}
	y := (x - d.mu) / d.sigma
	if d.xi == 0 {
		return -math.Expm1(-y)
	}
	return -math.Expm1(-math.Log1p(d.xi*y) / d.xi)
}

// Quantile works from p directly, so the lower tail keeps its precision.
func (d GPD) Quantile(p float64) float64 {
	var y float64
	if d.xi == 0 {
		y = -math.Log1p(-p)
	} else {
		y = math.Expm1(-d.xi*math.Log1p(-p)) / d.xi
	}
	return clamp(d.mu+d.sigma*y, d.mu, d.hi)
}

func (d GPD) QuantileUpper(q float64) float64 { return d.fromUpper(q) }
