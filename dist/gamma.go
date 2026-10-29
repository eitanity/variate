package dist

import (
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/mathext"
)

const (
	familyGamma = "gamma"
	familyBeta  = "beta"
	paramShape  = "shape"
	paramScale  = "scale"
	paramAlpha  = "alpha"
	paramBeta   = "beta"
)

// gammaShape holds the Marsaglia–Tsang constants for one shape. For a shape
// below 1 the constants are those of shape+1 and boost is set, so a draw is
// multiplied by U^(1/shape).
type gammaShape struct {
	a     float64 // the shape the constants were built for (shape, or shape+1)
	inv   float64 // 1/shape, used by the boost
	d, c  float64
	boost bool
}

func newGammaShape(shape float64) gammaShape {
	g := gammaShape{a: shape, inv: 1 / shape}
	if shape < 1 {
		g.a = shape + 1
		g.boost = true
	}
	g.d = g.a - 1.0/3
	g.c = 1 / math.Sqrt(9*g.d)
	return g
}

// std draws from Gamma(a, 1) for a ≥ 1 by Marsaglia and Tsang's method, with
// the quantile function as the fallback after maxRejections tries.
func (g gammaShape) std(r *rand.Rand) float64 {
	for try := 0; try < maxRejections; try++ {
		x := r.NormFloat64()
		v := 1 + g.c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := r.Float64()
		x2 := x * x
		if u < 1-0.0331*x2*x2 || math.Log(u) < 0.5*x2+g.d*(1-v+math.Log(v)) {
			return g.d * v
		}
	}
	return mathext.GammaIncRegInv(g.a, unit(r))
}

// draw returns a Gamma(shape, 1) draw, applying the boost for shapes below 1.
func (g gammaShape) draw(r *rand.Rand) float64 {
	x := g.std(r)
	if g.boost {
		x *= math.Pow(unit(r), g.inv)
	}
	return x
}

// logDraw returns the logarithm of a Gamma(shape, 1) draw. The boost is
// applied in log space so that shapes far below 1 do not underflow.
func (g gammaShape) logDraw(r *rand.Rand) float64 {
	lx := math.Log(g.std(r))
	if g.boost {
		lx += math.Log(unit(r)) * g.inv
	}
	return lx
}

// Gamma is the Gamma distribution with shape k and scale θ.
type Gamma struct {
	shape, scale float64
	g            gammaShape
}

// NewGamma validates shape > 0 and scale > 0.
func NewGamma(shape, scale float64) (Gamma, error) {
	if !(shape > 0) || math.IsInf(shape, 0) {
		return Gamma{}, paramError(familyGamma, paramShape, shape)
	}
	if !(scale > 0) || math.IsInf(scale, 0) {
		return Gamma{}, paramError(familyGamma, paramScale, scale)
	}
	return Gamma{shape: shape, scale: scale, g: newGammaShape(shape)}, nil
}

func (d Gamma) Fill(r *rand.Rand, dst []float64) {
	for i := range dst {
		dst[i] = d.g.draw(r) * d.scale
	}
}

func (d Gamma) Mean() float64 { return d.shape * d.scale }

func (d Gamma) Var() float64 { return d.shape * d.scale * d.scale }

func (d Gamma) Support() (lo, hi float64) { return 0, math.Inf(1) }

func (d Gamma) CDF(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return mathext.GammaIncReg(d.shape, x/d.scale)
}

func (d Gamma) Survival(x float64) float64 {
	if x <= 0 {
		return 1
	}
	return mathext.GammaIncRegComp(d.shape, x/d.scale)
}

// Beta is the Beta distribution on [0, 1], drawn as X/(X+Y) from two Gammas.
type Beta struct {
	alpha, beta float64
	ga, gb      gammaShape
	logSpace    bool // either shape below 1: combine in log space
}

// NewBeta validates alpha > 0 and beta > 0.
func NewBeta(alpha, beta float64) (Beta, error) {
	if !(alpha > 0) || math.IsInf(alpha, 0) {
		return Beta{}, paramError(familyBeta, paramAlpha, alpha)
	}
	if !(beta > 0) || math.IsInf(beta, 0) {
		return Beta{}, paramError(familyBeta, paramBeta, beta)
	}
	return Beta{
		alpha: alpha, beta: beta,
		ga: newGammaShape(alpha), gb: newGammaShape(beta),
		logSpace: alpha < 1 || beta < 1,
	}, nil
}

// draw returns one Beta variate.
func (d Beta) draw(r *rand.Rand) float64 {
	if d.logSpace {
		lx := d.ga.logDraw(r)
		ly := d.gb.logDraw(r)
		return 1 / (1 + math.Exp(ly-lx))
	}
	x := d.ga.std(r)
	y := d.gb.std(r)
	return x / (x + y)
}

func (d Beta) Fill(r *rand.Rand, dst []float64) {
	for i := range dst {
		dst[i] = d.draw(r)
	}
}

func (d Beta) Mean() float64 { return d.alpha / (d.alpha + d.beta) }

func (d Beta) Var() float64 {
	s := d.alpha + d.beta
	return d.alpha * d.beta / (s * s * (s + 1))
}

func (d Beta) Support() (lo, hi float64) { return 0, 1 }

func (d Beta) CDF(x float64) float64 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 1
	}
	return mathext.RegIncBeta(d.alpha, d.beta, x)
}

func (d Beta) Survival(x float64) float64 {
	if x <= 0 {
		return 1
	}
	if x >= 1 {
		return 0
	}
	return mathext.RegIncBeta(d.beta, d.alpha, 1-x)
}
