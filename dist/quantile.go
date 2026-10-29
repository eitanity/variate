package dist

import (
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/mathext"
)

// seedFloor is the probability below which gonum's inverses are not
// consulted for a seed: they fail in the far tails, and the series seeds
// are accurate there.
const seedFloor = 1e-12

// gammaStd is Gamma(a, 1) as seen by the inverter.
type gammaStd struct{ a, lg float64 }

func (g gammaStd) cdf(x float64) float64 { return mathext.GammaIncReg(g.a, x) }

func (g gammaStd) sf(x float64) float64 { return mathext.GammaIncRegComp(g.a, x) }

func (g gammaStd) pdf(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return math.Exp((g.a-1)*math.Log(x) - x - g.lg)
}

// Quantile inverts the regularised incomplete gamma function to full
// precision: gonum's inverse seeds a safeguarded Newton iteration, with a
// small-x series seed where gonum returns 0.
func (d Gamma) Quantile(p float64) float64 {
	if p <= 0 {
		return 0
	}
	if p >= 1 {
		return math.Inf(1)
	}
	g := gammaStd{a: d.shape, lg: lgamma(d.shape)}
	x0 := 0.0
	if p >= seedFloor {
		x0 = mathext.GammaIncRegInv(d.shape, p)
	}
	if !(x0 > 0) {
		// P(a, x) ≈ x^a / Γ(a+1) for small x.
		x0 = math.Exp((math.Log(p) + lgamma(d.shape+1)) / d.shape)
	}
	hi := bracketUpper(g, p, false, math.Max(2*x0, 1))
	return invert(g, p, false, 0, hi, x0) * d.scale
}

// QuantileUpper inverts the survival function the same way.
func (d Gamma) QuantileUpper(q float64) float64 {
	if q >= 1 {
		return 0
	}
	if q <= 0 {
		return math.Inf(1)
	}
	g := gammaStd{a: d.shape, lg: lgamma(d.shape)}
	x0 := 0.0
	if q >= seedFloor {
		x0 = mathext.GammaIncRegCompInv(d.shape, q)
	}
	if !(x0 > 0) || math.IsInf(x0, 0) {
		x0 = d.shape + 1
	}
	hi := bracketUpper(g, q, true, math.Max(2*x0, 1))
	return invert(g, q, true, 0, hi, x0) * d.scale
}

// betaPair is Beta(α, β) as seen by the inverter.
type betaPair struct{ a, b, lb float64 }

func (g betaPair) cdf(x float64) float64 { return mathext.RegIncBeta(g.a, g.b, x) }

func (g betaPair) sf(x float64) float64 { return mathext.RegIncBeta(g.b, g.a, 1-x) }

func (g betaPair) pdf(x float64) float64 {
	if x <= 0 || x >= 1 {
		return 0
	}
	return math.Exp((g.a-1)*math.Log(x) + (g.b-1)*math.Log1p(-x) - g.lb)
}

func (d Beta) pair() betaPair {
	return betaPair{a: d.alpha, b: d.beta, lb: lgamma(d.alpha) + lgamma(d.beta) - lgamma(d.alpha+d.beta)}
}

// Quantile inverts the regularised incomplete beta function to full
// precision, seeded by gonum's inverse or by the small-x series
// I_x(a, b) ≈ x^a / (a·B(a, b)) when that seed is 0.
func (d Beta) Quantile(p float64) float64 {
	if p <= 0 {
		return 0
	}
	if p >= 1 {
		return 1
	}
	g := d.pair()
	x0 := 0.0
	if p >= seedFloor {
		x0 = mathext.InvRegIncBeta(d.alpha, d.beta, p)
	}
	if !(x0 > 0) {
		x0 = math.Exp((math.Log(p) + math.Log(d.alpha) + g.lb) / d.alpha)
	}
	return invert(g, p, false, 0, 1, x0)
}

// QuantileUpper inverts the survival function the same way; the seed uses
// the mirrored inverse.
func (d Beta) QuantileUpper(q float64) float64 {
	if q >= 1 {
		return 0
	}
	if q <= 0 {
		return 1
	}
	g := d.pair()
	y0 := 0.0
	if q >= seedFloor {
		y0 = mathext.InvRegIncBeta(d.beta, d.alpha, q)
	}
	if !(y0 > 0) {
		y0 = math.Exp((math.Log(q) + math.Log(d.beta) + g.lb) / d.beta)
	}
	return invert(g, q, true, 0, 1, 1-y0)
}

// truncPair is the standardised truncated normal as seen by the inverter.
type truncPair struct {
	a, b, z float64
	gx, gw  [massNodes]float64
}

func (d TruncatedNormal) pair() truncPair {
	return truncPair{a: d.a, b: d.b, z: d.z, gx: d.gx, gw: d.gw}
}

func (g truncPair) cdf(z float64) float64 { return normMass(g.gx[:], g.gw[:], g.a, z) / g.z }

func (g truncPair) sf(z float64) float64 { return normMass(g.gx[:], g.gw[:], z, g.b) / g.z }

func (g truncPair) pdf(z float64) float64 { return normPDF(z) / g.z }

// moments are the closed forms, with Z = Φ(b) − Φ(a) taken on the side
// where it does not cancel.
func (d *TruncatedNormal) moments() (mean, variance float64) {
	a, b := d.a, d.b
	var z float64
	if a > 0 {
		z = normSurvival(a) - normSurvival(b)
	} else {
		z = normCDF(b) - normCDF(a)
	}
	pa, pb := normPDF(a), normPDF(b)
	apa, bpb := 0.0, 0.0
	if !math.IsInf(a, 0) {
		apa = a * pa
	}
	if !math.IsInf(b, 0) {
		bpb = b * pb
	}
	m := (pa - pb) / z
	v := 1 + (apa-bpb)/z - m*m
	return d.mu + d.sigma*m, d.sigma * d.sigma * v
}

// drawStd returns a standardised draw on the sampled (possibly reflected)
// interval.
func (d TruncatedNormal) drawStd(r *rand.Rand) float64 {
	switch {
	case d.robert:
		return d.drawRobert(r)
	case d.upperForm:
		return normQuantileUpper(d.sLo + r.Float64()*d.sSpan)
	default:
		return normQuantile(d.pLo + r.Float64()*d.pSpan)
	}
}

// drawRobert is the exponential rejection for a far upper tail, bounded by
// maxRejections; the fallback is the inverse of the limiting exponential.
func (d TruncatedNormal) drawRobert(r *rand.Rand) float64 {
	a, b := d.a, d.b
	if d.mirror {
		a, b = -d.b, -d.a
	}
	for try := 0; try < maxRejections; try++ {
		z := a - math.Log(unit(r))/d.alpha
		if z > b {
			continue
		}
		dz := z - d.alpha
		if math.Log(unit(r)) <= -dz*dz/2 {
			return z
		}
	}
	span := -math.Expm1(-a * (b - a))
	return a - math.Log1p(-r.Float64()*span)/a
}

// seedQuantile is the closed-form quantile, accurate except within a few
// ulps of a bound; the inverter refines it on the exact mass.
func (d TruncatedNormal) seedQuantile(p float64, upper bool) float64 {
	if upper {
		p = 1 - p
	}
	if d.a > 0 {
		sa, sb := normSurvival(d.a), normSurvival(d.b)
		return normQuantileUpper(sa - p*(sa-sb))
	}
	pa, pb := normCDF(d.a), normCDF(d.b)
	return normQuantile(pa + p*(pb-pa))
}

func (d TruncatedNormal) Quantile(p float64) float64 {
	if p <= 0 {
		return d.lo
	}
	if p >= 1 {
		return d.hi
	}
	lo, hi := d.bracket()
	z := invert(d.pair(), p, false, lo, hi, d.seedQuantile(p, false))
	return clamp(d.mu+d.sigma*z, d.lo, d.hi)
}

func (d TruncatedNormal) QuantileUpper(q float64) float64 {
	if q >= 1 {
		return d.lo
	}
	if q <= 0 {
		return d.hi
	}
	lo, hi := d.bracket()
	z := invert(d.pair(), q, true, lo, hi, d.seedQuantile(q, true))
	return clamp(d.mu+d.sigma*z, d.lo, d.hi)
}

// bracket is the finite standardised interval the inverter works on.
func (d TruncatedNormal) bracket() (lo, hi float64) {
	lo, hi = d.a, d.b
	if math.IsInf(lo, -1) {
		lo = -normalTailLimit - 2
	}
	if math.IsInf(hi, 1) {
		hi = normalTailLimit + 2
	}
	return lo, hi
}
