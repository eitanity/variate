package dist

import (
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/integrate/quad"
)

const (
	familyPERT    = "pert"
	familyLogPERT = "logpert"
	paramLow      = "low"
	paramPeak     = "peak"
	paramHigh     = "high"
	paramT        = "t"

	// DefaultPERTShape is the t of the modified PERT when none is given.
	DefaultPERTShape = 4
)

// PERT is the modified PERT distribution on [low, high] with mode peak and
// shape parameter t.
type PERT struct {
	low, high, span float64
	beta            Beta
	mean, variance  float64
}

// NewPERT validates low ≤ peak ≤ high, all finite with a positive finite
// span, and t ≥ 0 (t = 0 is the uniform).
func NewPERT(low, peak, high, t float64) (PERT, error) {
	if err := checkPERTParams(familyPERT, low, peak, high, t); err != nil {
		return PERT{}, err
	}
	span := high - low
	alpha := 1 + t*(peak-low)/span
	beta := 1 + t*(high-peak)/span
	b, err := NewBeta(alpha, beta)
	if err != nil {
		return PERT{}, err
	}
	s := alpha + beta
	return PERT{
		low: low, high: high, span: span, beta: b,
		mean:     (low + t*peak + high) / (t + 2),
		variance: span * span * alpha * beta / (s * s * (s + 1)),
	}, nil
}

func (d PERT) Fill(r *rand.Rand, dst []float64) {
	d.beta.Fill(r, dst)
	for i, b := range dst {
		dst[i] = clamp(d.low+d.span*b, d.low, d.high)
	}
}

func (d PERT) Mean() float64 { return d.mean }

func (d PERT) Var() float64 { return d.variance }

func (d PERT) Support() (lo, hi float64) { return d.low, d.high }

func (d PERT) CDF(x float64) float64 { return d.beta.CDF((x - d.low) / d.span) }

func (d PERT) Survival(x float64) float64 { return d.beta.Survival((x - d.low) / d.span) }

func (d PERT) Quantile(p float64) float64 {
	return clamp(d.low+d.span*d.beta.Quantile(p), d.low, d.high)
}

func (d PERT) QuantileUpper(q float64) float64 {
	return clamp(d.low+d.span*d.beta.QuantileUpper(q), d.low, d.high)
}

// Shape returns the Beta shapes behind the PERT.
func (d PERT) Shape() (alpha, beta float64) { return d.beta.alpha, d.beta.beta }

// LogPERT is a PERT on log10 of the value: the draw is 10 to a PERT draw on
// [log10 low, log10 high] with mode log10 peak.
type LogPERT struct {
	lp             PERT
	low, high      float64
	mean, variance float64
}

// NewLogPERT validates like NewPERT and additionally requires low > 0. Its
// moments are computed once by quadrature.
func NewLogPERT(low, peak, high, t float64) (LogPERT, error) {
	if !(low > 0) {
		return LogPERT{}, paramError(familyLogPERT, paramLow, low)
	}
	if err := checkPERTParams(familyLogPERT, low, peak, high, t); err != nil {
		return LogPERT{}, err
	}
	lp, err := NewPERT(math.Log10(low), math.Log10(peak), math.Log10(high), t)
	if err != nil {
		return LogPERT{}, err
	}
	d := LogPERT{lp: lp, low: low, high: high}
	d.mean, d.variance = d.moments()
	return d, nil
}

func (d LogPERT) Fill(r *rand.Rand, dst []float64) {
	d.lp.Fill(r, dst)
	for i, v := range dst {
		dst[i] = clamp(math.Exp(v*math.Ln10), d.low, d.high)
	}
}

func (d LogPERT) Mean() float64 { return d.mean }

func (d LogPERT) Var() float64 { return d.variance }

func (d LogPERT) Support() (lo, hi float64) { return d.low, d.high }

func (d LogPERT) CDF(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return d.lp.CDF(math.Log10(x))
}

func (d LogPERT) Survival(x float64) float64 {
	if x <= 0 {
		return 1
	}
	return d.lp.Survival(math.Log10(x))
}

func (d LogPERT) Quantile(p float64) float64 {
	return clamp(math.Exp(d.lp.Quantile(p)*math.Ln10), d.low, d.high)
}

func (d LogPERT) QuantileUpper(q float64) float64 {
	return clamp(math.Exp(d.lp.QuantileUpper(q)*math.Ln10), d.low, d.high)
}

// Quadrature over the Beta density behind the log10 PERT. The integral is
// split at ½ and each half substituted (b = w^(1/α), 1−b = w^(1/β)) so the
// endpoint singularities of shapes below 1 disappear and the integrand is
// smooth. Nodes double until both moments settle or the cap is reached.
const (
	quadStartNodes = 32
	quadDoublings  = 8
	quadTol        = 1e-13
)

// moments returns E[X] and Var[X] for X = 10^(low + span·B).
func (d LogPERT) moments() (mean, variance float64) {
	alpha, beta := d.lp.Shape()
	k := d.lp.span * math.Ln10
	base := math.Exp(d.lp.low * math.Ln10)
	norm := math.Exp(lgamma(alpha+beta) - lgamma(alpha) - lgamma(beta))
	var m1, prev float64
	n := quadStartNodes
	// bound: quadDoublings doublings of the node count
	for i := 0; i <= quadDoublings; i++ {
		m1 = betaExpect(alpha, beta, k, 0, 0, norm, n)
		if i > 0 && math.Abs(m1-prev) <= quadTol*math.Abs(m1) {
			break
		}
		prev = m1
		n *= 2
	}
	// The variance is a second pass centred on the mean, which avoids the
	// cancellation in E[X²] − E[X]² for a narrow range.
	variance = betaExpect(alpha, beta, k, 2, m1, norm, n)
	return base * m1, base * base * variance
}

// betaExpect integrates g(b)·Beta(b; α, β) over [0, 1] with Gauss–Legendre on
// each substituted half, where g(b) = e^(k·b) for power 0 and
// (e^(k·b) − centre)² for power 2.
func betaExpect(alpha, beta, k float64, power int, centre, norm float64, n int) float64 {
	x := make([]float64, n)
	w := make([]float64, n)
	// Lower half: b = v^(1/α), v ∈ [0, (½)^α]; the Jacobian cancels b^(α−1).
	quad.Legendre{}.FixedLocations(x, w, 0, math.Pow(0.5, alpha))
	sum := 0.0
	for i := range x {
		b := math.Pow(x[i], 1/alpha)
		sum += w[i] * gPower(k, b, power, centre) * math.Pow(1-b, beta-1) / alpha
	}
	// Upper half: 1−b = v^(1/β), v ∈ [0, (½)^β].
	quad.Legendre{}.FixedLocations(x, w, 0, math.Pow(0.5, beta))
	for i := range x {
		b := 1 - math.Pow(x[i], 1/beta)
		sum += w[i] * gPower(k, b, power, centre) * math.Pow(b, alpha-1) / beta
	}
	return norm * sum
}

func gPower(k, b float64, power int, centre float64) float64 {
	g := math.Exp(k * b)
	if power == 0 {
		return g
	}
	dlt := g - centre
	return dlt * dlt
}

func lgamma(x float64) float64 {
	v, _ := math.Lgamma(x)
	return v
}
