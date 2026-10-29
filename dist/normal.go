package dist

import (
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/integrate/quad"
)

const (
	familyNormal      = "normal"
	familyTruncNormal = "truncnormal"
	paramLo           = "lo"
	paramHi           = "hi"

	// robertTail is the standardised lower bound beyond which a one-sided
	// truncated normal is drawn by Robert's exponential rejection instead of
	// inversion, which loses resolution there.
	robertTail = 5
	// normalTailLimit is the standardised bound beyond which the tail mass
	// underflows and a truncated normal is refused.
	normalTailLimit = 37
	// massNodes is the Gauss–Legendre order used for the mass of a short
	// interval, where a difference of CDF values would cancel.
	massNodes = 32
	// massShort is the interval length below which that quadrature is used.
	massShort = 1
)

// Standard normal helpers, tail-safe: the survival function and its inverse
// are computed from erfc directly rather than through 1 − Φ.
func normCDF(z float64) float64 { return 0.5 * math.Erfc(-z/math.Sqrt2) }

func normSurvival(z float64) float64 { return 0.5 * math.Erfc(z/math.Sqrt2) }

// normPolishSteps bounds the Newton refinement of the inverse. Erfcinv is
// accurate to about 1e-5 relative in the far tail and returns +Inf below
// about 1e-16, where the asymptotic seed takes over; each Newton step on
// the exact survival function squares the error.
const normPolishSteps = 12

// normSeedUpper is a starting point for S(z) = q: Erfcinv where it works,
// otherwise the tail expansion z ≈ sqrt(−2 ln q − ln 2π − 2 ln z).
func normSeedUpper(q float64) float64 {
	if z := math.Sqrt2 * math.Erfcinv(2*q); !math.IsInf(z, 0) && !math.IsNaN(z) {
		return z
	}
	z := math.Sqrt(-2 * math.Log(q))
	// bound: 3 fixed-point refinements of the expansion
	for i := 0; i < 3; i++ {
		z = math.Sqrt(-2*math.Log(q) - math.Log(2*math.Pi) - 2*math.Log(z))
	}
	return z
}

// normQuantileUpper solves S(z) = q by Newton steps on the exact survival
// function from the seed.
func normQuantileUpper(q float64) float64 {
	if q <= 0 {
		return math.Inf(1)
	}
	if q >= 1 {
		return math.Inf(-1)
	}
	z := normSeedUpper(q)
	// bound: normPolishSteps
	for i := 0; i < normPolishSteps; i++ {
		g := normSurvival(z) - q
		if math.Abs(g) <= invertTol*q {
			break
		}
		z += g / normPDF(z)
	}
	return z
}

func normQuantile(p float64) float64 { return -normQuantileUpper(p) }

func normPDF(z float64) float64 { return math.Exp(-z*z/2) / math.Sqrt(2*math.Pi) }

// Normal is the normal distribution; its support is the whole line, so it
// needs truncation before it can sit in a model slot.
type Normal struct{ mu, sigma float64 }

// NewNormal validates sigma > 0 and a finite mu.
func NewNormal(mu, sigma float64) (Normal, error) {
	if !finite(mu) {
		return Normal{}, paramError(familyNormal, paramMu, mu)
	}
	if !(sigma > 0) || !finite(sigma) {
		return Normal{}, paramError(familyNormal, paramSigma, sigma)
	}
	return Normal{mu: mu, sigma: sigma}, nil
}

func (d Normal) Fill(r *rand.Rand, dst []float64) {
	for i := range dst {
		dst[i] = d.mu + d.sigma*r.NormFloat64()
	}
}

func (d Normal) Mean() float64 { return d.mu }

func (d Normal) Var() float64 { return d.sigma * d.sigma }

func (d Normal) Support() (lo, hi float64) { return math.Inf(-1), math.Inf(1) }

func (d Normal) CDF(x float64) float64 { return normCDF((x - d.mu) / d.sigma) }

func (d Normal) Survival(x float64) float64 { return normSurvival((x - d.mu) / d.sigma) }

func (d Normal) Quantile(p float64) float64 { return d.mu + d.sigma*normQuantile(p) }

func (d Normal) QuantileUpper(q float64) float64 { return d.mu + d.sigma*normQuantileUpper(q) }

// TruncatedNormal is Normal(mu, sigma) restricted to [lo, hi]; either bound
// may be infinite.
type TruncatedNormal struct {
	mu, sigma, lo, hi float64
	a, b              float64 // standardised bounds
	mirror            bool    // the interval lies in the lower tail: sample −Z on [−b, −a]
	robert            bool    // far upper tail: Robert's exponential rejection
	alpha             float64 // Robert's rate
	pLo, pSpan        float64 // CDF form: Φ(a) and Φ(b) − Φ(a) on the sampled side
	sLo, sSpan        float64 // survival form: S(b) and S(a) − S(b)
	upperForm         bool    // use the survival form (interval beyond 0)
	mean, variance    float64
	z                 float64 // mass of [a, b]
	gx, gw            [massNodes]float64
}

// setup chooses the sampling form. The interval is reflected into the upper
// half-line where needed so that one tail-safe form serves both tails.
func (d *TruncatedNormal) setup() {
	a, b := d.a, d.b
	if b <= 0 {
		d.mirror = true
		a, b = -b, -a
	}
	switch {
	case a > robertTail:
		d.robert = true
		d.alpha = (a + math.Sqrt(a*a+4)) / 2
	case a > 0:
		d.upperForm = true
		d.sLo = normSurvival(b)
		d.sSpan = normSurvival(a) - d.sLo
	default:
		d.pLo = normCDF(a)
		d.pSpan = normCDF(b) - d.pLo
	}
	quad.Legendre{}.FixedLocations(d.gx[:], d.gw[:], -1, 1)
	d.z = d.mass(d.a, d.b)
	d.mean, d.variance = d.moments()
}

// mass is P(lo < Z ≤ hi) for the standard normal.
func (d *TruncatedNormal) mass(lo, hi float64) float64 {
	return normMass(d.gx[:], d.gw[:], lo, hi)
}

// normMass is P(lo < Z ≤ hi) for the standard normal, lo ≤ hi, with
// Gauss–Legendre nodes gx and weights gw on [−1, 1]. A short interval is
// integrated directly so that nothing cancels; a long one uses the
// tail-safe difference on the side that does not cancel.
func normMass(gx, gw []float64, lo, hi float64) float64 {
	switch {
	case hi <= lo:
		return 0
	case math.IsInf(lo, -1):
		return normCDF(hi)
	case math.IsInf(hi, 1):
		return normSurvival(lo)
	case hi-lo <= massShort:
		mid, half := (lo+hi)/2, (hi-lo)/2
		sum := 0.0
		for i := range gx {
			sum += gw[i] * normPDF(mid+half*gx[i])
		}
		return sum * half
	case lo >= 0:
		return normSurvival(lo) - normSurvival(hi)
	default:
		return normCDF(hi) - normCDF(lo)
	}
}

func (d TruncatedNormal) Fill(r *rand.Rand, dst []float64) {
	for i := range dst {
		z := d.drawStd(r)
		if d.mirror {
			z = -z
		}
		dst[i] = clamp(d.mu+d.sigma*z, d.lo, d.hi)
	}
}

func (d TruncatedNormal) Mean() float64 { return d.mean }

func (d TruncatedNormal) Var() float64 { return d.variance }

func (d TruncatedNormal) Support() (lo, hi float64) { return d.lo, d.hi }

func (d TruncatedNormal) CDF(x float64) float64 {
	if x <= d.lo {
		return 0
	}
	if x >= d.hi {
		return 1
	}
	return clamp(d.pair().cdf((x-d.mu)/d.sigma), 0, 1)
}

func (d TruncatedNormal) Survival(x float64) float64 {
	if x <= d.lo {
		return 1
	}
	if x >= d.hi {
		return 0
	}
	return clamp(d.pair().sf((x-d.mu)/d.sigma), 0, 1)
}
