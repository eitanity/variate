package dist

import (
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/mathext"
)

const (
	familyBernoulli = "bernoulli"
	familyBinomial  = "binomial"
	paramP          = "p"
	paramN          = "n"

	// btpeThreshold is n·min(p, 1−p) at which the Binomial sampler switches
	// from inversion to BTPE.
	btpeThreshold = 10
	// binomialInversionSteps bounds the sequential search; below the
	// threshold the mass beyond it is negligible.
	binomialInversionSteps = 256
)

// Bernoulli is a single trial with success probability p.
type Bernoulli struct{ p float64 }

// NewBernoulli validates 0 ≤ p ≤ 1.
func NewBernoulli(p float64) (Bernoulli, error) {
	if !(p >= 0 && p <= 1) {
		return Bernoulli{}, paramError(familyBernoulli, paramP, p)
	}
	return Bernoulli{p: p}, nil
}

func (d Bernoulli) Fill(r *rand.Rand, dst []float64) {
	for i := range dst {
		if r.Float64() < d.p {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

func (d Bernoulli) Mean() float64 { return d.p }

func (d Bernoulli) Var() float64 { return d.p * (1 - d.p) }

func (d Bernoulli) Support() (lo, hi float64) { return 0, 1 }

func (d Bernoulli) CDF(x float64) float64 {
	switch {
	case x < 0:
		return 0
	case x < 1:
		return 1 - d.p
	default:
		return 1
	}
}

func (d Bernoulli) Survival(x float64) float64 {
	switch {
	case x < 0:
		return 1
	case x < 1:
		return d.p
	default:
		return 0
	}
}

func (d Bernoulli) Quantile(p float64) float64 {
	if p <= 1-d.p {
		return 0
	}
	return 1
}

func (d Bernoulli) QuantileUpper(q float64) float64 {
	if q >= d.p {
		return 0
	}
	return 1
}

func (d Bernoulli) PMF(x float64) float64 {
	switch x {
	case 0:
		return 1 - d.p
	case 1:
		return d.p
	default:
		return 0
	}
}

// btpe holds the constants of the BTPE method for one (n, r) with r ≤ ½.
type btpe struct {
	n, r, q                   float64
	m, p1, xm, xl, xr         float64
	c, laml, lamr, p2, p3, p4 float64
	s2                        float64 // n·r·q
	a, sq                     float64 // squeeze: a = s·(n+1), sq = r/q
}

func newBTPE(n, r float64) btpe {
	q := 1 - r
	fm := n*r + r
	m := math.Floor(fm)
	s2 := n * r * q
	p1 := math.Floor(2.195*math.Sqrt(s2)-4.6*q) + 0.5
	xm := m + 0.5
	xl := xm - p1
	xr := xm + p1
	c := 0.134 + 20.5/(15.3+m)
	al := (fm - xl) / (fm - xl*r)
	laml := al * (1 + al/2)
	al = (xr - fm) / (xr * q)
	lamr := al * (1 + al/2)
	p2 := p1 * (1 + 2*c)
	p3 := p2 + c/laml
	p4 := p3 + c/lamr
	return btpe{
		n: n, r: r, q: q, m: m, p1: p1, xm: xm, xl: xl, xr: xr,
		c: c, laml: laml, lamr: lamr, p2: p2, p3: p3, p4: p4, s2: s2,
		a: r / q * (n + 1), sq: r / q,
	}
}

// candidate draws a proposal y and the uniform v it must pass; ok is false
// when the proposal was rejected outright.
func (b btpe) candidate(r *rand.Rand) (y, v float64, ok bool) {
	u := r.Float64() * b.p4
	v = r.Float64()
	switch {
	case u <= b.p1:
		return math.Floor(b.xm - b.p1*v + u), -1, true // -1 marks the triangle: accepted
	case u <= b.p2:
		x := b.xl + (u-b.p1)/b.c
		v = v*b.c + 1 - math.Abs(b.m-x+0.5)/b.p1
		if v > 1 || v <= 0 {
			return 0, 0, false
		}
		return math.Floor(x), v, true
	case u <= b.p3:
		y = math.Floor(b.xl + math.Log(v)/b.laml)
		if y < 0 {
			return 0, 0, false
		}
		return y, v * (u - b.p2) * b.laml, true
	default:
		y = math.Floor(b.xr - math.Log(v)/b.lamr)
		if y > b.n {
			return 0, 0, false
		}
		return y, v * (u - b.p3) * b.lamr, true
	}
}

// accept runs the squeeze near the mode or the Stirling-corrected log test
// in the tails.
func (b btpe) accept(y, v float64) bool {
	k := math.Abs(y - b.m)
	if k <= 20 || k >= b.s2/2-1 {
		f := 1.0
		switch {
		case b.m < y:
			// bound: at most 20 steps or the far tail, both bounded by n
			for i := b.m + 1; i <= y; i++ {
				f *= b.a/i - b.sq
			}
		case b.m > y:
			// bound: as above
			for i := y + 1; i <= b.m; i++ {
				f /= b.a/i - b.sq
			}
		}
		return v <= f
	}
	rho := (k / b.s2) * ((k*(k/3+0.625)+1.0/6)/b.s2 + 0.5)
	t := -k * k / (2 * b.s2)
	alv := math.Log(v)
	if alv < t-rho {
		return true
	}
	if alv > t+rho {
		return false
	}
	x1, f1 := y+1, b.m+1
	z, w := b.n+1-b.m, b.n-y+1
	x2, f2, z2, w2 := x1*x1, f1*f1, z*z, w*w
	lhs := b.xm*math.Log(f1/x1) + (b.n-b.m+0.5)*math.Log(z/w) + (y-b.m)*math.Log(w*b.r/(x1*b.q))
	lhs += (13860-(462-(132-(99-140/f2)/f2)/f2)/f2)/f1/166320 +
		(13860-(462-(132-(99-140/z2)/z2)/z2)/z2)/z/166320 +
		(13860-(462-(132-(99-140/x2)/x2)/x2)/x2)/x1/166320 +
		(13860-(462-(132-(99-140/w2)/w2)/w2)/w2)/w/166320
	return alv <= lhs
}

func (b btpe) draw(r *rand.Rand) float64 {
	for try := 0; try < maxRejections; try++ {
		y, v, ok := b.candidate(r)
		if !ok {
			continue
		}
		if v < 0 || b.accept(y, v) {
			return y
		}
	}
	return binomialQuantile(b.n, b.r, unit(r))
}

// binomialInversion draws by sequential search from P(0) = q^n.
func binomialInversion(r *rand.Rand, n, p, q float64) float64 {
	u := r.Float64()
	pr := math.Pow(q, n)
	s := pr
	k := 0.0
	ratio := p / q
	// bound: binomialInversionSteps
	for step := 0; u > s && step < binomialInversionSteps && k < n; step++ {
		k++
		pr *= (n - k + 1) / k * ratio
		s += pr
	}
	return k
}

// binomialCDF is P(K ≤ k) = I_{1−p}(n − k, k + 1).
func binomialCDF(n, p, k float64) float64 {
	k = math.Floor(k)
	switch {
	case k < 0:
		return 0
	case k >= n:
		return 1
	default:
		return mathext.RegIncBeta(n-k, k+1, 1-p)
	}
}

// binomialSurvival is P(K > k) = I_p(k + 1, n − k).
func binomialSurvival(n, p, k float64) float64 {
	k = math.Floor(k)
	switch {
	case k < 0:
		return 1
	case k >= n:
		return 0
	default:
		return mathext.RegIncBeta(k+1, n-k, p)
	}
}

// binomialQuantile is the smallest k with CDF(k) ≥ u, by bisection.
func binomialQuantile(n, p, u float64) float64 {
	lo, hi := -1.0, n
	// Invariant: CDF(lo) < u ≤ CDF(hi).
	// bound: quantileSteps bisections on [0, n], n < 2^53
	for i := 0; i < quantileSteps && hi-lo > 1; i++ {
		mid := math.Floor((lo + hi) / 2)
		if binomialCDF(n, p, mid) >= u {
			hi = mid
		} else {
			lo = mid
		}
	}
	return hi
}

// Binomial counts successes in n trials with probability p.
type Binomial struct {
	n, p    float64
	reflect bool    // p > ½: draw with 1−p and reflect
	r, q    float64 // working probability (≤ ½) and its complement
	useBTPE bool
	b       btpe
}

// NewBinomial validates n ≥ 0 integer and 0 ≤ p ≤ 1.
func NewBinomial(n, p float64) (Binomial, error) {
	if !(n >= 0) || n != math.Floor(n) || math.IsInf(n, 0) {
		return Binomial{}, paramError(familyBinomial, paramN, n)
	}
	if !(p >= 0 && p <= 1) {
		return Binomial{}, paramError(familyBinomial, paramP, p)
	}
	d := Binomial{n: n, p: p, r: p, q: 1 - p}
	if p > 0.5 {
		d.reflect = true
		d.r, d.q = 1-p, p
	}
	if n*d.r >= btpeThreshold {
		d.useBTPE = true
		d.b = newBTPE(n, d.r)
	}
	return d, nil
}

func (d Binomial) Fill(r *rand.Rand, dst []float64) {
	for i := range dst {
		var k float64
		switch {
		case d.n == 0 || d.r == 0:
			k = 0
		case d.useBTPE:
			k = d.b.draw(r)
		default:
			k = binomialInversion(r, d.n, d.r, d.q)
		}
		if d.reflect {
			k = d.n - k
		}
		dst[i] = k
	}
}

func (d Binomial) Mean() float64 { return d.n * d.p }

func (d Binomial) Var() float64 { return d.n * d.p * (1 - d.p) }

func (d Binomial) Support() (lo, hi float64) { return 0, d.n }

func (d Binomial) CDF(x float64) float64 { return binomialCDF(d.n, d.p, x) }

func (d Binomial) Survival(x float64) float64 { return binomialSurvival(d.n, d.p, x) }

func (d Binomial) Quantile(p float64) float64 {
	if p <= 0 {
		return 0
	}
	return binomialQuantile(d.n, d.p, p)
}

// QuantileUpper is the smallest k with Survival(k) ≤ q.
func (d Binomial) QuantileUpper(q float64) float64 {
	if q >= 1 {
		return 0
	}
	lo, hi := -1.0, d.n
	// bound: quantileSteps bisections
	for i := 0; i < quantileSteps && hi-lo > 1; i++ {
		mid := math.Floor((lo + hi) / 2)
		if binomialSurvival(d.n, d.p, mid) <= q {
			hi = mid
		} else {
			lo = mid
		}
	}
	return hi
}

func (d Binomial) PMF(x float64) float64 {
	if x < 0 || x > d.n || x != math.Floor(x) {
		return 0
	}
	switch d.p {
	case 0:
		return boolToFloat(x == 0)
	case 1:
		return boolToFloat(x == d.n)
	}
	return math.Exp(lgamma(d.n+1) - lgamma(x+1) - lgamma(d.n-x+1) + x*math.Log(d.p) + (d.n-x)*math.Log1p(-d.p))
}

func boolToFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}
