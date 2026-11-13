package dist

import (
	"math"
	"sort"
)

// halfLog2Pi is ½·ln(2π), the normal density's constant.
const halfLog2Pi = 0.918938533204672741780329736406

// outside reports x off [lo, hi], with NaN and the infinities never inside.
func outside(x, lo, hi float64) bool {
	return !(x >= lo && x <= hi) || math.IsInf(x, 0)
}

// notCount reports x off the non-negative integers.
func notCount(x float64) bool {
	return outside(x, 0, math.Inf(1)) || x != math.Floor(x)
}

// hasMass reports a positive probability from its logarithm.
func hasMass(logMass float64) bool { return !math.IsInf(logMass, -1) }

// edgeLog is the log density at a support end where the density behaves
// like a power with exponent shape − 1: it diverges below shape 1,
// vanishes above it, and is atOne at it.
func edgeLog(shape, atOne float64) float64 {
	switch {
	case shape < 1:
		return math.Inf(1)
	case shape > 1:
		return math.Inf(-1)
	default:
		return atOne
	}
}

func logNormalKernel(z float64) float64 { return -z*z/2 - halfLog2Pi }

func (d Constant) LogDensity(x float64) float64 {
	if x == d.c {
		return 0
	}
	return math.Inf(-1)
}

func (d Uniform) LogDensity(x float64) float64 {
	if outside(x, d.a, d.b) {
		return math.Inf(-1)
	}
	return -math.Log(d.span)
}

func (d LogUniform) LogDensity(x float64) float64 {
	if outside(x, d.a, d.b) {
		return math.Inf(-1)
	}
	return -math.Log(x) - math.Log(d.logRatio)
}

func (d Triangular) LogDensity(x float64) float64 {
	switch {
	case outside(x, d.low, d.high):
		return math.Inf(-1)
	case x < d.mode:
		return math.Log(2 * (x - d.low) / d.cL)
	case x > d.mode:
		return math.Log(2 * (d.high - x) / d.cH)
	default:
		return math.Log(2 / d.span)
	}
}

func (d Exponential) LogDensity(x float64) float64 {
	if outside(x, 0, math.Inf(1)) {
		return math.Inf(-1)
	}
	return -math.Log(d.mean) - x/d.mean
}

func (d Lognormal) LogDensity(x float64) float64 {
	if outside(x, 0, math.Inf(1)) || x == 0 {
		return math.Inf(-1)
	}
	lx := math.Log(x)
	return -lx - math.Log(d.sigma) + logNormalKernel((lx-d.mu)/d.sigma)
}

func (d Weibull) LogDensity(x float64) float64 {
	switch {
	case outside(x, 0, math.Inf(1)):
		return math.Inf(-1)
	case x == 0:
		return edgeLog(d.shape, -math.Log(d.scale))
	}
	u := x / d.scale
	return math.Log(d.shape) - math.Log(d.scale) + (d.shape-1)*math.Log(u) - math.Pow(u, d.shape)
}

// LogDensity is ln β − ln x + l − 2·ln(1 + e^l) with l the log odds,
// the last term as a softplus so that neither tail overflows.
func (d LogLogistic) LogDensity(x float64) float64 {
	switch {
	case outside(x, 0, math.Inf(1)):
		return math.Inf(-1)
	case x == 0:
		return edgeLog(d.shape, -math.Log(d.scale))
	}
	l := d.logOdds(x)
	return math.Log(d.shape) - math.Log(x) + l - 2*softplus(l)
}

// softplus is ln(1 + e^l), exact where e^l overflows.
func softplus(l float64) float64 {
	if l > 0 {
		return l + math.Log1p(math.Exp(-l))
	}
	return math.Log1p(math.Exp(l))
}

func (d Gamma) LogDensity(x float64) float64 {
	switch {
	case outside(x, 0, math.Inf(1)):
		return math.Inf(-1)
	case x == 0:
		return edgeLog(d.shape, -math.Log(d.scale))
	}
	u := x / d.scale
	return (d.shape-1)*math.Log(u) - u - lgamma(d.shape) - math.Log(d.scale)
}

func (d Beta) LogDensity(x float64) float64 {
	switch {
	case outside(x, 0, 1):
		return math.Inf(-1)
	case x == 0:
		return edgeLog(d.alpha, math.Log(d.beta))
	case x == 1:
		return edgeLog(d.beta, math.Log(d.alpha))
	}
	lb := lgamma(d.alpha) + lgamma(d.beta) - lgamma(d.alpha+d.beta)
	return (d.alpha-1)*math.Log(x) + (d.beta-1)*math.Log1p(-x) - lb
}

func (d PERT) LogDensity(x float64) float64 {
	if outside(x, d.low, d.high) {
		return math.Inf(-1)
	}
	return d.beta.LogDensity(clamp((x-d.low)/d.span, 0, 1)) - math.Log(d.span)
}

func (d LogPERT) LogDensity(x float64) float64 {
	if outside(x, d.low, d.high) {
		return math.Inf(-1)
	}
	v := clamp(math.Log10(x), d.lp.low, d.lp.high)
	return d.lp.LogDensity(v) - math.Log(x) - math.Log(math.Ln10)
}

func (d Normal) LogDensity(x float64) float64 {
	if outside(x, math.Inf(-1), math.Inf(1)) {
		return math.Inf(-1)
	}
	return logNormalKernel((x-d.mu)/d.sigma) - math.Log(d.sigma)
}

func (d TruncatedNormal) LogDensity(x float64) float64 {
	if outside(x, d.lo, d.hi) {
		return math.Inf(-1)
	}
	return logNormalKernel((x-d.mu)/d.sigma) - math.Log(d.sigma) - math.Log(d.z)
}

func (d Pareto) LogDensity(x float64) float64 {
	if outside(x, d.xm, math.Inf(1)) {
		return math.Inf(-1)
	}
	return math.Log(d.alpha) - math.Log(x) - d.alpha*math.Log(x/d.xm)
}

func (d GPD) LogDensity(x float64) float64 {
	if outside(x, d.mu, d.hi) {
		return math.Inf(-1)
	}
	y := (x - d.mu) / d.sigma
	switch d.xi {
	case 0:
		return -math.Log(d.sigma) - y
	case -1:
		return -math.Log(d.sigma) // the uniform on the support
	}
	return -math.Log(d.sigma) - (1/d.xi+1)*math.Log1p(math.Max(d.xi*y, -1))
}

func (d Bernoulli) LogDensity(x float64) float64 {
	switch x {
	case 0:
		return math.Log1p(-d.p)
	case 1:
		return math.Log(d.p)
	default:
		return math.Inf(-1)
	}
}

func (d Binomial) LogDensity(x float64) float64 {
	if notCount(x) || x > d.n {
		return math.Inf(-1)
	}
	switch d.p {
	case 0:
		return logIf(x == 0)
	case 1:
		return logIf(x == d.n)
	}
	return lgamma(d.n+1) - lgamma(x+1) - lgamma(d.n-x+1) + x*math.Log(d.p) + (d.n-x)*math.Log1p(-d.p)
}

// logIf is the log mass of a certain outcome.
func logIf(certain bool) float64 {
	if certain {
		return 0
	}
	return math.Inf(-1)
}

func (d Geometric) LogDensity(x float64) float64 {
	switch {
	case notCount(x):
		return math.Inf(-1)
	case x == 0:
		return math.Log(d.p)
	}
	return math.Log(d.p) + x*d.logQ
}

func (d NegativeBinomial) LogDensity(x float64) float64 {
	if notCount(x) {
		return math.Inf(-1)
	}
	return lgamma(x+d.r) - lgamma(x+1) - lgamma(d.r) + d.r*math.Log(d.p) + x*math.Log1p(-d.p)
}

func (d Poisson) LogDensity(x float64) float64 {
	if notCount(x) {
		return math.Inf(-1)
	}
	if d.lambda == 0 {
		return logIf(x == 0)
	}
	return x*math.Log(d.lambda) - d.lambda - lgamma(x+1)
}

func (d Categorical) LogDensity(x float64) float64 {
	i := sort.SearchFloat64s(d.sortedV, x)
	if i == len(d.sortedV) || d.sortedV[i] != x {
		return math.Inf(-1)
	}
	return math.Log(d.mass[i])
}

// LogDensity is the parent's at the unscaled value, less ln f where the
// value is a density: scaling leaves a mass as it is.
func (d Scaled) LogDensity(x float64) float64 {
	z, atom := d.parentAt(x)
	if atom {
		return d.d.LogDensity(z)
	}
	return d.d.LogDensity(z) - math.Log(d.f)
}

// parentAt returns the parent's value behind x, and whether it is an
// atom: one of the floats f·z = x if any is, else the largest of them.
func (d Scaled) parentAt(x float64) (z float64, atom bool) {
	z = d.unscale(x)
	c := z
	// bound: unscaleSteps
	for i := 0; i < unscaleSteps && d.f*c == x; i++ {
		if d.d.IsAtom(c) {
			return c, true
		}
		c = math.Nextafter(c, math.Inf(-1))
	}
	return z, false
}

func (d Truncated) LogDensity(x float64) float64 {
	if outside(x, d.lo, d.hi) || x == d.lo && !d.closed && d.d.IsAtom(x) {
		return math.Inf(-1)
	}
	return d.d.LogDensity(x) - math.Log(d.mass)
}

// LogDensity is the parent's below the cap and the log of the mass the
// cap collects at it.
func (d Capped) LogDensity(x float64) float64 {
	if x > d.c {
		return math.Inf(-1)
	}
	if x == d.c && d.atom > 0 {
		mass := d.atom
		if d.d.IsAtom(x) {
			mass += math.Exp(d.d.LogDensity(x))
		}
		return math.Log(mass)
	}
	return d.d.LogDensity(x)
}

func (d Constant) IsAtom(x float64) bool         { return x == d.c }
func (d Bernoulli) IsAtom(x float64) bool        { return hasMass(d.LogDensity(x)) }
func (d Binomial) IsAtom(x float64) bool         { return hasMass(d.LogDensity(x)) }
func (d Geometric) IsAtom(x float64) bool        { return hasMass(d.LogDensity(x)) }
func (d NegativeBinomial) IsAtom(x float64) bool { return hasMass(d.LogDensity(x)) }
func (d Poisson) IsAtom(x float64) bool          { return hasMass(d.LogDensity(x)) }
func (d Categorical) IsAtom(x float64) bool      { return hasMass(d.LogDensity(x)) }
func (Uniform) IsAtom(float64) bool              { return false }
func (LogUniform) IsAtom(float64) bool           { return false }
func (Triangular) IsAtom(float64) bool           { return false }
func (Exponential) IsAtom(float64) bool          { return false }
func (Lognormal) IsAtom(float64) bool            { return false }
func (Weibull) IsAtom(float64) bool              { return false }
func (LogLogistic) IsAtom(float64) bool          { return false }
func (Gamma) IsAtom(float64) bool                { return false }
func (Beta) IsAtom(float64) bool                 { return false }
func (PERT) IsAtom(float64) bool                 { return false }
func (LogPERT) IsAtom(float64) bool              { return false }
func (Normal) IsAtom(float64) bool               { return false }
func (TruncatedNormal) IsAtom(float64) bool      { return false }
func (Pareto) IsAtom(float64) bool               { return false }
func (GPD) IsAtom(float64) bool                  { return false }

func (d Scaled) IsAtom(x float64) bool {
	_, atom := d.parentAt(x)
	return atom
}

func (d Truncated) IsAtom(x float64) bool {
	if outside(x, d.lo, d.hi) || x == d.lo && !d.closed {
		return false
	}
	return d.d.IsAtom(x)
}

func (d Capped) IsAtom(x float64) bool {
	if x > d.c {
		return false
	}
	return x == d.c && d.atom > 0 || d.d.IsAtom(x)
}

func (d Exponential) LogSurvival(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return -x / d.mean
}

func (d Weibull) LogSurvival(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return -math.Pow(x/d.scale, d.shape)
}

func (d LogLogistic) LogSurvival(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return -softplus(d.logOdds(x))
}

func (d Pareto) LogSurvival(x float64) float64 {
	if x <= d.xm {
		return 0
	}
	return -d.alpha * math.Log(x/d.xm)
}

func (d GPD) LogSurvival(x float64) float64 {
	switch {
	case x <= d.mu:
		return 0
	case x >= d.hi:
		return math.Inf(-1)
	}
	y := (x - d.mu) / d.sigma
	if d.xi == 0 {
		return -y
	}
	return -math.Log1p(d.xi*y) / d.xi
}

func (d Geometric) LogSurvival(x float64) float64 {
	if x < 0 {
		return 0
	}
	return (math.Floor(x) + 1) * d.logQ
}

func (d Constant) LogSurvival(x float64) float64         { return math.Log(d.Survival(x)) }
func (d Uniform) LogSurvival(x float64) float64          { return math.Log(d.Survival(x)) }
func (d LogUniform) LogSurvival(x float64) float64       { return math.Log(d.Survival(x)) }
func (d Triangular) LogSurvival(x float64) float64       { return math.Log(d.Survival(x)) }
func (d Lognormal) LogSurvival(x float64) float64        { return math.Log(d.Survival(x)) }
func (d Gamma) LogSurvival(x float64) float64            { return math.Log(d.Survival(x)) }
func (d Beta) LogSurvival(x float64) float64             { return math.Log(d.Survival(x)) }
func (d PERT) LogSurvival(x float64) float64             { return math.Log(d.Survival(x)) }
func (d LogPERT) LogSurvival(x float64) float64          { return math.Log(d.Survival(x)) }
func (d Normal) LogSurvival(x float64) float64           { return math.Log(d.Survival(x)) }
func (d TruncatedNormal) LogSurvival(x float64) float64  { return math.Log(d.Survival(x)) }
func (d Bernoulli) LogSurvival(x float64) float64        { return math.Log(d.Survival(x)) }
func (d Binomial) LogSurvival(x float64) float64         { return math.Log(d.Survival(x)) }
func (d NegativeBinomial) LogSurvival(x float64) float64 { return math.Log(d.Survival(x)) }
func (d Poisson) LogSurvival(x float64) float64          { return math.Log(d.Survival(x)) }
func (d Categorical) LogSurvival(x float64) float64      { return math.Log(d.Survival(x)) }

func (d Scaled) LogSurvival(x float64) float64 { return d.d.LogSurvival(d.unscale(x)) }

func (d Truncated) LogSurvival(x float64) float64 {
	if math.IsInf(d.hi, 1) && x > d.lo {
		return d.d.LogSurvival(x) - math.Log(d.mass)
	}
	return math.Log(d.Survival(x))
}

func (d Capped) LogSurvival(x float64) float64 {
	if x >= d.c {
		return math.Inf(-1)
	}
	return d.d.LogSurvival(x)
}
