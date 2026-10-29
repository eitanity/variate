package variate

import (
	"math"

	"code.eitanity.com/variate/dist"
)

// distribution is the subset of dist.Distribution the transform reads.
type distribution = dist.Distribution

// chooseTilt minimises α(θ) + ρ(x*; θ) over θ·M·h ∈ [0, tiltMax] by
// golden-section search, with x* at lattice index kStar, for the total's
// lattice moments.
func (t *transformer) chooseTilt(h float64, m, kStar int) float64 {
	mean, sd := t.latticeMoments(h)
	return t.chooseTiltFor(h, m, kStar, mean, sd)
}

// chooseTiltFor is chooseTilt for a quantity with the given mean and
// standard deviation (+Inf when the variance is infinite).
func (t *transformer) chooseTiltFor(h float64, m, kStar int, mean, sd float64) float64 {
	mh := float64(m) * h
	cost := func(thetaMh float64) float64 {
		theta := thetaMh / mh
		return aliasFor(mean, sd, mh, theta) + t.roundoffAt(h, m, theta, 0, kStar)
	}
	a, b := 0.0, float64(tiltMax)
	phi := (math.Sqrt(5) - 1) / 2
	x1 := b - phi*(b-a)
	x2 := a + phi*(b-a)
	f1, f2 := cost(x1), cost(x2)
	// bound: goldenIterations
	for i := 0; i < goldenIterations; i++ {
		if f1 < f2 {
			b, x2, f2 = x2, x1, f1
			x1 = b - phi*(b-a)
			f1 = cost(x1)
		} else {
			a, x1, f1 = x1, x2, f2
			x2 = a + phi*(b-a)
			f2 = cost(x2)
		}
	}
	return (a + b) / 2 / mh
}

// latticeMoments bounds the mean and standard deviation of T_up from the
// exact moments: every draw rounds up by at most h.
func (t *transformer) latticeMoments(h float64) (mean, sd float64) {
	e := t.e
	mean = e.Mean + h*e.DrawsMean
	if math.IsInf(e.Var, 1) || math.IsNaN(e.Var) {
		return mean, math.Inf(1)
	}
	return mean, math.Sqrt(e.Var) + h*math.Sqrt(e.DrawsVar)
}

// aliasBound is α = r^M · P̄(T_up ≥ M·h), with Cantelli (or Markov) on
// the lattice moments bounded from the exact ones.
func (t *transformer) aliasBound(h float64, m int, theta float64) float64 {
	mean, sd := t.latticeMoments(h)
	return aliasFor(mean, sd, float64(m)*h, theta)
}

// aliasFor is e^(−θ·Mh) times the Cantelli bound on P(X ≥ Mh), or the
// Markov bound when the variance is infinite.
func aliasFor(mean, sd, mh, theta float64) float64 {
	var tail float64
	switch {
	case math.IsInf(sd, 1) || math.IsNaN(sd):
		tail = math.Min(1, mean/mh)
	case mh <= mean:
		tail = 1
	default:
		d := mh - mean
		tail = sd * sd / (sd*sd + d*d)
	}
	return math.Exp(-theta*mh) * tail
}

// roundoffDelta is the 2-norm error estimate of the tilted masses.
func (t *transformer) roundoffDelta(m int, eps float64) float64 {
	maxLam := 1.0
	for _, l := range t.lam {
		maxLam = math.Max(maxLam, l)
	}
	depth := float64(t.m.maxPath + len(t.m.srcs) + 2)
	return roundoffSafety*unitRoundoff*(math.Log2(float64(m))+depth)*maxLam + eps
}

// roundoffAt is ρ(x) at lattice index k for a tilt.
func (t *transformer) roundoffAt(h float64, m int, theta, eps float64, k int) float64 {
	delta := t.roundoffDelta(m, eps)
	if theta == 0 {
		return delta * math.Sqrt(float64(k+1))
	}
	a := 2 * theta * h
	return delta * math.Sqrt(math.Expm1(a*float64(k+1))/math.Expm1(a))
}

// roundoff is ρ at every lattice point.
func (t *transformer) roundoff(h float64, m int, theta, eps float64) []float64 {
	rho := make([]float64, m)
	for k := range rho {
		rho[k] = t.roundoffAt(h, m, theta, eps, k)
	}
	return rho
}

// lower and upper curves of a bracket.
func (r *passResult) lowerAt(b brackets, k int) float64 {
	return math.Max(0, b.down[k]-r.rho[k]-r.beta)
}

func (r *passResult) upperAt(b brackets, k int) float64 {
	return math.Min(1, b.up[k]+r.rho[k]+r.alias+r.beta)
}

// withinTol reports whether the total's bracket is within tol up to x*.
func (r *passResult) withinTol(xStar, tol, tailP float64) bool {
	kStar := min(int(math.Ceil(xStar/r.h)), r.m-1)
	// bound: grid points up to x*
	for k := 0; k <= kStar; k++ {
		if !resolved(r.lowerAt(r.total, k), r.upperAt(r.total, k), tol, tailP) {
			return false
		}
	}
	return true
}

// resolved reports whether a bracket is within tol of its upper bound,
// or of the tail level when the upper bound is already below it.
func resolved(lo, up, tol, tailP float64) bool {
	return up-lo <= tol*math.Max(up, 1-tailP)
}

// dominant names the term that keeps the bracket open.
func (r *passResult) dominant(xStar float64) string {
	kStar := min(int(math.Ceil(xStar/r.h)), r.m-1)
	var gap, rho float64
	// bound: grid points up to x*
	for k := 0; k <= kStar; k++ {
		gap = math.Max(gap, r.total.up[k]-r.total.down[k])
		rho = math.Max(rho, r.rho[k])
	}
	switch {
	case gap >= rho && gap >= r.alias:
		return "the discretisation gap stays above Tol at MaxGrid: the range is too wide for a linear grid"
	case r.alias >= rho:
		return "the wrap-around bound stays above Tol at MaxGrid: the tail is too heavy for this method"
	default:
		return "the round-off estimate stays above Tol at MaxGrid: the tail is too heavy for this method"
	}
}

// fill writes the bracket parts into the result.
func (r *passResult) fill(out *Transformed) {
	out.H, out.M, out.Theta, out.PZero = r.h, r.m, r.theta, r.pzero
	out.Total = r.bounds(r.total)
	for _, c := range r.cats {
		out.ByCategory = append(out.ByCategory, r.bounds(c))
	}
	for _, s := range r.srcs {
		out.BySource = append(out.BySource, r.bounds(s))
	}
}

func (r *passResult) bounds(b brackets) Bounds {
	out := Bounds{H: r.h, Lower: make([]float64, r.m), Upper: make([]float64, r.m), DiscGap: make([]float64, r.m), Roundoff: r.rho, Alias: r.alias}
	// bound: grid points
	for k := 0; k < r.m; k++ {
		out.Lower[k] = r.lowerAt(b, k)
		out.Upper[k] = r.upperAt(b, k)
		out.DiscGap[k] = b.up[k] - b.down[k]
	}
	tighten(out.Lower, out.Upper)
	return out
}

// tighten makes the bounds monotone, which the true curve is: an upper
// bound at x also bounds every point beyond it, so the running minimum
// from the left is an upper bound too, and the running maximum from the
// right a lower bound. The round-off term grows with x, so without this
// the upper bound can rise where the curve is flat.
func tighten(lower, upper []float64) {
	for k := 1; k < len(upper); k++ {
		upper[k] = math.Min(upper[k], upper[k-1])
	}
	for k := len(lower) - 2; k >= 0; k-- {
		lower[k] = math.Max(lower[k], lower[k+1])
	}
}

// selfCheck: the computed means and total masses agree with the exact
// lattice values up to the wrapped mass and round-off; the zero mass of
// the upper pipeline is the exact P(T = 0).
func (r *passResult) selfCheck(t *transformer, meanUp, meanDown, dropMean float64) string {
	e := t.e
	top := r.rho[r.m-1] * float64(r.m) * r.h
	if !math.IsInf(e.Var, 0) && !math.IsNaN(e.Var) {
		second := e.Var + e.Mean*e.Mean
		wrap := math.Sqrt(second * r.alias / math.Exp(-r.theta*float64(r.m)*r.h))
		for _, c := range []struct {
			b    brackets
			want float64
		}{{r.total, meanUp}} {
			got := curveMean(c.b.up, r.h)
			if got > c.want+top+1e-9*c.want || got < c.want-wrap-top-1e-9*c.want {
				return "self-check: the computed upper mean " + ftoa(got) + " does not agree with the lattice mean " + ftoa(c.want)
			}
		}
		if meanDown > e.Mean+1e-9*e.Mean || meanUp+dropMean < e.Mean-1e-9*e.Mean {
			return "self-check: the lattice means " + ftoa(meanDown) + ", " + ftoa(meanUp) + " do not bracket the exact mean " + ftoa(e.Mean)
		}
	}
	if r.pzero > e.PZero()+r.rho[0]+r.alias+1e-12 || r.pzero < e.PZero()-r.rho[0]-1e-12 {
		return "self-check: P(T = 0) " + ftoa(r.pzero) + " differs from the exact " + ftoa(e.PZero())
	}
	return ""
}

// curveMean is Σ_k k·h·mass_k from an exceedance curve.
func curveMean(ec []float64, h float64) float64 {
	sum := 0.0
	for k := 1; k < len(ec); k++ {
		sum += ec[k] * h
	}
	return sum
}
