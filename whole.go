package variate

import (
	"math"

	"gonum.org/v1/gonum/dsp/fourier"
)

// A trigger whose inputs are every value the sources' rounds book sums
// the whole total of the sources, S, and the period total is then
// T = S + 1{S > t}·W, with W the trigger's walk. W is independent of S:
// a node the walk reaches whose values or routes fed the sum would be
// one of its inputs, which a trigger may not reach. So T's variance has
// a closed form but for one term, E[S·1{S > t}], which the transform of
// the sum gives; and T's distribution is the sum's split at the level,
// the part above convolved with W's.

// wholeSum is the one trigger of a model that sums its whole total.
type wholeSum struct {
	trig    int
	figures sumFigures
	varHi   float64 // the upper end of Var[T], which sizes the transform
	// The walk's value and draws: first and second moments per walk.
	wMean, wSecond, dMean, dSecond float64
}

// sumsWholeTotal reports whether trigger i is the model's only trigger
// and sums every value the sources' rounds book, in a model without
// delays or repair, whose total is then the sum and the walk alone.
func (m *Compiled) sumsWholeTotal(i int) bool {
	if len(m.trigs) != 1 || m.hasDelays || m.hasRepair || len(m.countedBy(index(i))) > 0 {
		return false
	}
	for v, n := range m.nodes {
		if !m.reachedBySources(index(v)) {
			continue
		}
		// bound: functions of one node
		for f := n.fnStart; f < n.rateEnd; f++ {
			if !m.feedsInto(f, index(i)) {
				return false
			}
		}
		if n.thresh != noIndex && !m.feedsInto(n.thresh, index(i)) {
			return false
		}
	}
	return true
}

// reachedBySources reports whether a source of the model's own, not a
// trigger, reaches node v.
func (m *Compiled) reachedBySources(v int32) bool {
	for k, s := range m.srcs {
		if s.trigger == noIndex && m.reach[k][v] > 0 {
			return true
		}
	}
	return false
}

// wholeVariance sets Var[T] for a trigger summing the whole total:
//
//	Var[T] = Var[S] + p·E[W²] − p²·E[W]² + 2·E[W]·(E[S·1{S > t}] − E[S]·p)
//
// with E[S·1{S > t}] = E[S] − ∫₀^t P(S ≥ x) dx + t·p. p and the integral
// are brackets from the transform; Var[T] is the middle of the range they
// give, its upper end kept to size the transform. The draws' variance is
// bounded above, by Minkowski's inequality, as the transform needs no
// more of it.
func (m *Compiled) wholeVariance(e *Expectation, m1, m2 []float64) {
	w := m.whole
	sf := w.figures
	node := m.trigs[w.trig].node
	w.wMean, w.wSecond = m1[node], m2[node]
	d1, d2 := m.valueToGo(valueFilter{kind: filterDraws})
	w.dMean, w.dSecond = d1[node], d2[node]
	t := m.trigs[w.trig].over
	at := func(p, below float64) float64 {
		part := sf.mean - below + t*p
		return sf.variance + p*w.wSecond - p*p*w.wMean*w.wMean + 2*w.wMean*(part-sf.mean*p)
	}
	ps := []float64{sf.f.Lo, sf.f.Hi}
	if w.wMean > 0 {
		// The stationary point of the quadratic in p.
		if p := (w.wSecond - 2*w.wMean*sf.mean) / (2 * w.wMean * w.wMean); p > sf.f.Lo && p < sf.f.Hi {
			ps = append(ps, p)
		}
	}
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, p := range ps {
		for _, below := range []float64{sf.belowLo, sf.belowHi} {
			v := at(p, below)
			lo, hi = math.Min(lo, v), math.Max(hi, v)
		}
	}
	e.Var, w.varHi = (lo+hi)/2, hi
	if math.IsInf(sf.variance, 1) || math.IsInf(w.wSecond, 1) {
		e.Var, w.varHi = math.Inf(1), math.Inf(1)
	}
	pHi := sf.f.Hi
	walkDraws := math.Sqrt(math.Max(0, pHi*w.dSecond-sf.f.Lo*sf.f.Lo*w.dMean*w.dMean))
	e.DrawsVar = math.Pow(math.Sqrt(sf.drawsVar)+walkDraws, 2)
}

// walkFunctions marks the functions only a trigger's walk draws: those
// on nodes no source of the model's own reaches.
func (m *Compiled) walkFunctions() []bool {
	out := make([]bool, len(m.fns))
	for f := range m.fns {
		out[f] = !m.reachedBySources(m.fnNode[f])
	}
	return out
}

// addWalk turns the pass's bracket on the sum into one on the total. In
// either rounding the total is S + 1{S > t}·W of that rounding's lattice
// S and W, which rises with both, so a lattice law below S's (its lower
// curve) with one below W's gives a total below T's, and the upper
// curves one above. P(T ≥ x) is P(S ≥ x) and, from the sum's mass above
// the level and below x, the chance the walk makes up the difference: a
// convolution, by an FFT on twice the grid so that nothing wraps, with
// its round-off bound added.
func (r *passResult) addWalk(t *transformer, down, up *pipeline, walkDown, walkUp []complex128) {
	w := t.m.whole
	pLo := w.figures.f.Lo
	if pLo <= 0 {
		return
	}
	sLo, sHi := make([]float64, r.m), make([]float64, r.m)
	for k := range r.m {
		sLo[k], sHi[k] = r.lowerAt(r.total, k), r.upperAt(r.total, k)
	}
	tighten(sLo, sHi)
	ecDown, ecUp := down.invert(walkDown), up.invert(walkUp)
	beta := math.Max(down.walkBeta, up.walkBeta) / pLo
	mean := w.wMean + r.h*w.dMean
	sd := math.Sqrt(math.Max(0, w.wSecond-w.wMean*w.wMean)) + r.h*math.Sqrt(math.Max(0, w.dSecond-w.dMean*w.dMean))
	alias := aliasFor(mean, sd, float64(r.m)*r.h, r.theta)
	wLo, wHi := make([]float64, r.m), make([]float64, r.m)
	for k := range r.m {
		wLo[k] = math.Max(0, ecDown[k]-r.rho[k]-beta)
		wHi[k] = math.Min(1, ecUp[k]+r.rho[k]+alias+beta)
	}
	tighten(wLo, wHi)
	over := t.m.trigs[w.trig].over
	plan := fourier.NewCmplxFFT(2 * r.m)
	lower := r.withWalk(t, plan, sLo, wLo, over, -1)
	upper := r.withWalk(t, plan, sHi, wHi, over, 1)
	r.total = brackets{down: lower, up: upper}
	// The curves are bounds now: no term is left to add.
	r.rho = make([]float64, r.m)
	r.alias, r.beta = 0, 0
}

// withWalk is P(T ≥ kh) for a lattice sum with exceedance ecS and a walk
// with exceedance ecW, independent, the walk added when the sum is above
// over; side −1 takes the round-off bound off, +1 adds it.
func (r *passResult) withWalk(t *transformer, plan *fourier.CmplxFFT, ecS, ecW []float64, over float64, side float64) []float64 {
	n := 2 * r.m
	a := make([]complex128, n)
	b := make([]complex128, n)
	first := int(math.Floor(over/r.h)) + 1
	normA, normB := 0.0, 0.0
	// bound: grid points
	for j := first; j < r.m-1; j++ {
		x := math.Max(0, ecS[j]-ecS[j+1])
		a[j] = complex(x, 0)
		normA += x * x
	}
	// bound: grid points
	for k := 1; k < r.m; k++ {
		b[k] = complex(ecW[k], 0)
		normB += ecW[k] * ecW[k]
	}
	fa := plan.Coefficients(nil, a)
	fb := plan.Coefficients(nil, b)
	for i := range fa {
		fa[i] *= fb[i]
	}
	c := plan.Sequence(nil, fa)
	t.work += int64(3 * n)
	delta := roundoffSafety * unitRoundoff * (3*math.Log2(float64(n)) + 3) * math.Sqrt(normA*normB)
	out := make([]float64, r.m)
	for k := range out {
		v := ecS[k] + real(c[k])/float64(n) + side*delta
		out[k] = math.Min(1, math.Max(0, v))
	}
	return out
}
