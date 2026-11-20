package variate

import (
	"slices"

	"code.eitanity.com/variate/dist"
)

// A factor on values joins the transform's mixture over the period
// draws as a rate factor does: at each of its nodes g the scaled
// functions take their laws times g, discretised on the pass's lattice,
// and their lattice means, at that node's draws per period, are mixed
// with the transforms by the same weights.

// valueFactorReason is why the transform does not take the model's
// value factors, or "": a factor that can be negative, or one scaling a
// value drawn once a period, a value a threshold tests, or a value of a
// model whose trigger sums the whole total.
func (m *Compiled) valueFactorReason() string {
	factors := m.valueFactors()
	if len(factors) == 0 {
		return ""
	}
	for _, i := range factors {
		if lo, _ := m.pools[m.factorPools[i]].Support(); lo < 0 {
			return "a factor that scales a function's value can be negative"
		}
	}
	if m.whole != nil {
		return "a factor scales a function's value in a model whose trigger sums the whole total"
	}
	for _, f := range m.fns {
		if f.factor != noIndex && f.pool != noIndex && m.poolPeriod[f.pool] != noIndex {
			return "a factor scales a function's value drawn once a period"
		}
	}
	for _, n := range m.nodes {
		if n.thresh != noIndex && m.fns[n.thresh].factor != noIndex {
			return "a factor scales a value a threshold tests"
		}
	}
	for _, o := range m.outs {
		if o.fn != noIndex && m.fns[o.fn].factor != noIndex {
			return "a factor scales a value a threshold tests"
		}
	}
	return ""
}

// valueFactorQuantities appends to qs each factor on values that no
// rate names, as a factor quantity of the mixture.
func (m *Compiled) valueFactorQuantities(qs []thetaQuantity) []thetaQuantity {
	for _, i := range m.valueFactors() {
		pool := m.factorPools[i]
		if !slices.ContainsFunc(qs, func(q thetaQuantity) bool { return q.factor && q.cycle == nil && q.pool == pool }) {
			qs = append(qs, thetaQuantity{pool: pool, fn: noIndex, factor: true})
		}
	}
	return qs
}

// rateShares fills, for each scaled function, each source's share of
// its draws per period, and nil for the rest.
func (t *transformer) rateShares() {
	m := t.m
	t.shares = make([][]float64, len(m.fns))
	for f, fn := range m.fns {
		if fn.factor == noIndex {
			continue
		}
		v := m.fnNode[f]
		sh := make([]float64, len(m.srcs))
		sum := 0.0
		for k := range m.srcs {
			sh[k] = m.countMean[k] * m.reach[k][v]
			sum += sh[k]
		}
		for k := range sh {
			if sum > 0 {
				sh[k] /= sum
			}
		}
		t.shares[f] = sh
	}
}

// drawsGiven is scaled function f's draws per period at the period
// draws in drawn: its mean draws, each source's share scaled by its
// rate's factor at drawn over the factor's mean.
func (t *transformer) drawsGiven(f int32, drawn []float64) float64 {
	m := t.m
	ratio := 0.0
	for k, sh := range t.shares[f] {
		if sh == 0 {
			continue
		}
		r := 1.0
		if mean := m.factorAt(k, m.poolMean); mean > 0 {
			r = m.factorAt(k, drawn) / mean
		}
		ratio += sh * r
	}
	return t.draws[f] * ratio
}

// fnLaw is function f's law at the period draws: its own, or times its
// factor's draw; ok is false for a factor drawn at 0, when the value is
// 0.
func (p *pipeline) fnLaw(f int32) (d distribution, g float64, ok bool) {
	m := p.t.m
	d = m.pools[m.fns[f].pool]
	if m.fns[f].factor == noIndex {
		return d, 1, true
	}
	g = p.mix.drawn[m.fns[f].factor]
	if g == 1 {
		return d, g, true
	}
	if !(g > 0) {
		return d, g, false
	}
	s, err := dist.NewScaled(d, g)
	if err != nil {
		return d, g, false
	}
	return s, g, true
}

// fnLattice is a function's lattice per draw: its mean, its dropped
// mass, and the bound on the mean its dropped tail carries.
type fnLattice struct {
	mean, dropped, drop float64
}

// latticeAcct is a lattice mean, the bound on the mean the dropped
// tails carry and the dropped mass, and the parts of each the walk's
// functions carry.
type latticeAcct struct {
	mean, drop, beta, walkMean, walkDrop, walkBeta float64
}

// take adds function f's lattice at the given draws per period.
func (a *latticeAcct) take(t *transformer, f int32, draws float64, l fnLattice) {
	a.mean += draws * l.mean
	a.drop += draws * l.drop
	a.beta += draws * l.dropped
	if t.walkFn != nil && t.walkFn[f] {
		a.walkMean += draws * l.mean
		a.walkDrop += draws * l.drop
		a.walkBeta += draws * l.dropped
	}
}

// add adds w times o.
func (a *latticeAcct) add(o latticeAcct, w float64) {
	a.mean += w * o.mean
	a.drop += w * o.drop
	a.beta += w * o.beta
	a.walkMean += w * o.walkMean
	a.walkDrop += w * o.walkDrop
	a.walkBeta += w * o.walkBeta
}

// merge adds a to the pipeline's lattice mean, dropped tails' mean and
// dropped mass.
func (p *pipeline) merge(a latticeAcct) {
	p.mean += a.mean
	p.dropMean += a.drop
	p.beta += a.beta
	p.walkMean += a.walkMean
	p.walkDrop += a.walkDrop
	p.walkBeta += a.walkBeta
}

// account takes function f's lattice into the total's: an unscaled
// function's once, in the first all-functions pass, and a scaled one's
// at every point of the mixture, at its draws there.
func (p *pipeline) account(f int32, l fnLattice) {
	if p.t.m.fns[f].factor == noIndex {
		var a latticeAcct
		a.take(p.t, f, p.t.draws[f], l)
		p.merge(a)
		return
	}
	p.point.take(p.t, f, p.t.drawsGiven(f, p.mix.drawn), l)
}
