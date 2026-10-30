package variate

import (
	"math"

	"code.eitanity.com/variate/dist"
)

// pairBudget bounds the pass over node pairs of the variance computation.
const pairBudget = 1 << 30

// Expectation holds the exact values of the model, indexed like Result.
type Expectation struct {
	Reach               [][]float64 // [source][node]
	EntriesPer          []float64   // [node]
	OutcomeRate         []float64   // [outcome]
	Terminal            [][]float64 // [source][node], 0 for non-leaves
	Mean                float64
	Var                 float64 // +Inf if infinite; NaN with delays or above the pair budget
	DrawsMean, DrawsVar float64
	MeanByCategory      []float64
	MeanBySource        []float64
	Contrib             []float64 // [node]
	VarByCategory       []float64 // NaN with delays or above the pair budget
	VarBySource         []float64
	ContribVar          []float64 // [node]
	StepsPerPeriod      float64
	ArrivalsPerPeriod   float64 // arrivals the renewal and chain sources draw beyond their rounds: the stream's events, the units' failures and repairs
	RoundsPerPeriod     float64 // Σ_k E[λ_k]
	pzero               float64
}

// valueFilter selects which function values the recursion books: every
// function (filterAll), one category, one node, or the constant 1 per
// function (the draws count).
type valueFilter struct {
	kind  int
	index int32
}

const (
	filterAll = iota
	filterCategory
	filterNode
	filterDraws
)

// nodeMoments are a node's own first and second moments under a filter:
// the entry value a (sum of OnEntry values), the rate ρ (sum of PerTime
// rates), each as a mean, a plain second moment and a Θ-second moment
// (second moments for period-scoped functions, squared means otherwise).
type nodeMoments struct {
	a, a2, a2Theta   float64
	rho, rho2, rho2T float64
}

// fnMoments returns E[X], E[X²] and whether the draw is period-scoped for
// function f under the filter; a filtered-out function contributes 0.
func (m *Compiled) fnMoments(f int32, filt valueFilter) (mean, second float64, period bool) {
	c := m.fns[f]
	switch filt.kind {
	case filterCategory:
		if c.category != filt.index {
			return 0, 0, false
		}
	case filterNode:
		if m.fnNode[f] != filt.index {
			return 0, 0, false
		}
	case filterDraws:
		return 1, 1, false
	}
	d := m.pools[c.pool]
	mean = d.Mean()
	return mean, d.Var() + mean*mean, m.poolPeriod[c.pool] != noIndex
}

// sumMoments combines independent terms: the mean, E[(Σ)²] and the
// Θ-version of E[(Σ)²].
func (m *Compiled) sumMoments(from, to int32, filt valueFilter) (mean, second, secondTheta float64) {
	// bound: table length
	for f := from; f < to; f++ {
		x, x2, period := m.fnMoments(f, filt)
		second += x2 + 2*mean*x
		if period {
			secondTheta += x2 + 2*mean*x
		} else {
			secondTheta += x*x + 2*mean*x
		}
		mean += x
	}
	return mean, second, secondTheta
}

func (m *Compiled) nodeMomentsFor(v int32, filt valueFilter) nodeMoments {
	n := m.nodes[v]
	var nm nodeMoments
	nm.a, nm.a2, nm.a2Theta = m.sumMoments(n.fnStart, n.fnEnd, filt)
	nm.rho, nm.rho2, nm.rho2T = m.sumMoments(n.rateStart, n.rateEnd, filt)
	return nm
}

// holdMoments returns E[τ], E[τ²] and whether τ is period-scoped for an
// outcome; 0 without a holding time.
func (m *Compiled) holdMoments(o cout) (mean, second float64, period bool) {
	if o.hold == noIndex {
		return 0, 0, false
	}
	d := m.pools[o.hold]
	mean = d.Mean()
	return mean, d.Var() + mean*mean, m.poolPeriod[o.hold] != noIndex
}

// valueToGo computes M1 and M2, the moments of the value an event books
// from each node onwards, backwards in reverse topological order.
func (m *Compiled) valueToGo(filt valueFilter) (m1, m2 []float64) {
	m1 = make([]float64, len(m.nodes))
	m2 = make([]float64, len(m.nodes))
	for i := len(m.topo) - 1; i >= 0; i-- {
		v := m.topo[i]
		nm := m.nodeMomentsFor(v, filt)
		n := m.nodes[v]
		var sumPY, sumPY2 float64
		// bound: outcomes of one node
		for j := n.outStart; j < n.outEnd; j++ {
			o := m.outs[j]
			p := m.outcomeMeanProb(v, j)
			tau, tau2, _ := m.holdMoments(o)
			ey := nm.rho*tau + m1[o.to]
			ey2 := rateSecond(nm.rho2, tau2) + m2[o.to] + 2*nm.rho*tau*m1[o.to]
			sumPY += weighted(p, ey)
			sumPY2 += weighted(p, ey2)
		}
		m1[v] = nm.a + sumPY
		m2[v] = nm.a2 + 2*nm.a*sumPY + sumPY2
	}
	return m1, m2
}

// weighted is w·x with a zero weight giving 0 even for an infinite x: a
// source that never fires contributes nothing, however heavy its values.
func weighted(w, x float64) float64 {
	if w == 0 {
		return 0
	}
	return w * x
}

// rateSecond is E[ρ²]·E[τ²], with no holding time booking nothing even
// when the rate's second moment is infinite (0·∞ would be NaN).
func rateSecond(rho2, tau2 float64) float64 {
	if tau2 == 0 {
		return 0
	}
	return rho2 * tau2
}

// Analytic computes the exact expectations of the model.
func (m *Compiled) Analytic() (*Expectation, error) {
	e := &Expectation{Reach: m.reach}
	e.EntriesPer = make([]float64, len(m.nodes))
	e.OutcomeRate = make([]float64, len(m.outs))
	e.Terminal = make([][]float64, len(m.srcs))
	e.MeanBySource = make([]float64, len(m.srcs))
	lam := make([]float64, len(m.srcs))
	m1, m2 := m.valueToGo(valueFilter{kind: filterAll})
	for k, s := range m.srcs {
		lam[k] = m.countMean[k]
		e.RoundsPerPeriod += lam[k]
		e.ArrivalsPerPeriod += m.arrivalsOf(k)
		e.Terminal[k] = make([]float64, len(m.nodes))
		for v := range m.nodes {
			e.EntriesPer[v] += lam[k] * m.reach[k][v]
			e.StepsPerPeriod += lam[k] * m.reach[k][v]
			if m.nodes[v].outStart == m.nodes[v].outEnd {
				e.Terminal[k][v] = m.reach[k][v]
			}
		}
		e.MeanBySource[k] = lam[k] * m1[s.to]
		e.Mean += e.MeanBySource[k]
	}
	for j := range m.outs {
		v := m.outcomeNode[j]
		e.OutcomeRate[j] = e.EntriesPer[v] * m.outcomeMeanProb(v, index(j))
	}
	e.MeanByCategory = make([]float64, len(m.categories))
	for c := range m.categories {
		e.MeanByCategory[c] = m.meanUnder(valueFilter{kind: filterCategory, index: index(c)}, e.EntriesPer)
	}
	e.Contrib = make([]float64, len(m.nodes))
	for v := range m.nodes {
		if m.nodes[v].fnStart != m.nodes[v].rateEnd {
			e.Contrib[v] = m.meanUnder(valueFilter{kind: filterNode, index: index(v)}, e.EntriesPer)
		}
	}
	e.DrawsMean = m.meanUnder(valueFilter{kind: filterDraws}, e.EntriesPer)
	e.Var = m.variance(valueFilter{kind: filterAll}, lam, m1, m2, e.Mean)
	d1, d2 := m.valueToGo(valueFilter{kind: filterDraws})
	e.DrawsVar = m.variance(valueFilter{kind: filterDraws}, lam, d1, d2, e.DrawsMean)
	m.breakdownVariances(e, lam, m1, m2)
	e.pzero = m.zeroProbability()
	return e, nil
}

// PZero is the exact probability of a zero period total.
func (e *Expectation) PZero() float64 { return e.pzero }

// quadRule is a fixed quadrature rule over a rate: nodes and weights.
type quadRule struct{ x, w []float64 }

// rateRule finds a rule for E[exp(a·λ)] over a frequency: the family's
// own Gauss rule where it has one, else Gauss–Legendre in the quantile
// domain with the tail substitution, doubling the nodes until the value
// at a = −1, the steepest case used, settles to quadTol.
func rateRule(d dist.Distribution) quadRule {
	prev := math.NaN()
	n := quadStart
	var rule quadRule
	// bound: quadDoublings
	for i := 0; i <= quadDoublings; i++ {
		rule = quadRule{x: make([]float64, n), w: make([]float64, n)}
		if !d.Nodes(rule.x, rule.w) {
			quantileNodes(d, rule.x, rule.w)
		}
		sum := rule.mixedExp(-1)
		if math.Abs(sum-prev) <= quadTol {
			return rule
		}
		prev = sum
		n *= 2
	}
	return rule
}

// mixedExp is E[exp(a·λ)] under the rule.
func (q quadRule) mixedExp(a float64) float64 {
	sum := 0.0
	for j := range q.x {
		sum += q.w[j] * math.Exp(a*q.x[j])
	}
	return sum
}

// breakdownVariances fills the per-source, per-category and per-node
// variances: sources from the total's own tables, the others by a filtered
// pass each. The pair budget covers all passes together.
func (m *Compiled) breakdownVariances(e *Expectation, lam, m1, m2 []float64) {
	e.VarBySource = make([]float64, len(m.srcs))
	e.VarByCategory = make([]float64, len(m.categories))
	e.ContribVar = make([]float64, len(m.nodes))
	nan := math.NaN()
	passes := 1 + len(m.categories)
	for v := range m.nodes {
		if m.nodes[v].fnStart != m.nodes[v].rateEnd {
			passes++
		}
	}
	if m.hasDelays || m.hasRepair || m.pairCost()*float64(passes) > pairBudget {
		for k := range e.VarBySource {
			e.VarBySource[k] = nan
		}
		for c := range e.VarByCategory {
			e.VarByCategory[c] = nan
		}
		for v := range e.ContribVar {
			e.ContribVar[v] = nan
		}
		return
	}
	q := m.pairPass(valueFilter{kind: filterAll}, m1)
	for k, s := range m.srcs {
		e.VarBySource[k] = lam[k]*m2[s.to] + m.countN2[k]*q.at(s.to, s.to) - e.MeanBySource[k]*e.MeanBySource[k]
	}
	for c := range m.categories {
		f := valueFilter{kind: filterCategory, index: index(c)}
		c1, c2 := m.valueToGo(f)
		e.VarByCategory[c] = m.variance(f, lam, c1, c2, e.MeanByCategory[c])
	}
	for v := range m.nodes {
		if m.nodes[v].fnStart == m.nodes[v].rateEnd {
			continue
		}
		f := valueFilter{kind: filterNode, index: index(v)}
		n1, n2 := m.valueToGo(f)
		e.ContribVar[v] = m.variance(f, lam, n1, n2, e.Contrib[v])
	}
}

// meanUnder is the expected value booked per period under a filter:
// entries per node times the node's own expected booking.
func (m *Compiled) meanUnder(filt valueFilter, entries []float64) float64 {
	total := 0.0
	for v := range m.nodes {
		nm := m.nodeMomentsFor(index(v), filt)
		held := 0.0
		n := m.nodes[v]
		// bound: outcomes of one node
		for j := n.outStart; j < n.outEnd; j++ {
			tau, _, _ := m.holdMoments(m.outs[j])
			held += m.outcomeMeanProb(index(v), j) * tau
		}
		total += entries[v] * (nm.a + nm.rho*held)
	}
	return total
}

// variance is Var[T] for a model without delays: E[Var[T|Θ]] + Var[E[T|Θ]].
// NaN with delays or above the pair budget; +Inf when any term is.
func (m *Compiled) variance(filt valueFilter, lam, m1, m2 []float64, mean float64) float64 {
	if m.hasDelays || m.hasRepair || m.pairCost() > pairBudget {
		return math.NaN()
	}
	within := 0.0
	for k, s := range m.srcs {
		within += weighted(lam[k], m2[s.to])
	}
	q := m.pairPass(filt, m1)
	between := -mean * mean
	for k, s := range m.srcs {
		for l, t := range m.srcs {
			if k == l {
				between += weighted(m.countN2[k], q.at(s.to, t.to))
			} else {
				between += weighted(m.countPair(k, l), q.at(s.to, t.to))
			}
		}
	}
	if math.IsInf(within, 0) || math.IsInf(between, 0) || math.IsNaN(between) {
		return math.Inf(1)
	}
	return within + between
}

// pairCost estimates the terms of the pair pass.
func (m *Compiled) pairCost() float64 {
	cost := 0.0
	for i, v := range m.topo {
		outs := float64(m.nodes[v].outEnd - m.nodes[v].outStart)
		cost += outs*float64(len(m.topo)-i) + outs*outs
	}
	return cost
}
