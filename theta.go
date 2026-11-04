package variate

import (
	"math"

	"code.eitanity.com/variate/dist"
)

const (
	// probNodesStart is the first node count of a period-scoped
	// probability's quadrature; it doubles until the mixture settles.
	probNodesStart = 4
	// zeroWork caps the scalar mixture behind the exact P(T = 0): its
	// points times the work of one.
	zeroWork = 1 << 23
	// passesCeiling is where the count of tensor points saturates, above
	// every cap on it and within a 32-bit int.
	passesCeiling = 1 << 30

	reasonNoProbRule = "a period-scoped probability has no quadrature rule"
	reasonProbCapped = "period-scoped probability quadrature did not converge within the pass cap"
	reasonProbNoConv = "period-scoped probability quadrature did not converge"
	reasonFreqNoConv = "frequency quadrature did not converge"
	totalSubject     = "total"
)

// thetaQuantity is one period-scoped quantity besides the frequencies:
// an outcome probability or a function value, with the nodes and weights
// its mixture is taken over.
type thetaQuantity struct {
	pool   int32
	fn     int32       // the function for a value; noIndex for a probability or a factor
	factor bool        // a shared factor on rates, or the schedules, which sit on no node
	cycle  *jointCycle // the schedules' joint cycle, its nodes the indices of its points; nil otherwise
	nodes  []float64
	weight []float64
}

// thetaMix enumerates the tensor product of the quantities' nodes,
// keeping the current draw of every pool in drawn.
type thetaMix struct {
	q     []thetaQuantity
	idx   []int
	drawn []float64
}

// periodQuantities lists the period-scoped quantities besides the
// frequencies, without nodes.
func (m *Compiled) periodQuantities() []thetaQuantity {
	var qs []thetaQuantity
	for f, c := range m.fns {
		if m.poolPeriod[c.pool] != noIndex {
			qs = append(qs, thetaQuantity{pool: c.pool, fn: index(f)})
		}
	}
	for _, o := range m.outs {
		if o.probPool != noIndex {
			qs = append(qs, thetaQuantity{pool: o.probPool, fn: noIndex})
		}
	}
	seen := map[int32]bool{}
	scheduled := false
	for _, s := range m.srcs {
		scheduled = scheduled || s.schedule != noIndex
		if s.factor != noIndex && !seen[s.factor] {
			seen[s.factor] = true
			qs = append(qs, thetaQuantity{pool: s.factor, fn: noIndex, factor: true})
		}
	}
	if scheduled {
		qs = append(qs, m.cycle.quantity())
	}
	return qs
}

// newThetaMix starts an enumeration with every pool at its mean.
func (m *Compiled) newThetaMix(qs []thetaQuantity) *thetaMix {
	x := &thetaMix{q: qs, idx: make([]int, len(qs)), drawn: make([]float64, len(m.poolMean))}
	copy(x.drawn, m.poolMean)
	return x
}

// passes is the number of points in the tensor, at most passesCeiling:
// the product of many quantities' node counts would otherwise wrap and
// pass under every cap.
func (x *thetaMix) passes() int {
	for _, q := range x.q {
		if q.cycle != nil && q.cycle.over {
			return passesCeiling
		}
	}
	for _, q := range x.q {
		if len(q.nodes) == 0 {
			return 0
		}
	}
	n := 1
	for _, q := range x.q {
		if n > passesCeiling/len(q.nodes) {
			return passesCeiling
		}
		n *= len(q.nodes)
	}
	return n
}

// probabilities counts the probability quantities.
func (x *thetaMix) probabilities() int {
	n := 0
	for _, q := range x.q {
		if q.fn == noIndex {
			n++
		}
	}
	return n
}

// setProbNodes gives every probability n quadrature nodes; false when a
// probability has no rule.
func (x *thetaMix) setProbNodes(m *Compiled, n int) bool {
	for i := range x.q {
		q := &x.q[i]
		if q.fn != noIndex || q.cycle != nil {
			continue
		}
		q.nodes = make([]float64, n)
		q.weight = make([]float64, n)
		if !probNodes(m.pools[q.pool], q.nodes, q.weight) {
			return false
		}
		q.trim()
	}
	return true
}

// trim drops nodes of zero weight, which a finite rule pads with.
func (q *thetaQuantity) trim() {
	kept := 0
	for i := range q.nodes {
		if q.weight[i] != 0 {
			q.nodes[kept], q.weight[kept] = q.nodes[i], q.weight[i]
			kept++
		}
	}
	q.nodes, q.weight = q.nodes[:kept], q.weight[:kept]
}

// probCounts returns the node count of every probability quantity.
func (x *thetaMix) probCounts(dst []int) []int {
	dst = dst[:0]
	for _, q := range x.q {
		if q.fn == noIndex {
			dst = append(dst, len(q.nodes))
		}
	}
	return dst
}

func sameCounts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// start positions the enumeration at its first point; false when a
// quantity has no nodes.
func (x *thetaMix) start() bool {
	for i, q := range x.q {
		if len(q.nodes) == 0 {
			return false
		}
		x.idx[i] = 0
		x.set(i)
	}
	return true
}

// next advances the enumeration; false once every point has been seen.
func (x *thetaMix) next() bool {
	for i := range x.q {
		q := &x.q[i]
		x.idx[i]++
		if x.idx[i] < len(q.nodes) {
			x.set(i)
			return true
		}
		x.idx[i] = 0
		x.set(i)
	}
	return false
}

// set draws quantity i at its current node: the joint cycle's point sets
// every schedule at once.
func (x *thetaMix) set(i int) {
	q := &x.q[i]
	if q.cycle == nil {
		x.drawn[q.pool] = q.nodes[x.idx[i]]
		return
	}
	for s, pool := range q.cycle.pools {
		x.drawn[pool] = q.cycle.points[x.idx[i]][s]
	}
}

// weight is the product of the current nodes' weights.
func (x *thetaMix) weight() float64 {
	w := 1.0
	for i, q := range x.q {
		w *= q.weight[x.idx[i]]
	}
	return w
}

// probNodes fills a quadrature rule for a probability: the family's own
// Gauss rule, else Gauss–Legendre through the quantile function.
func probNodes(d dist.Distribution, x, w []float64) bool {
	if d.Nodes(x, w) {
		return true
	}
	if len(x) < 2 {
		return false
	}
	quantileNodes(d, x, w)
	return true
}

// zeroProbability is the exact P(T = 0): every source's rounds book
// nothing, mixed over the period draws. A value's draw matters only
// through whether it is 0, so it costs two atoms; a probability's nodes
// double until the mixture settles. Quantities that no common source can
// reach are independent, so the mixture factors into one tensor per
// connected component of quantities and sources, and the product of the
// components is exact. A component whose mixture would exceed zeroWork
// keeps its last settled value, or else takes its probabilities at their
// means, and PZero is then an approximation for that component; when even
// that is above zeroWork, PZero is NaN, as it is when a trigger can fire
// on a zero total.
func (m *Compiled) zeroProbability() float64 {
	if m.fireOnZero() {
		// The trigger's walk may book in a period whose sources booked
		// nothing, with a probability the sources' mixture does not give.
		return math.NaN()
	}
	qs := m.periodQuantities()
	for i := range qs {
		if qs[i].fn == noIndex {
			continue
		}
		pz := m.pools[qs[i].pool].CDF(0)
		qs[i].nodes = []float64{0, 1}
		qs[i].weight = []float64{pz, 1 - pz}
		qs[i].trim() // a value never 0, or always, is one point
	}
	rules := make([]quadRule, len(m.srcs))
	for k, s := range m.srcs {
		if s.trigger != noIndex {
			continue
		}
		if _, ok := m.pools[s.pool].Laplace(0); !ok {
			rules[k] = rateRule(m.pools[s.pool])
		}
	}
	z := make([]float64, len(m.nodes))
	total := 1.0
	for _, comp := range m.zeroComponents(qs) {
		total *= m.zeroComponent(comp, rules, z)
	}
	return total
}

// zeroComponent is one factor of P(T = 0): the mixture over the
// component's quantities of the product over its sources.
func (m *Compiled) zeroComponent(comp zeroComp, rules []quadRule, z []float64) float64 {
	mix := m.newThetaMix(comp.qs)
	point := m.zeroPointWork(comp.srcs, rules)
	n := probNodesStart
	prev := math.NaN()
	// bound: quadDoublings
	for i := 0; i <= quadDoublings; i++ {
		if !mix.setProbNodes(m, n) || int64(mix.passes())*point > zeroWork {
			break
		}
		total := m.zeroMixed(mix, comp.srcs, rules, z)
		if mix.probabilities() == 0 || math.Abs(total-prev) <= quadTol {
			return total
		}
		prev = total
		n *= 2
	}
	if !math.IsNaN(prev) {
		return prev
	}
	for i := range mix.q {
		if mix.q[i].fn == noIndex && mix.q[i].cycle == nil {
			mix.q[i].nodes = []float64{m.poolMean[mix.q[i].pool]}
			mix.q[i].weight = []float64{1}
		}
	}
	if int64(mix.passes())*point > zeroWork {
		return math.NaN()
	}
	return m.zeroMixed(mix, comp.srcs, rules, z)
}

// zeroPointWork is the work of one point of a component's mixture: the
// nodes, outcomes and functions zeroGiven visits, and a frequency-rule
// node for each source without a count law.
func (m *Compiled) zeroPointWork(srcs []int, rules []quadRule) int64 {
	work := int64(len(m.topo) + len(m.outs) + len(m.fns))
	for _, k := range srcs {
		work += int64(len(rules[k].x))
	}
	return work
}

// zeroMixed sums the weighted conditional values over a mixture.
func (m *Compiled) zeroMixed(mix *thetaMix, srcs []int, rules []quadRule, z []float64) float64 {
	total := 0.0
	// bound: zeroWork
	for ok := mix.start(); ok; ok = mix.next() {
		total += mix.weight() * m.zeroGiven(mix.drawn, z, rules, srcs)
	}
	return total
}

// zeroComp is a connected component: quantities and the sources that
// can reach any of their nodes.
type zeroComp struct {
	qs   []thetaQuantity
	srcs []int
}

// zeroComponents groups quantities with the sources reaching them by
// union-find over sources; sources reaching no quantity form one
// component of their own.
func (m *Compiled) zeroComponents(qs []thetaQuantity) []zeroComp {
	parent := make([]int, len(m.srcs))
	for k := range parent {
		parent[k] = k
	}
	find := func(k int) int {
		// bound: the union-find depth, at most the number of sources
		for parent[k] != k {
			parent[k] = parent[parent[k]]
			k = parent[k]
		}
		return k
	}
	qSources := make([][]int, len(qs))
	for i, q := range qs {
		qSources[i] = m.quantitySources(q)
		for j := 1; j < len(qSources[i]); j++ {
			parent[find(qSources[i][j])] = find(qSources[i][0])
		}
	}
	byRoot := map[int]*zeroComp{}
	var order []int
	for k := range m.srcs {
		if m.srcs[k].trigger != noIndex {
			// A trigger fires only on a sum above a level that is not
			// negative, so never in a period whose total is 0, unless it
			// counts entries that book nothing, which is NaN above.
			continue
		}
		r := find(k)
		c, ok := byRoot[r]
		if !ok {
			c = &zeroComp{}
			byRoot[r] = c
			order = append(order, r)
		}
		c.srcs = append(c.srcs, k)
	}
	for i, q := range qs {
		if len(qSources[i]) == 0 {
			continue // a quantity no source reaches does not matter
		}
		byRoot[find(qSources[i][0])].qs = append(byRoot[find(qSources[i][0])].qs, q)
	}
	out := make([]zeroComp, 0, len(order))
	for _, r := range order {
		out = append(out, *byRoot[r])
	}
	return out
}

// quantitySources lists the sources a period-scoped quantity touches:
// those naming a factor, or those reaching the node a value or
// probability sits on.
func (m *Compiled) quantitySources(q thetaQuantity) []int {
	var out []int
	v := m.quantityNode(q)
	for k := range m.srcs {
		s := m.srcs[k]
		if s.trigger != noIndex {
			continue
		}
		if (q.cycle != nil && s.schedule != noIndex) || (q.factor && q.cycle == nil && s.factor == q.pool) || (!q.factor && m.reach[k][v] > 0) {
			out = append(out, k)
		}
	}
	return out
}

// quantityNode is the node a period-scoped quantity sits on.
func (m *Compiled) quantityNode(q thetaQuantity) int32 {
	if q.fn != noIndex {
		return m.fnNode[q.fn]
	}
	for j, o := range m.outs {
		if o.probPool == q.pool {
			return m.outcomeNode[j]
		}
	}
	return 0
}

// zeroMix is the probability that the rest of the round from node v
// books nothing, over its outcomes: a threshold's branch also needs its
// half of the tested value to be 0.
func (m *Compiled) zeroMix(v int32, n cnode, drawn, z []float64) float64 {
	mix := 0.0
	// bound: outcomes of one node
	for j := n.outStart; j < n.outEnd; j++ {
		zero := 1.0
		if f := m.outs[j].fn; f != noIndex {
			zero = m.pools[m.fns[f].pool].CDF(0)
		}
		mix += m.outcomeProbWith(v, j, drawn) * zero * z[m.outs[j].to]
	}
	// A sequential node's outcome probabilities can sum to just above 1,
	// and then e^(f·(z−1)·λ) overflows over a heavy frequency's far nodes.
	return math.Min(mix, 1)
}

// zeroGiven is P(T = 0 | Θ) with the period draws in drawn; z is scratch
// for P(C(v) = 0 | Θ).
func (m *Compiled) zeroGiven(drawn, z []float64, rules []quadRule, srcs []int) float64 {
	for i := len(m.topo) - 1; i >= 0; i-- {
		v := m.topo[i]
		n := m.nodes[v]
		pz := 1.0
		// bound: functions of one node
		for f := n.fnStart; f < n.fnEnd; f++ {
			pool := m.fns[f].pool
			switch {
			case m.poolPeriod[pool] == noIndex:
				pz *= m.pools[pool].CDF(0)
			case drawn[pool] != 0:
				pz = 0
			}
		}
		if n.outStart != n.outEnd {
			pz *= m.zeroMix(v, n, drawn, z)
		}
		z[v] = pz
	}
	total := 1.0
	for _, k := range srcs {
		s := m.srcs[k]
		// P_N(z) with the source's own count law, at z = P(C = 0),
		// the rate scaled by the factor drawn.
		f := m.factorAt(k, drawn)
		if g, ok := m.countGF(k, complex(z[s.to], 0), f); ok {
			total *= real(g)
		} else if s.repair != noIndex || s.chainEnd > s.chainStart {
			total = math.NaN()
		} else {
			total *= rules[k].mixedExp(f * (z[s.to] - 1))
		}
	}
	return total
}
