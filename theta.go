package variate

import (
	"math"

	"code.eitanity.com/variate/dist"
)

const (
	// probNodesStart is the first node count of a period-scoped
	// probability's quadrature; it doubles until the mixture settles.
	probNodesStart = 4
	// zeroPasses caps the scalar mixture behind the exact P(T = 0).
	zeroPasses = 1 << 22

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
	fn     int32 // the function for a value; noIndex for a probability or a factor
	factor bool  // a shared factor on rates, which sits on no node
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
	for _, s := range m.srcs {
		for _, pool := range [2]int32{s.factor, s.schedule} {
			if pool != noIndex && !seen[pool] {
				seen[pool] = true
				qs = append(qs, thetaQuantity{pool: pool, fn: noIndex, factor: true})
			}
		}
	}
	return qs
}

// newThetaMix starts an enumeration with every pool at its mean.
func (m *Compiled) newThetaMix(qs []thetaQuantity) *thetaMix {
	x := &thetaMix{q: qs, idx: make([]int, len(qs)), drawn: make([]float64, len(m.poolMean))}
	copy(x.drawn, m.poolMean)
	return x
}

// passes is the number of points in the tensor.
func (x *thetaMix) passes() int {
	n := 1
	for _, q := range x.q {
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
		if q.fn != noIndex {
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
		x.drawn[q.pool] = q.nodes[0]
	}
	return true
}

// next advances the enumeration; false once every point has been seen.
func (x *thetaMix) next() bool {
	for i := range x.q {
		q := &x.q[i]
		x.idx[i]++
		if x.idx[i] < len(q.nodes) {
			x.drawn[q.pool] = q.nodes[x.idx[i]]
			return true
		}
		x.idx[i] = 0
		x.drawn[q.pool] = q.nodes[0]
	}
	return false
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
// components is exact. A component whose tensor would exceed zeroPasses
// takes its probabilities at their means, and PZero is then an
// approximation for that component.
func (m *Compiled) zeroProbability() float64 {
	qs := m.periodQuantities()
	for i := range qs {
		if qs[i].fn == noIndex {
			continue
		}
		pz := m.pools[qs[i].pool].CDF(0)
		qs[i].nodes = []float64{0, 1}
		qs[i].weight = []float64{pz, 1 - pz}
	}
	rules := make([]quadRule, len(m.srcs))
	for k, s := range m.srcs {
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
	n := probNodesStart
	prev := math.NaN()
	// bound: quadDoublings
	for i := 0; i <= quadDoublings; i++ {
		if !mix.setProbNodes(m, n) || mix.passes() > zeroPasses {
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
		if mix.q[i].fn == noIndex {
			mix.q[i].nodes = []float64{m.poolMean[mix.q[i].pool]}
			mix.q[i].weight = []float64{1}
		}
	}
	return m.zeroMixed(mix, comp.srcs, rules, z)
}

// zeroMixed sums the weighted conditional values over a mixture.
func (m *Compiled) zeroMixed(mix *thetaMix, srcs []int, rules []quadRule, z []float64) float64 {
	total := 0.0
	// bound: zeroPasses
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
		if (q.factor && (m.srcs[k].factor == q.pool || m.srcs[k].schedule == q.pool)) || (!q.factor && m.reach[k][v] > 0) {
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
			mix := 0.0
			// bound: outcomes of one node
			for j := n.outStart; j < n.outEnd; j++ {
				mix += m.outcomeProbWith(v, j, drawn) * z[m.outs[j].to]
			}
			pz *= mix
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
