package variate

import (
	"code.eitanity.com/variate/dist"
)

// flatten turns the validated graph into index arrays. Pool indices are
// assigned in a fixed order: source frequencies, then per node its
// functions, outcome probabilities and holding times.
func (b *builder) flatten() (*Compiled, error) {
	m := &Compiled{nodeIDs: make([]string, len(b.g.Nodes)), sourceIDs: make([]string, len(b.g.Sources))}
	cats := map[string]int32{}
	factorPool := make([]int32, len(b.g.Factors))
	for i, f := range b.g.Factors {
		factorPool[i] = m.addPool(f.Model, PerPeriod)
	}
	for _, sc := range b.g.Schedules {
		// Over whole cycles every value is seen equally often: the pool's
		// law is the uniform categorical over the values; the walk sets
		// the period's value by its place in the cycle instead of drawing.
		probs := make([]float64, len(sc.Values))
		for i := range probs {
			probs[i] = 1 / float64(len(probs))
		}
		d, err := dist.NewCategorical(sc.Values, probs)
		if err != nil {
			return nil, modelError("schedule", sc.ID, err.Error())
		}
		m.schedules = append(m.schedules, schedule{pool: m.addPool(d, PerPeriod), values: sc.Values})
	}
	if err := b.flattenSources(m, factorPool); err != nil {
		return nil, err
	}
	for i, n := range b.g.Nodes {
		m.nodeIDs[i] = n.ID
		cn := cnode{mode: n.Mode, alias: noIndex}
		cn.fnStart = index(len(m.fns))
		b.flattenFunctions(m, index(i), n, OnEntry, cats)
		cn.fnEnd = index(len(m.fns))
		cn.rateStart = cn.fnEnd
		b.flattenFunctions(m, index(i), n, PerTime, cats)
		cn.rateEnd = index(len(m.fns))
		cn.outStart = index(len(m.outs))
		b.flattenOutcomes(m, index(i), n, &cn)
		cn.outEnd = index(len(m.outs))
		m.nodes = append(m.nodes, cn)
	}
	if int64(len(m.pools)) >= maxDistributions {
		return nil, modelError("graph", b.g.ID, "too many distributions")
	}
	m.categories = make([]string, len(cats))
	// unordered: each category fills its own slot
	for name, i := range cats {
		m.categories[i] = name
	}
	m.topo = b.topoOrder()
	for _, y := range b.g.Layers {
		m.layers = append(m.layers, y.normalised())
	}
	if err := m.countLaws(); err != nil {
		return nil, err
	}
	return m, nil
}

// flattenSources compiles the sources: their target, count law, chains
// and factor, with the rate pool of each.
func (b *builder) flattenSources(m *Compiled, factorPool []int32) error {
	for i, s := range b.g.Sources {
		m.sourceIDs[i] = s.ID
		src := csrc{to: b.nodeIndex[s.To], every: index(max(s.Every, 1)), repair: noIndex, factor: noIndex, schedule: noIndex}
		if s.Factor != "" {
			src.factor = factorPool[b.factorIndex[s.Factor]]
		}
		if s.Schedule != "" {
			src.schedule = m.schedules[b.scheduleIndex[s.Schedule]].pool
		}
		src.chainStart = index(len(m.chains))
		if len(s.Chains) > 0 {
			// The chains' stationary rate stands as the frequency pool, so
			// every reading of a source's rate finds the right mean.
			rate, _ := ChainFigures(s.Chains)
			d, err := dist.NewConstant(rate)
			if err != nil {
				return modelError("source", s.ID, "chain rate: "+err.Error())
			}
			src.pool = m.addPool(d, PerPeriod)
			m.chains = append(m.chains, s.Chains...)
			m.hasRepair = true
		} else {
			src.pool = m.addPool(s.Frequency, PerPeriod)
		}
		src.chainEnd = index(len(m.chains))
		if s.Repair != nil {
			src.repair = m.addPool(s.Repair, PerEvent)
			m.hasRepair = true
		}
		m.renewal = m.renewal || src.every > 1 || src.repair != noIndex || src.chainEnd > src.chainStart
		m.srcs = append(m.srcs, src)
	}
	return nil
}

// addPool registers a pool with its scope.
func (m *Compiled) addPool(d dist.Distribution, scope Scope) int32 {
	k := index(len(m.pools))
	m.pools = append(m.pools, d)
	m.poolMean = append(m.poolMean, d.Mean())
	slot := noIndex
	if scope == PerPeriod {
		slot = index(m.periodSlots)
		m.periodSlots++
	}
	m.poolPeriod = append(m.poolPeriod, slot)
	return k
}

func (b *builder) flattenFunctions(m *Compiled, node int32, n Node, basis Basis, cats map[string]int32) {
	for _, f := range n.Functions {
		if f.Basis != basis {
			continue
		}
		c, ok := cats[f.Category]
		if !ok {
			c = index(len(cats))
			cats[f.Category] = c
		}
		m.fns = append(m.fns, cfn{pool: m.addPool(f.Model, f.Scope), category: c})
		m.fnNames = append(m.fnNames, n.ID+":"+f.Name)
		m.fnNode = append(m.fnNode, node)
		m.fnPerTime = append(m.fnPerTime, basis == PerTime)
	}
}

// flattenOutcomes compiles a node's selection. Next becomes a one-outcome
// selection; fixed absolute outcomes are ordered by descending probability
// with cumulative thresholds; the remainder of a two-way node goes second.
func (b *builder) flattenOutcomes(m *Compiled, node int32, n Node, cn *cnode) {
	if n.Next != "" {
		m.appendOutcome(m.outcomeName(n.ID, n.Next, 0), node, cout{to: b.nodeIndex[n.Next], cum: 1, fixed: 1, probPool: noIndex, hold: m.holdPool(n.NextHolding, n.NextScope), outage: n.NextOutage})
		return
	}
	order := make([]int, len(n.Outcomes))
	probs := make([]float64, len(n.Outcomes))
	uncertain := -1
	for i, o := range n.Outcomes {
		order[i] = i
		switch {
		case o.P == nil:
			probs[i] = -1
		case isFixed(o.P):
			probs[i] = o.P.Mean()
		default:
			uncertain = i
		}
	}
	switch {
	case n.Mode == Sequential:
	case uncertain >= 0:
		cn.twoWay = true
		order = []int{uncertain, 1 - uncertain}
	default:
		if r := remainderIndex(probs); r >= 0 {
			probs[r] = 1 - fixedSum(probs)
		}
		sortOutcomesDesc(order, probs)
	}
	cum := 0.0
	dupes := map[string]int{}
	for _, i := range order {
		o := n.Outcomes[i]
		co := cout{to: b.nodeIndex[o.To], probPool: noIndex, hold: m.holdPool(o.Holding, o.HoldingScope), outage: o.OutageHolding}
		switch {
		case n.Mode == Sequential && o.P == nil:
			co.fixed = 1
		case n.Mode == Sequential && isFixed(o.P):
			co.fixed = o.P.Mean()
		case n.Mode == Sequential || cn.twoWay && o.P != nil:
			co.probPool = m.addPool(o.P, PerPeriod)
		case cn.twoWay:
			co.fixed = 1
		default:
			co.fixed = probs[i]
			cum += probs[i]
			co.cum = cum
		}
		dupes[o.To]++
		m.appendOutcome(m.outcomeName(n.ID, o.To, dupes[o.To]), node, co)
	}
	m.closeFixedSelection(n, cn)
}

// closeFixedSelection sets the last threshold to exactly 1 and builds an
// alias table for a wide fixed selection.
func (m *Compiled) closeFixedSelection(n Node, cn *cnode) {
	last := len(m.outs) - 1
	if n.Mode != Sequential && !cn.twoWay && last >= 0 {
		m.outs[last].cum = 1
	}
	count := len(m.outs) - int(cn.outStart)
	if n.Mode != Sequential && !cn.twoWay && count > aliasThreshold {
		cn.alias = index(len(m.aliasProb))
		probs := make([]float64, count)
		for i := range probs {
			probs[i] = m.outs[int(cn.outStart)+i].fixed
		}
		prob, alias := buildAlias(probs)
		m.aliasProb = append(m.aliasProb, prob...)
		m.aliasIdx = append(m.aliasIdx, alias...)
	}
}

func (m *Compiled) appendOutcome(name string, node int32, co cout) {
	m.outs = append(m.outs, co)
	m.outcomeNode = append(m.outcomeNode, node)
	m.outcomeNames = append(m.outcomeNames, name)
}

func (m *Compiled) holdPool(d dist.Continuous, scope Scope) int32 {
	if d == nil {
		return noIndex
	}
	m.hasDelays = true
	return m.addPool(d, scope)
}

// outcomeName is "<node>→<target>", suffixed #n for repeated targets.
func (m *Compiled) outcomeName(node, to string, n int) string {
	name := node + "→" + to
	if n > 1 {
		name += "#" + itoa(n)
	}
	return name
}

func remainderIndex(probs []float64) int {
	for i, p := range probs {
		if p < 0 {
			return i
		}
	}
	return -1
}

func fixedSum(probs []float64) float64 {
	s := 0.0
	for _, p := range probs {
		if p >= 0 {
			s += p
		}
	}
	return s
}

// buildAlias is Vose's construction for fixed outcome probabilities.
func buildAlias(probs []float64) (prob []float64, alias []int32) {
	n := len(probs)
	prob = make([]float64, n)
	alias = make([]int32, n)
	scaled := make([]float64, n)
	small := make([]int32, 0, n)
	large := make([]int32, 0, n)
	for i, p := range probs {
		scaled[i] = p * float64(n)
		if scaled[i] < 1 {
			small = append(small, index(i))
		} else {
			large = append(large, index(i))
		}
	}
	// bound: each iteration retires one entry
	for len(small) > 0 && len(large) > 0 {
		s, l := small[len(small)-1], large[len(large)-1]
		small, large = small[:len(small)-1], large[:len(large)-1]
		prob[s], alias[s] = scaled[s], l
		scaled[l] += scaled[s] - 1
		if scaled[l] < 1 {
			small = append(small, l)
		} else {
			large = append(large, l)
		}
	}
	for _, i := range append(large, small...) {
		prob[i] = 1
	}
	return prob, alias
}

// topoOrder is a Kahn order of the nodes; validation guarantees a DAG.
func (b *builder) topoOrder() []int32 {
	n := len(b.g.Nodes)
	indeg := make([]int, n)
	for _, nd := range b.g.Nodes {
		for _, to := range b.targets(nd) {
			indeg[to]++
		}
	}
	order := make([]int32, 0, n)
	for v := range indeg {
		if indeg[v] == 0 {
			order = append(order, index(v))
		}
	}
	// bound: each node is appended once
	for i := 0; i < len(order); i++ {
		for _, to := range b.targets(b.g.Nodes[order[i]]) {
			indeg[to]--
			if indeg[to] == 0 {
				order = append(order, to)
			}
		}
	}
	return order
}
