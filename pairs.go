package variate

// pairTable holds Q(u, w) = E[m₁(u|Θ)·m₁(w|Θ)] for every pair of nodes,
// stored once per unordered pair by topological position.
type pairTable struct {
	pos []int32 // topological position of each node
	q   []float64
	n   int
}

func (t *pairTable) at(u, w int32) float64 {
	a, b := t.pos[u], t.pos[w]
	if a > b {
		a, b = b, a
	}
	return t.q[int(a)*t.n+int(b)]
}

func (t *pairTable) set(u, w int32, v float64) {
	a, b := t.pos[u], t.pos[w]
	if a > b {
		a, b = b, a
	}
	t.q[int(a)*t.n+int(b)] = v
}

// pairPass computes Q in reverse topological order of the earlier node, so
// that every Q(child, ·) a pair needs is already known.
func (m *Compiled) pairPass(filt valueFilter, m1 []float64) *pairTable {
	n := len(m.topo)
	t := &pairTable{pos: make([]int32, n), q: make([]float64, n*n), n: n}
	for i, v := range m.topo {
		t.pos[v] = index(i)
	}
	for i := n - 1; i >= 0; i-- {
		u := m.topo[i]
		t.set(u, u, m.selfPair(u, filt, m1, t))
		// bound: nodes
		for j := i + 1; j < n; j++ {
			w := m.topo[j]
			t.set(u, w, m.crossPair(u, w, filt, m1, t))
		}
	}
	return t
}

// crossPair is Q(u, w) for u before w: u's own draws are independent of
// everything in m₁(w|Θ).
func (m *Compiled) crossPair(u, w int32, filt valueFilter, m1 []float64, t *pairTable) float64 {
	nm := m.nodeMomentsFor(u, filt)
	n := m.nodes[u]
	q := nm.a * m1[w]
	// bound: outcomes of one node
	for j := n.outStart; j < n.outEnd; j++ {
		o := m.outs[j]
		tau, _, _ := m.holdMoments(o)
		q += m.outcomeMeanProb(u, j) * (nm.rho*tau*m1[w] + t.at(o.to, w))
	}
	return q
}

// selfPair is Q(u, u), where u's period-scoped draws appear squared.
func (m *Compiled) selfPair(u int32, filt valueFilter, m1 []float64, t *pairTable) float64 {
	nm := m.nodeMomentsFor(u, filt)
	n := m.nodes[u]
	q := nm.a2Theta
	sumPB := 0.0
	// bound: outcomes of one node
	for j := n.outStart; j < n.outEnd; j++ {
		o := m.outs[j]
		tau, _, _ := m.holdMoments(o)
		sumPB += m.outcomeMeanProb(u, j) * (nm.rho*tau + m1[o.to])
	}
	q += 2 * nm.a * sumPB
	// bound: outcomes of one node
	for i := n.outStart; i < n.outEnd; i++ {
		// bound: outcomes of one node
		for j := n.outStart; j < n.outEnd; j++ {
			q += m.probPair(u, i, j) * m.bPair(nm, i, j, m1, t)
		}
	}
	return q
}

// bPair is E[B_i B_j] with B_j = ρ·τ_j + m₁(child_j|Θ).
func (m *Compiled) bPair(nm nodeMoments, i, j int32, m1 []float64, t *pairTable) float64 {
	oi, oj := m.outs[i], m.outs[j]
	ti, ti2, periodI := m.holdMoments(oi)
	tj, _, _ := m.holdMoments(oj)
	var tauPair float64
	switch {
	case i == j && periodI:
		tauPair = ti2
	default:
		tauPair = ti * tj
	}
	return rateSecond(nm.rho2T, tauPair) + nm.rho*ti*m1[oj.to] + nm.rho*tj*m1[oi.to] + t.at(oi.to, oj.to)
}

// probPair is E[π_i π_j] for two outcomes of one node.
func (m *Compiled) probPair(u, i, j int32) float64 {
	n := m.nodes[u]
	switch {
	case n.mode == Sequential:
		return m.sequentialPair(u, i, j)
	case n.twoWay:
		p, p2 := m.probMoments(m.outs[n.outStart])
		first := n.outStart
		switch {
		case i == first && j == first:
			return p2
		case i != first && j != first:
			return 1 - 2*p + p2
		default:
			return p - p2
		}
	default:
		return m.outs[i].fixed * m.outs[j].fixed
	}
}

// probMoments returns E[q] and E[q²] for an outcome's own probability.
func (m *Compiled) probMoments(o cout) (mean, second float64) {
	if o.probPool == noIndex {
		return o.fixed, o.fixed * o.fixed
	}
	d := m.pools[o.probPool]
	mean = d.Mean()
	return mean, d.Var() + mean*mean
}

// sequentialPair expands π_j = q_j·Π_{l<j}(1 − q_l) with independent q.
func (m *Compiled) sequentialPair(u, i, j int32) float64 {
	if i > j {
		i, j = j, i
	}
	n := m.nodes[u]
	last := n.outEnd - 1
	v := 1.0
	// bound: outcomes of one node
	for l := n.outStart; l < i; l++ {
		q, q2 := m.probMoments(m.outs[l])
		v *= 1 - 2*q + q2 // E[(1 − q)²]
	}
	qi, qi2 := m.probMoments(m.outs[i])
	if i == last {
		qi, qi2 = 1, 1
	}
	if i == j {
		return v * qi2
	}
	v *= qi - qi2 // E[q_i(1 − q_i)]
	// bound: outcomes of one node
	for l := i + 1; l < j; l++ {
		q, _ := m.probMoments(m.outs[l])
		v *= 1 - q
	}
	qj, _ := m.probMoments(m.outs[j])
	if j == last {
		qj = 1
	}
	return v * qj
}
