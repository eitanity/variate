package variate

import (
	"math"
	"strings"

	"code.eitanity.com/variate/dist"
)

// Values that move together: each group draws one standard normal W per
// round, the first time the round draws a member, and each member is
// Q(Φ(√ρ·W + √(1−ρ)·E)) of its own law Q and its own normal E, so it
// keeps its law and every pair has normal correlation ρ, the rank
// correlation r's 2·sin(πr/6). No booking multiplies two members (a
// node's rates and holding are not both members), so every mean is
// unchanged; the second moment of a round gains each pair's covariance
// times the chance the round books both, and their coefficients.

const (
	// depNodesStart and depNodesMax are the Gauss–Hermite rules the
	// covariances double through; a covariance unsettled at the last is
	// NaN.
	depNodesStart = 8
	depNodesMax   = 256
	// depSettle is the change, relative to the members' standard
	// deviations, that counts as settled.
	depSettle = 1e-9
)

// notPerRound is why a value drawn once a period cannot be a member.
const notPerRound = "drawn once a period, not per round"

// flattenRounds compiles what concerns a round's draws as a whole: the
// dependence groups, then the given state.
func (b *builder) flattenRounds(m *Compiled) {
	b.flattenDependence(m)
	b.flattenGiven(m)
}

// cgroup is a compiled dependence group.
type cgroup struct {
	sq, sq1 float64     // √ρ and √(1−ρ)
	members []depMember // functions, then holdings
	cov     [][]float64 // the members' covariances; NaN unsettled
}

// depMember is a function or, with out set, an outcome's holding.
type depMember struct {
	fn, out int32 // one is noIndex
	node    int32
}

// checkDependence: a group has two members or more and a rank in
// [0, 1], and each member is a value, rate or holding a round draws
// once that a copula may draw instead.
func (b *builder) checkDependence() error {
	seen := map[string]bool{}
	behind := b.behindQueues()
	read := b.triggerRead()
	rates, holds := map[string]bool{}, map[string]bool{}
	for i, d := range b.g.Dependence {
		subject := "dependence " + itoa(i)
		if !(d.Rank >= 0 && d.Rank <= 1) {
			return modelError("graph", subject, "rank must be in [0, 1]")
		}
		if len(d.Members) < 2 {
			return modelError("graph", subject, "a group needs two members or more")
		}
		for _, s := range d.Members {
			if why := b.checkGroupMember(s, seen, behind, read, rates, holds); why != "" {
				return modelError("graph", subject, why)
			}
		}
	}
	return nil
}

// checkGroupMember checks one member against the members before it:
// named once, and not a rate of a node whose holding is a member or the
// other way round.
func (b *builder) checkGroupMember(s string, seen map[string]bool, behind, read []bool, rates, holds map[string]bool) string {
	if seen[s] {
		return s + " is named twice"
	}
	seen[s] = true
	node, rate, why := b.checkMember(s, behind, read)
	if why != "" {
		return s + ": " + why
	}
	if strings.Contains(s, "->") {
		holds[node] = true
	} else if rate {
		rates[node] = true
	}
	if rates[node] && holds[node] {
		return "node " + node + " has a rate and its holding as members, and their product would move its mean"
	}
	return ""
}

// checkMember resolves a member, giving its node, whether it is a
// per-time rate, and why it cannot be one.
func (b *builder) checkMember(s string, behind, read []bool) (string, bool, string) {
	node, to, isHold := strings.Cut(s, "->")
	name := ""
	if !isHold {
		var ok bool
		if node, name, ok = strings.Cut(s, "."); !ok {
			return "", false, "not node.function or node->to"
		}
	}
	v, ok := b.nodeIndex[node]
	switch {
	case !ok:
		return node, false, "node " + node + " not found"
	case behind[v]:
		return node, false, "a member is drawn before a round waits at a queue, and the wait parts it from its group"
	case read[v]:
		return node, false, "a trigger reads the node's values on their own"
	case isHold:
		return node, false, holdingMember(b.g.Nodes[v], to)
	}
	return functionMember(b.g.Nodes[v], name)
}

// functionMember checks a function as a member.
func functionMember(n Node, name string) (string, bool, string) {
	for _, f := range n.Functions {
		if f.Name != name {
			continue
		}
		why := ""
		switch {
		case f.Expr != "":
			why = "an expression is computed from its operands, not drawn"
		case n.Threshold != nil && n.Threshold.Function == name:
			why = "the value a threshold tests"
		case f.Scope == PerPeriod:
			why = notPerRound
		case isOperand(n, name):
			why = "an expression's operand"
		case f.Model == nil || math.IsInf(f.Model.Var(), 1):
			why = "its variance is infinite"
		}
		return n.ID, f.Basis == PerTime, why
	}
	return n.ID, false, "function " + name + " not found"
}

// holdingMember checks the holding from n to to as a member.
func holdingMember(n Node, to string) string {
	if n.Next != "" {
		switch {
		case n.Next != to:
			return "no holding to " + to
		case n.NextHolding == nil:
			return "the way to " + to + " has no holding time"
		case n.NextOutage:
			return "an outage holding is the block's down time"
		case n.NextScope == PerPeriod:
			return notPerRound
		}
		return ""
	}
	why, count := "no holding to "+to, 0
	for _, o := range n.Outcomes {
		if o.To != to {
			continue
		}
		count++
		switch {
		case o.Holding == nil:
			why = "the way to " + to + " has no holding time"
		case o.OutageHolding:
			why = "an outage holding is the block's down time"
		case o.HoldingScope == PerPeriod:
			why = notPerRound
		default:
			why = ""
		}
	}
	if count > 1 {
		return "two outcomes go to " + to
	}
	return why
}

// isOperand reports whether an expression of n names the function.
func isOperand(n Node, name string) bool {
	for _, f := range n.Functions {
		if f.Expr == "" {
			continue
		}
		if rpn, msg := parseExpr(f.Expr); msg == "" {
			for _, x := range exprNames(rpn) {
				if x == name {
					return true
				}
			}
		}
	}
	return false
}

// behindQueues marks the queue nodes and every node after one.
func (b *builder) behindQueues() []bool {
	out := make([]bool, len(b.g.Nodes))
	for i, n := range b.g.Nodes {
		if n.Queue == nil {
			continue
		}
		for v, r := range b.reachedFrom(index(i)) {
			out[v] = out[v] || r
		}
	}
	return out
}

// triggerRead marks the nodes a trigger's sum reads values of.
func (b *builder) triggerRead() []bool {
	out := make([]bool, len(b.g.Nodes))
	for _, n := range b.g.Nodes {
		if n.Trigger == nil {
			continue
		}
		for _, in := range n.Trigger.Inputs {
			if v, ok := b.nodeIndex[in.Node]; ok {
				out[v] = true
			}
		}
	}
	return out
}

// flattenDependence compiles the groups of positive rank, a group of
// rank 0 being none.
func (b *builder) flattenDependence(m *Compiled) {
	for _, d := range b.g.Dependence {
		if d.Rank == 0 {
			continue
		}
		rho := math.Min(1, 2*math.Sin(math.Pi*d.Rank/6))
		g := cgroup{sq: math.Sqrt(rho), sq1: math.Sqrt(math.Max(0, 1-rho))}
		for _, s := range d.Members {
			g.members = append(g.members, m.resolveMember(b, s))
		}
		if m.fnGroup == nil {
			m.fnGroup, m.fnSlot = filled(len(m.fns)), filled(len(m.fns))
			m.outGroup, m.outSlot = filled(len(m.outs)), filled(len(m.outs))
		}
		gi := index(len(m.groups))
		for k, mb := range g.members {
			if mb.fn != noIndex {
				m.fnGroup[mb.fn], m.fnSlot[mb.fn] = gi, index(k)
			} else {
				m.outGroup[mb.out], m.outSlot[mb.out] = gi, index(k)
			}
		}
		g.cov = depCovariances(m.memberLaws(g.members), g.sq, g.sq1)
		m.groups = append(m.groups, g)
	}
}

// filled is n noIndex values.
func filled(n int) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = noIndex
	}
	return out
}

// resolveMember finds a checked member's compiled function or outcome.
func (m *Compiled) resolveMember(b *builder, s string) depMember {
	if node, to, ok := strings.Cut(s, "->"); ok {
		v := b.nodeIndex[node]
		n := m.nodes[v]
		// bound: outcomes of one node
		for j := n.outStart; j < n.outEnd; j++ {
			if m.outs[j].to == b.nodeIndex[to] {
				return depMember{fn: noIndex, out: j, node: v}
			}
		}
	}
	node, name, _ := strings.Cut(s, ".")
	want := node + ":" + name
	for f, got := range m.fnNames {
		if got == want {
			return depMember{fn: index(f), out: noIndex, node: m.fnNode[f]}
		}
	}
	return depMember{fn: noIndex, out: noIndex}
}

// memberLaws is each on-entry value member's law, nil for a rate or a
// holding, whose covariances no exact figure uses.
func (m *Compiled) memberLaws(ms []depMember) []dist.Distribution {
	out := make([]dist.Distribution, len(ms))
	for k, mb := range ms {
		if mb.fn != noIndex && !m.fnPerTime[mb.fn] {
			out[k] = m.pools[m.fns[mb.fn].pool]
		}
	}
	return out
}

// copulaValue is Q(Φ(z)) of law d, the upper tail read from above.
func copulaValue(d dist.Distribution, z float64) float64 {
	if z <= 0 {
		return d.Quantile(0.5 * math.Erfc(-z/math.Sqrt2))
	}
	return d.QuantileUpper(0.5 * math.Erfc(z/math.Sqrt2))
}

// depCovariances is the members' covariances over the Gauss–Hermite
// rules for W, and for each member's own normal, doubling until each
// settles.
func depCovariances(laws []dist.Distribution, sq, sq1 float64) [][]float64 {
	k := len(laws)
	var prev [][]float64
	// bound: the rules from depNodesStart to depNodesMax
	for n := depNodesStart; n <= depNodesMax; n *= 2 {
		cov := depCovAt(laws, sq, sq1, n)
		if prev != nil && depSettled(laws, prev, cov) {
			return cov
		}
		if n == depNodesMax {
			for i := range k {
				for j := range k {
					if !depClose(laws, prev, cov, i, j) {
						cov[i][j] = math.NaN()
					}
				}
			}
			return cov
		}
		prev = cov
	}
	return prev
}

// depCovAt is the covariances on the n-point rule: g_i(w) = E[Q_i(Φ(√ρ·w
// + √(1−ρ)·E))], and Cov_ij = Σ W·(g_i − ḡ_i)(g_j − ḡ_j).
func depCovAt(laws []dist.Distribution, sq, sq1 float64, n int) [][]float64 {
	x, w := make([]float64, n), make([]float64, n)
	dist.HermiteRule(x, w)
	k := len(laws)
	g := make([][]float64, k)
	mean := make([]float64, k)
	for i, d := range laws {
		g[i] = make([]float64, n)
		if d == nil {
			continue
		}
		// bound: n nodes
		for a := range n {
			if sq1 == 0 {
				g[i][a] = copulaValue(d, x[a])
			} else {
				// bound: n nodes
				for c := range n {
					g[i][a] += w[c] * copulaValue(d, sq*x[a]+sq1*x[c])
				}
			}
			mean[i] += w[a] * g[i][a]
		}
	}
	cov := make([][]float64, k)
	for i := range k {
		cov[i] = make([]float64, k)
		for j := range k {
			// bound: n nodes
			for a := range n {
				cov[i][j] += w[a] * (g[i][a] - mean[i]) * (g[j][a] - mean[j])
			}
		}
	}
	return cov
}

// depSettled reports whether every covariance is within depSettle of
// prev's.
func depSettled(laws []dist.Distribution, prev, cov [][]float64) bool {
	for i := range cov {
		for j := range cov {
			if !depClose(laws, prev, cov, i, j) {
				return false
			}
		}
	}
	return true
}

func depClose(laws []dist.Distribution, prev, cov [][]float64, i, j int) bool {
	if laws[i] == nil || laws[j] == nil {
		return true
	}
	scale := math.Sqrt(laws[i].Var() * laws[j].Var())
	return math.Abs(cov[i][j]-prev[i][j]) <= depSettle*math.Max(scale, 1e-300)
}

// depCorrection is, per node, what the dependence adds to the second
// moment of the value booked from it onwards under the filter:
// Σ_{i≠j} Cov_ij·s_i·s_j·P(the round books both), s the members'
// signs, for the on-entry values that are members. A rate or holding
// member needs a holding time, and the exact variance of a model with
// one is not computed, so only values count here.
func (m *Compiled) depCorrection(filt valueFilter) []float64 {
	corr := make([]float64, len(m.nodes))
	if len(m.groups) == 0 || filt.kind == filterDraws {
		return corr
	}
	reach := make([][][]float64, len(m.groups))
	for gi := range m.groups {
		reach[gi] = m.depReach(&m.groups[gi], filt)
	}
	for i := len(m.topo) - 1; i >= 0; i-- {
		v := m.topo[i]
		n := m.nodes[v]
		// bound: outcomes of one node
		for j := n.outStart; j < n.outEnd; j++ {
			corr[v] += weighted(m.outcomeMeanProb(v, j), corr[m.outs[j].to])
		}
		for gi := range m.groups {
			corr[v] += m.depAt(&m.groups[gi], reach[gi], v, filt)
		}
	}
	return corr
}

// depSign is a member's sign under the filter: a value's, 0 for a rate,
// a holding, or a value outside the filter.
func (m *Compiled) depSign(mb depMember, filt valueFilter) float64 {
	if mb.fn == noIndex || m.fnPerTime[mb.fn] {
		return 0
	}
	s, ok := m.fnSign(mb.fn, filt)
	if !ok {
		return 0
	}
	return s
}

// depReach is, per member j and node x, s_j·P(the round books j | the
// round at x).
func (m *Compiled) depReach(g *cgroup, filt valueFilter) [][]float64 {
	out := make([][]float64, len(g.members))
	for k, mb := range g.members {
		e := make([]float64, len(m.nodes))
		for i := len(m.topo) - 1; i >= 0; i-- {
			v := m.topo[i]
			if v == mb.node {
				e[v] = m.depSign(mb, filt)
				continue
			}
			n := m.nodes[v]
			// bound: outcomes of one node
			for j := n.outStart; j < n.outEnd; j++ {
				e[v] += weighted(m.outcomeMeanProb(v, j), e[m.outs[j].to])
			}
		}
		out[k] = e
	}
	return out
}

// depAt is group g's addition at node v: its pairs there, and its pairs
// from a member there to one later on the path, each counted once in
// each order.
func (m *Compiled) depAt(g *cgroup, reach [][]float64, v int32, filt valueFilter) float64 {
	sum := 0.0
	n := m.nodes[v]
	for a, ma := range g.members {
		sa := m.depSign(ma, filt)
		if ma.node != v || sa == 0 {
			continue
		}
		for b, mb := range g.members {
			switch {
			case b == a:
			case mb.node == v:
				sum += g.cov[a][b] * sa * m.depSign(mb, filt)
			default:
				later := 0.0
				// bound: outcomes of one node
				for j := n.outStart; j < n.outEnd; j++ {
					later += weighted(m.outcomeMeanProb(v, j), reach[b][m.outs[j].to])
				}
				sum += 2 * g.cov[a][b] * sa * later
			}
		}
	}
	return sum
}

// depZeroKnown reports whether P(T = 0) can take the members' draws as
// independent: no member function is 0 with positive chance.
func (m *Compiled) depZeroKnown() bool {
	for _, g := range m.groups {
		for _, mb := range g.members {
			if mb.fn != noIndex && m.pools[m.fns[mb.fn].pool].CDF(0) > 0 {
				return false
			}
		}
	}
	return true
}
