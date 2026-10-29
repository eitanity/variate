package variate

import (
	"math"
	"sort"

	"code.eitanity.com/variate/dist"
	"code.eitanity.com/variate/stats"
)

const (
	noIndex int32 = -1

	// probSumTol is the tolerance on fixed probabilities summing to 1.
	probSumTol = 1e-12
	// pathLevel is the tail probability the path-duration bound allows.
	pathLevel = 1e-15
	// spillWarn is the window width above which Compile warns.
	spillWarn = 1000
	// aliasThreshold is the number of fixed outcomes above which a node
	// selects with an alias table.
	aliasThreshold = 8
	// maxDistributions keeps the stream key injective.
	maxDistributions = 1 << 32
	maxReachEntries  = 1 << 26 // sources × nodes a model may have: the reach tables
)

// cnode is a compiled node: index ranges into the outcome and function
// tables, and the selection mode.
type cnode struct {
	outStart, outEnd   int32 // outcomes
	fnStart, fnEnd     int32 // OnEntry functions
	rateStart, rateEnd int32 // PerTime functions
	mode               Mode
	twoWay             bool  // absolute mode with one uncertain probability and a remainder
	alias              int32 // alias table start, or noIndex
}

// cout is a compiled outcome. For a fixed probability, cum is the
// cumulative threshold and probPool is noIndex; otherwise probPool is the
// pool of the period-scoped probability. fixed is the probability itself
// (sequential mode) or the remainder.
type cout struct {
	to       int32
	cum      float64
	fixed    float64
	probPool int32
	hold     int32 // holding-time pool, or noIndex
	outage   bool  // the holding is the firing block's own down time; hold then carries its bound
}

// cfn is a compiled function.
type cfn struct {
	pool     int32
	category int32
}

// csrc is a compiled source.
type csrc struct {
	to     int32
	pool   int32
	every  int32 // 1 = every arrival
	repair int32 // repair holding pool, or noIndex
	// The source's chains in m.chains, when it is a reliability diagram.
	chainStart, chainEnd int32
	factor               int32 // the shared factor's pool, or noIndex
	schedule             int32 // the schedule's pool, or noIndex
}

// Compiled is a validated, flattened model.
type Compiled struct {
	nodeIDs, sourceIDs, outcomeNames, categories []string
	nodes                                        []cnode
	outs                                         []cout
	fns                                          []cfn
	srcs                                         []csrc
	outcomeNode                                  []int32 // owning node of each outcome
	fnNode                                       []int32 // owning node of each function
	fnNames                                      []string
	fnPerTime                                    []bool
	aliasProb                                    []float64
	aliasIdx                                     []int32
	pools                                        []dist.Distribution
	poolMean                                     []float64
	poolPeriod                                   []int32 // per-period slot, or noIndex for event scope
	periodSlots                                  int
	topo                                         []int32
	reach                                        [][]float64
	spill                                        int
	hasDelays                                    bool
	renewal                                      bool        // a source with the Erlang or repair law
	hasRepair                                    bool        // a source with the repair law or a reliability diagram: no closed variance
	chains                                       []Chain     // every source's chains, in source order
	layers                                       []Layer     // the model's layers, defaults filled
	schedules                                    []schedule  // the model's schedules, with their pools
	countMean, countN2, countTail                []float64   // per source: E[N], E[N(N−1)], truncated mass of the Erlang law
	countPMF                                     [][]float64 // per source: the Erlang law's mass function, or nil
	cfg                                          Config
	crit                                         criteria
	plan                                         []planRow
	expect                                       *Expectation
	watchNode                                    []bool // nodes a round must reach to be delivered; nil = all
	watchLeaf                                    []bool // leaves a round must end at; nil = all
	deadLeaf, deadNode                           []bool // nodes from which no watched leaf or node is reachable
	minValue                                     float64
	maxPath                                      int // nodes on the longest source-to-leaf path
	transformed                                  *Transformed
	tailX                                        float64   // upper bound on the TailP quantile of T; 0 when unknown
	tailThresholds                               []float64 // levels the tail shares are counted against
}

// Compile validates the graph and flattens it; the warnings are the
// non-fatal findings.
func (g Graph) Compile(cfg Config) (*Compiled, []Warning, error) {
	b := &builder{g: g, cfg: cfg, nodeIndex: map[string]int32{}}
	if err := b.validate(); err != nil {
		return nil, nil, err
	}
	m, err := b.flatten()
	if err != nil {
		return nil, nil, err
	}
	m.cfg = cfg
	if err := reachCap(g.ID, len(m.srcs), len(m.nodes)); err != nil {
		return nil, nil, err
	}
	m.reach = m.reachTable()
	if err := m.outageHolds(); err != nil {
		return nil, nil, err
	}
	warnings := b.warnings
	warnings = append(warnings, m.unreachableWarnings()...)
	warnings = append(warnings, m.workWarning())
	if err := m.setSpill(cfg.Spill, &warnings); err != nil {
		return nil, nil, err
	}
	e, err := m.Analytic()
	if err != nil {
		return nil, nil, err
	}
	m.expect = e
	m.tailThresholds = tailLevels(e.Mean)
	m.setWatch(cfg.Watch, b.nodeIndex)
	m.maxPath = m.longestNodePath()
	m.crit = criteriaFrom(cfg)
	m.plan = m.buildPlan(e, m.crit)
	if err := m.checkCaps(m.crit, e.StepsPerPeriod+e.ArrivalsPerPeriod, cfg.AllowUnconverged); err != nil {
		return nil, nil, err
	}
	if cfg.Transform {
		if err := m.transformAtCompile(&warnings); err != nil {
			return nil, nil, err
		}
	}
	return m, warnings, nil
}

// reachCap refuses a model whose reach tables, sources × nodes, would
// not fit.
func reachCap(graph string, srcs, nodes int) error {
	if int64(srcs)*int64(nodes) > maxReachEntries {
		return modelError("graph", graph, "sources × nodes = "+itoa(srcs)+" × "+itoa(nodes)+" exceeds "+itoa(maxReachEntries)+"; the reach tables would not fit")
	}
	return nil
}

// transformAtCompile runs the transform and, where it applies, takes its
// upper bound on the TailP quantile for the tail criterion, warning when
// that lies beyond the histogram range.
func (m *Compiled) transformAtCompile(ws *[]Warning) error {
	tr, err := m.Transform(TransformConfig{TailP: m.crit.tailP, MaxGrid: m.cfg.TransformGrid, Tol: m.cfg.TransformTol, MaxWork: m.cfg.TransformWork, MaxBytes: m.cfg.TransformBytes})
	if err != nil {
		return err
	}
	m.transformed = tr
	if !tr.Applicable {
		*ws = append(*ws, Warning{Code: "no-transform", Subject: totalSubject, Message: "the transform does not apply: " + tr.Reason})
		return nil
	}
	_, m.tailX = tr.Total.Quantile(m.crit.tailP)
	if _, top := stats.LogRange(m.expect.Mean); m.tailX >= top {
		*ws = append(*ws, Warning{Code: "tail-beyond-range", Subject: totalSubject, Message: "the TailP quantile bound " + ftoa(m.tailX) + " lies above the histogram range"})
	}
	return nil
}

// Transformed returns the transform computed at compile, or nil.
func (m *Compiled) Transformed() *Transformed { return m.transformed }

// TailThresholds are the totals the tail shares are counted against.
func (m *Compiled) TailThresholds() []float64 { return m.tailThresholds }

// NodeIDs returns the node names in index order.
func (m *Compiled) NodeIDs() []string { return m.nodeIDs }

// NodeOrder is the node identifiers in topological order, every node
// after the nodes that lead to it.
func (m *Compiled) NodeOrder() []string {
	out := make([]string, len(m.topo))
	for i, v := range m.topo {
		out[i] = m.nodeIDs[v]
	}
	return out
}

// SourceIDs returns the source names in index order.
func (m *Compiled) SourceIDs() []string { return m.sourceIDs }

// OutcomeNames returns the outcome names in index order.
func (m *Compiled) OutcomeNames() []string { return m.outcomeNames }

// Categories returns the category names in index order.
func (m *Compiled) Categories() []string { return m.categories }

// Spill returns the window width S.
func (m *Compiled) Spill() int { return m.spill }

// HasDelays reports whether any outcome has a holding time.
func (m *Compiled) HasDelays() bool { return m.hasDelays }

// HasRepair reports whether a source is silent for a repair after each
// firing, which leaves the variance and P(T = 0) without a closed form.
func (m *Compiled) HasRepair() bool { return m.hasRepair }

// Pools returns the distributions by pool index.
func (m *Compiled) Pools() []dist.Distribution { return m.pools }

// reachTable is R_k(v): the probability that an event from source k
// reaches node v, by a forward pass in topological order.
func (m *Compiled) reachTable() [][]float64 {
	reach := make([][]float64, len(m.srcs))
	for k := range m.srcs {
		reach[k] = make([]float64, len(m.nodes))
	}
	m.reachWith(m.poolMean, reach)
	return reach
}

// reachWith fills the reach of every source with the period-scoped
// probabilities at the values in drawn, indexed by pool.
func (m *Compiled) reachWith(drawn []float64, reach [][]float64) {
	for k, s := range m.srcs {
		r := reach[k]
		clear(r)
		r[s.to] = 1
		for _, v := range m.topo {
			if r[v] == 0 {
				continue
			}
			n := m.nodes[v]
			// bound: outcomes of one node
			for j := n.outStart; j < n.outEnd; j++ {
				r[m.outs[j].to] += r[v] * m.outcomeProbWith(v, j, drawn)
			}
		}
	}
}

// outcomeMeanProb is E[P(j)] for outcome j of node v.
func (m *Compiled) outcomeMeanProb(v, j int32) float64 {
	return m.outcomeProbWith(v, j, m.poolMean)
}

// outcomeProbWith is P(j) for outcome j of node v with every
// period-scoped probability at its value in drawn, indexed by pool.
func (m *Compiled) outcomeProbWith(v, j int32, drawn []float64) float64 {
	n := m.nodes[v]
	o := m.outs[j]
	if n.mode == Sequential {
		p := m.outcomeProbOf(o, drawn)
		if j == n.outEnd-1 {
			p = 1
		}
		// bound: outcomes of one node
		for i := n.outStart; i < j; i++ {
			p *= 1 - m.outcomeProbOf(m.outs[i], drawn)
		}
		return p
	}
	if n.twoWay {
		if o.probPool != noIndex {
			return drawn[o.probPool]
		}
		return 1 - drawn[m.outs[n.outStart].probPool]
	}
	return o.fixed
}

// outcomeProbOf is an outcome's own probability at the drawn values.
func (m *Compiled) outcomeProbOf(o cout, drawn []float64) float64 {
	if o.probPool != noIndex {
		return drawn[o.probPool]
	}
	return o.fixed
}

func (m *Compiled) unreachableWarnings() []Warning {
	var ws []Warning
	for v := range m.nodes {
		reached := false
		for k := range m.srcs {
			if m.reach[k][v] > 0 {
				reached = true
				break
			}
		}
		if !reached {
			ws = append(ws, Warning{Code: "unreachable-node", Subject: m.nodeIDs[v], Message: "no source reaches this node"})
		}
	}
	return ws
}

// workWarning reports the expected rounds and walk steps per period.
func (m *Compiled) workWarning() Warning {
	var rounds, steps float64
	for k := range m.srcs {
		lam := m.countMean[k]
		rounds += lam
		for v := range m.nodes {
			steps += lam * m.reach[k][v]
		}
	}
	arrivals := 0.0
	for k := range m.srcs {
		arrivals += m.arrivalsOf(k)
	}
	msg := "expected rounds per period " + ftoa(rounds) + ", steps per period " + ftoa(steps)
	if arrivals > 0 {
		msg += ", arrivals drawn per period " + ftoa(arrivals)
	}
	return Warning{Code: "expected-work", Subject: "model", Message: msg}
}

// setSpill computes the path-duration bound S, or takes the override.
func (m *Compiled) setSpill(override int, ws *[]Warning) error {
	if override < 0 {
		return modelError("config", "Spill", "must be >= 0")
	}
	if !m.hasDelays {
		m.spill = 0
		return nil
	}
	if override > 0 {
		m.spill = override
		return nil
	}
	// L: the largest number of non-zero holding times on any path.
	counts := make([]float64, len(m.nodes))
	m.longestPath(counts, func(cout) float64 { return 1 })
	maxL := 0.0
	for _, s := range m.srcs {
		maxL = math.Max(maxL, counts[s.to])
	}
	level := pathLevel / maxL
	weights := make([]float64, len(m.nodes))
	driver := m.longestPath(weights, func(o cout) float64 {
		return m.pools[o.hold].QuantileUpper(level)
	})
	q := 0.0
	for _, s := range m.srcs {
		q = math.Max(q, weights[s.to])
	}
	if math.IsInf(q, 0) || math.IsNaN(q) {
		return modelError("outcome", m.outcomeNames[driver], "holding time has no finite quantile at the path-duration level; add a cap")
	}
	m.spill = int(math.Ceil(q))
	if m.spill > spillWarn {
		*ws = append(*ws, Warning{Code: "wide-window", Subject: m.outcomeNames[driver], Message: "path-duration bound S = " + itoa(m.spill) + " periods; a cap on this holding time would shorten it"})
	}
	return nil
}

// longestPath fills best[v] with the heaviest path from v to a leaf, where
// an outcome with a holding time weighs w(o) and others 0, by a backward
// pass in reverse topological order. It returns the outcome that starts the
// heaviest path from any source target.
func (m *Compiled) longestPath(best []float64, w func(cout) float64) int32 {
	via := make([]int32, len(m.nodes))
	for i := len(m.topo) - 1; i >= 0; i-- {
		v := m.topo[i]
		via[v] = noIndex
		n := m.nodes[v]
		// bound: outcomes of one node
		for j := n.outStart; j < n.outEnd; j++ {
			o := m.outs[j]
			c := best[o.to]
			if o.hold != noIndex {
				c += w(o)
			}
			if c > best[v] || via[v] == noIndex {
				best[v], via[v] = c, j
			}
		}
	}
	driver := noIndex
	top := -1.0
	for _, s := range m.srcs {
		if best[s.to] > top && via[s.to] != noIndex {
			top = best[s.to]
			driver = m.heaviestOutcome(via, s.to)
		}
	}
	return driver
}

// heaviestOutcome follows via from v to the first outcome with a holding
// time on the heaviest path.
func (m *Compiled) heaviestOutcome(via []int32, v int32) int32 {
	// bound: a path visits each node at most once
	for step := 0; step < len(m.nodes); step++ {
		j := via[v]
		if j == noIndex {
			return noIndex
		}
		if m.outs[j].hold != noIndex {
			return j
		}
		v = m.outs[j].to
	}
	return noIndex
}

// sortOutcomesDesc orders fixed outcomes by descending probability,
// stable.
func sortOutcomesDesc(idx []int, probs []float64) {
	sort.SliceStable(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })
}

// setWatch resolves the Watch names; validation has checked they exist.
func (m *Compiled) setWatch(w Watch, ix map[string]int32) {
	if len(w.Nodes) > 0 {
		m.watchNode = make([]bool, len(m.nodes))
		for _, id := range w.Nodes {
			m.watchNode[ix[id]] = true
		}
	}
	if len(w.Leaves) > 0 {
		m.watchLeaf = make([]bool, len(m.nodes))
		for _, id := range w.Leaves {
			m.watchLeaf[ix[id]] = true
		}
	}
	m.minValue = w.MinValue
	m.deadLeaf = m.deadFor(m.watchLeaf)
	m.deadNode = m.deadFor(m.watchNode)
}

// deadFor marks the nodes from which no node of the set can be reached,
// so a round entering one can no longer match; nil for no set.
func (m *Compiled) deadFor(set []bool) []bool {
	if set == nil {
		return nil
	}
	dead := make([]bool, len(m.nodes))
	for i := len(m.topo) - 1; i >= 0; i-- {
		v := m.topo[i]
		live := set[v]
		n := m.nodes[v]
		// bound: outcomes of one node
		for j := n.outStart; j < n.outEnd && !live; j++ {
			live = !dead[m.outs[j].to]
		}
		dead[v] = !live
	}
	return dead
}

// longestNodePath counts the nodes on the longest source-to-leaf path.
func (m *Compiled) longestNodePath() int {
	depth := make([]int, len(m.nodes))
	for i := len(m.topo) - 1; i >= 0; i-- {
		v := m.topo[i]
		depth[v] = 1
		n := m.nodes[v]
		// bound: outcomes of one node
		for j := n.outStart; j < n.outEnd; j++ {
			depth[v] = max(depth[v], 1+depth[m.outs[j].to])
		}
	}
	longest := 0
	for _, src := range m.srcs {
		longest = max(longest, depth[src.to])
	}
	return longest
}

// LongestPath is the number of nodes on the longest source-to-leaf path:
// the step capacity a round needs.
func (m *Compiled) LongestPath() int { return m.maxPath }

// expectedMatches estimates the delivered rounds per period under the
// Watch: an upper bound from reach, or every round when MinValue alone
// filters.
func (m *Compiled) expectedMatches() float64 {
	total := 0.0
	for k, src := range m.srcs {
		lam := m.pools[src.pool].Mean()
		share := 1.0
		switch {
		case m.watchLeaf != nil:
			share = 0
			for v := range m.nodes {
				if m.watchLeaf[v] {
					share += m.reach[k][v]
				}
			}
		case m.watchNode != nil:
			share = 0
			for v := range m.nodes {
				if m.watchNode[v] {
					share += m.reach[k][v]
				}
			}
			share = math.Min(share, 1)
		}
		total += lam * share
	}
	return total
}
