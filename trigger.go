package variate

import (
	"math"

	"code.eitanity.com/variate/dist"
)

// A trigger is compiled as one more source after the model's own: its
// count per period is 1 when the period's sum of its inputs is above its
// level and 0 otherwise, and its round enters the trigger's node. The
// walk adds each input booking to the sums of the triggers it feeds, and
// once every source has walked the period, each trigger above its level
// walks one round. So the reach, the counts and the breakdowns by source
// take the trigger as they take any source, with its count's mean,
// P(fire), standing as its frequency.

const (
	// fireTail is the share of P(fire) the transform's tail level leaves
	// above the level, when the first transform resolves short of it.
	fireTail = 0.5
	// fireTails is the number of transforms that may lower the tail level.
	fireTails = 6
	// fireTailMax is the deepest tail level asked of the transform.
	fireTailMax = 1 - 1e-15
	// fireCategory books every input of the sum's model.
	fireCategory = "sum"
)

// ctrig is a compiled trigger.
type ctrig struct {
	node     int32
	src      int32 // the source the trigger stands as
	over     float64
	upstream string // a trigger whose walk feeds this one's sum, or ""
	firing   Firing
}

// Firing is P(fire) of a trigger: within [Lo, Hi] from the transform of
// its sum, or NaN with the reason the transform does not apply.
type Firing struct {
	Source int // the trigger's index among the sources
	Lo, Hi float64
	Reason string
}

// Exact reports whether the bracket is known.
func (f Firing) Exact() bool { return !math.IsNaN(f.Lo) }

// flattenTriggers appends each trigger as a source entering its node,
// with a pool whose mean is set to P(fire) once that is known, and lists
// per function the triggers its bookings feed.
func (b *builder) flattenTriggers(m *Compiled) {
	var pairs, counted [][2]int32 // function or node, trigger
	// In evaluation order, so each trigger's source walks after those
	// whose walks feed its sum.
	for _, i := range b.trigOrder {
		n := b.g.Nodes[i]
		tr := index(len(m.trigs))
		k := index(len(m.srcs))
		pool := m.addPool(point(0), PerEvent)
		m.srcs = append(m.srcs, csrc{to: i, pool: pool, every: 1, repair: noIndex, factor: noIndex, schedule: noIndex, trigger: tr, chainStart: index(len(m.chains)), chainEnd: index(len(m.chains))})
		m.sourceIDs = append(m.sourceIDs, n.ID)
		m.trigs = append(m.trigs, ctrig{node: i, src: k, over: n.Trigger.Over, upstream: b.trigUp[i]})
		for _, in := range n.Trigger.Inputs {
			for _, f := range m.nodeFunctions(b.nodeIndex[in.Node], in.Node, in.Function) {
				pairs = append(pairs, [2]int32{f, tr})
			}
		}
		for _, id := range n.Trigger.Entries {
			counted = append(counted, [2]int32{b.nodeIndex[id], tr})
		}
	}
	m.feedStart, m.feeds = groupBy(pairs, len(m.fns))
	m.countStart, m.counts = groupBy(counted, len(m.nodes))
}

// groupBy lists the second of each pair by the first, below n: the
// start of each first's run, with one more entry than n, and the runs.
func groupBy(pairs [][2]int32, n int) (start, list []int32) {
	start = make([]int32, n+1)
	for _, p := range pairs {
		start[p[0]+1]++
	}
	for i := range n {
		start[i+1] += start[i]
	}
	list = make([]int32, len(pairs))
	next := append([]int32(nil), start[:n]...)
	for _, p := range pairs {
		list[next[p[0]]] = p[1]
		next[p[0]]++
	}
	return start, list
}

// nodeFunctions lists the compiled functions of node v named name, or
// all of them for "": its OnEntry and PerTime functions and the value a
// threshold tests, without the threshold's halves.
func (m *Compiled) nodeFunctions(v int32, id, name string) []int32 {
	n := m.nodes[v]
	var out []int32
	match := func(f int32) {
		if name == "" || m.fnNames[f] == id+":"+name {
			out = append(out, f)
		}
	}
	// bound: functions of one node
	for f := n.fnStart; f < n.rateEnd; f++ {
		match(f)
	}
	if n.thresh != noIndex {
		match(n.thresh)
	}
	return out
}

// fireProbabilities sets each trigger's P(fire) from the transform of
// its sum, and its source's count mean with it: the bracket's middle,
// or where the transform does not apply, an upper bound by Markov's
// inequality, or 1 for a chained trigger, which sizes the histograms
// while the figures it reaches are NaN.
func (m *Compiled) fireProbabilities(g Graph, cfg Config) {
	for i := range m.trigs {
		tr := &m.trigs[i]
		var f Firing
		p := 1.0
		if tr.upstream != "" {
			// The sum's model leaves the triggers out, so it has no
			// part of a chained trigger's sum.
			f = Firing{Lo: math.NaN(), Hi: math.NaN(), Reason: "the sum takes in the walk of trigger " + tr.upstream + ", which the transform of the sum leaves out"}
		} else {
			var mean float64
			f, mean = fireOf(g, m.nodeIDs[tr.node], tr.over, cfg)
			if f.Exact() {
				p = (f.Lo + f.Hi) / 2
			} else if tr.over > 0 {
				p = math.Min(1, mean/tr.over)
			}
		}
		f.Source = int(tr.src)
		tr.firing = f
		k := tr.src
		m.pools[m.srcs[k].pool] = point(p)
		m.poolMean[m.srcs[k].pool] = p
		m.countMean[k] = p
	}
}

// fireOf brackets P(S > over) for the trigger on node id, with the mean
// of S. The transform resolves the sum up to its TailP quantile; when
// the level lies beyond that, the tail level is raised until the
// quantile passes the level or P(S > over) is bounded to 0.
func fireOf(g Graph, id string, over float64, cfg Config) (Firing, float64) {
	nan := math.NaN()
	fg := fireGraph(g, id)
	sub := Config{NoBreakdown: true, TailP: cfg.TailP}
	dm, _, err := fg.flattenReached(sub)
	if err != nil {
		return Firing{Lo: nan, Hi: nan, Reason: "the sum's model: " + err.Error()}, nan
	}
	e, err := dm.Analytic()
	if err != nil {
		return Firing{Lo: nan, Hi: nan, Reason: "the sum's model: " + err.Error()}, nan
	}
	dm.expect = e
	dm.crit = criteriaFrom(sub)
	tail := dm.crit.tailP
	// bound: fireTails
	for range fireTails {
		t := newTransformer(dm, TransformConfig{TailP: tail, MaxGrid: cfg.TransformGrid, Tol: cfg.TransformTol, MaxWork: cfg.TransformWork, MaxBytes: cfg.TransformBytes})
		t.totalOnly = true
		out := &Transformed{}
		if reason := t.applicable(); reason != "" {
			return Firing{Lo: nan, Hi: nan, Reason: "the transform of the sum does not apply: " + reason}, e.Mean
		}
		if err := t.run(out); err != nil {
			return Firing{Lo: nan, Hi: nan, Reason: "the transform of the sum: " + err.Error()}, e.Mean
		}
		if !out.Applicable {
			return Firing{Lo: nan, Hi: nan, Reason: "the transform of the sum does not apply: " + out.Reason}, e.Mean
		}
		lo, hi := exceedsBracket(out.Total, over)
		_, resolved := out.Total.Quantile(tail)
		if over < resolved || hi == 0 || tail >= fireTailMax {
			return Firing{Lo: lo, Hi: hi}, e.Mean
		}
		share := lo
		if share == 0 {
			share = hi / 16
		}
		tail = math.Min(fireTailMax, 1-fireTail*share)
	}
	return Firing{Lo: nan, Hi: nan, Reason: "the sum's tail above the level was not resolved"}, e.Mean
}

// exceedsBracket brackets P(S > x) from a bracket on P(S ≥ kh): rounded
// up, S lies on the lattice and exceeds x exactly when it reaches the
// next point above x, and rounded down it exceeds x no more often; so
// the bracket at that point holds P(S > x). Beyond the lattice the upper
// curve's last value bounds it.
func exceedsBracket(b Bounds, x float64) (lo, hi float64) {
	k := int(math.Floor(x/b.H)) + 1
	if k >= len(b.Lower) {
		return 0, b.Upper[len(b.Upper)-1]
	}
	return b.Lower[k], b.Upper[k]
}

// fireGraph is the model of a trigger's sum: the graph with only the
// trigger's inputs booked, or a value of 1 at each node whose entries it
// counts, in one category, and no layers or triggers. A threshold that
// tests a value outside the sum becomes a selection with its fixed
// probabilities. Holding times go when no input is booked per time,
// since a booking counts toward its event's period wherever it lands.
func fireGraph(g Graph, id string) Graph {
	var tr *Trigger
	for _, n := range g.Nodes {
		if n.ID == id {
			tr = n.Trigger
		}
	}
	perTime := false
	nodes := make([]Node, len(g.Nodes))
	for i, n := range g.Nodes {
		nodes[i] = sumNode(n, tr)
		for _, f := range nodes[i].Functions {
			perTime = perTime || f.Basis == PerTime
		}
	}
	if !perTime {
		for i := range nodes {
			dropHoldings(&nodes[i])
		}
	}
	return Graph{ID: g.ID, Sources: g.Sources, Nodes: nodes, Factors: g.Factors, Schedules: g.Schedules, Phased: g.Phased}
}

// sumNode is node n in the model of trigger tr's sum: its functions in
// the sum, in its one category, and no trigger of its own.
func sumNode(n Node, tr *Trigger) Node {
	in := func(fn string) bool {
		for _, x := range tr.Inputs {
			if x.Node == n.ID && (x.Function == "" || x.Function == fn) {
				return true
			}
		}
		return false
	}
	n.Trigger = nil
	var fns []Function
	for _, f := range n.Functions {
		if in(f.Name) {
			f.Category, f.Range = fireCategory, nil
			fns = append(fns, f)
		}
	}
	for _, c := range tr.Entries {
		if c == n.ID {
			fns = append(fns, Function{Name: "entries", Category: fireCategory, Model: point(1)})
		}
	}
	if th := n.Threshold; th != nil && !in(th.Function) {
		n.Outcomes = thresholdOutcomes(n)
		n.Threshold = nil
	}
	n.Functions = fns
	return n
}

// thresholdOutcomes are a threshold's branches as a selection, above
// with P(X > Over) and below with the rest.
func thresholdOutcomes(n Node) []Outcome {
	th := n.Threshold
	var x dist.Distribution
	for _, f := range n.Functions {
		if f.Name == th.Function {
			x = f.Model
		}
	}
	return []Outcome{{To: th.Above, P: point(x.Survival(th.Over))}, {To: th.Below}}
}

// dropHoldings removes a node's holding times.
func dropHoldings(n *Node) {
	n.NextHolding, n.NextOutage, n.NextRange = nil, false, nil
	outs := make([]Outcome, len(n.Outcomes))
	for j, o := range n.Outcomes {
		o.Holding, o.OutageHolding, o.HoldingRange = nil, false, nil
		outs[j] = o
	}
	n.Outcomes = outs
}

// point is the constant distribution at a finite value.
func point(v float64) dist.Constant {
	d, _ := dist.NewConstant(v) // the callers' values are finite
	return d
}

// maskFirings sets to NaN every mean a trigger with an unknown P(fire)
// reaches: the total's, its source's, and the entries, outcomes,
// contributions and categories of the nodes its walk can reach.
func (m *Compiled) maskFirings(e *Expectation) {
	nan := math.NaN()
	for _, tr := range m.trigs {
		if tr.firing.Exact() {
			continue
		}
		e.Mean, e.DrawsMean, e.StepsPerPeriod, e.RoundsPerPeriod = nan, nan, nan, nan
		e.MeanBySource[tr.src] = nan
		for v, r := range m.reach[tr.src] {
			if r > 0 {
				m.maskNode(e, index(v))
			}
		}
	}
}

// maskNode sets node v's entries, outcomes, contribution and the
// categories it books to NaN.
func (m *Compiled) maskNode(e *Expectation, v int32) {
	nan := math.NaN()
	n := m.nodes[v]
	e.EntriesPer[v], e.Contrib[v] = nan, nan
	// bound: outcomes of one node
	for j := n.outStart; j < n.outEnd; j++ {
		e.OutcomeRate[j] = nan
		if f := m.outs[j].fn; f != noIndex {
			e.MeanByCategory[m.fns[f].category] = nan
		}
	}
	// bound: functions of one node
	for f := n.fnStart; f < n.rateEnd; f++ {
		e.MeanByCategory[m.fns[f].category] = nan
	}
	if n.thresh != noIndex {
		e.MeanByCategory[m.fns[n.thresh].category] = nan
	}
}

// periodEnd is the time from a period's start at which a trigger's
// round enters its node: the period's last instant, the float below 1,
// so that it books in that period.
const periodEnd = 1 - 0x1p-53

// feed adds a booking of function f to the sums of the triggers it is
// an input of. A trigger's walk reaches only the inputs of triggers
// evaluated after it, the order following the dependencies, so every
// booking arrives before the sum it feeds is read.
func (w *Workspace) feed(wk *worker, f int32, x float64) {
	m := w.m
	// bound: the triggers one function feeds
	for i := m.feedStart[f]; i < m.feedStart[f+1]; i++ {
		wk.sums[m.feeds[i]] += x
	}
}

// count adds an entry into node v to the counts of the triggers that
// count it.
func (w *Workspace) count(wk *worker, v int32) {
	m := w.m
	// bound: the triggers one node's entries feed
	for i := m.countStart[v]; i < m.countStart[v+1]; i++ {
		wk.sums[m.counts[i]]++
	}
}

// runTrigger walks source k's trigger once if its sum over period p is
// above its level. The sources walk before the triggers, and each
// trigger after those whose walks feed it, so the sum is the period's
// whole.
func (w *Workspace) runTrigger(wk *worker, sl *slot, k int32, p, first int64) {
	tr := w.m.trigs[w.m.srcs[k].trigger]
	if !(wk.sums[w.m.srcs[k].trigger] > tr.over) {
		return
	}
	if p >= 0 {
		sl.events[k]++
	}
	if w.stream {
		w.beginRound(wk, periodEnd)
	}
	w.walk(wk, sl, k, tr.node, periodEnd, p, first)
}

// triggerInputs lists the nodes whose functions or entries feed
// trigger i, once each, in node order.
func (m *Compiled) triggerInputs(i int32) []string {
	var out []string
	for v, n := range m.nodes {
		fed := false
		// bound: the triggers one node's entries feed
		for j := m.countStart[v]; j < m.countStart[v+1] && !fed; j++ {
			fed = m.counts[j] == i
		}
		// bound: functions of one node
		for f := n.fnStart; f < n.rateEnd && !fed; f++ {
			fed = m.feedsInto(f, i)
		}
		if n.thresh != noIndex && !fed {
			fed = m.feedsInto(n.thresh, i)
		}
		if fed {
			out = append(out, m.nodeIDs[v])
		}
	}
	return out
}

// feedsInto reports whether function f feeds trigger i.
func (m *Compiled) feedsInto(f, i int32) bool {
	// bound: the triggers one function feeds
	for j := m.feedStart[f]; j < m.feedStart[f+1]; j++ {
		if m.feeds[j] == i {
			return true
		}
	}
	return false
}

// fireOnZero reports whether a trigger can fire in a period whose
// total is otherwise 0: it counts entries into a node that may book
// nothing, or the firings of such a trigger. A trigger on booked values
// fires only on a positive sum.
func (m *Compiled) fireOnZero() bool {
	trigOf := map[int32]int{}
	zero := make([]bool, len(m.trigs))
	for i, tr := range m.trigs {
		trigOf[tr.node] = i
		for _, v := range m.countedBy(index(i)) {
			if j, ok := trigOf[v]; ok {
				zero[i] = zero[i] || zero[j]
			} else {
				zero[i] = zero[i] || !m.alwaysBooks(v)
			}
		}
		if zero[i] {
			return true
		}
	}
	return false
}

// alwaysBooks reports whether an entry into node v books a positive
// value for certain: one of its OnEntry values is never 0.
func (m *Compiled) alwaysBooks(v int32) bool {
	n := m.nodes[v]
	// bound: functions of one node
	for f := n.fnStart; f < n.fnEnd; f++ {
		if m.pools[m.fns[f].pool].CDF(0) == 0 {
			return true
		}
	}
	return n.thresh != noIndex && m.pools[m.fns[n.thresh].pool].CDF(0) == 0
}

// countedBy lists the nodes whose entries trigger i counts.
func (m *Compiled) countedBy(i int32) []int32 {
	var out []int32
	for v := range m.nodes {
		// bound: the triggers one node's entries feed
		for j := m.countStart[v]; j < m.countStart[v+1]; j++ {
			if m.counts[j] == i {
				out = append(out, index(v))
			}
		}
	}
	return out
}

// HasTriggers reports whether the model has a trigger, whose firing
// moves with the rest of the total and leaves the variance without a
// closed form.
func (m *Compiled) HasTriggers() bool { return len(m.trigs) > 0 }

// Firings returns each trigger's P(fire).
func (m *Compiled) Firings() []Firing {
	out := make([]Firing, len(m.trigs))
	for i, tr := range m.trigs {
		out[i] = tr.firing
	}
	return out
}
