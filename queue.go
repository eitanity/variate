package variate

import (
	"math"
	"slices"

	"code.eitanity.com/variate/stats"
)

const (
	// maxServers bounds a queue's servers.
	maxServers = 1 << 16
	// queueSlack and queueFloor size a queue's line: a multiple of the
	// rounds expected to reach it in a task, plus a floor.
	queueSlack = 2
	queueFloor = 1024
	// maxQueueItems bounds a queue's line.
	maxQueueItems = 1 << 26
	// maxDrain bounds the periods a started run's reset queues take to
	// empty after the horizon.
	maxDrain = 1 << 24
)

// The row columns of a queue, each by period: rounds that joined, rounds
// served of those that joined, their waits' and stays' sums, the line's
// length at the period's end, the servers' busy time, and the rounds
// sent to the unfinished node at its end.
const (
	qJoined = iota
	qServed
	qWait
	qStay
	qLength
	qBusy
	qUnfinished
	queueCols
)

// cqueue is a compiled queue.
type cqueue struct {
	node       int32
	servers    int32
	speed      float64
	work       int32 // the work's pool
	reset      bool
	unfinished int32 // the node the unserved rounds go to with reset, or noIndex
	cap        int   // rounds the line, and the arrivals still to join, may hold
}

// qitem is a round at a queue: when it reached the queue, from the start
// of period 0, before it for a round of a given state; its arrival
// period and source; its work; and the order it was parked in, which
// breaks ties in time.
type qitem struct {
	t     float64
	p     int64
	k     int32
	given int32 // its entry of the given state at this queue, from 1; −1 for a given state's round from a holding; 0 for an arrival
	work  float64
	seq   uint64
	rkey  uint64  // the round's key, which its draws after the queue keep
	rated float64 // when the queue node's rates over its stay are booked to
}

// qserver is a server: the round it serves, when it began and when it
// ends.
type qserver struct {
	item          qitem
	start, finish float64
	spanned       float64 // when its busy time is booked to
	busy          bool
}

// wqueue is a worker's state of one queue within a task: the rounds still
// to reach it, as a heap by time; the line, a ring in order of joining;
// and the servers.
type wqueue struct {
	arr     []qitem
	nArr    int
	line    []qitem
	head    int
	nLine   int
	servers []qserver
	nBusy   int
}

func newWQueue(q cqueue) wqueue {
	return wqueue{arr: make([]qitem, q.cap), line: make([]qitem, q.cap), servers: make([]qserver, q.servers)}
}

// ---- compile ----

// checkQueue: a queue has at least one server of a positive, finite
// speed, a work law on [0, ∞) with a finite mean, an unfinished node
// that exists exactly when it resets, and no selection the line cannot
// hold: no threshold or trigger at the node.
func (b *builder) checkQueue(n Node) error {
	q := n.Queue
	if q == nil {
		return nil
	}
	switch {
	case q.Servers < 1 || q.Servers > maxServers:
		return modelError("queue", n.ID, "needs between 1 and "+itoa(maxServers)+" servers")
	case !(q.Speed > 0) || math.IsInf(q.Speed, 1):
		return modelError("queue", n.ID, "needs a positive, finite speed")
	case n.Threshold != nil || n.Trigger != nil:
		return modelError("queue", n.ID, "a queue node chooses its outcome when served, so it takes no threshold or trigger")
	case q.Reset && q.Unfinished == "":
		return modelError("queue", n.ID, "a queue that resets needs an unfinished node for the rounds unserved at a period's end")
	case !q.Reset && q.Unfinished != "":
		return modelError("queue", n.ID, "only a queue that resets sends rounds to an unfinished node")
	}
	if q.Unfinished != "" {
		if _, ok := b.nodeIndex[q.Unfinished]; !ok {
			return modelError("queue", n.ID, "unfinished node "+q.Unfinished+" not found")
		}
	}
	return checkSlot("queue", n.ID+" work", q.Work, 0, math.Inf(1), false)
}

// checkQueues checks what the queues ask of the rest of the graph. A
// round reaching a queue loses its walk's own state, so no outage holding
// is reached from one. No trigger's inputs are reached from a queue,
// whose rounds may book periods after they arrived. In a steady-state
// run each queue is reached within the period its rounds arrive: no
// holding time, trigger, unfinished exit or queue that carries its line
// on the way to it.
func (b *builder) checkQueues() error {
	queues := false
	for _, n := range b.g.Nodes {
		queues = queues || n.Queue != nil
	}
	if !queues {
		return nil
	}
	if err := b.checkAfterQueues(); err != nil {
		return err
	}
	if b.g.Start.Horizon > 0 {
		return nil
	}
	return b.checkSteadyQueues()
}

// checkAfterQueues refuses an outage holding after a queue, and a
// trigger whose inputs a queue reaches.
func (b *builder) checkAfterQueues() error {
	after := b.afterQueues()
	for i, n := range b.g.Nodes {
		if after[i] && (n.NextOutage || hasOutage(n)) {
			return modelError("node", n.ID, "an outage holding cannot follow a queue: the round leaves the outage behind while it waits")
		}
		if n.Trigger == nil {
			continue
		}
		for _, id := range triggerNodes(n.Trigger) {
			if v, ok := b.nodeIndex[id]; ok && (after[v] || b.g.Nodes[v].Queue != nil) {
				return modelError("trigger", n.ID, "input "+id+" is reached through a queue, whose rounds may book periods after they arrived; a trigger after a queue is not supported")
			}
		}
	}
	return nil
}

// checkSteadyQueues refuses, in a steady-state run, a queue reached
// after its rounds' arrival period.
func (b *builder) checkSteadyQueues() error {
	late := b.lateNodes()
	for i, n := range b.g.Nodes {
		switch {
		case n.Queue == nil:
		case late[i]:
			return modelError("queue", n.ID, "in a steady-state run a queue must be reached within the period its rounds arrive: no holding time, trigger or unfinished exit on the way to it; a started run has no such limit")
		}
	}
	return nil
}

func hasOutage(n Node) bool {
	for _, o := range n.Outcomes {
		if o.OutageHolding {
			return true
		}
	}
	return false
}

// afterQueues marks the nodes a round can reach after leaving a queue.
func (b *builder) afterQueues() []bool {
	after := make([]bool, len(b.g.Nodes))
	for _, v := range b.topoOrder() {
		n := b.g.Nodes[v]
		if !after[v] && n.Queue == nil {
			continue
		}
		for _, to := range b.targets(n) {
			after[to] = true
		}
	}
	return after
}

// lateNodes marks the nodes a round can reach after its arrival period:
// past a holding time, as a trigger's round at the period's end, by a
// queue's unfinished exit at its end, or past a queue that carries its
// line.
func (b *builder) lateNodes() []bool {
	late := make([]bool, len(b.g.Nodes))
	for i, n := range b.g.Nodes {
		late[i] = late[i] || n.Trigger != nil
	}
	for _, v := range b.topoOrder() {
		n := b.g.Nodes[v]
		mark := func(to string, held bool) {
			if t, ok := b.nodeIndex[to]; ok && (late[v] || held) {
				late[t] = true
			}
		}
		carries := n.Queue != nil && !n.Queue.Reset
		for _, o := range n.Outcomes {
			mark(o.To, o.Holding != nil || o.OutageHolding || carries)
		}
		if n.Next != "" {
			mark(n.Next, n.NextHolding != nil || n.NextOutage || carries)
		}
		if n.Threshold != nil {
			mark(n.Threshold.Above, false)
			mark(n.Threshold.Below, false)
		}
		if n.Queue != nil && n.Queue.Unfinished != "" {
			mark(n.Queue.Unfinished, true)
		}
	}
	return late
}

// flattenQueue compiles node i's queue.
func (b *builder) flattenQueue(m *Compiled, i int32, n Node, cn *cnode) {
	cn.queue = noIndex
	if n.Queue == nil {
		return
	}
	q := n.Queue
	cq := cqueue{node: i, servers: int32(q.Servers), speed: q.Speed, work: m.addPool(q.Work, PerEvent, "work "+n.ID), reset: q.Reset, unfinished: noIndex} //nolint:gosec // servers are checked below 2^16
	if q.Unfinished != "" {
		cq.unfinished = b.nodeIndex[q.Unfinished]
	}
	cn.queue = index(len(m.queues))
	m.queues = append(m.queues, cq)
	m.hasDelays = true
}

// sizeQueues sets each queue's capacity from the rounds expected to reach
// it in a task: a started run's horizon, or one period, since a reset
// queue empties at each period's end, or carriedPeriods for a queue that
// carries its line in a steady-state run.
func (m *Compiled) sizeQueues() error {
	for i := range m.queues {
		q := &m.queues[i]
		periods := float64(max(m.horizon, 1))
		if !q.reset && m.horizon == 0 {
			periods = carriedPeriods
		}
		in := 0.0
		for k := range m.srcs {
			in += m.countMean[k] * m.reach[k][q.node]
		}
		need := queueSlack*(in*periods+m.givenRounds()) + queueFloor
		if !(need <= maxQueueItems) {
			return modelError("queue", m.nodeIDs[q.node], "expects "+figure(in*periods)+" rounds in a task, more than a queue can hold ("+itoa(maxQueueItems)+")")
		}
		q.cap = int(need)
	}
	return nil
}

// queueScale adds to the sizing means what each queue node's per-time
// rates book over a stay, taken at its least, the service alone.
func (m *Compiled) queueScale(e *Expectation) {
	for _, q := range m.queues {
		n := m.nodes[q.node]
		service := m.poolMean[q.work] / q.speed
		// bound: functions of one node
		for f := n.rateStart; f < n.rateEnd; f++ {
			x := m.poolMean[m.fns[f].pool] * service
			e.MeanByCategory[m.fns[f].category] += x * e.EntriesPer[q.node]
			if !m.fnInTotal(f) {
				continue
			}
			e.Mean += x * e.EntriesPer[q.node]
			e.Contrib[q.node] += x * e.EntriesPer[q.node]
			for k := range m.srcs {
				e.MeanBySource[k] += x * m.countMean[k] * m.reach[k][q.node]
			}
		}
	}
}

// HasQueues reports whether any node is a queue.
func (m *Compiled) HasQueues() bool { return len(m.queues) > 0 }

// QueueIDs returns the queue nodes' names in queue order.
func (m *Compiled) QueueIDs() []string {
	out := make([]string, len(m.queues))
	for i, q := range m.queues {
		out[i] = m.nodeIDs[q.node]
	}
	return out
}

// QueueServers returns queue i's servers.
func (m *Compiled) QueueServers(i int) int { return int(m.queues[i].servers) }

// queueReached marks the nodes a round reaches after a queue, and the
// queue nodes themselves.
func (m *Compiled) queueReached() (after, queue []bool) {
	after = make([]bool, len(m.nodes))
	queue = make([]bool, len(m.nodes))
	for _, q := range m.queues {
		queue[q.node] = true
	}
	for _, v := range m.topo {
		if !after[v] && !queue[v] {
			continue
		}
		n := m.nodes[v]
		// bound: outcomes of one node
		for j := n.outStart; j < n.outEnd; j++ {
			after[m.outs[j].to] = true
		}
		if c := n.queue; c != noIndex && m.queues[c].unfinished != noIndex {
			after[m.queues[c].unfinished] = true
		}
	}
	return after, queue
}

// maskQueues sets to NaN the steady-state means a queue leaves without a
// closed form: what is booked after it, and over its stay; the entries,
// reach and outcomes after it, and the sources, categories and total
// these reach.
func (m *Compiled) maskQueues(e *Expectation) {
	if !m.HasQueues() {
		return
	}
	after, queue := m.queueReached()
	nan := math.NaN()
	// The reach is the compiled table's; the expectations get their own.
	reach := make([][]float64, len(e.Reach))
	for k, r := range e.Reach {
		reach[k] = slices.Clone(r)
	}
	e.Reach = reach
	for v := range m.nodes {
		if after[v] {
			e.EntriesPer[v] = nan
			for k := range e.Reach {
				e.Reach[k][v] = nan
			}
		}
		n := m.nodes[v]
		if n.hasFunctions() && (after[v] || (queue[v] && n.rateStart != n.rateEnd)) {
			m.maskBooked(e, index(v), after[v])
		}
	}
	for j := range m.outs {
		v := m.outcomeNode[j]
		if after[v] || queue[v] {
			e.OutcomeRate[j] = nan
		}
	}
}

// maskBooked sets to NaN what node v books: its contribution, the total,
// the sources reaching it, and its categories, at a queue node only
// those of its rates over the stay.
func (m *Compiled) maskBooked(e *Expectation, v int32, after bool) {
	nan := math.NaN()
	e.Contrib[v] = nan
	e.Mean = nan
	for c := range m.categories {
		if (after && m.nodeBooks(v, index(c))) || (!after && m.ratesBook(m.nodes[v], index(c))) {
			e.MeanByCategory[c] = nan
		}
	}
	for k := range m.srcs {
		if m.reach[k][v] > 0 {
			e.MeanBySource[k] = nan
		}
	}
}

// ---- results ----

// itemAccum is the statistics of one per-round quantity of a queue, its
// wait or its stay, over the rounds served: their sum and count, a
// histogram, and per batch group, the tasks dealt round robin, a coarse
// histogram, the count and the sum, for the errors.
type itemAccum struct {
	Sum      stats.Neumaier
	N        int64
	Hist     *stats.LogHist
	Groups   []*stats.LogHist
	GroupN   []int64
	GroupSum []stats.Neumaier
}

func newItemAccum(mean, res float64) itemAccum {
	a := itemAccum{Hist: stats.NewLogHist(mean, res), GroupN: make([]int64, batchGroups), GroupSum: make([]stats.Neumaier, batchGroups)}
	a.Groups = make([]*stats.LogHist, batchGroups)
	a.Groups[0] = stats.NewCoarse(a.Hist, groupRes)
	for g := 1; g < batchGroups; g++ {
		a.Groups[g] = a.Groups[0].Empty()
	}
	return a
}

func (a *itemAccum) reset() {
	a.Sum, a.N = stats.Neumaier{}, 0
	a.Hist.Reset()
	for _, h := range a.Groups {
		h.Reset()
	}
	clear(a.GroupN)
	clear(a.GroupSum)
}

// merge adds a task's histograms and its sum, which its row columns hold.
func (a *itemAccum) merge(task int64, h, coarse *stats.LogHist, sum float64) {
	g := task % batchGroups
	a.Hist.Merge(h)
	a.Groups[g].Merge(coarse)
	a.GroupN[g] += h.N
	a.GroupSum[g].Add(sum)
	a.Sum.Add(sum)
	a.N += h.N
}

// ItemStat summarises a per-round quantity of a queue over the rounds
// served: the count, the mean with its standard error over the batch
// groups, and the quantiles with their 95% intervals and errors, which
// the groups widen as they do a period's.
type ItemStat struct {
	N                        int64
	Mean, StdErr             float64
	P50, P90, P95, P99, P999 float64
	Clipped                  bool
	QLo, QHi, QSE            [5]float64
	StdErrUnreliable         bool // the variance is infinite, so the mean's standard error means nothing
}

func (a *itemAccum) stat() ItemStat {
	s := ItemStat{N: a.N, Mean: math.NaN(), StdErr: math.NaN()}
	if a.N == 0 {
		return s
	}
	s.Mean = a.Sum.Value() / float64(a.N)
	s.StdErr = groupErr(a.GroupN, func(g int) float64 { return a.GroupSum[g].Value() })
	var q [5]float64
	for i, p := range QuantileLevels() {
		var c, ci bool
		q[i], c = a.Hist.Quantile(p)
		d := quantileZ * a.inflation(q[i]) * math.Sqrt(p*(1-p)/float64(a.N))
		s.QLo[i], s.QHi[i], ci = a.Hist.QuantileInterval(p, d)
		s.QSE[i] = (s.QHi[i] - s.QLo[i]) / (2 * quantileZ)
		s.Clipped = s.Clipped || c || ci
	}
	s.P50, s.P90, s.P95, s.P99, s.P999 = q[0], q[1], q[2], q[3], q[4]
	return s
}

// inflation is the ratio of the groups' standard error of the share of
// rounds at or above x to its binomial one: rounds of one task share its
// line, so they are not independent.
func (a *itemAccum) inflation(x float64) float64 {
	var n, c int64
	for g, h := range a.Groups {
		n += a.GroupN[g]
		c += h.AtOrAbove(x)
	}
	if c == 0 || c == n {
		return 1
	}
	p := float64(c) / float64(n)
	batch := groupErr(a.GroupN, func(g int) float64 { return float64(a.Groups[g].AtOrAbove(x)) })
	return batch / math.Sqrt(p*(1-p)/float64(n))
}

// successors lists the nodes a round goes to from node v: its outcomes'
// targets, and a queue's unfinished node.
func (m *Compiled) successors(v int32) []int32 {
	n := m.nodes[v]
	out := make([]int32, 0, n.outEnd-n.outStart+1)
	// bound: outcomes of one node
	for j := n.outStart; j < n.outEnd; j++ {
		out = append(out, m.outs[j].to)
	}
	if c := n.queue; c != noIndex && m.queues[c].unfinished != noIndex {
		out = append(out, m.queues[c].unfinished)
	}
	return out
}

// ratesBook reports whether a per-time function of node n is in
// category c.
func (m *Compiled) ratesBook(n cnode, c int32) bool {
	// bound: functions of one node
	for f := n.rateStart; f < n.rateEnd; f++ {
		if m.fns[f].category == c {
			return true
		}
	}
	return false
}
