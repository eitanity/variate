package variate

import (
	"math"

	"code.eitanity.com/variate/dist"
)

// maxGiven bounds the rounds one entry of a given state may draw in a
// repeat.
const maxGiven = 1 << 24

// Given is one part of what is under way at the start of a started run:
// Count rounds of Source at node Node. Without To, Node is a queue and
// the rounds are in it, each there for Age: the oldest in service, the
// rest waiting, oldest first. With To, the rounds are part-way through
// the holding time of Node's outcome to To, each held for Elapsed so
// far. Nothing they booked before period 0 is booked again: no on-entry
// value at Node, and the per-time values over what is left of the stay
// or the holding.
type Given struct {
	Node, To string
	Source   string            // whose rounds they are; "" for the model's one source
	Count    dist.Distribution // rounds, drawn once a repeat and rounded to a whole number
	Age      dist.Distribution // at a queue: how long each round has been there, in periods
	// InService is, at a queue, how long a round in service has been
	// served; nil draws its remaining work from the work's equilibrium
	// residual, as in a queue that has long been running.
	InService dist.Distribution
	// Elapsed is, in a holding, how long a round has been held; nil
	// draws what is left from the holding's equilibrium residual.
	Elapsed dist.Distribution
}

// cgiven is a compiled entry of a given state: its source, its queue or
// its holding outcome, and its pools, noIndex where not given; resid
// draws the equilibrium residual of the work or the holding when the
// entry leaves the progress unknown.
type cgiven struct {
	k, node, queue, out            int32
	count, age, inService, elapsed int32
	law                            dist.Distribution // the work's or the holding's
	resid                          residTable
	equilibrium                    bool
}

// checkGiven checks a given state: a started run's, each entry at a
// queue with the rounds' ages or in a holding time drawn from a law, of
// a source the model has, with counts and times that are not negative.
func (b *builder) checkGiven() error {
	st := b.g.Start
	if len(st.State) > 0 && st.Horizon == 0 {
		return modelError("start", "state", "a given state starts a started run: give a horizon")
	}
	for _, g := range st.State {
		if err := b.checkGivenEntry(g); err != nil {
			return err
		}
	}
	return nil
}

func (b *builder) checkGivenEntry(g Given) error {
	subject := givenSubject(g)
	v, ok := b.nodeIndex[g.Node]
	if !ok {
		return modelError("start", subject, "node "+g.Node+" not found")
	}
	if _, err := b.givenSource(g); err != nil {
		return err
	}
	if err := checkSlot("start", subject+" count", g.Count, 0, math.Inf(1), false); err != nil {
		return err
	}
	n := b.g.Nodes[v]
	if g.To == "" {
		return b.checkGivenQueue(g, n, subject)
	}
	return b.checkGivenHolding(g, n, subject)
}

func (b *builder) checkGivenQueue(g Given, n Node, subject string) error {
	switch {
	case n.Queue == nil:
		return modelError("start", subject, g.Node+" is not a queue: give to, the outcome whose holding time the rounds are in")
	case g.Elapsed != nil:
		return modelError("start", subject, "elapsed is a holding's; rounds at a queue take age and in_service")
	}
	if err := checkSlot("start", subject+" age", g.Age, 0, math.Inf(1), false); err != nil {
		return err
	}
	if err := checkSlot("start", subject+" in_service", g.InService, 0, math.Inf(1), true); err != nil {
		return err
	}
	return checkResidual(subject, "the work", "in_service", g.InService, n.Queue.Work)
}

func (b *builder) checkGivenHolding(g Given, n Node, subject string) error {
	if g.Age != nil || g.InService != nil {
		return modelError("start", subject, "age and in_service are a queue's; rounds in a holding time take elapsed")
	}
	hold, outage, found := givenHold(n, g.To)
	switch {
	case found != 1:
		return modelError("start", subject, g.Node+" needs exactly one outcome to "+g.To+" with a holding time; it has "+itoa(found))
	case outage:
		return modelError("start", subject, "an outage holding lasts as long as its block is down, which a given state does not hold")
	case hold == nil:
		return modelError("start", subject, "the outcome to "+g.To+" has no holding time")
	}
	if err := checkSlot("start", subject+" elapsed", g.Elapsed, 0, math.Inf(1), true); err != nil {
		return err
	}
	return checkResidual(subject, "the holding time", "elapsed", g.Elapsed, hold)
}

// checkResidual refuses an entry that leaves the time so far unknown when
// the law d's equilibrium residual, of mean E[X²]/(2E[X]), has an infinite
// mean.
func checkResidual(subject, law, field string, given, d dist.Distribution) error {
	if given != nil || d == nil || !math.IsInf(d.Var(), 1) {
		return nil
	}
	return modelError("start", subject, law+" has infinite variance, so what is left drawn from its long-run residual has an infinite mean: give "+field)
}

// givenHold finds node n's outcomes to the node to: the holding of the
// last found, whether it is an outage, and how many there are.
func givenHold(n Node, to string) (hold dist.Continuous, outage bool, found int) {
	for _, o := range n.Outcomes {
		if o.To == to {
			hold, outage, found = o.Holding, o.OutageHolding, found+1
		}
	}
	if n.Next == to && to != "" {
		hold, outage, found = n.NextHolding, n.NextOutage, found+1
	}
	return hold, outage, found
}

// givenSource is the source an entry names, or the model's one source.
func (b *builder) givenSource(g Given) (int32, error) {
	if g.Source == "" {
		if len(b.g.Sources) != 1 {
			return 0, modelError("start", givenSubject(g), "the model has "+itoa(len(b.g.Sources))+" sources: name the rounds' source")
		}
		return 0, nil
	}
	for i, s := range b.g.Sources {
		if s.ID == g.Source {
			return index(i), nil
		}
	}
	return 0, modelError("start", givenSubject(g), "source "+g.Source+" not found")
}

func givenSubject(g Given) string {
	if g.To == "" {
		return g.Node
	}
	return g.Node + "→" + g.To
}

// flattenGiven compiles the given state, its pools after the model's.
func (b *builder) flattenGiven(m *Compiled) {
	for _, g := range b.g.Start.State {
		k, _ := b.givenSource(g)
		v := b.nodeIndex[g.Node]
		cg := cgiven{k: k, node: v, queue: m.nodes[v].queue, out: noIndex, age: noIndex, inService: noIndex, elapsed: noIndex}
		name := "given " + g.Node + "→" + g.To
		cg.count = m.addPool(g.Count, PerEvent, name+" count")
		m.givenKey = append(m.givenKey, m.poolKey[cg.count])
		if g.To == "" {
			cg.law = m.pools[m.queues[cg.queue].work]
			cg.age = m.addPool(g.Age, PerEvent, name+" age")
			cg.inService = m.optPool(g.InService, name+" in service")
		} else {
			cg.queue = noIndex
			cg.out = m.givenOutcome(v, b.nodeIndex[g.To])
			cg.law = m.pools[m.outs[cg.out].hold]
			cg.elapsed = m.optPool(g.Elapsed, name+" elapsed")
		}
		cg.equilibrium = cg.inService == noIndex && cg.elapsed == noIndex
		if lo, hi := cg.law.Support(); cg.equilibrium && lo != hi {
			cg.resid = newResidTable(cg.law)
		}
		m.given = append(m.given, cg)
	}
}

func (m *Compiled) optPool(d dist.Distribution, name string) int32 {
	if d == nil {
		return noIndex
	}
	return m.addPool(d, PerEvent, name)
}

// givenOutcome is node v's outcome to node to with a holding time.
func (m *Compiled) givenOutcome(v, to int32) int32 {
	n := m.nodes[v]
	// bound: outcomes of one node
	for j := n.outStart; j < n.outEnd; j++ {
		if m.outs[j].to == to && m.outs[j].hold != noIndex {
			return j
		}
	}
	return noIndex
}

// givenRounds is the mean rounds of a repeat's given state.
func (m *Compiled) givenRounds() float64 {
	n := 0.0
	for _, g := range m.given {
		n += m.poolMean[g.count]
	}
	return n
}

// markGiven sets to NaN, in the exact per-period means, what the given
// state's rounds book: the rates over the rest of a stay or holding,
// and everything after it.
func (m *Compiled) markGiven(rows [][]float64, l layout) {
	for _, g := range m.given {
		if n := m.nodes[g.node]; n.rateStart != n.rateEnd {
			m.markValues(rows, l, int(g.k), g.node)
		}
		var from []int32
		if g.out != noIndex {
			from = []int32{m.outs[g.out].to}
		} else {
			from = m.successors(g.node)
		}
		for v, ok := range m.reachedFrom(from) {
			if ok {
				m.markNode(rows, l, int(g.k), index(v))
			}
		}
	}
}

// reachedFrom marks the nodes a round reaches from the nodes from, them
// included.
func (m *Compiled) reachedFrom(from []int32) []bool {
	seen := make([]bool, len(m.nodes))
	for _, v := range from {
		seen[v] = true
	}
	for _, v := range m.topo {
		if !seen[v] {
			continue
		}
		for _, to := range m.successors(v) {
			seen[to] = true
		}
	}
	return seen
}

// givenScale adds to the sizing means of a repeat what the given
// state's rounds book from their node on.
func (m *Compiled) givenScale(e *Expectation) {
	if len(m.given) == 0 {
		return
	}
	all, _ := m.valueToGo(valueFilter{kind: filterAll})
	cats := make([][]float64, len(m.categories))
	for c := range cats {
		cats[c], _ = m.valueToGo(valueFilter{kind: filterCategory, index: index(c)})
	}
	for _, g := range m.given {
		n := m.poolMean[g.count]
		e.Mean += n * all[g.node]
		e.MeanBySource[g.k] += n * all[g.node]
		for c := range cats {
			e.MeanByCategory[c] += n * cats[c][g.node]
		}
	}
}

// StartState describes a started run's given state, an entry a line:
// the mean rounds and where they are, and whether what is left of a
// service or a holding follows the time so far or, not given, the
// long-run residual.
func (m *Compiled) StartState() []string {
	out := make([]string, 0, len(m.given))
	for _, g := range m.given {
		s := figure(m.poolMean[g.count]) + " rounds of " + m.sourceIDs[g.k]
		if g.out != noIndex {
			s += " part-way through the holding " + m.outcomeNames[g.out] + ", what is left "
		} else {
			s += " at queue " + m.nodeIDs[g.node] + ", the oldest in service with the work left "
		}
		if g.equilibrium {
			s += "drawn from the long-run residual, the time so far not given"
		} else {
			s += "drawn given the time so far"
		}
		out = append(out, s)
	}
	return out
}

// GivenState reports whether a started run starts from a given state.
func (m *Compiled) GivenState() bool { return len(m.given) > 0 }
