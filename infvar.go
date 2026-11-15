package variate

import "math"

// markInfiniteVariance finds the row quantities whose variance is
// infinite: those a function books into when its value, or its rate,
// has infinite variance, and those a source reaches when its count per
// period has. A period's per-time booking of one round is at most its
// rate, so a long holding or stay adds no infinite variance. Each cause
// is warned of once.
func (m *Compiled) markInfiniteVariance(ws *[]Warning) {
	l := newLayout(m)
	m.infVar = make([]bool, l.width)
	for f := range m.fns {
		if !math.IsInf(m.pools[m.fns[f].pool].Var(), 1) {
			continue
		}
		v := m.fnNode[f]
		reached := false
		for k := range m.srcs {
			if m.countMean[k] > 0 && m.reach[k][v] > 0 {
				reached = true
				m.markBooked(&l, index(k), int32(f))
			}
		}
		if reached {
			*ws = append(*ws, Warning{Code: infVarCode, Subject: m.fnNames[f], Message: "the value has a finite mean and infinite variance: " + infVarNote})
		}
	}
	m.markQueues(ws)
	for k := range m.srcs {
		if m.countMean[k] > 0 && math.IsInf(m.countN2[k], 1) {
			m.markSource(&l, index(k))
			*ws = append(*ws, Warning{Code: infVarCode, Subject: m.sourceIDs[k], Message: "the count per period has a finite mean and infinite variance: " + infVarNote})
		}
	}
}

const infVarCode = "infinite-variance"

const infVarNote = "the standard errors of the quantities it reaches are unreliable, and their target-rse and contrib-rse criteria do not apply"

// markBooked marks what function f books for source k's rounds.
func (m *Compiled) markBooked(l *layout, k, f int32) {
	m.infVar[l.total()] = true
	m.infVar[l.cat(m.fns[f].category)] = true
	m.infVar[l.src(k)] = true
	m.infVar[l.contrib(m.fnNode[f])] = true
}

// markSource marks what source k's rounds count and book.
func (m *Compiled) markSource(l *layout, k int32) {
	for v := range m.nodes {
		if m.reach[k][v] == 0 {
			continue
		}
		m.infVar[l.entry(index(v))] = true
		n := m.nodes[v]
		for _, r := range [2][2]int32{{n.fnStart, n.fnEnd}, {n.rateStart, n.rateEnd}} {
			// bound: the node's functions
			for f := r[0]; f < r[1]; f++ {
				if d := m.pools[m.fns[f].pool]; d.Mean() != 0 || d.Var() != 0 {
					m.markBooked(l, k, f)
				}
			}
		}
	}
}

// markQueues finds the carried queues whose wait and stay have infinite
// variance: a round waits for the work of every round ahead of it, so
// the work's infinite variance, or a count's of a source reaching the
// queue, makes the wait's infinite. A reset queue's rounds stay at most
// a period.
func (m *Compiled) markQueues(ws *[]Warning) {
	m.queueInfVar = make([]bool, len(m.queues))
	for c, q := range m.queues {
		if q.reset {
			continue
		}
		cause := ""
		if math.IsInf(m.pools[q.work].Var(), 1) {
			cause = "the work has"
		}
		for k := range m.srcs {
			if m.countMean[k] > 0 && m.reach[k][q.node] > 0 && math.IsInf(m.countN2[k], 1) {
				cause = "the count of source " + m.sourceIDs[k] + " has"
			}
		}
		if cause != "" {
			m.queueInfVar[c] = true
			*ws = append(*ws, Warning{Code: infVarCode, Subject: m.nodeIDs[q.node], Message: cause + " a finite mean and infinite variance: the standard errors of the queue's wait and stay are unreliable"})
		}
	}
}

// QueueStdErrUnreliable reports whether queue c's wait and stay have
// infinite variance, so that their standard errors mean nothing.
func (m *Compiled) QueueStdErrUnreliable(c int) bool {
	return c < len(m.queueInfVar) && m.queueInfVar[c]
}

// StdErrUnreliable reports whether the quantity in column col, in the
// order of the series' columns, has infinite variance, so that its
// standard error means nothing.
func (m *Compiled) StdErrUnreliable(col int) bool {
	return col < len(m.infVar) && m.infVar[col]
}
