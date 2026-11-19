package variate

import (
	"math"
	"math/rand/v2"
	"slices"

	"code.eitanity.com/variate/dist"
)

// The given state put in place at a repeat's start, on the hot path:
// nothing here allocates.

// placeGiven puts a started run's given state in place at the start of
// period 0: the rounds at each queue, in line oldest first and the
// oldest in service, then the rounds in holding times, which walk on as
// each holding ends.
func (w *Workspace) placeGiven(wk *worker, sl *slot) {
	m := w.m
	for i := range m.given {
		if m.given[i].queue != noIndex {
			w.lineUp(wk, sl, index(i))
		}
	}
	for c := range wk.queues {
		w.seatGiven(wk, index(c))
	}
	for i := range m.given {
		if m.given[i].out != noIndex {
			w.releaseHeld(wk, sl, index(i))
		}
	}
}

// givenRound keys entry i's round j of the repeat; j = 0 keys its count.
func (w *Workspace) givenRound(wk *worker, i int32, j int) {
	wk.rkey = mix(mix(wk.pkey, w.m.givenKey[i]), uint64(j)) //nolint:gosec // a count
}

// givenCount draws entry i's rounds for the repeat, a whole number.
func (w *Workspace) givenCount(wk *worker, sl *slot, i int32) int {
	g := &w.m.given[i]
	w.givenRound(wk, i, 0)
	x := math.Round(w.draw(wk, g.count))
	if x > maxGiven {
		if sl.err == nil {
			sl.err = givenCountError(w.m.nodeIDs[g.node], x)
		}
		return 0
	}
	return int(x)
}

// lineUp draws entry i's rounds into its queue's line, each with its age
// and the work an arrival brings, and counts them as joining in period 0.
func (w *Workspace) lineUp(wk *worker, sl *slot, i int32) {
	m := w.m
	g := &m.given[i]
	q := &wk.queues[g.queue]
	n := w.givenCount(wk, sl, i)
	// bound: the entry's rounds, at most maxGiven
	for j := range n {
		if sl.err != nil {
			return
		}
		w.givenRound(wk, i, j+1)
		wk.qseq++
		it := qitem{t: -w.draw(wk, g.age), k: g.k, given: i + 1, seq: wk.qseq, rkey: wk.rkey}
		it.work = w.draw(wk, m.queues[g.queue].work)
		w.queueCol(sl, g.queue, qJoined, 0, 1)
		if !q.join(it) {
			sl.err = queueFullError(m.nodeIDs[g.node], len(q.line))
		}
	}
}

// byArrival orders a queue's rounds by when they reached it, then by the
// order they were drawn.
func byArrival(a, b qitem) int {
	switch {
	case before(a, b):
		return -1
	case before(b, a):
		return 1
	}
	return 0
}

// seatGiven orders queue c's given rounds oldest first and puts the
// oldest in service, each with the work it has left; its service began
// as long ago as it has been served, but not before it arrived.
func (w *Workspace) seatGiven(wk *worker, c int32) {
	q := &wk.queues[c]
	if q.nLine == 0 {
		return
	}
	speed := w.m.queues[c].speed
	slices.SortFunc(q.line[:q.nLine], byArrival)
	for s := range q.servers {
		if q.nLine == 0 {
			return
		}
		it := q.leave()
		wk.rkey = it.rkey
		left, served := w.inService(wk, &w.m.given[it.given-1], speed)
		it.work = left
		start := -math.Min(served, -it.t)
		q.servers[s] = qserver{item: it, start: start, finish: left / speed, busy: true}
		q.nBusy++
	}
}

// inService draws a given round's work left and its time served: with
// its time served given, the work beyond what that time did, given the
// work was more; without, the work's equilibrium residual and then the
// work done, which given the residual r is the work less r, given it is
// more than r.
func (w *Workspace) inService(wk *worker, g *cgiven, speed float64) (left, served float64) {
	if g.inService != noIndex {
		e := w.draw(wk, g.inService)
		return residualAfter(g.law, e*speed, wk.k.at(mix(wk.rkey, tagGiven))), e
	}
	left = w.equilibrium(wk, g)
	return left, residualAfter(g.law, left, wk.k.at(mix(wk.rkey, tagServed))) / speed
}

// releaseHeld walks an entry's rounds on from the end of their holding
// times, the holding node's rates booked over what is left of it; they
// count in no source's reach, having arrived before the run.
func (w *Workspace) releaseHeld(wk *worker, sl *slot, i int32) {
	m := w.m
	g := &m.given[i]
	n := m.nodes[g.node]
	to := m.outs[g.out].to
	cnt := w.givenCount(wk, sl, i)
	// bound: the entry's rounds, at most maxGiven
	for j := range cnt {
		if sl.err != nil {
			return
		}
		w.givenRound(wk, i, j+1)
		var left float64
		if g.elapsed != noIndex {
			e := w.draw(wk, g.elapsed)
			left = residualAfter(g.law, e, wk.k.at(mix(wk.rkey, tagGiven)))
		} else {
			left = w.equilibrium(wk, g)
		}
		if n.rateStart != n.rateEnd && left > 0 {
			w.bookRate(wk, sl, g.k, g.node, n, 0, left, 0)
		}
		wk.given = true
		w.walk(wk, sl, g.k, to, left, 0)
		wk.given = false
	}
}

// equilibrium draws the equilibrium residual of an entry's law, keyed
// by the round: a uniform share of a constant, else from its table.
func (w *Workspace) equilibrium(wk *worker, g *cgiven) float64 {
	r := wk.k.at(mix(wk.rkey, tagGiven))
	if lo, hi := g.law.Support(); lo == hi {
		return r.Float64() * hi
	}
	return g.resid.draw(r)
}

// residualAfter draws X − a given X > a by inverting the upper tail:
// X = Q(u·P(X > a)) for u uniform on (0, 1]. Past the law's support it
// is 0.
func residualAfter(d dist.Distribution, a float64, r *rand.Rand) float64 {
	s := d.Survival(a)
	if !(s > 0) {
		return 0
	}
	return math.Max(d.QuantileUpper((1-r.Float64())*s)-a, 0)
}
