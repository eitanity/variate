package variate

import "math"

// The queues' event loop and line, on the hot path: nothing here
// allocates.

func (q *wqueue) reset() {
	q.nArr, q.head, q.nLine, q.nBusy = 0, 0, 0, 0
	clear(q.servers)
}

func (q *wqueue) empty() bool { return q.nArr == 0 && q.nLine == 0 && q.nBusy == 0 }

// before orders two parked rounds by time, then by the order parked.
func before(a, b qitem) bool { return a.t < b.t || (a.t == b.t && a.seq < b.seq) }

// push parks a round to reach the queue, reporting false when full.
func (q *wqueue) push(it qitem) bool {
	if q.nArr == len(q.arr) {
		return false
	}
	i := q.nArr
	q.arr[i] = it
	q.nArr++
	// bound: the heap's depth
	for i > 0 {
		up := (i - 1) / 2
		if !before(q.arr[i], q.arr[up]) {
			break
		}
		q.arr[i], q.arr[up] = q.arr[up], q.arr[i]
		i = up
	}
	return true
}

// pop takes the earliest parked round.
func (q *wqueue) pop() qitem {
	top := q.arr[0]
	q.nArr--
	q.arr[0] = q.arr[q.nArr]
	i := 0
	// bound: the heap's depth
	for {
		l, r, m := 2*i+1, 2*i+2, i
		if l < q.nArr && before(q.arr[l], q.arr[m]) {
			m = l
		}
		if r < q.nArr && before(q.arr[r], q.arr[m]) {
			m = r
		}
		if m == i {
			return top
		}
		q.arr[i], q.arr[m] = q.arr[m], q.arr[i]
		i = m
	}
}

// join appends a round to the line, reporting false when full.
func (q *wqueue) join(it qitem) bool {
	if q.nLine == len(q.line) {
		return false
	}
	q.line[(q.head+q.nLine)%len(q.line)] = it
	q.nLine++
	return true
}

// leave takes the round that has waited longest.
func (q *wqueue) leave() qitem {
	it := q.line[q.head]
	q.head = (q.head + 1) % len(q.line)
	q.nLine--
	return it
}

// next is the queue's next event before until: the earliest server to
// finish, or else the earliest round to arrive, a finish first at a tie.
func (q *wqueue) next(until float64) (t float64, server int, ok bool) {
	t, server = until, -1
	for s := range q.servers {
		if q.servers[s].busy && q.servers[s].finish < t {
			t, server = q.servers[s].finish, s
		}
	}
	if q.nArr > 0 && q.arr[0].t < t {
		return q.arr[0].t, -1, true
	}
	return t, server, server >= 0
}

// free is a server not serving, or −1.
func (q *wqueue) free() int {
	for s := range q.servers {
		if !q.servers[s].busy {
			return s
		}
	}
	return -1
}

// parkRound puts a round reaching queue c at time t from the start of its
// arrival period p into the queue's arrivals, to join in time order; a
// given state's round keeps its mark.
func (w *Workspace) parkRound(wk *worker, sl *slot, c int32, k int32, t float64, p int64) {
	wk.qseq++
	it := qitem{t: float64(p) + t, p: p, k: k, seq: wk.qseq, rkey: wk.rkey, rated: float64(p) + t}
	if wk.given {
		it.given = -1
	}
	if !wk.queues[c].push(it) && sl.err == nil {
		sl.err = queueFullError(w.m.nodeIDs[w.m.queues[c].node], len(wk.queues[c].arr))
	}
}

// queueCol adds x to queue c's column col in the row of period q, when
// the row is in the block.
func (w *Workspace) queueCol(sl *slot, c int32, col int, q int64, x float64) {
	if q < sl.a || q >= sl.b {
		return
	}
	sl.rowAt(w.l, q)[w.l.queue(int(c), col)] += x
}

// queueSpan adds the time [s, f) overlaps each period of the block to
// queue c's column col.
func (w *Workspace) queueSpan(sl *slot, c int32, col int, s, f float64) {
	lo := math.Max(s, float64(sl.a))
	hi := math.Min(f, float64(sl.b))
	// bound: the periods of the block the span overlaps
	for q := math.Floor(lo); q < hi; q++ {
		w.queueCol(sl, c, col, int64(q), math.Min(hi, q+1)-math.Max(lo, q))
	}
}

// runQueues serves every queue's events before time until, from the start
// of period 0, in time order across the queues: a server finishing or a
// round arriving, the lower queue first at a tie.
func (w *Workspace) runQueues(wk *worker, sl *slot, until float64) {
	// bound: each event serves or seats one parked round, and the rounds
	// parked in a task are fewer than the queues' capacity
	for sl.err == nil {
		best, at, server := -1, until, -1
		for c := range wk.queues {
			if t, s, ok := wk.queues[c].next(at); ok && t < at {
				best, at, server = c, t, s
			}
		}
		if best < 0 {
			return
		}
		if server >= 0 {
			w.finishService(wk, sl, index(best), server)
		} else {
			w.arrive(wk, sl, index(best))
		}
	}
}

// arrive seats queue c's earliest parked round: it draws its work and
// joins the line, or goes to a free server.
func (w *Workspace) arrive(wk *worker, sl *slot, c int32) {
	q := &wk.queues[c]
	it := q.pop()
	wk.rkey = it.rkey
	it.work = w.draw(wk, w.m.queues[c].work)
	w.queueCol(sl, c, qJoined, int64(math.Floor(it.t)), 1)
	if s := q.free(); s >= 0 {
		w.startService(wk, c, s, it, it.t)
		return
	}
	if !q.join(it) && sl.err == nil {
		sl.err = queueFullError(w.m.nodeIDs[w.m.queues[c].node], len(q.line))
	}
}

// startService puts a round on server s of queue c at time t.
func (w *Workspace) startService(wk *worker, c int32, s int, it qitem, t float64) {
	q := &wk.queues[c]
	q.servers[s] = qserver{item: it, start: t, finish: t + it.work/w.m.queues[c].speed, spanned: t, busy: true}
	q.nBusy++
}

// finishService ends server s's service at queue c: the round's wait and
// stay count to the period it reached the queue, period 0 for a round of
// a given state and the block's first for one carried from an earlier
// block, the node's rates book over the stay not yet booked, the server
// takes the next round in line, and the round goes on through the
// node's outcomes.
func (w *Workspace) finishService(wk *worker, sl *slot, c int32, s int) {
	m := w.m
	q := &wk.queues[c]
	sv := q.servers[s]
	q.servers[s].busy = false
	q.nBusy--
	it := sv.item
	arrived := max(int64(math.Floor(it.t)), it.p, sl.a)
	w.queueCol(sl, c, qServed, arrived, 1)
	w.queueCol(sl, c, qWait, arrived, sv.start-it.t)
	w.queueCol(sl, c, qStay, arrived, sv.finish-it.t)
	w.queueSpan(sl, c, qBusy, math.Max(sv.start, sv.spanned), sv.finish)
	if arrived < sl.b && sl.task >= 0 {
		sl.qwait[c].AddPair(sv.start-it.t, sl.qwaitC[c])
		sl.qstay[c].AddPair(sv.finish-it.t, sl.qstayC[c])
	}
	if q.nLine > 0 {
		w.startService(wk, c, s, q.leave(), sv.finish)
	}
	v := m.queues[c].node
	n := m.nodes[v]
	p := it.p
	wk.rkey = it.rkey
	if n.rateStart != n.rateEnd {
		w.bookRate(wk, sl, it.k, v, n, math.Max(math.Max(it.t, it.rated)-float64(p), 0), sv.finish-float64(p), p)
	}
	if n.outStart == n.outEnd || sl.err != nil {
		return
	}
	t := sv.finish - float64(p)
	wk.given = it.given != 0
	next, tau, _, ok := w.advance(wk, sl, it.k, v, n, t, p, 0)
	if ok {
		w.walk(wk, sl, it.k, next.to, t+tau, p)
	}
	wk.given = false
}

// periodEnd records each queue's line at the end of period p and, for a
// reset queue, sends the rounds still waiting or in service to its
// unfinished node, their rates booked to the end.
func (w *Workspace) queuePeriodEnd(wk *worker, sl *slot, p int64) {
	m := w.m
	end := float64(p + 1)
	for c := range wk.queues {
		q := &wk.queues[c]
		ci := index(c)
		w.queueCol(sl, ci, qLength, p, float64(q.nBusy+q.nLine))
		cq := m.queues[c]
		if !cq.reset {
			continue
		}
		for s := range q.servers {
			if q.servers[s].busy {
				w.queueSpan(sl, ci, qBusy, q.servers[s].start, end)
				q.servers[s].busy = false
				w.unfinished(wk, sl, ci, q.servers[s].item, end, p)
			}
		}
		q.nBusy = 0
		// bound: the line
		for q.nLine > 0 && sl.err == nil {
			w.unfinished(wk, sl, ci, q.leave(), end, p)
		}
	}
}

// unfinished sends a round unserved at time end to queue c's unfinished
// node, the queue node's rates booked over its stay.
func (w *Workspace) unfinished(wk *worker, sl *slot, c int32, it qitem, end float64, p int64) {
	m := w.m
	cq := m.queues[c]
	w.queueCol(sl, c, qUnfinished, p, 1)
	wk.rkey = it.rkey
	n := m.nodes[cq.node]
	if n.rateStart != n.rateEnd {
		w.bookRate(wk, sl, it.k, cq.node, n, math.Max(it.t-float64(it.p), 0), end-float64(it.p), it.p)
	}
	wk.given = it.given != 0
	w.walk(wk, sl, it.k, cq.unfinished, end-float64(it.p), it.p)
	wk.given = false
}

// drainQueues serves a started run's queues to empty after its horizon:
// the rounds already on their way still arrive, no new ones come, and a
// reset queue still resets at each period's end.
func (w *Workspace) drainQueues(wk *worker, sl *slot) {
	resets := false
	for _, q := range w.m.queues {
		resets = resets || q.reset
	}
	if !resets {
		w.runQueues(wk, sl, math.Inf(1))
		return
	}
	// bound: maxDrain periods
	for p := sl.b; p < sl.b+maxDrain && sl.err == nil; p++ {
		if w.queuesEmpty(wk) {
			return
		}
		w.runQueues(wk, sl, float64(p+1))
		w.queuePeriodEnd(wk, sl, p)
	}
	if !w.queuesEmpty(wk) && sl.err == nil {
		sl.err = drainError(maxDrain)
	}
}

func (w *Workspace) queuesEmpty(wk *worker) bool {
	for c := range wk.queues {
		if !wk.queues[c].empty() {
			return false
		}
	}
	return true
}
