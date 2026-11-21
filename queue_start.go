package variate

import (
	"math"
	"math/rand/v2"

	"code.eitanity.com/variate/dist"
)

// A queue that carries its line in a steady-state run starts each
// timeline in its long-run state, drawn exactly. Its rounds are a Poisson
// stream, and a first-come queue of c servers is never longer than c
// one-server queues each sent the rounds at random, a c-th of the
// stream, and serving them by processor sharing, whose long-run state is
// known: a geometric number of rounds in each, a round's work drawn in
// proportion to its size and its work done a uniform share of it. Run
// backwards in time those queues are again such queues, the work done
// and the work left swapped. Once all are empty at once the first-come
// queue was empty then too, and serving it forward from there over the
// rounds the backward queues let go, as rounds arriving, gives its state
// now.

const (
	// maxLookBack bounds the rounds an exact start serves forward.
	maxLookBack = 1 << 22
	// carriedPeriods is the periods of rounds a carried queue's line holds
	// in a steady-state run, where it may be long at a block's start.
	carriedPeriods = 1024
	// tailFrom and tailTo are where a work's tail is read, in means.
	tailFrom = 0x1p200
	tailTo   = 0x1p210
)

// qstart is what a carried queue's exact start draws from.
type qstart struct {
	sub   float64    // the rate of a dominating queue's rounds
	rho   float64    // its load
	resid residTable // the work's equilibrium residual, for a work not constant
	fixed bool       // the work is constant
	cum   []float64  // the sources' cumulative shares of the queue's rounds
}

// buildStarts readies each queue that carries its line in a
// steady-state run for its exact start, refusing one whose rounds are
// not a Poisson stream at a constant rate, whose load is not below 1, or
// whose work's fourth moment is infinite, which the time average's error
// needs.
func (m *Compiled) buildStarts() error {
	if m.Started() {
		return nil
	}
	m.qstarts = make([]qstart, len(m.queues))
	for c, q := range m.queues {
		if q.reset {
			continue
		}
		id := m.nodeIDs[q.node]
		const started = "; a started run (start: { from: empty, horizon: n }) has no such limit"
		if why := m.poissonInto(q.node); why != "" {
			return modelError("queue", id, "a queue that carries its line in a steady-state run starts from its long-run state, which needs the rounds reaching it to be a Poisson stream at a constant rate: "+why+started)
		}
		cum := make([]float64, len(m.srcs))
		lam := 0.0
		for k := range m.srcs {
			lam += m.countMean[k] * m.reach[k][q.node]
			cum[k] = lam
		}
		for k := range cum {
			cum[k] /= lam
		}
		work := m.pools[q.work]
		rho := lam * work.Mean() / q.speed / float64(q.servers)
		if !(rho < 1) {
			return modelError("queue", id, "its load "+figure(rho)+" is not below 1, so the line grows without bound and has no long-run state"+started)
		}
		if !fourthMomentFinite(work) {
			return modelError("queue", id, "its work's tail, far out, falls no faster than x^-4, so its fourth moment is infinite or too large for the long-run wait's error to be measured"+started)
		}
		s := qstart{sub: lam / float64(q.servers), rho: rho, cum: cum}
		if lo, hi := work.Support(); lo == hi {
			s.fixed = true
		} else {
			s.resid = newResidTable(work)
		}
		m.qstarts[c] = s
		m.carried = true
	}
	return nil
}

// fourthMomentFinite reports a law whose fourth moment is finite, read
// from its tail: a bounded support, or a log-survival falling faster
// than −4 log x from 2^200 to 2^210 means, where a power tail shows its
// index and a lighter one falls far faster or underflows.
func fourthMomentFinite(d dist.Distribution) bool {
	if _, hi := d.Support(); !math.IsInf(hi, 1) {
		return true
	}
	mean := d.Mean()
	if !(mean > 0) || math.IsInf(mean, 1) {
		return false
	}
	l1, l2 := d.LogSurvival(mean*tailFrom), d.LogSurvival(mean*tailTo)
	if math.IsInf(l2, -1) {
		return true
	}
	return (l2-l1)/math.Log(tailTo/tailFrom) < -4
}

// psTag is a round at a processor-sharing queue: the virtual time it
// leaves at, the work each round present has had when it does, and its
// work.
type psTag struct{ f, work float64 }

// psQueue is a dominating queue, run backwards: its clock, its virtual
// time, its rounds as a heap by the virtual time they leave at, and its
// next round's arrival.
type psQueue struct {
	at, v, next float64
	tags        []psTag
}

// qdep is a round a dominating queue let go: when, backwards, and its
// work.
type qdep struct{ t, work float64 }

// starter is a worker's scratch for exact starts.
type starter struct {
	pcg  rand.PCG
	r    *rand.Rand
	ps   []psQueue
	heap []int32 // the dominating queues by their next event
	pos  []int32
	key  []float64
	deps []qdep
	free []float64 // the servers' free times, a heap
}

func newStarter() *starter {
	s := &starter{}
	s.r = rand.New(&s.pcg)
	return s
}

// exactStart puts queue c in its long-run state at the start of the
// slot's block, and returns how far back, in periods, the dominating
// queues were last all empty.
func (w *Workspace) exactStart(wk *worker, sl *slot, c int32) (float64, error) {
	m := w.m
	cq := m.queues[c]
	qs := &m.qstarts[c]
	st := wk.start
	st.pcg.Seed(wk.k.seed, mix(mix(wk.lane, tagExact), uint64(c))) //nolint:gosec // a queue's index
	back, err := st.coalesce(m, cq, qs)
	if err != nil {
		return back, err
	}
	w.replay(wk, sl, c, back)
	return back, nil
}

// coalesce draws the dominating queues' long-run state and runs them
// backwards until all are empty at once, keeping the rounds they let go
// in the order they left; it returns that time.
func (st *starter) coalesce(m *Compiled, cq cqueue, qs *qstart) (float64, error) {
	n := int(cq.servers)
	busy := st.seat(m, cq, qs)
	if busy == 0 {
		return 0, nil
	}
	st.buildHeap(n)
	work := m.pools[cq.work]
	// bound: each event lets a round go or takes one, and the rounds
	// taken are fewer than maxLookBack
	for len(st.deps) < maxLookBack {
		j := st.heap[0]
		q := &st.ps[j]
		t := st.key[j]
		if len(q.tags) > 0 && q.departure() == t {
			q.v = q.tags[0].f
			q.at = t
			st.deps = append(st.deps, qdep{t: t, work: q.pop().work})
			if len(q.tags) == 0 {
				busy--
			}
			if busy == 0 {
				return t, nil
			}
		} else {
			if len(q.tags) == 0 {
				busy++
			}
			q.arrive(t, m.drawPool(work, st.r), cq.speed, st.r.ExpFloat64()/qs.sub)
		}
		st.key[j] = q.event()
		st.down(st.pos[j])
	}
	return 0, lookBackError(m.nodeIDs[cq.node], maxLookBack)
}

// seat draws each dominating queue's long-run state and first arrival,
// and returns how many hold a round.
func (st *starter) seat(m *Compiled, cq cqueue, qs *qstart) int {
	n := int(cq.servers)
	st.ps = growPS(st.ps, n)
	st.deps = st.deps[:0]
	busy := 0
	// bound: the servers
	for j := 0; j < n; j++ {
		q := &st.ps[j]
		q.at, q.v, q.tags = 0, 0, q.tags[:0]
		q.next = st.r.ExpFloat64() / qs.sub
		// bound: a geometric count, below maxLookBack rounds
		for len(q.tags) < maxLookBack && st.r.Float64() < qs.rho {
			done, work := st.inService(m, cq, qs)
			q.push(psTag{f: done / cq.speed, work: work})
		}
		if len(q.tags) > 0 {
			busy++
		}
	}
	return busy
}

// arrive takes a round of work x at time t, the next due gap later.
func (q *psQueue) arrive(t, x, speed, gap float64) {
	if len(q.tags) > 0 {
		q.v += (t - q.at) / float64(len(q.tags))
	}
	q.at = t
	q.push(psTag{f: q.v + x/speed, work: x})
	q.next = t + gap
}

// inService draws a round of a dominating queue's long-run state: its
// work done, from the work's equilibrium residual, and its work, drawn
// given it exceeds the work done.
func (st *starter) inService(m *Compiled, cq cqueue, qs *qstart) (done, work float64) {
	d := m.pools[cq.work]
	if qs.fixed {
		w := d.Mean()
		return st.r.Float64() * w, w
	}
	done = qs.resid.draw(st.r)
	work = math.Max(d.QuantileUpper(unitOpen(st.r)*d.Survival(done)), done)
	return done, work
}

// drawPool draws one value of law d from r.
func (m *Compiled) drawPool(d dist.Distribution, r *rand.Rand) float64 {
	var one [1]float64
	d.Fill(r, one[:])
	return one[0]
}

// replay serves queue c forward from empty, back periods before the block's
// start, over the rounds the dominating queues let go, the latest
// backwards the earliest forwards, and seats at the block's start the
// rounds still in service or in line, each from a source drawn by its
// share of the queue's rounds.
func (w *Workspace) replay(wk *worker, sl *slot, c int32, back float64) {
	m := w.m
	cq := m.queues[c]
	st := wk.start
	q := &wk.queues[c]
	a := float64(sl.a)
	n := int(cq.servers)
	st.free = growF64(st.free, n)
	for s := range st.free {
		st.free[s] = -back
	}
	server := 0
	// bound: the rounds let go, fewer than maxLookBack
	for i := len(st.deps) - 1; i >= 0 && sl.err == nil; i-- {
		d := st.deps[i]
		arr := -d.t
		start := math.Max(arr, st.free[0])
		finish := start + d.work/cq.speed
		st.free[0] = finish
		siftDownF64(st.free, 0)
		if finish <= 0 {
			continue
		}
		wk.qseq++
		it := qitem{t: a + arr, p: int64(math.Floor(a + arr)), k: st.source(m.qstarts[c].cum), work: d.work, seq: wk.qseq, rkey: st.r.Uint64(), rated: a}
		if start <= 0 {
			q.servers[server] = qserver{item: it, start: a + start, finish: a + finish, spanned: a, busy: true}
			server++
			q.nBusy++
			continue
		}
		if !q.join(it) {
			sl.err = queueFullError(m.nodeIDs[cq.node], len(q.line))
		}
	}
}

// source draws a source by its cumulative share.
func (st *starter) source(cum []float64) int32 {
	u := st.r.Float64()
	for k, x := range cum {
		if u < x {
			return index(k)
		}
	}
	return index(len(cum) - 1)
}

// departure is when the queue's next round leaves, with one present.
func (q *psQueue) departure() float64 {
	return q.at + (q.tags[0].f-q.v)*float64(len(q.tags))
}

// event is the queue's next event: a round leaving, or one arriving.
func (q *psQueue) event() float64 {
	if len(q.tags) == 0 {
		return q.next
	}
	return math.Min(q.departure(), q.next)
}

func (q *psQueue) push(x psTag) {
	q.tags = append(q.tags, x)
	i := len(q.tags) - 1
	// bound: the heap's depth
	for i > 0 {
		up := (i - 1) / 2
		if q.tags[up].f <= q.tags[i].f {
			return
		}
		q.tags[i], q.tags[up] = q.tags[up], q.tags[i]
		i = up
	}
}

func (q *psQueue) pop() psTag {
	top := q.tags[0]
	last := len(q.tags) - 1
	q.tags[0] = q.tags[last]
	q.tags = q.tags[:last]
	i := 0
	// bound: the heap's depth
	for {
		l, r, s := 2*i+1, 2*i+2, i
		if l < last && q.tags[l].f < q.tags[s].f {
			s = l
		}
		if r < last && q.tags[r].f < q.tags[s].f {
			s = r
		}
		if s == i {
			return top
		}
		q.tags[i], q.tags[s] = q.tags[s], q.tags[i]
		i = s
	}
}

// buildHeap orders the n dominating queues by their next event.
func (st *starter) buildHeap(n int) {
	st.heap = growI32(st.heap, n)
	st.pos = growI32(st.pos, n)
	st.key = growF64(st.key, n)
	// bound: the servers
	for j := 0; j < n; j++ {
		st.heap[j], st.pos[j] = index(j), index(j)
		st.key[j] = st.ps[j].event()
	}
	// bound: the servers
	for i := n/2 - 1; i >= 0; i-- {
		st.down(index(i))
	}
}

// down sifts the queue at heap position i down after its key rose.
func (st *starter) down(i int32) {
	n := index(len(st.heap))
	// bound: the heap's depth
	for {
		l, r, s := 2*i+1, 2*i+2, i
		if l < n && st.key[st.heap[l]] < st.key[st.heap[s]] {
			s = l
		}
		if r < n && st.key[st.heap[r]] < st.key[st.heap[s]] {
			s = r
		}
		if s == i {
			return
		}
		st.heap[i], st.heap[s] = st.heap[s], st.heap[i]
		st.pos[st.heap[i]], st.pos[st.heap[s]] = i, s
		i = s
	}
}

// siftDownF64 sifts h[i] down a min-heap.
func siftDownF64(h []float64, i int) {
	n := len(h)
	// bound: the heap's depth
	for {
		l, r, s := 2*i+1, 2*i+2, i
		if l < n && h[l] < h[s] {
			s = l
		}
		if r < n && h[r] < h[s] {
			s = r
		}
		if s == i {
			return
		}
		h[i], h[s] = h[s], h[i]
		i = s
	}
}

func growPS(s []psQueue, n int) []psQueue {
	if cap(s) < n {
		return append(s[:cap(s)], make([]psQueue, n-cap(s))...)[:n]
	}
	return s[:n]
}

func growF64(s []float64, n int) []float64 {
	if cap(s) < n {
		return make([]float64, n)
	}
	return s[:n]
}

func growI32(s []int32, n int) []int32 {
	if cap(s) < n {
		return make([]int32, n)
	}
	return s[:n]
}

// qcarry is a carried queue's state between a timeline's blocks: its
// line in order, and its servers.
type qcarry struct {
	line    []qitem
	servers []qserver
}

// startQueues seats each carried queue at the start of the slot's
// block: drawn exactly at a timeline's first block, else as the
// timeline's last block left it.
func (w *Workspace) startQueues(wk *worker, sl *slot) {
	m := w.m
	first := sl.task-w.lines < -w.burnIn
	for c, cq := range m.queues {
		if cq.reset {
			continue
		}
		q := &wk.queues[c]
		if first {
			T, err := w.exactStart(wk, sl, index(c))
			if err != nil && sl.err == nil {
				sl.err = err
			}
			sl.qback[c] += T
			sl.qstarts[c]++
			continue
		}
		cr := &w.carry[sl.line][c]
		copy(q.servers, cr.servers)
		for _, s := range cr.servers {
			if s.busy {
				q.nBusy++
			}
		}
		for _, it := range cr.line {
			if !q.join(it) && sl.err == nil {
				sl.err = queueFullError(m.nodeIDs[cq.node], len(q.line))
			}
		}
	}
}

// carryQueues books what each carried queue's rounds have run up to the
// block's end, the queue node's rates over their stay and the servers'
// busy time, and keeps the line and servers for the timeline's next
// block.
func (w *Workspace) carryQueues(wk *worker, sl *slot) {
	m := w.m
	end := float64(sl.b)
	for c, cq := range m.queues {
		if cq.reset {
			continue
		}
		q := &wk.queues[c]
		ci := index(c)
		cr := &w.carry[sl.line][c]
		cr.servers = append(cr.servers[:0], q.servers...)
		for s := range cr.servers {
			sv := &cr.servers[s]
			if !sv.busy {
				continue
			}
			w.queueSpan(sl, ci, qBusy, math.Max(sv.start, sv.spanned), end)
			sv.spanned = end
			w.rateTo(wk, sl, cq, &sv.item, end)
		}
		cr.line = cr.line[:0]
		// bound: the line
		for i := 0; i < q.nLine; i++ {
			it := q.line[(q.head+i)%len(q.line)]
			w.rateTo(wk, sl, cq, &it, end)
			cr.line = append(cr.line, it)
		}
	}
}

// rateTo books the queue node's rates over a round's stay from where
// they were booked to, up to end, and moves that mark to end.
func (w *Workspace) rateTo(wk *worker, sl *slot, cq cqueue, it *qitem, end float64) {
	n := w.m.nodes[cq.node]
	if n.rateStart != n.rateEnd {
		wk.rkey = it.rkey
		p := float64(it.p)
		w.bookRate(wk, sl, it.k, cq.node, n, math.Max(math.Max(it.t, it.rated)-p, 0), end-p, it.p)
	}
	it.rated = end
}

func lookBackError(queue string, rounds int) error {
	return modelError("queue", queue, "its exact start served more than "+itoa(rounds)+" rounds forward; a started run (start: { from: empty, horizon: n }) has no such limit")
}
