package variate

import (
	"math"

	"code.eitanity.com/variate/dist"
	"code.eitanity.com/variate/stats"
)

// runTask simulates the block [a, b) into the slot: task 0 also the
// burn-in arrivals in [−S, 0). Rows close as the task leaves them, except
// the first S periods of the block, which move to the head.
func (w *Workspace) runTask(wk *worker, sl *slot, seed uint64, hs []*stats.LogHist) {
	m := w.m
	sl.reset()
	wk.ss.Reset(seed, sl.id)
	first := sl.a
	if sl.task == 0 {
		first = -int64(w.s)
	}
	if m.renewal {
		w.startSources(wk, float64(first))
	}
	// bound: the block plus the burn-in
	for p := first; p < sl.b && sl.err == nil; p++ {
		sl.base = int((p - first) % int64(w.s+1))
		w.drawPeriod(wk, p)
		for k := range m.srcs {
			w.runSource(wk, sl, index(k), p, first)
		}
		if w.stream {
			w.flushPeriod(wk, sl)
		}
		w.leavePeriod(sl, p, first, hs)
	}
}

// drawPeriod takes this period's value from every period-scoped pool,
// and sets each schedule's value by the period's place in its cycle.
func (w *Workspace) drawPeriod(wk *worker, p int64) {
	m := w.m
	for k, slot := range m.poolPeriod {
		if slot != noIndex {
			wk.period[slot] = wk.ss.Sampler(k).Next()
		}
	}
	for _, sc := range m.schedules {
		wk.period[m.poolPeriod[sc.pool]] = sc.at(p)
	}
}

// draw returns a pool's value for this round: the period's draw for a
// period-scoped pool, the next pooled value otherwise.
func (w *Workspace) draw(wk *worker, pool int32) float64 {
	if slot := w.m.poolPeriod[pool]; slot != noIndex {
		return wk.period[slot]
	}
	return wk.ss.Sampler(int(pool)).Next()
}

// runSource emits the period's rounds of source k and walks each one.
func (w *Workspace) runSource(wk *worker, sl *slot, k int32, p, first int64) {
	m := w.m
	src := m.srcs[k]
	if src.chainEnd > src.chainStart {
		w.runChainSource(wk, sl, k, p, first)
		return
	}
	if src.every > 1 || src.repair != noIndex {
		w.runRenewalSource(wk, sl, k, p, first)
		return
	}
	lam := wk.period[m.poolPeriod[src.pool]]
	if src.factor != noIndex {
		lam *= wk.period[m.poolPeriod[src.factor]]
	}
	if src.schedule != noIndex {
		lam *= wk.period[m.poolPeriod[src.schedule]]
	}
	n := dist.PoissonCount(wk.ss.Control(), lam)
	if p >= 0 {
		sl.events[k] += n
	}
	// bound: the Poisson count
	for i := uint64(0); i < n && sl.err == nil; i++ {
		t := arrivalTime(p, wk.ss.Control().Float64())
		if w.stream {
			w.beginRound(wk, t)
		}
		w.walk(wk, sl, k, src.to, t, p, first)
	}
}

// walk follows one round from node v at time t to a leaf, booking as it
// goes. The round's arrival period is p; counts are kept for p ≥ 0.
func (w *Workspace) walk(wk *worker, sl *slot, k, v int32, t float64, p, first int64) {
	m := w.m
	nodes := len(m.nodes)
	// bound: a path visits each node at most once
	for step := 0; step <= nodes; step++ {
		sl.steps++
		if p >= 0 {
			sl.reach[int(k)*nodes+int(v)]++
		}
		row, ok := w.rowFor(sl, t, p, noIndex)
		if !ok {
			return
		}
		row[w.l.entry(v)]++
		n := m.nodes[v]
		booked := w.bookEntry(wk, row, k, v, n)
		if w.stream && wk.live {
			w.prune(wk, v)
		}
		if n.outStart == n.outEnd {
			if w.stream && wk.live {
				w.recordStep(wk, v, noIndex, t, booked)
				w.endRound(wk, sl, k, v, p, t)
			}
			return
		}
		next, tau, booked2, ok := w.advance(wk, sl, k, v, n, t, p, first, booked)
		if !ok {
			return
		}
		booked = booked2
		if w.stream && wk.live {
			w.recordStep(wk, v, next.out, t, booked)
		}
		t += tau
		v = next.to
	}
}

// hop is the outcome taken out of a node and its target.
type hop struct{ out, to int32 }

// advance selects the outcome out of node v, counts it, draws the holding
// time and books the PerTime functions over it. It returns the hop, the
// holding time and the value booked so far; ok is false on a window error.
func (w *Workspace) advance(wk *worker, sl *slot, k, v int32, n cnode, t float64, p, first int64, booked float64) (hop, float64, float64, bool) {
	m := w.m
	j := w.choose(wk, v, n)
	if p >= 0 {
		sl.outs[int(k)*len(m.outs)+int(j)]++
	}
	o := m.outs[j]
	tau := 0.0
	switch {
	case o.outage:
		tau = wk.outage
	case o.hold != noIndex:
		tau = w.draw(wk, o.hold)
	}
	if n.rateStart != n.rateEnd && tau > 0 {
		rated, ok := w.bookRate(wk, sl, k, v, n, t, t+tau, p, first, j)
		if !ok {
			return hop{}, 0, 0, false
		}
		booked += rated
	}
	return hop{out: j, to: o.to}, tau, booked, true
}

// bookEntry books node v's OnEntry functions into the row and returns the
// value booked.
func (w *Workspace) bookEntry(wk *worker, row []float64, k, v int32, n cnode) float64 {
	m := w.m
	booked := 0.0
	// bound: functions of one node
	for f := n.fnStart; f < n.fnEnd; f++ {
		x := w.draw(wk, m.fns[f].pool)
		w.book(row, k, v, m.fns[f].category, x)
		booked += x
	}
	return booked
}

// arrivalTime is the time of an arrival at fraction u of period p. At a
// large period number p + u can round up to p + 1, outside the period,
// so the time is kept below it.
func arrivalTime(p int64, u float64) float64 {
	t := float64(p) + u
	if end := float64(p) + 1; t >= end {
		return math.Nextafter(end, math.Inf(-1))
	}
	return t
}

// rowFor returns the window row for a booking at time t, or ok = false
// after recording a window error: the booking lands more than S periods
// after the arrival period.
func (w *Workspace) rowFor(sl *slot, t float64, p int64, via int32) ([]float64, bool) {
	limit := float64(p) + float64(w.s) + 1
	if !(t < limit) {
		sl.err = w.windowError(p, t, via)
		return nil, false
	}
	// The row of period q = ⌊t⌋ lies d = q − p rows past the period's own
	// row in the ring; d is within [0, S] by the check above, so one
	// subtraction wraps it.
	r := sl.base + int(int64(math.Floor(t))-p)
	if r > w.s {
		r -= w.s + 1
	}
	return sl.window[r*w.l.width : (r+1)*w.l.width], true
}

// book adds a value to a row's total, category, source and node
// contribution.
func (w *Workspace) book(row []float64, k, v, cat int32, x float64) {
	l := &w.l
	row[l.total()] += x
	row[l.cat(cat)] += x
	row[l.src(k)] += x
	row[l.contrib(v)] += x
}

// bookRate spreads each PerTime rate of node v over [a, b), one row per
// period overlapped, and returns the value booked.
func (w *Workspace) bookRate(wk *worker, sl *slot, k, v int32, n cnode, a, b float64, p, first int64, via int32) (float64, bool) {
	m := w.m
	total := 0.0
	// bound: functions of one node
	for f := n.rateStart; f < n.rateEnd; f++ {
		rate := w.draw(wk, m.fns[f].pool)
		// bound: the periods the holding time spans, within the window
		for q := math.Floor(a); q < b; q++ {
			row, ok := w.rowFor(sl, q, p, via)
			if !ok {
				return total, false
			}
			lo := math.Max(a, q)
			hi := math.Min(b, q+1)
			x := rate * (hi - lo)
			w.book(row, k, v, m.fns[f].category, x)
			total += x
		}
	}
	return total, true
}

// choose selects an outcome of node v by its selection mode.
func (w *Workspace) choose(wk *worker, v int32, n cnode) int32 {
	m := w.m
	u := wk.ss.Control().Float64()
	switch {
	case n.mode == Sequential:
		last := n.outEnd - 1
		// bound: outcomes of one node
		for j := n.outStart; j < last; j++ {
			q := m.outs[j].fixed
			if m.outs[j].probPool != noIndex {
				q = wk.period[m.poolPeriod[m.outs[j].probPool]]
			}
			if u < q {
				return j
			}
			u = wk.ss.Control().Float64()
		}
		return last
	case n.twoWay:
		if u < wk.period[m.poolPeriod[m.outs[n.outStart].probPool]] {
			return n.outStart
		}
		return n.outStart + 1
	case n.alias != noIndex:
		count := int(n.outEnd - n.outStart)
		scaled := u * float64(count)
		i := int(scaled)
		if scaled-float64(i) < m.aliasProb[int(n.alias)+i] {
			return n.outStart + index(i)
		}
		return n.outStart + m.aliasIdx[int(n.alias)+i]
	default:
		// bound: outcomes of one node
		for j := n.outStart; j < n.outEnd-1; j++ {
			if u < m.outs[j].cum {
				return j
			}
		}
		return n.outEnd - 1
	}
}

// leavePeriod closes the row of period p, or parks it in the head when it
// may still receive the previous task's spill, or drops it in the burn-in.
func (w *Workspace) leavePeriod(sl *slot, p, first int64, hs []*stats.LogHist) {
	row := sl.row(w.l, w.s, first, p)
	switch {
	case p < 0:
	case p < sl.a+int64(w.s):
		h := int(p-sl.a) * w.l.width
		copy(sl.head[h:h+w.l.width], row)
	default:
		w.closeRow(sl, row, p, hs)
		sl.closed++
	}
	clear(row)
}

// closeRow folds a finished period into the slot statistics and the
// histograms, after checking that the breakdowns sum to the total.
func (w *Workspace) closeRow(sl *slot, row []float64, p int64, hs []*stats.LogHist) {
	l := &w.l
	total := row[l.total()]
	var cats, srcs float64
	// bound: row width
	for c := 0; c < l.cats; c++ {
		cats += row[l.cat(index(c))]
	}
	// bound: row width
	for k := 0; k < l.srcs; k++ {
		srcs += row[l.src(index(k))]
	}
	tol := invariantTol * math.Max(1, math.Abs(total))
	if math.Abs(cats-total) > tol || math.Abs(srcs-total) > tol {
		sl.err = invariantError(cats, srcs, total)
		return
	}
	w.applyLayers(row, total)
	for i, x := range row {
		sl.stats[i].add(x)
		if h := hs[i]; h != nil {
			h.Add(x)
		}
	}
	if w.m.tailX > 0 && total >= w.m.tailX {
		sl.tail++
	}
	w.addTail(sl, row, total)
	if s := w.result.Series; s != nil {
		copy(s.rows[int(p)*l.width:(int(p)+1)*l.width], row)
	}
}

// invariantTol is the rounding allowance on the breakdown sums.
const invariantTol = 1e-9
