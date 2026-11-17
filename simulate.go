package variate

import (
	"math"

	"code.eitanity.com/variate/dist"
	"code.eitanity.com/variate/stats"
)

// runTask simulates the block [a, b) into the slot's rows; a booking past
// the block goes to the slot's pending list. A burn-in task's block is
// before period 0. The rows are closed later, once the finaliser has
// handed the slot what earlier tasks booked into its block.
func (w *Workspace) runTask(wk *worker, sl *slot, seed uint64) {
	m := w.m
	sl.reset()
	wk.ss.Reset(seed, sl.id)
	if m.renewal {
		w.startSources(wk)
	}
	for c := range wk.queues {
		wk.queues[c].reset()
	}
	wk.qseq = 0
	// bound: the block
	for p := sl.a; p < sl.b && sl.err == nil; p++ {
		w.drawPeriod(wk, p)
		if p == sl.a && len(m.given) > 0 {
			w.placeGiven(wk, sl)
		}
		for k := range m.srcs {
			w.runSource(wk, sl, index(k), p)
		}
		if len(wk.queues) > 0 {
			w.runQueues(wk, sl, float64(p+1))
			w.queuePeriodEnd(wk, sl, p)
		}
		if w.stream {
			w.flushPeriod(wk, sl)
		}
	}
	if w.started {
		if len(wk.queues) > 0 {
			w.drainQueues(wk, sl)
		}
		w.dropPending(sl)
	}
}

// dropPending drops what a started run's repeat booked past its
// horizon, keeping its total.
func (w *Workspace) dropPending(sl *slot) {
	q := &sl.pend
	// bound: the slot's points
	for i := 0; i < q.nPts; i++ {
		if pt := q.pts[i]; pt.col == int32(w.l.total()) { //nolint:gosec // column 0
			sl.dropped += pt.x
		}
	}
	// bound: the slot's segments
	for i := 0; i < q.nSeg; i++ {
		if s := q.segs[i]; w.m.inTotal[s.cat] {
			sl.dropped += s.rate * (s.hi - s.lo)
		}
	}
	q.reset()
}

// closeRows closes the block's rows in period order, once the slot holds
// everything booked into them; a burn-in task's are dropped.
func (w *Workspace) closeRows(sl *slot, hs []*stats.LogHist) {
	if sl.task < 0 {
		return
	}
	if w.started {
		w.closeHorizon(sl, hs)
		return
	}
	// bound: the block
	for p := sl.a; p < sl.b && sl.err == nil; p++ {
		w.closeRow(sl, sl.rowAt(w.l, p), p, hs)
		sl.closed++
	}
}

// closeHorizon closes a started run's repeat as one row, its periods
// summed, in the scratch row, which is free once the block is simulated.
// The rows per period, their named totals set, stay for the finaliser.
func (w *Workspace) closeHorizon(sl *slot, hs []*stats.LogHist) {
	row := sl.far
	clear(row)
	// bound: the horizon
	for p := sl.a; p < sl.b; p++ {
		pr := sl.rowAt(w.l, p)
		w.applyTotals(pr)
		for i, x := range pr {
			row[i] += x
		}
	}
	w.closeRow(sl, row, sl.task, hs)
	clear(row)
	sl.closed = 1
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
	clear(wk.sums)
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
func (w *Workspace) runSource(wk *worker, sl *slot, k int32, p int64) {
	m := w.m
	src := m.srcs[k]
	if src.trigger != noIndex {
		w.runTrigger(wk, sl, k, p)
		return
	}
	// The times a source carries are from the period's start; they move
	// to the next period's start by one exact subtraction.
	if src.chainEnd > src.chainStart {
		w.runChainSource(wk, sl, k, p)
		wk.chainT[k]--
		return
	}
	if src.every > 1 || src.repair != noIndex {
		w.runRenewalSource(wk, sl, k, p)
		wk.downUntil[k]--
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
		t := wk.ss.Control().Float64()
		if w.stream {
			w.beginRound(wk, t)
		}
		w.walk(wk, sl, k, src.to, t, p)
	}
}

// walk follows one round from node v at time t to a leaf, booking as it
// goes. The round's arrival period is p, and t is from its start, so a
// time within a period keeps its resolution at any p; counts are kept
// for p ≥ 0.
func (w *Workspace) walk(wk *worker, sl *slot, k, v int32, t float64, p int64) {
	m := w.m
	nodes := len(m.nodes)
	// bound: a path visits each node at most once
	for step := 0; step <= nodes; step++ {
		sl.steps++
		if p >= 0 && !wk.given {
			sl.reach[int(k)*nodes+int(v)]++
		}
		row := w.rowFor(sl, t, p)
		row[w.l.entry(v)]++
		w.count(wk, v)
		n := m.nodes[v]
		booked := w.bookEntry(wk, row, k, v, n)
		if w.stopsAt(wk, sl, k, n, t, p) {
			return
		}
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
		next, tau, booked2, ok := w.advance(wk, sl, k, v, n, t, p, booked)
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

// stopsAt ends a walk's step at node n after its entry is booked: on an
// error, or at a queue, where the round waits its turn.
func (w *Workspace) stopsAt(wk *worker, sl *slot, k int32, n cnode, t float64, p int64) bool {
	w.flushFar(sl)
	if sl.err != nil {
		return true
	}
	if n.queue != noIndex {
		w.parkRound(wk, sl, n.queue, k, t, p)
		return true
	}
	return false
}

// chooseSequential tries the outcomes in order, each with its own
// probability and a fresh uniform after the first, u; the last takes
// what is left.
func (w *Workspace) chooseSequential(wk *worker, n cnode, u float64) int32 {
	m := w.m
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
}

// hop is the outcome taken out of a node and its target.
type hop struct{ out, to int32 }

// advance selects the outcome out of node v, counts it, draws the holding
// time and books the PerTime functions over it. It returns the hop, the
// holding time and the value booked so far; ok is false on an error.
func (w *Workspace) advance(wk *worker, sl *slot, k, v int32, n cnode, t float64, p int64, booked float64) (hop, float64, float64, bool) {
	m := w.m
	j := w.choose(wk, v, n)
	if p >= 0 && !wk.given {
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
		booked += w.bookRate(wk, sl, k, v, n, t, t+tau, p)
		if sl.err != nil {
			return hop{}, 0, 0, false
		}
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
		w.feed(wk, f, x)
		booked += x
	}
	if n.thresh != noIndex {
		x := w.draw(wk, m.fns[n.thresh].pool)
		w.book(row, k, v, m.fns[n.thresh].category, x)
		w.feed(wk, n.thresh, x)
		booked += x
		wk.tested = x
	}
	return booked
}

// rowFor returns the row for a booking at time t from the start of
// arrival period p: the period's row within the block, else the scratch
// row, which the caller flushes to the pending list.
func (w *Workspace) rowFor(sl *slot, t float64, p int64) []float64 {
	q := p + int64(math.Floor(t))
	if q < sl.b {
		return sl.rowAt(w.l, q)
	}
	return sl.farRow(q)
}

// book adds a value to a row's total, category, source and node
// contribution; a category outside the total gets it alone.
func (w *Workspace) book(row []float64, k, v, cat int32, x float64) {
	l := &w.l
	if !w.m.inTotal[cat] {
		row[l.cat(cat)] += x
		return
	}
	row[l.total()] += x
	row[l.cat(cat)] += x
	row[l.src(k)] += x
	row[l.contrib(v)] += x
}

// bookRate spreads each PerTime rate of node v over [a, b), times from
// the start of arrival period p: a row per period overlapped within the
// block, and the rest past it as one pending segment. It returns the
// value booked.
func (w *Workspace) bookRate(wk *worker, sl *slot, k, v int32, n cnode, a, b float64, p int64) float64 {
	m := w.m
	total := 0.0
	end := float64(sl.b - p)
	// bound: functions of one node
	for f := n.rateStart; f < n.rateEnd; f++ {
		rate := w.draw(wk, m.fns[f].pool)
		cat := m.fns[f].category
		// bound: the periods the holding time spans within the block
		for q := math.Floor(a); q < math.Min(b, end); q++ {
			lo := math.Max(a, q)
			hi := math.Min(b, q+1)
			x := rate * (hi - lo)
			w.book(sl.rowAt(w.l, p+int64(q)), k, v, cat, x)
			w.feed(wk, f, x)
			total += x
		}
		if b > end && w.started {
			// A started run drops what lands past its horizon.
			if m.inTotal[cat] {
				sl.dropped += rate * (b - math.Max(a, end))
			}
		} else if b > end {
			lo := math.Max(a, end)
			x := rate * (b - lo)
			if !sl.pend.addSeg(pendSeg{p0: p, from: sl.task, lo: lo, hi: b, rate: rate, k: k, v: v, cat: cat}) && sl.err == nil {
				sl.err = pendingError(len(sl.pend.segs), sl.b)
			}
			w.feed(wk, f, x)
			total += x
		}
	}
	return total
}

// choose selects an outcome of node v by its selection mode.
func (w *Workspace) choose(wk *worker, v int32, n cnode) int32 {
	m := w.m
	if n.thresh != noIndex {
		// Above, the first outcome, when the value is greater.
		if wk.tested > n.over {
			return n.outStart
		}
		return n.outStart + 1
	}
	u := wk.ss.Control().Float64()
	switch {
	case n.mode == Sequential:
		return w.chooseSequential(wk, n, u)
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

// closeRow folds a finished period into the slot statistics and the
// histograms, after checking that the breakdowns sum to the total; with
// the breakdowns off, only the total, the entries, the named totals,
// the layers and the triggers' sums.
func (w *Workspace) closeRow(sl *slot, row []float64, p int64, hs []*stats.LogHist) {
	l := &w.l
	total := row[l.total()]
	w.applyTotals(row)
	w.applyLevels(row)
	if !w.m.crit.breakdown {
		w.applyLayers(row, total)
		w.fold(sl, row, 0, 1, hs)
		w.fold(sl, row, l.entryAt, l.contribAt, hs)
		w.fold(sl, row, l.totalAt, l.width, hs)
		if w.m.tailX > 0 && total >= w.m.tailX {
			sl.tail++
		}
		w.addWhen(sl, row)
		w.keepSeries(row, p)
		return
	}
	var cats, srcs float64
	// bound: row width
	for c := 0; c < l.cats; c++ {
		if w.m.inTotal[c] {
			cats += row[l.cat(index(c))]
		}
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
	w.fold(sl, row, 0, l.width, hs)
	if w.m.tailX > 0 && total >= w.m.tailX {
		sl.tail++
	}
	w.addWhen(sl, row)
	w.keepSeries(row, p)
}

// applyTotals sets each named total's column to the sum of its
// categories' columns, less those it subtracts.
func (w *Workspace) applyTotals(row []float64) {
	l := &w.l
	for i := range w.m.totals {
		t := &w.m.totals[i]
		s := 0.0
		for c, in := range t.has {
			if in {
				s += t.sign(c) * row[l.cat(index(c))]
			}
		}
		row[l.named(i)] = s
	}
}

// applyLevels sets each of Config.At's columns to 1 where its quantity
// reaches the value, else 0.
func (w *Workspace) applyLevels(row []float64) {
	// bound: the levels
	for i, lv := range w.m.levels {
		x := 0.0
		if row[lv.row] >= lv.x {
			x = 1
		}
		row[w.l.level(i)] = x
	}
}

// fold adds the row's quantities lo to hi to the slot statistics and
// their histograms.
func (w *Workspace) fold(sl *slot, row []float64, lo, hi int, hs []*stats.LogHist) {
	if sl.coarse != nil {
		// bound: row width
		for i := lo; i < hi; i++ {
			sl.stats[i].add(row[i])
			if h := hs[i]; h != nil {
				h.AddPair(row[i], sl.coarse[i])
			}
		}
		return
	}
	// bound: row width
	for i := lo; i < hi; i++ {
		sl.stats[i].add(row[i])
		if h := hs[i]; h != nil {
			h.Add(row[i])
		}
	}
}

// keepSeries copies a closed period's row into the series, if one is kept.
func (w *Workspace) keepSeries(row []float64, p int64) {
	if s := w.result.Series; s != nil && !w.started {
		width := w.l.width
		copy(s.rows[int(p)*width:(int(p)+1)*width], row)
	}
}

// invariantTol is the rounding allowance on the breakdown sums.
const invariantTol = 1e-9
