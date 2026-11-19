package variate

import (
	"math"
	"math/rand/v2"

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
	wk.k.seed = seed
	wk.lane = splitmix(uint64(sl.line)) //nolint:gosec // a timeline's number
	if w.started {
		wk.lane = splitmix(uint64(sl.task) + 1<<63) //nolint:gosec // a repeat's number
	}
	if m.renewal {
		w.startSources(wk, sl.a)
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

// drawPeriod draws this period's value of every period-scoped pool,
// sets each schedule's value by the period's place in its cycle, and
// keys each source's rounds and events of the period.
func (w *Workspace) drawPeriod(wk *worker, p int64) {
	m := w.m
	wk.pkey = mix(wk.lane, uint64(p-w.keyShift)) //nolint:gosec // a period's number, negative in the burn-in
	for k, slot := range m.poolPeriod {
		if slot == noIndex {
			continue
		}
		if m.poolConst[k] {
			wk.period[slot] = m.poolMean[k]
		} else {
			wk.period[slot] = wk.k.value(m.pools[k], within(wk.pkey, m.poolKey[k]))
		}
	}
	for k := range m.srcs {
		wk.spk[k] = within(wk.pkey, m.srcKey[k])
		wk.nround[k] = 0
		wk.evPCG[k].Seed(wk.k.seed, mix(wk.spk[k], tagEvents))
	}
	for _, sc := range m.schedules {
		wk.period[m.poolPeriod[sc.pool]] = sc.at(p)
	}
	clear(wk.sums)
}

// draw returns a pool's value for this round: the period's draw for a
// period-scoped pool, else a draw keyed by the round and the pool.
func (w *Workspace) draw(wk *worker, pool int32) float64 {
	m := w.m
	if slot := m.poolPeriod[pool]; slot != noIndex {
		return wk.period[slot]
	}
	if m.poolConst[pool] {
		return m.poolMean[pool]
	}
	return wk.k.value(m.pools[pool], within(wk.rkey, m.poolKey[pool]))
}

// drawFrom draws a pool's value from r, a source's events stream.
func (w *Workspace) drawFrom(wk *worker, pool int32, r *rand.Rand) float64 {
	m := w.m
	if m.poolConst[pool] {
		return m.poolMean[pool]
	}
	return wk.k.from(m.pools[pool], r)
}

// newRound keys the next round of source k in the period.
func (wk *worker) newRound(k int32) {
	wk.rkey = within(wk.spk[k], uint64(wk.nround[k]))
	wk.nround[k]++
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
	n := dist.PoissonCountAt(lam, unitOpen(wk.k.at(within(wk.spk[k], tagCount))))
	if p >= 0 {
		sl.events[k] += n
	}
	// bound: the Poisson count
	for i := uint64(0); i < n && sl.err == nil; i++ {
		wk.newRound(k)
		t := wk.k.at(within(wk.rkey, tagTime)).Float64()
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
		q := p + int64(math.Floor(t))
		row := w.rowFor(sl, q)
		row[w.l.entry(v)]++
		w.count(wk, v)
		n := m.nodes[v]
		booked := w.bookEntry(wk, row, k, v, n, w.factor(q))
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
// probability and a fresh uniform from r after the first, u; the last
// takes what is left.
func (w *Workspace) chooseSequential(wk *worker, n cnode, u float64, r *rand.Rand) int32 {
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
		u = r.Float64()
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
	case m.outGroup != nil && m.outGroup[j] != noIndex:
		tau = w.together(wk, m.outGroup[j], o.hold)
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

// bookEntry books node v's OnEntry functions into the row, d the row's
// discount factor, and returns the
// value booked.
func (w *Workspace) bookEntry(wk *worker, row []float64, k, v int32, n cnode, d float64) float64 {
	m := w.m
	booked := 0.0
	// bound: functions of one node
	for f := n.fnStart; f < n.fnEnd; f++ {
		x := w.value(wk, f)
		if wk.vals != nil {
			wk.vals[f] = x
		}
		w.book(row, k, v, m.fns[f].category, x, d)
		w.feed(wk, f, x)
		booked += x
	}
	if n.thresh != noIndex {
		x := w.value(wk, n.thresh)
		w.book(row, k, v, m.fns[n.thresh].category, x, d)
		w.feed(wk, n.thresh, x)
		booked += x
		wk.tested = x
	}
	return booked
}

// value draws function f's value, or computes an expression's from the
// node's values drawn before it, times the period's draw of the factor
// on it.
func (w *Workspace) value(wk *worker, f int32) float64 {
	fn := w.m.fns[f]
	var x float64
	switch {
	case fn.expr != noIndex:
		x = w.m.exprs[fn.expr].eval(wk.vals)
	case w.m.fnGroup != nil && w.m.fnGroup[f] != noIndex:
		x = w.together(wk, w.m.fnGroup[f], fn.pool)
	default:
		x = w.draw(wk, fn.pool)
	}
	if fn.factor != noIndex {
		x *= wk.period[w.m.poolPeriod[fn.factor]]
	}
	return x
}

// together draws a member of group g from law pool: the group's normal
// for the round, keyed by the round and the group, mixed with the
// member's own.
func (w *Workspace) together(wk *worker, g, pool int32) float64 {
	m := w.m
	c := &m.groups[g]
	z := c.sq * wk.k.at(within(wk.rkey, m.groupKey[g])).NormFloat64()
	if c.sq1 > 0 {
		z += c.sq1 * wk.k.at(within(wk.rkey, mix(m.poolKey[pool], tagOwn))).NormFloat64()
	}
	return copulaValue(m.pools[pool], z)
}

// rowFor returns the row for a booking in period q: the period's row
// within the block, else the scratch row, which the caller flushes to the
// pending list.
func (w *Workspace) rowFor(sl *slot, q int64) []float64 {
	if q < sl.b {
		return sl.rowAt(w.l, q)
	}
	return sl.farRow(q)
}

// book adds a value to a row's total, category, source and node
// contribution; a category outside the total gets it alone. A
// discounted category books it times d, the row's period's factor: the
// one place a value is discounted, so what it feeds on is not.
func (w *Workspace) book(row []float64, k, v, cat int32, x, d float64) {
	l := &w.l
	if w.m.discounted[cat] {
		x *= d
	}
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
		rate := w.value(wk, f)
		cat := m.fns[f].category
		// bound: the periods the holding time spans within the block
		for q := math.Floor(a); q < math.Min(b, end); q++ {
			lo := math.Max(a, q)
			hi := math.Min(b, q+1)
			x := rate * (hi - lo)
			w.book(sl.rowAt(w.l, p+int64(q)), k, v, cat, x, w.factor(p+int64(q)))
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
	r := wk.k.at(within(wk.rkey, m.chooseKey[v]))
	u := r.Float64()
	switch {
	case n.mode == Sequential:
		return w.chooseSequential(wk, n, u, r)
	case n.twoWay:
		if u < wk.period[m.poolPeriod[m.outs[n.outStart].probPool]] {
			return n.outStart
		}
		return n.outStart + 1
	case n.outEnd-n.outStart > searchOutcomes:
		// The first outcome whose cumulative passes u, by bisection.
		lo, hi := n.outStart, n.outEnd-1
		// bound: log2 of the node's outcomes
		for lo < hi {
			mid := lo + (hi-lo)/2
			if u < m.outs[mid].cum {
				hi = mid
			} else {
				lo = mid + 1
			}
		}
		return lo
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

// searchOutcomes is the most outcomes a node's choice tries in order;
// more are found by bisection.
const searchOutcomes = 8

// unitOpen is a uniform on (0, 1).
func unitOpen(r *rand.Rand) float64 {
	return (float64(r.Uint64()>>11) + 0.5) / (1 << 53)
}

// invariantTol is the rounding allowance on the breakdown sums.
const invariantTol = 1e-9
