package variate

import "math"

// pendPoint is a booking past the block of the task that made it: a
// value for one column of period p's row.
type pendPoint struct {
	p    int64   // the period
	from int64   // the task that booked it
	x    float64 // the value
	col  int32   // the row column
}

// pendSeg is a rate over a holding time past the block of the task that
// booked it: the span [lo, hi), in periods from the start of arrival
// period p0, at rate per period, booked as source k, node v, category
// cat.
type pendSeg struct {
	p0        int64
	from      int64
	lo, hi    float64
	rate      float64
	k, v, cat int32
}

// The sizes of a pending point and segment, for the footprint.
const (
	pendPointBytes = 32
	pendSegBytes   = 56
)

// pending is a fixed-capacity list of bookings past a block, kept in the
// order they were made. It never grows: a full list is an error.
type pending struct {
	pts  []pendPoint
	segs []pendSeg
	nPts int
	nSeg int
}

func newPending(capacity int) pending {
	return pending{pts: make([]pendPoint, capacity), segs: make([]pendSeg, capacity)}
}

func (q *pending) reset() { q.nPts, q.nSeg = 0, 0 }

// addPoint appends a point, reporting false when the list is full.
func (q *pending) addPoint(pt pendPoint) bool {
	if q.nPts == len(q.pts) {
		return false
	}
	q.pts[q.nPts] = pt
	q.nPts++
	return true
}

// addSeg appends a segment, reporting false when the list is full.
func (q *pending) addSeg(s pendSeg) bool {
	if q.nSeg == len(q.segs) {
		return false
	}
	q.segs[q.nSeg] = s
	q.nSeg++
	return true
}

// farRow returns the slot's scratch row for a booking in period q past
// its block. The caller books into it and then calls flushFar.
func (sl *slot) farRow(q int64) []float64 {
	sl.farAt = q
	return sl.far
}

// flushFar moves what the scratch row holds into the slot's pending
// points, a column at a time, and clears it.
func (w *Workspace) flushFar(sl *slot) {
	if sl.farAt < sl.b {
		return
	}
	if w.started {
		// A started run drops what lands past its horizon.
		sl.dropped += sl.far[w.l.total()]
		clear(sl.far)
		sl.farAt = math.MinInt64
		return
	}
	for c, x := range sl.far {
		if x == 0 {
			continue
		}
		if !sl.pend.addPoint(pendPoint{p: sl.farAt, from: sl.task, x: x, col: int32(c)}) && sl.err == nil { //nolint:gosec // a row column
			sl.err = pendingError(len(sl.pend.pts), sl.farAt)
		}
		sl.far[c] = 0
	}
	sl.farAt = math.MinInt64
}

// handOff gives slot sl what the tasks before it booked into its block,
// in the order they booked it, and keeps the rest; then it takes the
// slot's own pending bookings. Each timeline has its own store, so a
// slot sees only what its timeline booked. It runs on the finaliser in
// task order. A
// burn-in task's rows are never closed, so what lands in them is
// dropped. The slot records, by the task each came from, the total it
// was handed, which an early stop counts beyond the horizon.
func (w *Workspace) handOff(sl *slot) error {
	st := &w.stores[sl.line]
	keep := 0
	// bound: the store's points
	for i := 0; i < st.nPts; i++ {
		pt := st.pts[i]
		if pt.p >= sl.b {
			st.pts[keep] = pt
			keep++
			continue
		}
		if sl.task >= 0 {
			sl.rowAt(w.l, pt.p)[pt.col] += pt.x
			if pt.col == int32(w.l.total()) { //nolint:gosec // column 0
				sl.addIncoming(pt.from, pt.x)
			}
		}
	}
	st.nPts = keep
	keep = 0
	// bound: the store's segments
	for i := 0; i < st.nSeg; i++ {
		s := st.segs[i]
		end := float64(sl.b - s.p0)
		if sl.task >= 0 {
			sl.addIncoming(s.from, w.bookSeg(sl, s, math.Min(s.hi, end)))
		}
		if s.hi > end {
			s.lo = math.Max(s.lo, end)
			st.segs[keep] = s
			keep++
		}
	}
	st.nSeg = keep
	// bound: the slot's points
	for i := 0; i < sl.pend.nPts; i++ {
		if !st.addPoint(sl.pend.pts[i]) {
			return pendingError(len(st.pts), sl.b)
		}
	}
	// bound: the slot's segments
	for i := 0; i < sl.pend.nSeg; i++ {
		if !st.addSeg(sl.pend.segs[i]) {
			return pendingError(len(st.segs), sl.b)
		}
	}
	sl.pend.reset()
	return nil
}

// bookSeg books segment s's span up to hi into the slot's rows, a row per
// period overlapped, and returns the value booked into the total.
func (w *Workspace) bookSeg(sl *slot, s pendSeg, hi float64) float64 {
	total := 0.0
	// bound: the periods of the block the segment overlaps
	for q := math.Floor(s.lo); q < hi; q++ {
		lo := math.Max(s.lo, q)
		x := s.rate * (math.Min(hi, q+1) - lo)
		w.book(sl.rowAt(w.l, s.p0+int64(q)), s.k, s.v, s.cat, x, w.factor(s.p0+int64(q)))
		total += x
	}
	if !w.m.inTotal[s.cat] {
		return 0
	}
	return total
}

// addIncoming adds x, handed from task from, to the slot's incoming
// totals: by distance back from the slot's task up to the ring's length,
// and in one bucket beyond it.
func (sl *slot) addIncoming(from int64, x float64) {
	d := sl.task - from
	if d >= int64(len(sl.incoming)) {
		d = 0
	}
	sl.incoming[d] += x
}

// beyond returns the total a handed slot was given by tasks before
// task stop: bucket 0 holds those further back than the ring, all before
// it, since a handed slot is never more slots ahead of the finaliser
// than there are slots.
func (sl *slot) beyond(stop int64) float64 {
	v := sl.incoming[0]
	for d := 1; d < len(sl.incoming); d++ {
		if sl.task-int64(d) < stop {
			v += sl.incoming[d]
		}
	}
	return v
}

// flushPending adds to BeyondHorizon what the tasks before stop booked at
// or after its timeline's first period from task stop on, and no slot
// closed: what the stores still hold, and what was handed to slots that
// were never finalised.
func (w *Workspace) flushPending(stop int64) {
	r := &w.result
	for i := range w.stores {
		st := &w.stores[i]
		// bound: the store's points
		for i := 0; i < st.nPts; i++ {
			if pt := st.pts[i]; pt.from < stop && pt.col == int32(w.l.total()) { //nolint:gosec // column 0
				r.BeyondHorizon += pt.x
			}
		}
		// bound: the store's segments
		for i := 0; i < st.nSeg; i++ {
			if s := st.segs[i]; s.from < stop && w.m.inTotal[s.cat] {
				r.BeyondHorizon += s.rate * (s.hi - s.lo)
			}
		}
	}
	for i, sl := range w.slots {
		if st := w.sched.slotState[i]; (st == slotHanded || st == slotDone) && sl.task >= stop {
			r.BeyondHorizon += sl.beyond(stop)
		}
	}
}
