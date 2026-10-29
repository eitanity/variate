package variate

// finalise folds a finished task into the result, strictly in task order:
// the carry buffer into the head, the head rows closed, the slot
// statistics merged, the spill copied to the carry buffer.
func (w *Workspace) finalise(sl *slot) {
	l := w.l
	width := l.width
	first := sl.a
	if sl.task == 0 {
		first = -int64(w.s)
	}
	// bound: S head rows
	for i := 0; i < w.s; i++ {
		h := sl.head[i*width : (i+1)*width]
		if sl.task > 0 {
			c := w.carry[i*width : (i+1)*width]
			for j := range h {
				h[j] += c[j]
			}
		}
		if int64(i) < sl.b-sl.a {
			w.closeRow(sl, h, sl.a+int64(i), w.hists)
			sl.closed++
			continue
		}
		// A block shorter than S: the row belongs to the spill.
		spill := sl.row(l, w.s, first, sl.a+int64(i))
		for j := range h {
			spill[j] += h[j]
		}
	}
	w.mergeSlot(sl)
	// The spill: the window rows of [b, b+S) in period order.
	// bound: S spill rows
	for i := 0; i < w.s; i++ {
		copy(w.carry[i*width:(i+1)*width], sl.row(l, w.s, first, sl.b+int64(i)))
	}
	w.carryAt = sl.b
}

// mergeSlot adds the slot's statistics and counts into the result.
func (w *Workspace) mergeSlot(sl *slot) {
	r := &w.result
	for i := range sl.stats {
		a := w.accum(i)
		q := &sl.stats[i]
		a.Sum.Merge(q.sum)
		a.W.Merge(q.w)
		if q.min < a.Min {
			a.Min = q.min
		}
		if q.max > a.Max {
			a.Max = q.max
		}
		a.Zero += q.zero
		if sl.closed == w.block {
			a.Block.Add(q.w.Mean)
		}
	}
	nodes, outs := len(w.m.nodes), len(w.m.outs)
	for k := range w.m.srcs {
		r.Events[k] += sl.events[k]
		// bound: row width
		for v := 0; v < nodes; v++ {
			r.Reach[k][v] += sl.reach[k*nodes+v]
		}
		// bound: row width
		for j := 0; j < outs; j++ {
			r.OutcomeCount[k][j] += sl.outs[k*outs+j]
		}
	}
	r.Steps += sl.steps
	r.TailPeriods += sl.tail
	width := w.l.tailWidth()
	for j := range r.TailCounts {
		r.TailCounts[j] += sl.tailCounts[j]
		sums := sl.tailSums[j*width : (j+1)*width]
		for i, v := range sums {
			r.TailSums[j][i] += v
		}
	}
}

// flushCarry adds whatever the carry buffer holds at or beyond the horizon
// to BeyondHorizon.
func (w *Workspace) flushCarry(h int64) {
	width := w.l.width
	// bound: S rows
	for i := 0; i < w.s; i++ {
		if w.carryAt+int64(i) >= h {
			w.result.BeyondHorizon += w.carry[i*width]
		}
	}
}
