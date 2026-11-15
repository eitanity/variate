package variate

import "code.eitanity.com/variate/stats"

// finalise folds a closed task into the result, strictly in task order.
// A burn-in task adds only its steps.
func (w *Workspace) finalise(sl *slot) {
	if sl.task < 0 {
		w.result.Steps += sl.steps
		return
	}
	w.mergeSlot(sl)
}

// addGroup adds a full block's quantity i to its batch group: its
// periods, its coarse histogram and its sum, each where the Accum keeps
// it.
func addGroup(a *Accum, sl *slot, q *qstat, coarse []*stats.LogHist, i int) {
	g := sl.task % batchGroups
	if a.GroupN != nil {
		a.GroupN[g] += sl.closed
	}
	if a.Groups != nil {
		a.Groups[g].Merge(coarse[i])
	}
	if a.GroupSum != nil {
		a.GroupSum[g].Merge(q.sum)
	}
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
			addGroup(a, sl, q, sl.coarse, i)
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
	if w.started {
		r.BeyondHorizon += sl.dropped
		r.ByPeriod.add(sl.rows)
		if s := r.Series; s != nil {
			copy(s.rows[int(sl.task)*len(sl.rows):], sl.rows)
		}
	}
	width := w.l.tailWidth()
	for j := range r.TailCounts {
		r.TailCounts[j] += sl.tailCounts[j]
		sums := sl.tailSums[j*width : (j+1)*width]
		for i, v := range sums {
			r.TailSums[j][i] += v
		}
	}
}
