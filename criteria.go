package variate

import "math"

// evaluate checks every criterion against the observed counts and
// statistics after periods have been finalised, filling the plan rows'
// Observed and Met, and returns whether all are met. unmet is reused.
func (w *Workspace) evaluate(periods int64) (allMet bool, unmet []string) {
	r := &w.result
	c := w.m.crit
	unmet = w.unmet[:0]
	allMet = true
	for i := range w.rows {
		row := &w.rows[i]
		src := w.m.plan[i]
		switch src.kind {
		case critMinHits:
			var hits uint64
			for k := range r.Sources {
				hits += r.OutcomeCount[k][src.index]
			}
			row.Observed = float64(hits)
			row.Met = hits >= c.minHits
		case critContrib:
			row.Observed = rse(&r.Contrib[src.index], w.delays)
			row.Met = row.Observed <= c.contribRSE
		case critTarget:
			row.Observed = rse(w.accum(src.index), w.delays)
			row.Met = row.Observed <= c.targetRSE
		case critTail:
			row.Observed = float64(periods)
			row.Met = float64(periods) >= float64(c.tailHits)/(1-c.tailP)
			if w.m.tailX > 0 {
				row.Observed = float64(r.TailPeriods)
				row.Met = r.TailPeriods >= int64(c.tailHits) //nolint:gosec // TailHits is validated far below 2^63
			}
		}
		if !row.Met {
			allMet = false
			unmet = append(unmet, w.labels[i])
		}
	}
	return allMet, unmet
}

// rse is the relative standard error of an accumulated mean; +Inf when
// the error is not yet estimable.
func rse(a *Accum, delays bool) float64 {
	se := a.StdErr(delays)
	if math.IsNaN(se) || a.Mean() <= 0 {
		return math.Inf(1)
	}
	return se / a.Mean()
}

// adaptiveStop decides, at a batch boundary of an adaptive run, whether to
// stop: every criterion met, or a cap reached.
func (w *Workspace) adaptiveStop(periods int64) bool {
	c := w.m.crit
	allMet, unmet := w.evaluate(periods)
	r := &w.result
	switch {
	case allMet:
		r.Converged = true
		r.Binding = w.lastUnmet
		if r.Binding == "" {
			r.Binding = w.planBinding
		}
		r.Unmet = r.Unmet[:0]
		return true
	case periods >= c.maxPeriods || r.Steps >= c.maxSteps:
		r.Converged = false
		r.Binding = "cap"
		r.Unmet = append(r.Unmet[:0], unmet...)
		return true
	default:
		w.lastUnmet = unmet[0]
		return false
	}
}

// fixedVerdict records the criteria of a fixed run without changing it.
func (w *Workspace) fixedVerdict(periods int64) {
	allMet, unmet := w.evaluate(periods)
	r := &w.result
	if r.Stopped {
		return
	}
	r.Converged = allMet
	r.Binding = "fixed"
	r.Unmet = append(r.Unmet[:0], unmet...)
}
