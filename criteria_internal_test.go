package variate

import (
	"math"
	"testing"
)

// Criterion boundaries: a count exactly at its target is met, one below
// is not; mutation testing showed no test held the boundary.
func TestCriteriaBoundaries(t *testing.T) {
	// TailP 0.5 keeps TailHits/(1 − TailP) exact in floating point, so
	// the boundary is reachable.
	cfg := Config{Periods: 10, MinHits: 7, ContribRSE: 0.05, TargetRSE: 0.02, TailP: 0.5, TailHits: 3}
	m, _, err := diamond().Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	w.resetResult(1)
	r := &w.result
	// Hits exactly at MinHits on every outcome.
	for k := range r.OutcomeCount {
		for j := range r.OutcomeCount[k] {
			r.OutcomeCount[k][j] = 0
		}
		r.OutcomeCount[k][0] = 7
	}
	for j := range m.outs {
		r.OutcomeCount[0][j] = 7
	}
	// Exactly the periods the tail rule needs: TailHits/(1 − TailP) = 6.
	w.evaluate(6)
	for _, row := range w.rows {
		if row.Criterion == critMinHits && (!row.Met || row.Observed != 7) {
			t.Errorf("min-hits at exactly %d: %+v", 7, row)
		}
		if row.Criterion == critTail && !row.Met {
			t.Errorf("tail at exactly the periods needed not met: %+v", row)
		}
	}
	r.OutcomeCount[0][0] = 6
	w.evaluate(5)
	for _, row := range w.rows {
		if row.Criterion == critMinHits && row.Driver == w.rows[0].Driver && row.Met && row.Observed == 6 {
			t.Errorf("min-hits one below met: %+v", row)
		}
		if row.Criterion == critTail && row.Met {
			t.Errorf("tail one period short met: %+v", row)
		}
	}
	// With a tail bound, TailPeriods exactly at TailHits is met.
	w.m.tailX = 1
	r.TailPeriods = 3
	w.evaluate(1)
	for _, row := range w.rows {
		if row.Criterion == critTail && (!row.Met || row.Observed != 3) {
			t.Errorf("tail periods at exactly TailHits: %+v", row)
		}
	}
	r.TailPeriods = 2
	w.evaluate(1)
	for _, row := range w.rows {
		if row.Criterion == critTail && row.Met {
			t.Errorf("tail periods one short met: %+v", row)
		}
	}
	w.m.tailX = 0
	// An RSE exactly at its target is met: a run whose total has a known
	// standard error. rse() with a zero mean is infinite.
	var a Accum
	if !math.IsInf(rse(&a, false), 1) {
		t.Errorf("rse of an empty accumulator %g", rse(&a, false))
	}
}

// Bound formulas at their definitions, which the statistical checks are
// too loose to pin: the lattice standard deviation grows with the span,
// the round-off term at zero tilt is δ·√(K+1), and δ follows its formula.
func TestBoundFormulas(t *testing.T) {
	m, _, err := diamond().Compile(Config{})
	if err != nil {
		t.Fatal(err)
	}
	tr := newTransformer(m, TransformConfig{})
	e := tr.e
	mean0, sd0 := tr.latticeMoments(0)
	mean1, sd1 := tr.latticeMoments(1)
	if mean0 != e.Mean || mean1 != e.Mean+e.DrawsMean || sd0 != math.Sqrt(e.Var) || sd1 != math.Sqrt(e.Var)+math.Sqrt(e.DrawsVar) {
		t.Errorf("lattice moments %g %g / %g %g", mean0, sd0, mean1, sd1)
	}
	delta := tr.roundoffDelta(1<<16, 0)
	want := roundoffSafety * unitRoundoff * (16 + float64(m.maxPath+len(m.srcs)+2)) * math.Max(1, tr.lam[0])
	if delta != want {
		t.Errorf("delta %g want %g", delta, want)
	}
	if got := tr.roundoffAt(0.1, 1<<16, 0, 0, 99); got != delta*10 {
		t.Errorf("rho at zero tilt %g want %g", got, delta*10)
	}
	if got := tr.roundoffAt(0.1, 1<<16, 0, 1e-9, 99); got != (delta+1e-9)*10 {
		t.Errorf("rho with eps %g want %g", got, (delta+1e-9)*10)
	}
	lower := []float64{0.5, 0.4, 0.6, 0.1}
	upper := []float64{1, 0.9, 0.95, 0.5}
	tighten(lower, upper)
	if lower[0] != 0.6 || lower[1] != 0.6 || lower[2] != 0.6 || lower[3] != 0.1 || upper[2] != 0.9 {
		t.Errorf("tighten %v %v", lower, upper)
	}
}
