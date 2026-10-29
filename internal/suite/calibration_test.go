package suite

import (
	"context"
	"math"
	"testing"

	"code.eitanity.com/variate"
)

// calibration runs adaptive runs over many seeds and counts how often the
// reported 95% interval covers the analytic mean total and each node's
// contribution. The stopping rule biases coverage down for heavy tails;
// the check is that it stays within the statistical test policy.
func calibration(t *testing.T, seeds int, delays bool) {
	t.Helper()
	g := genGraph(t, 9, delays)
	cfg := variate.Config{MinHits: 30, TargetRSE: 0.1, ContribRSE: 0.2, TailHits: 10, TailP: 0.99, BlockPeriods: 50, Batch: 4, Workers: 8}
	m, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	e, _ := m.Analytic()
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	coverTotal := 0
	coverContrib, nContrib := 0, 0
	for seed := 1; seed <= seeds; seed++ {
		r, err := w.Run(context.Background(), uint64(seed))
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(r.Total.Mean()-e.Mean) <= 1.96*r.Total.StdErr(delays) {
			coverTotal++
		}
		for v := range r.Nodes {
			if e.Contrib[v] <= 0 {
				continue
			}
			nContrib++
			if math.Abs(r.Contrib[v].Mean()-e.Contrib[v]) <= 1.96*r.Contrib[v].StdErr(delays) {
				coverContrib++
			}
		}
	}
	tol := 4.5 * math.Sqrt(0.95*0.05/float64(seeds))
	if c := float64(coverTotal) / float64(seeds); math.Abs(c-0.95) > tol {
		t.Errorf("delays %v: total coverage %.3f over %d seeds (tolerance %.3f)", delays, c, seeds, tol)
	}
	if c := float64(coverContrib) / float64(nContrib); math.Abs(c-0.95) > tol {
		t.Errorf("delays %v: contribution coverage %.3f over %d intervals (tolerance %.3f)", delays, c, nContrib, tol)
	}
}

func TestCalibrationSmoke(t *testing.T) {
	calibration(t, calibrationSeeds, false)
	calibration(t, calibrationSeeds, true)
}
