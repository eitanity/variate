package suite

import (
	"context"
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
)

// rarePath: 600 events per period; the outcome a→rare has probability p.
// The rare value is kept moderate so that the min-hits criterion, not the
// variance of the total, sets the run length.
func rarePath(p float64) variate.Graph {
	return variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(600), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Outcomes: []variate.Outcome{{To: "rare", P: C(p)}, {To: "common"}}},
			{ID: "rare", Functions: []variate.Function{Fn("big", "loss", C(1e3))}},
			{ID: "common", Functions: []variate.Function{Fn("small", "loss", Pert(1, 2, 3))}},
		},
	}
}

func TestPlanRarePath(t *testing.T) {
	m, _, err := rarePath(1e-6).Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	crit, driver := m.Binding()
	if crit != "min-hits" || driver != "a→rare" {
		t.Errorf("binding %s %s", crit, driver)
	}
	rows := m.Plan()
	var need int64
	for _, r := range rows {
		if r.Criterion == "min-hits" && r.Driver == "a→rare" {
			need = r.PeriodsNeeded
			if math.Abs(r.PlannedPerPeriod-6e-4) > 1e-12 {
				t.Errorf("planned per period %g", r.PlannedPerPeriod)
			}
		}
	}
	// 1000 hits at 6e-4 per period.
	if need != int64(math.Ceil(1000/6e-4)) {
		t.Errorf("periods needed %d", need)
	}
	if m.PlanPeriods(4096*64) < need {
		t.Error("plan not rounded up to whole batches")
	}
}

func TestPlanCaps(t *testing.T) {
	_, _, err := rarePath(1e-6).Compile(variate.Config{MaxPeriods: 1000})
	if err == nil || !strings.Contains(err.Error(), "min-hits a→rare") {
		t.Errorf("got %v", err)
	}
	m, _, err := rarePath(1e-6).Compile(variate.Config{MaxPeriods: 1000, AllowUnconverged: true})
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(variate.Config{MaxPeriods: 1000, AllowUnconverged: true, BlockPeriods: 10, Batch: 4, Workers: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 1)
	if err != nil {
		t.Fatal(err)
	}
	if r.Converged || r.Binding != "cap" || r.Periods != 1000 {
		t.Errorf("converged %v binding %s periods %d", r.Converged, r.Binding, r.Periods)
	}
	if len(r.Unmet) == 0 || !strings.Contains(strings.Join(r.Unmet, ","), "min-hits a→rare") {
		t.Errorf("unmet %v", r.Unmet)
	}
}

// TestAdaptiveRarePath: the adaptive run stops at the first batch boundary
// where every outcome has MinHits hits, naming the rare outcome as binding.
func TestAdaptiveRarePath(t *testing.T) {
	cfg := variate.Config{MinHits: 50, TargetRSE: 0.5, ContribRSE: 0.5, TailHits: 1, TailP: 0.9, BlockPeriods: 100, Batch: 4, Workers: 4}
	m, _, err := rarePath(1e-4).Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 7)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Converged || r.Binding != "min-hits a→rare" {
		t.Errorf("converged %v binding %q", r.Converged, r.Binding)
	}
	batch := int64(100 * 4)
	if r.Periods%batch != 0 || r.Periods == 0 {
		t.Errorf("periods %d not a batch multiple", r.Periods)
	}
	hits := r.OutcomeCount[0][1]
	if hits < 50 {
		t.Errorf("rare hits %d", hits)
	}
	// Agreement with the analytic hit rate: 600 × 1e-4 = 0.06 per period.
	if z := (float64(hits) - 0.06*float64(r.Periods)) / math.Sqrt(0.06*float64(r.Periods)); math.Abs(z) > 4.5 {
		t.Errorf("rare hit rate z=%.2f", z)
	}
	var s variate.Summary
	w.Summarize(&s)
	if len(s.Plan) == 0 || !s.Plan[0].Met {
		t.Errorf("plan rows %v", s.Plan)
	}
}

// TestAdaptiveDeterministicStop: the stopping period and the results are
// identical across worker counts.
func TestAdaptiveDeterministicStop(t *testing.T) {
	g := genGraph(t, 3, false)
	var ref *variate.Result
	var refPeriods int64
	var refFp []float64
	for _, workers := range []int{1, 2, 7, 32} {
		cfg := variate.Config{MinHits: 20, TargetRSE: 0.05, ContribRSE: 0.2, TailHits: 10, TailP: 0.99, BlockPeriods: 50, Batch: 4, Workers: workers}
		m, _, err := g.Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		w, err := m.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		r, err := w.Run(context.Background(), 11)
		if err != nil {
			t.Fatal(err)
		}
		fp := fingerprint(r)
		if ref == nil {
			ref, refPeriods, refFp = r, r.Periods, fp
			w.Close()
			continue
		}
		if r.Periods != refPeriods || !r.Converged {
			t.Errorf("%d workers: periods %d vs %d, converged %v", workers, r.Periods, refPeriods, r.Converged)
		}
		for i := range fp {
			if fp[i] != refFp[i] {
				t.Errorf("%d workers: field %d differs", workers, i)
				break
			}
		}
		w.Close()
	}
}

func TestFixedRunVerdict(t *testing.T) {
	cfg := variate.Config{Periods: 200, MinHits: 1000, BlockPeriods: 100}
	m, _, err := Diamond().Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	r, err := w.Run(context.Background(), 1)
	if err != nil {
		t.Fatal(err)
	}
	if r.Converged || r.Binding != "fixed" || len(r.Unmet) == 0 {
		t.Errorf("converged %v binding %s unmet %v", r.Converged, r.Binding, r.Unmet)
	}
}
