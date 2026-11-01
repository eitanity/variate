package variate

import (
	"testing"

	"code.eitanity.com/variate/dist"
)

// TestTimesWithinPeriodAtAnyPeriodNumber: a block simulates the same
// whatever its period numbers, so times within a period keep their full
// resolution however far into the run it is. The same task at period
// 4096 and at period 2^50 + 4096, from the same seed, books identical
// statistics, with holding times far below a period (where 2^50 resolves
// a quarter) spread over rows by a per-time rate, a renewal source with
// a repair law, and a chain whose outages run past the period's end.
func TestTimesWithinPeriodAtAnyPeriodNumber(t *testing.T) {
	c := func(x float64) dist.Constant { d, _ := dist.NewConstant(x); return d }
	exp := func(mean float64) dist.Exponential { d, _ := dist.NewExponential(mean); return d }
	short, _ := dist.NewUniform(0.01, 0.6)
	g := Graph{
		Sources: []Source{
			{ID: "poisson", Frequency: c(3), To: "held"},
			{ID: "repaired", Frequency: c(4), Repair: exp(0.05), To: "held"},
			{ID: "chain", Chains: []Chain{{K: 1, N: 1, Fail: 2, Repair: 1}}, To: "outage"},
		},
		Nodes: []Node{
			{ID: "held", Outcomes: []Outcome{{To: "rated", Holding: exp(1.0 / 30)}}},
			{ID: "rated", Functions: []Function{{Name: "r", Category: "c", Model: c(1), Basis: PerTime}},
				Outcomes: []Outcome{{To: "leaf", Holding: short}}},
			{ID: "outage", Functions: []Function{{Name: "down", Category: "c", Model: c(1), Basis: PerTime}},
				Outcomes: []Outcome{{To: "leaf", OutageHolding: true}}},
			{ID: "leaf"},
		},
	}
	m, _, err := g.Compile(Config{})
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(Config{Workers: 1, Periods: 1 << 16})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	wk := w.workers[0]
	sl := w.slots[0]
	simulate := func(a int64) ([]qstat, []uint64) {
		sl.id, sl.task, sl.a, sl.b = 1, 1, a, a+w.block
		w.runTask(wk, sl, 7, wk.hists[0])
		if sl.err != nil {
			t.Fatalf("block at %d: %v", a, sl.err)
		}
		return append([]qstat(nil), sl.stats...), append([]uint64(nil), sl.events...)
	}
	stats, events := simulate(w.block)
	far, farEvents := simulate(1<<50 + w.block)
	for k := range events {
		if events[k] != farEvents[k] {
			t.Errorf("source %d: %d rounds at the start, %d at 2^50", k, events[k], farEvents[k])
		}
	}
	for i := range stats {
		if stats[i] != far[i] {
			t.Errorf("quantity %d: %+v at the start, %+v at 2^50", i, stats[i], far[i])
		}
	}
}
