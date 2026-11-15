package suite

import (
	"context"
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// rampGraph: 10 rounds a period wait in a holding at a rate of 3 per
// period held, then book 1 on reaching done.
func rampGraph(hold dist.Distribution, horizon int) variate.Graph {
	return variate.Graph{
		ID:      "ramp",
		Start:   variate.Start{Horizon: horizon},
		Sources: []variate.Source{{ID: "work", Frequency: C(10), To: "queue"}},
		Nodes: []variate.Node{
			{ID: "queue", Functions: []variate.Function{{Name: "wait", Category: "c", Model: C(3), Basis: variate.PerTime}},
				Outcomes: []variate.Outcome{{To: "done", Holding: hold}}},
			{ID: "done", Functions: []variate.Function{Fn("cost", "c", C(1))}},
		},
	}
}

// TestStartedExactExponential: from an empty start, with an exponential
// holding of mean 2, the rounds reaching done by time t number
// 10·(t − 2(1 − e^(−t/2))) and the time held 2·10·(t − 2(1 − e^(−t/2)));
// each period's exact mean is the difference across it.
func TestStartedExactExponential(t *testing.T) {
	hold, _ := dist.NewExponential(2)
	m, _, err := rampGraph(hold, 12).Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, err := m.Analytic()
	if err != nil {
		t.Fatal(err)
	}
	cum := func(t float64) float64 { return t - 2*(1-math.Exp(-t/2)) }
	nodes := m.NodeIDs()
	done, queue := indexOf(nodes, "done"), indexOf(nodes, "queue")
	horizon := 0.0
	for k, row := range e.ByPeriod {
		passed := 10 * (cum(float64(k+1)) - cum(float64(k)))
		want := passed + 3*2*passed
		if got := row[entryColumn(m, done)]; math.Abs(got-passed) > 1e-9*passed {
			t.Errorf("period %d: entries at done %v, want %v", k, got, passed)
		}
		if got := row[entryColumn(m, queue)]; math.Abs(got-10) > 1e-12 {
			t.Errorf("period %d: entries at queue %v, want 10", k, got)
		}
		if got := row[0]; math.Abs(got-want) > 1e-9*want {
			t.Errorf("period %d: total %v, want %v", k, got, want)
		}
		horizon += want
	}
	if math.Abs(e.Mean-horizon) > 1e-9*horizon {
		t.Errorf("horizon mean %v, want %v", e.Mean, horizon)
	}
	if !math.IsNaN(e.Var) || !math.IsNaN(e.PZero()) {
		t.Errorf("variance %v and P(T = 0) %v, want NaN in a started run", e.Var, e.PZero())
	}
}

// TestStartedExactConstant: a constant holding of 1.5 periods passes
// rounds arriving in period i in periods i + 1 and i + 2, half each, and
// holds a round 1/8, 1, 3/8 of a period in periods i, i + 1, i + 2 on
// top of the arrival's own.
func TestStartedExactConstant(t *testing.T) {
	m, _, err := rampGraph(C(1.5), 5).Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := m.Analytic()
	done := indexOf(m.NodeIDs(), "done")
	wantPassed := []float64{0, 5, 10, 10, 10}
	// Held per arrival: 0.5 in its own period on average, then 0.875,
	// then 0.125; summed over arrivals of every earlier period.
	held := []float64{0.5, 0.5 + 0.875, 0.5 + 0.875 + 0.125, 1.5, 1.5}
	for k, row := range e.ByPeriod {
		if got := row[entryColumn(m, done)]; math.Abs(got-wantPassed[k]) > 1e-12 {
			t.Errorf("period %d: entries at done %v, want %v", k, got, wantPassed[k])
		}
		want := wantPassed[k] + 3*10*held[k]
		if got := row[0]; math.Abs(got-want) > 1e-9 {
			t.Errorf("period %d: total %v, want %v", k, got, want)
		}
	}
}

// startedMix has a scheduled source through a lognormal holding, a
// repair source, a block source through a constant holding, and a path
// through two holdings, which has no exact mean per period.
func startedMix(horizon int) variate.Graph {
	logn, _ := dist.NewLognormalFromMeanSD(1.2, 2)
	repair, _ := dist.NewExponential(0.4)
	return variate.Graph{
		ID:        "mix",
		Start:     variate.Start{Horizon: horizon},
		Schedules: []variate.Schedule{{ID: "season", Values: []float64{0.5, 1, 2}}},
		Sources: []variate.Source{
			{ID: "s", Frequency: C(4), To: "a", Schedule: "season"},
			{ID: "r", Frequency: C(3), To: "b", Repair: repair},
			{ID: "u", Chains: []variate.Chain{{K: 1, N: 2, Fail: 2, Repair: 6}}, To: "c"},
		},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{{Name: "fa", Category: "x", Model: Pert(1, 2, 4), Basis: variate.PerTime}},
				Outcomes: []variate.Outcome{{To: "b", P: C(0.6), Holding: logn}, {To: "e", P: C(0.4)}}},
			{ID: "b", Functions: []variate.Function{Fn("fb", "y", Pert(5, 10, 20))},
				Outcomes: []variate.Outcome{{To: "d", Holding: C(0.7)}}},
			{ID: "c", Functions: []variate.Function{Fn("fc", "y", C(2))},
				Outcomes: []variate.Outcome{{To: "e", Holding: C(1.5)}}},
			{ID: "d", Functions: []variate.Function{Fn("fd", "x", C(7))}},
			{ID: "e", Functions: []variate.Function{Fn("fe", "y", C(1))}},
		},
	}
}

// TestStartedSimulatedAgainstExact: each period's simulated mean of
// every quantity with an exact mean lies within four standard errors of
// it; the quantities behind two holdings have none and are simulated.
func TestStartedSimulatedAgainstExact(t *testing.T) {
	g := startedMix(6)
	m, _, err := g.Compile(variate.Config{Repeats: 40000, Seed: 11})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := m.Analytic()
	r, err := m.Simulate(context.Background(), variate.Config{Repeats: 40000, Seed: 11})
	if err != nil {
		t.Fatal(err)
	}
	ps := r.ByPeriod
	if ps == nil || ps.Periods() != 6 {
		t.Fatalf("per-period statistics %v, want 6 periods", ps)
	}
	nodes := m.NodeIDs()
	d := indexOf(nodes, "d")
	exact := 0
	for k := range 6 {
		row := e.ByPeriod[k]
		if !math.IsNaN(row[entryColumn(m, d)]) {
			t.Errorf("period %d: entries at d are exact at %v, but d is behind two holdings", k, row[entryColumn(m, d)])
		}
		check := func(name string, want float64, got variate.PeriodStat) {
			if math.IsNaN(want) {
				return
			}
			exact++
			if math.Abs(got.Mean-want) > 4*got.StdErr+1e-12 {
				t.Errorf("period %d %s: simulated %v ± %v, exact %v", k, name, got.Mean, got.StdErr, want)
			}
		}
		for v, id := range nodes {
			check("entries "+id, row[entryColumn(m, v)], ps.Entries(v, k))
		}
		for c, id := range m.Categories() {
			check("category "+id, row[1+c], ps.Category(c, k))
		}
		for s, id := range m.SourceIDs() {
			check("source "+id, row[1+len(m.Categories())+s], ps.Source(s, k))
		}
	}
	// Per period: entries at a, b, c and e; category y; sources r and u.
	if exact != 6*7 {
		t.Errorf("%d exact per-period means, want 42", exact)
	}
	if got, want := r.Periods, int64(40000); got != want {
		t.Errorf("rows %d, want one per repeat, %d", got, want)
	}
}

// TestStartedRepeatsExactly: the same seed repeats a started run, on any
// number of workers.
func TestStartedRepeatsExactly(t *testing.T) {
	m, _, err := startedMix(4).Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	var totals [2][4]float64
	for i, workers := range []int{1, 4} {
		r, err := m.Simulate(context.Background(), variate.Config{Repeats: 3000, Seed: 5, Workers: workers})
		if err != nil {
			t.Fatal(err)
		}
		for k := range 4 {
			totals[i][k] = r.ByPeriod.Total(k).Mean
		}
	}
	if totals[0] != totals[1] {
		t.Errorf("per-period totals %v on one worker, %v on four", totals[0], totals[1])
	}
}

// TestStartedRefusals: a started run refuses the settings it has no use
// for, and a steady-state run refuses repeats.
func TestStartedRefusals(t *testing.T) {
	hold, _ := dist.NewExponential(2)
	cases := []struct {
		name string
		g    variate.Graph
		cfg  variate.Config
		want string
	}{
		{"periods", rampGraph(hold, 3), variate.Config{Periods: 10}, "counts repeats"},
		{"burn-in", rampGraph(hold, 3), variate.Config{BurnIn: 10}, "no burn-in"},
		{"block", rampGraph(hold, 3), variate.Config{BlockPeriods: 10}, "one repeat"},
		{"transform", rampGraph(hold, 3), variate.Config{Transform: true}, "stationary"},
		{"repeats", rampGraph(hold, 0), variate.Config{Repeats: 10}, "started run"},
		{"horizon", rampGraph(hold, -1), variate.Config{}, "between 1"},
	}
	for _, c := range cases {
		_, _, err := c.g.Compile(c.cfg)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want one containing %q", c.name, err, c.want)
		}
	}
}

func indexOf(xs []string, x string) int {
	for i, y := range xs {
		if y == x {
			return i
		}
	}
	return -1
}

// entryColumn is node v's entries column in the rows of ByPeriod: after
// the total, the categories and the sources.
func entryColumn(m *variate.Compiled, v int) int {
	return 1 + len(m.Categories()) + len(m.SourceIDs()) + v
}
