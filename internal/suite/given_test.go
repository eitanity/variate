package suite

import (
	"context"
	"math"
	"strconv"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// heldGraph: rounds of source s held on a → b, booking 1 a period held
// in category r and 1 on reaching b in category c, with no arrivals, so
// only the given state books.
func heldGraph(hold dist.Continuous, horizon int, state ...variate.Given) variate.Graph {
	return variate.Graph{
		ID:      "held",
		Start:   variate.Start{Horizon: horizon, State: state},
		Sources: []variate.Source{{ID: "s", Frequency: C(0), To: "a"}},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{{Name: "hold", Category: "r", Model: C(1), Basis: variate.PerTime}},
				Outcomes: []variate.Outcome{{To: "b", Holding: hold}}},
			{ID: "b", Functions: []variate.Function{Fn("done", "c", C(1))}},
		},
	}
}

// startedRun compiles and runs a started run.
func startedRun(t *testing.T, g variate.Graph, cfg variate.Config) (*variate.Compiled, *variate.Result) {
	t.Helper()
	m, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	r, err := m.Simulate(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	return m, r
}

// nearMean reports a simulated mean more than four standard errors from its
// closed form.
func nearMean(t *testing.T, what string, got variate.PeriodStat, want float64) {
	t.Helper()
	if !(math.Abs(got.Mean-want) <= 4*got.StdErr+1e-9) {
		t.Errorf("%s: simulated %v ± %v, closed form %v", what, got.Mean, got.StdErr, want)
	}
}

// TestGivenHeldElapsed: 50 rounds held for 1.5 periods so far on a
// lognormal holding leave in period k with probability
// (S(1.5 + k) − S(1.5 + k + 1)) / S(1.5), and are held within it for
// ∫ S(1.5 + t) dt / S(1.5) over [k, k + 1].
func TestGivenHeldElapsed(t *testing.T) {
	hold, _ := dist.NewLognormalFromMeanSD(2, 3)
	g := heldGraph(hold, 8, variate.Given{Node: "a", To: "b", Count: C(50), Elapsed: C(1.5)})
	m, r := startedRun(t, g, variate.Config{Repeats: 20000, Seed: 3})
	s := func(x float64) float64 { return hold.Survival(1.5+x) / hold.Survival(1.5) }
	b := indexOf(m.NodeIDs(), "b")
	cr, cc := indexOf(m.Categories(), "r"), indexOf(m.Categories(), "c")
	for k := range 8 {
		fk := float64(k)
		nearMean(t, "entries at b, period "+strconv.Itoa(k), r.ByPeriod.Entries(b, k), 50*(s(fk)-s(fk+1)))
		nearMean(t, "held, period "+strconv.Itoa(k), r.ByPeriod.Category(cr, k), 50*simpson(s, fk, fk+1))
		nearMean(t, "category c, period "+strconv.Itoa(k), r.ByPeriod.Category(cc, k), 50*(s(fk)-s(fk+1)))
	}
}

// TestGivenHeldEquilibrium: without the time held so far, a constant
// holding of 3 leaves a uniform share of it, so a Poisson count of mean
// 12 leaves 4 a period for three periods, held 10, 6 and 2 within them.
func TestGivenHeldEquilibrium(t *testing.T) {
	count, _ := dist.NewPoisson(12)
	g := heldGraph(C(3), 5, variate.Given{Node: "a", To: "b", Count: count})
	m, r := startedRun(t, g, variate.Config{Repeats: 20000, Seed: 4})
	b := indexOf(m.NodeIDs(), "b")
	cr := indexOf(m.Categories(), "r")
	for k, want := range []float64{4, 4, 4, 0, 0} {
		nearMean(t, "entries at b, period "+strconv.Itoa(k), r.ByPeriod.Entries(b, k), want)
	}
	for k, want := range []float64{10, 6, 2, 0, 0} {
		nearMean(t, "held, period "+strconv.Itoa(k), r.ByPeriod.Category(cr, k), want)
	}
}

// lineGraph: a queue of c servers of speed 1 with constant or random
// work and no arrivals, which a given state fills.
func lineGraph(c int, work dist.Distribution, horizon int, state ...variate.Given) variate.Graph {
	return variate.Graph{
		ID:      "line",
		Start:   variate.Start{Horizon: horizon, State: state},
		Sources: []variate.Source{{ID: "s", Frequency: C(0), To: "q"}},
		Nodes: []variate.Node{
			{ID: "q", Queue: &variate.Queue{Servers: c, Speed: 1, Work: work}, Outcomes: []variate.Outcome{{To: "done"}}},
			{ID: "done", Functions: []variate.Function{Fn("v", "c", C(1))}},
		},
	}
}

// TestGivenLineEquilibrium: five rounds aged 2 at one server with work 1
// and the time served unknown: the oldest has a uniform U left, so one
// leaves each period from period 0; the one in service has waited
// 1 + U, the j-th in line 1.5 + j on average, a mean wait of 3.5 and a
// mean stay of 4.5 over the five.
func TestGivenLineEquilibrium(t *testing.T) {
	g := lineGraph(1, C(1), 6, variate.Given{Node: "q", Count: C(5), Age: C(2)})
	cfg := variate.Config{Repeats: 4000, Seed: 5}
	m, r := startedRun(t, g, cfg)
	done := indexOf(m.NodeIDs(), "done")
	for k, want := range []float64{1, 1, 1, 1, 1, 0} {
		nearMean(t, "served, period "+strconv.Itoa(k), r.ByPeriod.Entries(done, k), want)
	}
	q := queueRun(t, g, cfg)
	if !(math.Abs(q.Wait.Mean-3.5) <= 4*q.Wait.StdErr && math.Abs(q.Stay.Mean-4.5) <= 4*q.Stay.StdErr) {
		t.Errorf("wait %v ± %v, stay %v ± %v; want 3.5 and 4.5", q.Wait.Mean, q.Wait.StdErr, q.Stay.Mean, q.Stay.StdErr)
	}
	if q.Joined.Mean != 5 || q.Served.Mean != 5 {
		t.Errorf("joined %v, served %v a repeat; want the 5 given", q.Joined.Mean, q.Served.Mean)
	}
}

// TestGivenLineInService: four rounds aged 3 at two servers with work 2,
// served 0.5 so far: the two oldest leave at 1.5 having waited 2.5, the
// others at 3.5 having waited 4.5, so a mean wait of 3.5 and stay of 5.5.
func TestGivenLineInService(t *testing.T) {
	g := lineGraph(2, C(2), 5, variate.Given{Node: "q", Count: C(4), Age: C(3), InService: C(0.5)})
	cfg := variate.Config{Repeats: 50, Seed: 6}
	m, r := startedRun(t, g, cfg)
	done := indexOf(m.NodeIDs(), "done")
	for k, want := range []float64{0, 2, 0, 2, 0} {
		if got := r.ByPeriod.Entries(done, k).Mean; got != want {
			t.Errorf("served in period %d: %v, want %v", k, got, want)
		}
	}
	q := queueRun(t, g, cfg)
	if !(math.Abs(q.Wait.Mean-3.5) <= 1e-12 && math.Abs(q.Stay.Mean-5.5) <= 1e-12) {
		t.Errorf("wait %v, stay %v; want 3.5 and 5.5", q.Wait.Mean, q.Stay.Mean)
	}
}

// TestGivenLineJointEquilibrium: a round in service with its time served
// unknown has work left R and work done A from the equilibrium pair,
// each of mean E[W²]/(2E[W]): 1 for a lognormal work of mean and
// deviation 1. Aged 100 alone at a server it waits 100 − A and stays
// 100 + R, on average 99 and 101.
func TestGivenLineJointEquilibrium(t *testing.T) {
	w, _ := dist.NewLognormalFromMeanSD(1, 1)
	g := lineGraph(1, w, 4, variate.Given{Node: "q", Count: C(1), Age: C(100)})
	q := queueRun(t, g, variate.Config{Repeats: 40000, Seed: 7})
	if !(math.Abs(q.Wait.Mean-99) <= 4*q.Wait.StdErr && math.Abs(q.Stay.Mean-101) <= 4*q.Stay.StdErr) {
		t.Errorf("wait %v ± %v, stay %v ± %v; want 99 and 101", q.Wait.Mean, q.Wait.StdErr, q.Stay.Mean, q.Stay.StdErr)
	}
}

// TestGivenExactElsewhere: a given state takes the exact per-period
// means from what its rounds reach, and leaves another source's.
func TestGivenExactElsewhere(t *testing.T) {
	g := heldGraph(C(3), 4, variate.Given{Node: "a", To: "b", Count: C(5)})
	g.Sources = append(g.Sources, variate.Source{ID: "u", Frequency: C(2), To: "x"})
	g.Nodes = append(g.Nodes, variate.Node{ID: "x", Functions: []variate.Function{Fn("w", "y", C(1))}})
	g.Start.State[0].Source = "s"
	m, _, err := g.Compile(variate.Config{Repeats: 100})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := m.Analytic()
	b, x := indexOf(m.NodeIDs(), "b"), indexOf(m.NodeIDs(), "x")
	for k, row := range e.ByPeriod {
		if !math.IsNaN(row[entryColumn(m, b)]) {
			t.Errorf("period %d: entries at b exact at %v, but the given rounds reach it", k, row[entryColumn(m, b)])
		}
		if got := row[entryColumn(m, x)]; got != 2 {
			t.Errorf("period %d: entries at x %v, want the exact 2", k, got)
		}
	}
}

// TestGivenRepeatsExactly: the same seed repeats a run from a given
// state on any number of workers.
func TestGivenRepeatsExactly(t *testing.T) {
	w, _ := dist.NewExponential(1)
	g := lineGraph(2, w, 6, variate.Given{Node: "q", Count: C(9), Age: C(1)})
	g.Sources[0].Frequency = C(1.5)
	var totals [2][6]float64
	for i, workers := range []int{1, 4} {
		_, r := startedRun(t, g, variate.Config{Repeats: 2000, Seed: 8, Workers: workers})
		for k := range 6 {
			totals[i][k] = r.ByPeriod.Total(k).Mean
		}
	}
	if totals[0] != totals[1] {
		t.Errorf("per-period totals %v on one worker, %v on four", totals[0], totals[1])
	}
}

// TestGivenRefusals: a given state needs a started run, a queue for ages
// and a holding drawn from a law for elapsed times, a named source when
// there are several, and counts that are not negative.
func TestGivenRefusals(t *testing.T) {
	neg, _ := dist.NewUniform(-1, 3)
	at := func(g variate.Given) variate.Graph { return lineGraph(1, C(1), 3, g) }
	held := func(g variate.Given) variate.Graph { return heldGraph(C(1), 3, g) }
	steady := held(variate.Given{Node: "a", To: "b", Count: C(1)})
	steady.Start.Horizon = 0
	two := held(variate.Given{Node: "a", To: "b", Count: C(1)})
	two.Sources = append(two.Sources, variate.Source{ID: "u", Frequency: C(1), To: "b"})
	outage := held(variate.Given{Node: "a", To: "b", Count: C(1)})
	outage.Nodes[0].Outcomes[0] = variate.Outcome{To: "b", OutageHolding: true}
	par, _ := dist.NewPareto(0.2, 1.5) // infinite variance: its residual has an infinite mean
	cases := []struct {
		name string
		g    variate.Graph
		want string
	}{
		{"steady", steady, "give a horizon"},
		{"node", at(variate.Given{Node: "z", Count: C(1), Age: C(1)}), "not found"},
		{"source", at(variate.Given{Node: "q", Source: "z", Count: C(1), Age: C(1)}), "source z not found"},
		{"not a queue", held(variate.Given{Node: "a", Count: C(1), Age: C(1)}), "not a queue"},
		{"no age", at(variate.Given{Node: "q", Count: C(1)}), "age"},
		{"elapsed at a queue", at(variate.Given{Node: "q", Count: C(1), Age: C(1), Elapsed: C(1)}), "elapsed is a holding's"},
		{"age in a holding", held(variate.Given{Node: "a", To: "b", Count: C(1), Age: C(1)}), "age and in_service are a queue's"},
		{"no such outcome", held(variate.Given{Node: "a", To: "a", Count: C(1)}), "exactly one outcome"},
		{"outage", outage, "outage"},
		{"two sources", two, "name the rounds' source"},
		{"negative count", held(variate.Given{Node: "a", To: "b", Count: neg}), "does not fit"},
		{"residual in service", lineGraph(1, par, 3, variate.Given{Node: "q", Count: C(1), Age: C(1)}), "give in_service"},
		{"residual held", heldGraph(par, 3, variate.Given{Node: "a", To: "b", Count: C(1)}), "give elapsed"},
	}
	for _, c := range cases {
		_, _, err := c.g.Compile(variate.Config{})
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want one containing %q", c.name, err, c.want)
		}
	}
	for _, g := range []variate.Graph{
		lineGraph(1, par, 3, variate.Given{Node: "q", Count: C(1), Age: C(1), InService: C(1)}),
		heldGraph(par, 3, variate.Given{Node: "a", To: "b", Count: C(1), Elapsed: C(1)}),
	} {
		if _, _, err := g.Compile(variate.Config{}); err != nil {
			t.Errorf("%s with the time so far given: %v", g.ID, err)
		}
	}
	m, _, err := held(variate.Given{Node: "a", To: "b", Count: C(1)}).Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.NewWorkspace(variate.Config{Observer: observerFunc(nil)}); err == nil || !strings.Contains(err.Error(), "given state") {
		t.Errorf("round stream: error %v, want one naming the given state", err)
	}
}

// simpson integrates f over [a, b] by Simpson's rule on 2000 pieces.
func simpson(f func(float64) float64, a, b float64) float64 {
	const n = 2000
	h := (b - a) / n
	s := f(a) + f(b)
	for i := 1; i < n; i++ {
		w := 2.0
		if i%2 == 1 {
			w = 4
		}
		s += w * f(a+float64(i)*h)
	}
	return s * h / 3
}

// TestGivenNotReached: the given state's rounds count in the entries but
// not in their source's reach, which stays a share of its arrivals.
func TestGivenNotReached(t *testing.T) {
	g := lineGraph(1, C(0.1), 4, variate.Given{Node: "q", Count: C(30), Age: C(1)})
	g.Sources[0].Frequency = C(2)
	cfg := variate.Config{Repeats: 500, Seed: 9}
	m, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if _, err := w.Run(context.Background(), cfg.Seed); err != nil {
		t.Fatal(err)
	}
	var s variate.Summary
	w.Summarize(&s)
	if p := s.ReachProb["s"]["done"]; p.P != 1 {
		t.Errorf("reach of done from s %v, want 1: every arrival is served, and the given rounds are not arrivals", p.P)
	}
}

// TestGivenHeldIntoQueue: rounds released from a holding into a queue
// are served there and still count in no source's reach.
func TestGivenHeldIntoQueue(t *testing.T) {
	g := heldGraph(C(1), 6, variate.Given{Node: "a", To: "b", Count: C(10)})
	g.Sources[0].Frequency = C(1)
	g.Nodes[1].Queue = &variate.Queue{Servers: 1, Speed: 1, Work: C(0.1)}
	g.Nodes[1].Outcomes = []variate.Outcome{{To: "z"}}
	g.Nodes = append(g.Nodes, variate.Node{ID: "z"})
	cfg := variate.Config{Repeats: 300, Seed: 10}
	q := queueRun(t, g, cfg)
	if q.Joined.Mean != q.Served.Mean || q.Served.Mean < 10 {
		t.Errorf("joined %v, served %v a repeat; want all served, the 10 given among them", q.Joined.Mean, q.Served.Mean)
	}
	m, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := m.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if _, err := w.Run(context.Background(), cfg.Seed); err != nil {
		t.Fatal(err)
	}
	var s variate.Summary
	w.Summarize(&s)
	if p := s.ReachProb["s"]["z"]; p.P != 1 {
		t.Errorf("reach of z from s %v, want 1", p.P)
	}
}
