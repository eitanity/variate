package suite

import (
	"context"
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// laneGraph: Poisson rounds at rate lam a period reach a carried queue of
// c servers of speed s, each round bringing work w, in a started run of
// the horizon; the queue's per-time rate is 1, so a round books its stay.
func laneGraph(lam float64, c int, s float64, w dist.Distribution, horizon int) variate.Graph {
	return variate.Graph{
		ID:      "lane",
		Start:   variate.Start{Horizon: horizon},
		Sources: []variate.Source{{ID: "items", Frequency: C(lam), To: "lane"}},
		Nodes: []variate.Node{
			{ID: "lane", Queue: &variate.Queue{Servers: c, Speed: s, Work: w},
				Functions: []variate.Function{{Name: "open", Category: "risk", Model: C(1), Basis: variate.PerTime}},
				Outcomes:  []variate.Outcome{{To: "fixed"}}},
			{ID: "fixed"},
		},
	}
}

// queueRun runs g and returns its first queue's summary.
func queueRun(t *testing.T, g variate.Graph, cfg variate.Config) variate.QueueSummary {
	t.Helper()
	c, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if _, err := w.Run(context.Background(), cfg.Seed); err != nil {
		t.Fatal(err)
	}
	var s variate.Summary
	w.Summarize(&s)
	return s.Queues[c.QueueIDs()[0]]
}

// erlangCWait is the mean wait of M/M/c with arrival rate lam and service
// rate mu per server.
func erlangCWait(lam, mu float64, c int) float64 {
	a := lam / mu
	r := a / float64(c)
	term, sum := 1.0, 0.0
	for k := range c {
		if k > 0 {
			term *= a / float64(k)
		}
		sum += term
	}
	last := term * a / float64(c) / (1 - r)
	return last / (sum + last) / (float64(c)*mu - lam)
}

// TestQueueClosedForms: over a long horizon from empty the mean wait of
// one server with Poisson input is Pollaczek–Khinchine's λE[S²]/(2(1 − ρ))
// for constant, exponential and lognormal work, and of three exponential
// servers Erlang C's, each within four standard errors.
func TestQueueClosedForms(t *testing.T) {
	expo, _ := dist.NewExponential(1.233)
	logn, _ := dist.NewLognormalFromMeanSD(1.233, 1.233)
	lam := 6.12
	pk := func(es, es2 float64) float64 { return lam * es2 / (2 * (1 - lam*es)) }
	sp := 10.85
	cases := []struct {
		name string
		g    variate.Graph
		want float64
	}{
		{"M/D/1", laneGraph(lam, 1, sp, C(1.233), 200), pk(1.233/sp, (1.233/sp)*(1.233/sp))},
		{"M/M/1", laneGraph(lam, 1, sp, expo, 200), pk(1.233/sp, 2*(1.233/sp)*(1.233/sp))},
		{"M/LN/1", laneGraph(lam, 1, sp, logn, 200), pk(1.233/sp, 2*(1.233/sp)*(1.233/sp))},
		{"M/M/3", laneGraph(lam, 3, sp/3, expo, 200), erlangCWait(lam, sp/3/1.233, 3)},
	}
	for _, c := range cases {
		q := queueRun(t, c.g, variate.Config{Repeats: 300, Seed: 4})
		if math.Abs(q.Wait.Mean-c.want) > 4*q.Wait.StdErr {
			t.Errorf("%s: mean wait %v ± %v, closed form %v", c.name, q.Wait.Mean, q.Wait.StdErr, c.want)
		}
		if math.Abs(q.Utilisation-lam*1.233/sp) > 0.01 {
			t.Errorf("%s: utilisation %v, want ρ = %v", c.name, q.Utilisation, lam*1.233/sp)
		}
	}
}

// TestQueueLane: the fast-track lane of the remediation paper, at a
// reservation of 25.4%, has a mean stay of 136.8 days by
// Pollaczek–Khinchine with a lognormal S of mean 123.3 days and E[S²] of
// 1.47e5 days²; in periods of 100 days, within four standard errors.
func TestQueueLane(t *testing.T) {
	s, _ := dist.NewLognormalFromMeanSD(1.233, math.Sqrt(14.7-1.233*1.233))
	q := queueRun(t, laneGraph(6.12, 1, 10.85, s, 1000), variate.Config{Repeats: 80, Seed: 1})
	es := 1.233 / 10.85
	want := 6.12*14.7/(10.85*10.85)/(2*(1-6.12*es)) + es
	if math.Abs(q.Stay.Mean-want) > 4*q.Stay.StdErr {
		t.Errorf("mean stay %v ± %v periods, Pollaczek–Khinchine %v", q.Stay.Mean, q.Stay.StdErr, want)
	}
}

// TestQueueOverload: above capacity the line grows at the arrival rate
// less the service rate, λ − c·speed/E[work] a period, the fluid limit.
func TestQueueOverload(t *testing.T) {
	expo, _ := dist.NewExponential(1)
	g := laneGraph(12, 2, 5, expo, 40)
	c, _, err := g.Compile(variate.Config{Repeats: 400, Seed: 6})
	if err != nil {
		t.Fatal(err)
	}
	r, err := c.Simulate(context.Background(), variate.Config{Repeats: 400, Seed: 6})
	if err != nil {
		t.Fatal(err)
	}
	late := r.ByPeriod.Queue(0, 4, 39).Mean - r.ByPeriod.Queue(0, 4, 19).Mean
	if want := 20.0 * (12 - 2*5); math.Abs(late-want) > 0.05*want {
		t.Errorf("line grew %v over periods 20 to 40, want about %v", late, want)
	}
}

// resetGraph is a steady-state queue that resets each period, sending
// its unserved rounds to a node of their own.
func resetGraph(lam float64, work dist.Distribution) variate.Graph {
	h, _ := dist.NewExponential(0.5)
	return variate.Graph{
		ID:      "desk",
		Sources: []variate.Source{{ID: "jobs", Frequency: C(lam), To: "desk"}},
		Nodes: []variate.Node{
			{ID: "desk", Queue: &variate.Queue{Servers: 2, Speed: 1, Work: work, Reset: true, Unfinished: "over"},
				Functions: []variate.Function{{Name: "handling", Category: "cost", Model: C(100), Basis: variate.PerTime}},
				Outcomes:  []variate.Outcome{{To: "done", Holding: h}}},
			{ID: "done", Functions: []variate.Function{Fn("fee", "cost", C(5))}},
			{ID: "over", Functions: []variate.Function{Fn("penalty", "cost", C(20))}},
		},
	}
}

// TestQueueReset: a reset queue sends what is left at each period's end
// to its unfinished node, so its rounds are served or sent on, and the
// line at a period's end is the rounds sent on; a period much longer
// than a service gives the carried queue's mean wait, two exponential
// servers' Erlang C within five per cent.
func TestQueueReset(t *testing.T) {
	work, _ := dist.NewExponential(0.00028)
	g := resetGraph(5000, work)
	cfg := variate.Config{Periods: 400, Seed: 3}
	c, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	r, err := c.Simulate(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	joined, served, over := r.QueueCols[0].Sum.Value(), r.QueueCols[1].Sum.Value(), r.QueueCols[6].Sum.Value()
	if joined != served+over {
		t.Errorf("joined %v, served %v + unfinished %v", joined, served, over)
	}
	if line := r.QueueCols[4].Sum.Value(); line != over {
		t.Errorf("line at the periods' ends %v, sent on %v", line, over)
	}
	// The unfinished node's entries in period 0 come from the burn-in's
	// last period, and the run's last period sends its rounds past it.
	if got := r.EntriesPer[2].Sum.Value(); math.Abs(got-over) > 0.05*over {
		t.Errorf("entries at the unfinished node %v, sent on %v", got, over)
	}
	q := queueRun(t, g, cfg)
	want := erlangCWait(5000, 1/0.00028, 2)
	if math.Abs(q.Wait.Mean-want) > 0.05*want {
		t.Errorf("mean wait %v, Erlang C %v", q.Wait.Mean, want)
	}
}

// TestQueueExactMeans: in steady state what a queue books after it, and
// over its stay, has no exact mean; what is booked before it keeps one.
func TestQueueExactMeans(t *testing.T) {
	work, _ := dist.NewExponential(0.03)
	g := resetGraph(50, work)
	g.Nodes[0].Functions = append(g.Nodes[0].Functions, Fn("intake", "intake", C(2)))
	c, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := c.Analytic()
	cats := c.Categories()
	for i, name := range cats {
		switch got := e.MeanByCategory[i]; name {
		case "intake":
			if got != 100 {
				t.Errorf("intake, booked on arrival: exact %v, want 100", got)
			}
		default:
			if !math.IsNaN(got) {
				t.Errorf("%s, booked after the queue: exact %v, want NaN", name, got)
			}
		}
	}
	if e.EntriesPer[0] != 50 || !math.IsNaN(e.EntriesPer[1]) {
		t.Errorf("entries %v: the queue's exact, after it NaN", e.EntriesPer)
	}
}

// TestQueueRepeatsExactly: a model with queues repeats at a seed on any
// number of workers, started and steady.
func TestQueueRepeatsExactly(t *testing.T) {
	expo, _ := dist.NewExponential(1.233)
	work, _ := dist.NewExponential(0.03)
	for _, c := range []struct {
		g   variate.Graph
		cfg variate.Config
	}{
		{laneGraph(6.12, 2, 5, expo, 30), variate.Config{Repeats: 500, Seed: 9}},
		{resetGraph(50, work), variate.Config{Periods: 3000, Seed: 9}},
	} {
		var waits [2]float64
		for i, workers := range []int{1, 4} {
			cfg := c.cfg
			cfg.Workers = workers
			waits[i] = queueRun(t, c.g, cfg).Wait.Mean
		}
		if waits[0] != waits[1] {
			t.Errorf("%s: mean wait %v on one worker, %v on four", c.g.ID, waits[0], waits[1])
		}
	}
}

// TestQueueRefusals: what a queue cannot do, each refused with a reason.
func TestQueueRefusals(t *testing.T) {
	expo, _ := dist.NewExponential(1)
	carry := laneGraph(1, 1, 2, expo, 0)
	held := resetGraph(1, expo)
	held.Sources[0].To = "in"
	held.Nodes = append(held.Nodes, variate.Node{ID: "in", Outcomes: []variate.Outcome{{To: "desk", Holding: expo}}})
	noUnfinished := resetGraph(1, expo)
	noUnfinished.Nodes[0].Queue.Unfinished = ""
	stray := laneGraph(1, 1, 2, expo, 5)
	stray.Nodes[0].Queue.Unfinished = "fixed"
	zero := laneGraph(1, 0, 2, expo, 5)
	trig := laneGraph(1, 1, 2, expo, 5)
	trig.Nodes[1].Functions = []variate.Function{Fn("f", "c", C(1))}
	trig.Nodes = append(trig.Nodes, variate.Node{ID: "audit", Trigger: &variate.Trigger{Over: 1, Inputs: []variate.TriggerInput{{Node: "fixed"}}}})
	cases := []struct {
		name string
		g    variate.Graph
		cfg  variate.Config
		want string
	}{
		{"carry in steady state", carry, variate.Config{}, "needs a started run"},
		{"holding into a steady queue", held, variate.Config{}, "within the period"},
		{"reset without unfinished", noUnfinished, variate.Config{}, "needs an unfinished node"},
		{"unfinished without reset", stray, variate.Config{}, "only a queue that resets"},
		{"no servers", zero, variate.Config{}, "servers"},
		{"trigger after a queue", trig, variate.Config{}, "reached through a queue"},
	}
	for _, c := range cases {
		_, _, err := c.g.Compile(c.cfg)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want one containing %q", c.name, err, c.want)
		}
	}
	m, _, err := laneGraph(1, 1, 2, expo, 5).Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.NewWorkspace(variate.Config{Repeats: 10, Observer: nopObserver{}}); err == nil || !strings.Contains(err.Error(), "round stream") {
		t.Errorf("stream with a queue: error %v", err)
	}
}

// TestQueueStayQuantiles: a category booked only over a queue's stays has
// a histogram range, so its quantiles are finite.
func TestQueueStayQuantiles(t *testing.T) {
	expo, _ := dist.NewExponential(1)
	g := laneGraph(0.8, 1, 1, expo, 20)
	cfg := variate.Config{Repeats: 2000, Seed: 12}
	c, _, err := g.Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := c.NewWorkspace(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if _, err := w.Run(context.Background(), cfg.Seed); err != nil {
		t.Fatal(err)
	}
	var s variate.Summary
	w.Summarize(&s)
	st := s.ByCategory["risk"]
	if math.IsInf(st.P50, 0) || math.IsInf(st.P99, 0) || !(st.P50 > 0) {
		t.Errorf("category risk p50 %v, p99 %v; want finite quantiles of the time held", st.P50, st.P99)
	}
}

// TestQueueStartedSteps: a started run's expected steps count every
// arrival over the nodes it reaches, a queue or not: two nodes, two
// rounds a period, ten periods.
func TestQueueStartedSteps(t *testing.T) {
	m, _, err := laneGraph(2, 1, 1, C(0.1), 10).Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	e, _ := m.Analytic()
	if e.StepsPerPeriod != 40 {
		t.Errorf("steps per repeat %v, want 40", e.StepsPerPeriod)
	}
}
