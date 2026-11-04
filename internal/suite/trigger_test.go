package suite

import (
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// triggered is failures at rate lam, each booking a repair; a year with
// more than over failures sets off an inspection, which books its cost
// and finds a defect with probability 0.3.
func triggered(lam, over float64) variate.Graph {
	return variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(lam), To: "failure"}},
		Nodes: []variate.Node{
			{ID: "failure", Functions: []variate.Function{Fn("repair", "cost", Pert(1e3, 2e3, 6e3))}},
			{ID: "inspection", Functions: []variate.Function{Fn("inspect", "cost", Pert(2e4, 3e4, 6e4))},
				Trigger:  &variate.Trigger{Over: over, Entries: []string{"failure"}},
				Outcomes: []variate.Outcome{{To: "redesign", P: C(0.3)}, {To: "clear"}}},
			{ID: "redesign", Functions: []variate.Function{Fn("redesign", "cost", C(2e5))}},
			{ID: "clear"},
		},
	}
}

// poissonAbove is P(N > k) for N Poisson with mean lam.
func poissonAbove(lam float64, k int) float64 {
	term, below := math.Exp(-lam), 0.0
	for i := 0; i <= k; i++ {
		below += term
		term *= lam / float64(i+1)
	}
	if below > 0.5 {
		// The upper tail summed directly keeps its relative precision.
		above := 0.0
		for i := k + 1; term > 1e-300*above || above == 0; i++ {
			above += term
			term *= lam / float64(i+1)
			if i > k+10000 {
				break
			}
		}
		return above
	}
	return 1 - below
}

// TestTriggerExact: a trigger on a count fires with P(N > over) for N
// the period's Poisson count, from the transform of the sum, for a
// common level and a rare one beyond the default tail level; the mean
// is the failures' plus P(fire) times the inspection's walk.
func TestTriggerExact(t *testing.T) {
	for _, tc := range []struct {
		lam  float64
		over int
	}{{4, 3}, {4, 14}, {0.5, 0}} {
		c, _, err := triggered(tc.lam, float64(tc.over)).Compile(variate.Config{})
		if err != nil {
			t.Fatal(err)
		}
		e, _ := c.Analytic()
		if len(e.Firing) != 1 {
			t.Fatalf("firings %v", e.Firing)
		}
		f := e.Firing[0]
		want := poissonAbove(tc.lam, tc.over)
		if !f.Exact() || f.Lo > want*(1+1e-9) || f.Hi < want*(1-1e-9) || f.Hi-f.Lo > 1e-3*want {
			t.Errorf("λ %g over %d: P(fire) [%g, %g], want %g", tc.lam, tc.over, f.Lo, f.Hi, want)
		}
		p := (f.Lo + f.Hi) / 2
		mean := tc.lam*Pert(1e3, 2e3, 6e3).Mean() + p*(Pert(2e4, 3e4, 6e4).Mean()+0.3*2e5)
		if !near(e.Mean, mean, 1e-9) {
			t.Errorf("λ %g over %d: mean %g, want %g", tc.lam, tc.over, e.Mean, mean)
		}
		if !math.IsNaN(e.Var) {
			t.Errorf("variance %g, want NaN", e.Var)
		}
		if pz := math.Exp(-tc.lam); !near(e.PZero(), pz, 1e-9) {
			t.Errorf("P(T = 0) %g, want %g", e.PZero(), pz)
		}
		if got := e.EntriesPer[2]; !near(got, 0.3*p, 1e-12) {
			t.Errorf("redesign entries %g, want %g", got, 0.3*p)
		}
	}
}

// TestTriggerContinuous: on a compound sum, P(S > t) is the Poisson
// mixture of the Erlang survival of the exponential costs.
func TestTriggerContinuous(t *testing.T) {
	cost, _ := dist.NewExponential(1000)
	lam, over := 5.0, 12000.0
	g := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(lam), To: "loss"}},
		Nodes: []variate.Node{
			{ID: "loss", Functions: []variate.Function{Fn("cost", "cost", cost)}},
			{ID: "review", Functions: []variate.Function{Fn("fee", "cost", C(500))},
				Trigger: &variate.Trigger{Over: over, Inputs: []variate.TriggerInput{{Node: "loss"}}}},
		},
	}
	c, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	want, pn := 0.0, math.Exp(-lam)
	for n := 1; n < 200; n++ {
		pn *= lam / float64(n)
		// P(Erlang(n) > t) = P(Poisson(t/θ) < n).
		x, term, below := over/1000, math.Exp(-over/1000), 0.0
		for i := 0; i < n; i++ {
			below += term
			term *= x / float64(i+1)
		}
		want += pn * below
	}
	f := c.Firings()[0]
	if !f.Exact() || f.Lo > want || f.Hi < want || f.Hi-f.Lo > 1e-3*want {
		t.Errorf("P(fire) [%g, %g], want %g", f.Lo, f.Hi, want)
	}
}

// TestTriggerThresholdOutside: a threshold whose tested value is not in
// the sum still decides which rounds reach the input, with P(X > over).
func TestTriggerThresholdOutside(t *testing.T) {
	cost := lognormal(2000, 8000)
	g := thresholded(cost)
	g.Nodes = append(g.Nodes, variate.Node{ID: "review", Functions: []variate.Function{Fn("fee", "admin", C(1e4))},
		Trigger: &variate.Trigger{Over: 2, Entries: []string{"claim"}}})
	g.Sources[0].Frequency = C(5)
	c, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	want := poissonAbove(5*cost.Survival(5000), 2)
	if f := c.Firings()[0]; !f.Exact() || f.Lo > want*(1+1e-9) || f.Hi < want*(1-1e-9) || f.Hi-f.Lo > 1e-3*want {
		t.Errorf("P(fire) [%g, %g], want %g", f.Lo, f.Hi, want)
	}
}

// TestTriggerRun: a run's firings per period agree with P(fire), and
// its mean with the exact one, including with a holding time before the
// input, whose bookings count toward the period their event arrived in.
func TestTriggerRun(t *testing.T) {
	for _, delayed := range []bool{false, true} {
		g := triggered(4, 5)
		if delayed {
			g.Sources[0].To = "alarm"
			hold, _ := dist.NewUniform(0.2, 0.9)
			g.Nodes = append(g.Nodes, variate.Node{ID: "alarm", Next: "failure", NextHolding: hold})
		}
		c, r := run(t, g, variate.Config{Periods: 200000, Workers: 4}, 5)
		f := c.Firings()[0]
		p := (f.Lo + f.Hi) / 2
		n := float64(r.Periods)
		fired := float64(r.Events[1]) / n
		if se := math.Sqrt(p * (1 - p) / n); math.Abs(fired-p) > 5*se {
			t.Errorf("delayed %v: firings %g a period, P(fire) %g ± %g", delayed, fired, p, se)
		}
		e, _ := c.Analytic()
		if z := (r.Total.Mean() - e.Mean) / r.Total.StdErr(delayed); math.Abs(z) > 5 {
			t.Errorf("delayed %v: mean %g, exact %g (z %.2f)", delayed, r.Total.Mean(), e.Mean, z)
		}
	}
}

// TestTriggerUnknown: where the transform of the sum does not apply,
// P(fire) and the means it reaches are NaN with the reason, the means
// it does not reach stay exact, and the run still fires.
func TestTriggerUnknown(t *testing.T) {
	g := triggered(4, 5)
	rate, _ := dist.NewConstant(4)
	g.Sources[0] = variate.Source{ID: "s", Frequency: rate, Repair: Pert(0.01, 0.02, 0.05), To: "failure"}
	c, r := run(t, g, variate.Config{Periods: 20000, Workers: 2}, 7)
	e, _ := c.Analytic()
	f := e.Firing[0]
	if f.Exact() || !strings.Contains(f.Reason, "repair") {
		t.Errorf("firing %+v, want unknown with the repair reason", f)
	}
	if !math.IsNaN(e.Mean) || !math.IsNaN(e.EntriesPer[1]) || !math.IsNaN(e.MeanBySource[1]) {
		t.Errorf("mean %g, inspection entries %g, source %g, want NaN", e.Mean, e.EntriesPer[1], e.MeanBySource[1])
	}
	if math.IsNaN(e.EntriesPer[0]) || math.IsNaN(e.Contrib[0]) {
		t.Errorf("failure entries %g, contribution %g, want exact", e.EntriesPer[0], e.Contrib[0])
	}
	if r.Events[1] == 0 {
		t.Errorf("no firings in %d periods", r.Periods)
	}
}

// TestTriggerErrors: a trigger entered by an outcome or a source, with
// a negative level, no inputs, both inputs and entries, a missing input,
// an input named twice or a cycle of triggers is refused.
func TestTriggerErrors(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(g *variate.Graph)
		want string
	}{
		{"outcome", func(g *variate.Graph) { g.Nodes[0].Next = "inspection" }, "entered only by firing"},
		{"source", func(g *variate.Graph) { g.Sources[0].To = "inspection" }, "entered only by firing"},
		{"negative", func(g *variate.Graph) { g.Nodes[1].Trigger.Over = -1 }, "not negative"},
		{"nothing", func(g *variate.Graph) { g.Nodes[1].Trigger.Entries = nil }, "needs inputs or entries"},
		{"both", func(g *variate.Graph) { g.Nodes[1].Trigger.Inputs = []variate.TriggerInput{{Node: "failure"}} }, "not both"},
		{"missing entries", func(g *variate.Graph) { g.Nodes[1].Trigger.Entries[0] = "nope" }, "entries node nope not found"},
		{"missing node", func(g *variate.Graph) {
			g.Nodes[1].Trigger.Entries, g.Nodes[1].Trigger.Inputs = nil, []variate.TriggerInput{{Node: "nope"}}
		}, "input node nope not found"},
		{"missing function", func(g *variate.Graph) {
			g.Nodes[1].Trigger.Entries, g.Nodes[1].Trigger.Inputs = nil, []variate.TriggerInput{{Node: "failure", Function: "nope"}}
		}, "failure.nope not found"},
		{"twice", func(g *variate.Graph) {
			g.Nodes[1].Trigger.Entries, g.Nodes[1].Trigger.Inputs = nil, []variate.TriggerInput{{Node: "failure"}, {Node: "failure", Function: "repair"}}
		}, "named twice"},
		{"own walk", func(g *variate.Graph) { g.Nodes[1].Trigger.Entries[0] = "redesign" }, "trigger cycle: inspection <- inspection"},
		{"through another", func(g *variate.Graph) {
			g.Nodes[1].Trigger.Entries[0] = "audit"
			g.Nodes = append(g.Nodes, variate.Node{ID: "audit", Trigger: &variate.Trigger{Entries: []string{"redesign"}}})
		}, "trigger cycle:"},
	} {
		g := triggered(4, 3)
		tr := *g.Nodes[1].Trigger
		tr.Entries = append([]string(nil), tr.Entries...)
		g.Nodes[1].Trigger = &tr
		tc.edit(&g)
		_, _, err := g.Compile(variate.Config{})
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error %v, want %q", tc.name, err, tc.want)
		}
	}
}

// pumps is three pumps failing independently at rate lam each, a
// trigger per pump over 0 marking it failed in the period, and a top
// trigger over k − 1 on their firings: k of the three failing in the
// same period sets off a loss of cooling.
func pumps(lam float64, k int) variate.Graph {
	g := variate.Graph{}
	down := make([]string, 0, 3)
	for _, id := range []string{"a", "b", "c"} {
		g.Sources = append(g.Sources, variate.Source{ID: id, Frequency: C(lam), To: id})
		g.Nodes = append(g.Nodes,
			variate.Node{ID: id, Functions: []variate.Function{Fn("repair", "repair", C(100))}},
			variate.Node{ID: id + "_down", Trigger: &variate.Trigger{Entries: []string{id}}})
		down = append(down, id+"_down")
	}
	g.Nodes = append(g.Nodes, variate.Node{ID: "loss", Functions: []variate.Function{Fn("shutdown", "loss", C(1e5))},
		Trigger: &variate.Trigger{Over: float64(k - 1), Entries: down}})
	return g
}

// TestTriggerChain: the gates of three independent pumps fire as their
// laws say: OR (one of three), two of three and AND, each with the exact
// P(fire) 1 − (1 − p)³, the binomial tail and p³, with p the marks'
// exact 1 − e^(−λ), and a run's share of periods agreeing.
func TestTriggerChain(t *testing.T) {
	lam := 0.4
	p := 1 - math.Exp(-lam)
	for _, tc := range []struct {
		k    int
		want float64
	}{{1, 1 - (1-p)*(1-p)*(1-p)}, {2, 3*p*p*(1-p) + p*p*p}, {3, p * p * p}} {
		c, r := run(t, pumps(lam, tc.k), variate.Config{Periods: 200000, Workers: 4}, 9)
		ids := c.SourceIDs()
		if ids[len(ids)-1] != "loss" {
			t.Fatalf("the top trigger is not last: %v", ids)
		}
		for _, f := range c.Firings() {
			want := p
			if ids[f.Source] == "loss" {
				want = tc.want
			}
			if !f.Exact() || f.Lo > want*(1+1e-9) || f.Hi < want*(1-1e-9) || f.Hi-f.Lo > 1e-3*want {
				t.Errorf("k %d: %s P(fire) [%g, %g] %q, want %g", tc.k, ids[f.Source], f.Lo, f.Hi, f.Reason, want)
			}
		}
		n := float64(r.Periods)
		top := float64(r.Events[len(ids)-1]) / n
		if se := math.Sqrt(tc.want * (1 - tc.want) / n); math.Abs(top-tc.want) > 5*se {
			t.Errorf("k %d: top fires %g a period, want %g ± %g", tc.k, top, tc.want, se)
		}
		e, _ := c.Analytic()
		if mean := 3*lam*100 + tc.want*1e5; !near(e.Mean, mean, 1e-9) {
			t.Errorf("k %d: mean %g, want %g", tc.k, e.Mean, mean)
		}
	}
}

// TestTriggerNested: a gate over a gate, (a AND b) OR c, is exact as
// 1 − (1 − p²)(1 − p).
func TestTriggerNested(t *testing.T) {
	lam := 0.4
	p := 1 - math.Exp(-lam)
	g := pumps(lam, 1)
	g.Nodes[len(g.Nodes)-1].Trigger = &variate.Trigger{Entries: []string{"ab", "c_down"}}
	g.Nodes = append(g.Nodes, variate.Node{ID: "ab", Trigger: &variate.Trigger{Over: 1, Entries: []string{"a_down", "b_down"}}})
	c, _, err := g.Compile(variate.Config{})
	if err != nil {
		t.Fatal(err)
	}
	want := 1 - (1-p*p)*(1-p)
	for _, f := range c.Firings() {
		if c.SourceIDs()[f.Source] == "loss" && (!f.Exact() || f.Lo > want*(1+1e-9) || f.Hi < want*(1-1e-9)) {
			t.Errorf("P(fire) [%g, %g] %q, want %g", f.Lo, f.Hi, f.Reason, want)
		}
	}
}

// TestTriggerRepeated: a gate whose marks share a source (the repeated
// event), a shared factor or a period-scoped value is not a product of
// independent marks: its P(fire) is NaN naming what they share, and so
// is one summing values another trigger's walk books.
func TestTriggerRepeated(t *testing.T) {
	gamma, _ := dist.NewGamma(2, 0.5)
	for _, tc := range []struct {
		name string
		edit func(g *variate.Graph)
		want string
	}{
		{"source", func(g *variate.Graph) { g.Nodes[0].Next = "b" }, "both depend on source a"},
		{"factor", func(g *variate.Graph) {
			g.Factors = []variate.Factor{{ID: "climate", Model: gamma}}
			g.Sources[0].Factor, g.Sources[1].Factor = "climate", "climate"
		}, "both depend on a shared factor"},
		{"period value", func(g *variate.Graph) {
			g.Nodes = append(g.Nodes, variate.Node{ID: "shared", Functions: []variate.Function{{Name: "w", Category: "repair", Model: Pert(1, 2, 3), Scope: variate.PerPeriod}}})
			g.Nodes[0].Next, g.Nodes[2].Next = "shared", "shared"
		}, "both depend on the period-scoped quantities of node shared"},
		{"values", func(g *variate.Graph) {
			g.Nodes[len(g.Nodes)-1].Trigger = &variate.Trigger{Inputs: []variate.TriggerInput{{Node: "a_down"}}}
			g.Nodes[1].Functions = []variate.Function{Fn("log", "repair", C(1))}
		}, "only when it counts other triggers' firings"},
	} {
		g := pumps(0.4, 2)
		tc.edit(&g)
		c, _, err := g.Compile(variate.Config{})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		for _, f := range c.Firings() {
			if c.SourceIDs()[f.Source] == "loss" && (f.Exact() || !strings.Contains(f.Reason, tc.want)) {
				t.Errorf("%s: P(fire) [%g, %g] %q, want NaN with %q", tc.name, f.Lo, f.Hi, f.Reason, tc.want)
			}
		}
	}
}

// TestTriggerZero: a trigger counting entries into a node that books
// nothing fires in periods whose sources booked nothing, so P(T = 0) is
// NaN rather than the sources' own 1; counting a node that
// always books keeps it exact, and a run agrees with it.
func TestTriggerZero(t *testing.T) {
	g := variate.Graph{
		Sources: []variate.Source{{ID: "s", Frequency: C(1), To: "alarm"}},
		Nodes: []variate.Node{
			{ID: "alarm"},
			{ID: "inspect", Functions: []variate.Function{Fn("fee", "cost", C(100))}, Trigger: &variate.Trigger{Entries: []string{"alarm"}}},
		},
	}
	c, r := run(t, g, variate.Config{Periods: 100000, Workers: 2}, 3)
	e, _ := c.Analytic()
	if !math.IsNaN(e.PZero()) {
		t.Errorf("P(T = 0) %g, want NaN: the run's zero share is %g", e.PZero(), 1-r.Total.Hist.Exceedance(1e-9))
	}
	g.Nodes[0].Functions = []variate.Function{Fn("call", "cost", Pert(10, 20, 40))}
	c, r = run(t, g, variate.Config{Periods: 100000, Workers: 2}, 3)
	e, _ = c.Analytic()
	zero := 1 - r.Total.Hist.Exceedance(1e-9)
	if want := math.Exp(-1); !near(e.PZero(), want, 1e-9) || math.Abs(zero-want) > 5*math.Sqrt(want*(1-want)/1e5) {
		t.Errorf("P(T = 0) %g, run %g, want %g", e.PZero(), zero, want)
	}
}
