package suite

import (
	"context"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/dist"
)

// summaryOf compiles and runs g, returning its warnings and summary.
func summaryOf(t *testing.T, g variate.Graph, cfg variate.Config) ([]variate.Warning, variate.Summary) {
	t.Helper()
	c, ws, err := g.Compile(cfg)
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
	return ws, s
}

func infVarWarnings(ws []variate.Warning) []string {
	var out []string
	for _, w := range ws {
		if w.Code == "infinite-variance" {
			out = append(out, w.Subject)
		}
	}
	return out
}

// TestInfiniteVarianceValue: a Pareto value of shape 1.5 at one node has
// a finite mean and infinite variance. The total, its category, the
// source reaching it and its node's contribution have their standard
// errors marked and their relative-error criteria not applicable; the
// other source's and category's are untouched, and the tail criterion
// still applies.
func TestInfiniteVarianceValue(t *testing.T) {
	par, _ := dist.NewPareto(0.2, 1.5)
	g := variate.Graph{
		ID: "heavy",
		Sources: []variate.Source{
			{ID: "s", Frequency: C(1), To: "a"},
			{ID: "u", Frequency: C(1), To: "b"},
		},
		Nodes: []variate.Node{
			{ID: "a", Functions: []variate.Function{Fn("loss", "heavy", par)}},
			{ID: "b", Functions: []variate.Function{Fn("fee", "light", C(1))}},
		},
	}
	ws, s := summaryOf(t, g, variate.Config{Periods: 2000, Seed: 3})
	if got := infVarWarnings(ws); len(got) != 1 || got[0] != "a:loss" {
		t.Errorf("warnings %v, want one for a:loss", got)
	}
	for name, want := range map[string]bool{"total": true, "category heavy": true, "source s": true, "category light": false, "source u": false} {
		var st variate.Stat
		switch name {
		case "total":
			st = s.Total
		case "category heavy", "category light":
			st = s.ByCategory[name[len("category "):]]
		default:
			st = s.BySource[name[len("source "):]]
		}
		if st.StdErrUnreliable != want {
			t.Errorf("%s: stderr unreliable %v, want %v", name, st.StdErrUnreliable, want)
		}
	}
	if !s.ContribPer["a"].StdErrUnreliable || s.ContribPer["b"].StdErrUnreliable || s.EntriesPer["a"].StdErrUnreliable {
		t.Errorf("contributions a %v b %v, entries a %v: want only a's contribution marked",
			s.ContribPer["a"].StdErrUnreliable, s.ContribPer["b"].StdErrUnreliable, s.EntriesPer["a"].StdErrUnreliable)
	}
	na := map[string]bool{"target-rse total": true, "target-rse category heavy": true, "target-rse source s": true, "contrib-rse a": true}
	for _, row := range s.Plan {
		key := row.Criterion + " " + row.Driver
		want := ""
		if na[key] {
			want = "infinite variance"
		}
		if row.NotApplicable != want {
			t.Errorf("%s: not applicable %q, want %q", key, row.NotApplicable, want)
		}
		if row.NotApplicable != "" && (!row.Met || row.PeriodsNeeded != 0) {
			t.Errorf("%s: met %v, periods %d; want met and left out of the plan", key, row.Met, row.PeriodsNeeded)
		}
	}
}

// TestInfiniteVarianceCount: a source whose rate per period is a Pareto
// of shape 1.5 has a count of infinite variance, which marks the entries
// and values its rounds reach.
func TestInfiniteVarianceCount(t *testing.T) {
	par, _ := dist.NewPareto(0.5, 1.5)
	g := variate.Graph{
		ID:      "heavy count",
		Sources: []variate.Source{{ID: "s", Frequency: par, To: "a"}},
		Nodes:   []variate.Node{{ID: "a", Functions: []variate.Function{Fn("fee", "c", C(1))}}},
	}
	ws, s := summaryOf(t, g, variate.Config{Periods: 2000, Seed: 3})
	if got := infVarWarnings(ws); len(got) != 1 || got[0] != "s" {
		t.Errorf("warnings %v, want one for source s", got)
	}
	if !s.Total.StdErrUnreliable || !s.EntriesPer["a"].StdErrUnreliable || !s.BySource["s"].StdErrUnreliable {
		t.Errorf("total %v, entries %v, source %v: want all marked", s.Total.StdErrUnreliable, s.EntriesPer["a"].StdErrUnreliable, s.BySource["s"].StdErrUnreliable)
	}
}

// TestInfiniteVarianceHolding: a per-time rate of 1 over a holding of
// infinite variance books at most 1 a round in a period, so nothing is
// marked.
func TestInfiniteVarianceHolding(t *testing.T) {
	par, _ := dist.NewPareto(0.2, 1.5)
	ws, s := summaryOf(t, heldGraphFrom(par), variate.Config{Periods: 2000, Seed: 3})
	if got := infVarWarnings(ws); len(got) != 0 || s.Total.StdErrUnreliable {
		t.Errorf("warnings %v, total marked %v: want neither", got, s.Total.StdErrUnreliable)
	}
}

// heldGraphFrom is heldGraph fed one round a period, in steady state.
func heldGraphFrom(hold dist.Continuous) variate.Graph {
	g := heldGraph(hold, 0)
	g.Sources[0].Frequency = C(1)
	return g
}

// TestInfiniteVarianceQueue: a carried queue's wait and stay have
// infinite variance when its work has, and their errors are marked; the
// horizon total, its per-time values cut at the horizon, is not. A reset
// queue's rounds stay at most a period, so nothing is marked.
func TestInfiniteVarianceQueue(t *testing.T) {
	par, _ := dist.NewPareto(0.2, 1.5)
	g := lineGraph(1, par, 12)
	g.Sources[0].Frequency = C(1)
	g.Nodes[0].Functions = []variate.Function{{Name: "open", Category: "c", Model: C(1), Basis: variate.PerTime}}
	ws, s := summaryOf(t, g, variate.Config{Repeats: 200, Seed: 3})
	q := s.Queues["q"]
	if got := infVarWarnings(ws); len(got) != 1 || got[0] != "q" {
		t.Errorf("warnings %v, want one for q", got)
	}
	if !q.Wait.StdErrUnreliable || !q.Stay.StdErrUnreliable || s.Total.StdErrUnreliable {
		t.Errorf("wait %v, stay %v, total %v: want the wait and stay marked, not the total", q.Wait.StdErrUnreliable, q.Stay.StdErrUnreliable, s.Total.StdErrUnreliable)
	}
	g.Nodes[0].Queue.Reset = true
	g.Nodes[0].Queue.Unfinished = "done"
	ws, s = summaryOf(t, g, variate.Config{Repeats: 200, Seed: 3})
	if got := infVarWarnings(ws); len(got) != 0 || s.Queues["q"].Wait.StdErrUnreliable {
		t.Errorf("reset: warnings %v, wait marked %v: want neither", got, s.Queues["q"].Wait.StdErrUnreliable)
	}
}
