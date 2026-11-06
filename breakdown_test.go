package variate_test

import (
	"context"
	"math"
	"strings"
	"testing"

	"code.eitanity.com/variate"
	"code.eitanity.com/variate/config"
)

// TestNoBreakdown: without the breakdowns a run at the same seed and
// length gives the same total, entries and layers, keeps no category,
// source or contribution figures or criteria, and an adaptive run stops
// on the total's own criteria.
func TestNoBreakdown(t *testing.T) {
	m, err := config.Load("examples/shift.yaml")
	if err != nil {
		t.Fatal(err)
	}
	run := func(cfg variate.Config) (*variate.Compiled, *variate.Workspace, *variate.Result) {
		t.Helper()
		c, _, err := m.Graph.Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		w, err := c.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(w.Close)
		r, err := w.Run(context.Background(), cfg.Seed)
		if err != nil {
			t.Fatal(err)
		}
		return c, w, r
	}
	cfg := m.Config
	cfg.Seed, cfg.Periods = 3, 20000
	_, wa, a := run(cfg)
	cfg.NoBreakdown = true
	c, wb, b := run(cfg)

	var sa, sb variate.Summary
	wa.Summarize(&sa)
	wb.Summarize(&sb)
	if sa.Total != sb.Total {
		t.Errorf("total differs:\n%+v\n%+v", sa.Total, sb.Total)
	}
	for name, s := range sa.EntriesPer {
		if sb.EntriesPer[name] != s {
			t.Errorf("entries at %s differ", name)
		}
	}
	if len(sa.ByCategory) == 0 || len(sa.TailShares) == 0 {
		t.Fatal("the breakdown run has no breakdown")
	}
	if len(sb.ByCategory)+len(sb.BySource)+len(sb.ContribPer)+len(sb.TailShares) != 0 {
		t.Errorf("breakdowns kept: %d categories, %d sources, %d nodes, %d tail shares", len(sb.ByCategory), len(sb.BySource), len(sb.ContribPer), len(sb.TailShares))
	}
	if a.Steps != b.Steps || a.Periods != b.Periods {
		t.Errorf("steps %d/%d, periods %d/%d", a.Steps, b.Steps, a.Periods, b.Periods)
	}
	for _, kind := range []string{"category", "source", "contrib"} {
		if _, ok := wb.QuantityRow(kind, "parts"); ok {
			t.Errorf("a %s row is found", kind)
		}
	}
	for _, row := range c.Plan() {
		if row.Criterion == "contrib-rse" || strings.HasPrefix(row.Driver, "category ") || strings.HasPrefix(row.Driver, "source ") {
			t.Errorf("a breakdown criterion is planned: %+v", row)
		}
	}
	e, err := c.Analytic()
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(e.VarByCategory[0]) || math.IsNaN(e.Var) {
		t.Errorf("category variance %g, total variance %g", e.VarByCategory[0], e.Var)
	}

	// Adaptive: the parts category binds the full run; without it the
	// total's tail does.
	cfg = m.Config
	cfg.Seed = 3
	full, _, _ := run(cfg)
	cfg.NoBreakdown = true
	lean, _, _ := run(cfg)
	if pf, pl := full.PlanPeriods(1), lean.PlanPeriods(1); pl >= pf {
		t.Errorf("planned %d periods without the breakdown, %d with it", pl, pf)
	}
}

// TestNoBreakdownVariance: leaving the breakdown variances uncomputed
// changes no figure a run at the same seed and length reports, keeps
// the breakdown criteria unplanned rather than dropped, and leaves the
// total's variance as it was.
func TestNoBreakdownVariance(t *testing.T) {
	m, err := config.Load("examples/shift.yaml")
	if err != nil {
		t.Fatal(err)
	}
	run := func(cfg variate.Config) (*variate.Compiled, *variate.Summary) {
		t.Helper()
		c, _, err := m.Graph.Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		w, err := c.NewWorkspace(cfg)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(w.Close)
		if _, err := w.Run(context.Background(), cfg.Seed); err != nil {
			t.Fatal(err)
		}
		var s variate.Summary
		w.Summarize(&s)
		return c, &s
	}
	cfg := m.Config
	cfg.Seed, cfg.Periods = 3, 20000
	ca, sa := run(cfg)
	cfg.NoBreakdownVariance = true
	cb, sb := run(cfg)

	if sa.Total != sb.Total || len(sa.ByCategory) == 0 {
		t.Errorf("total differs or no categories:\n%+v\n%+v", sa.Total, sb.Total)
	}
	for _, kv := range []struct {
		name string
		a, b map[string]variate.Stat
	}{{"category", sa.ByCategory, sb.ByCategory}, {"source", sa.BySource, sb.BySource}, {"node", sa.ContribPer, sb.ContribPer}} {
		if len(kv.a) != len(kv.b) {
			t.Errorf("%d %s figures, %d without the variances", len(kv.a), kv.name, len(kv.b))
		}
		for name, s := range kv.a { // unordered: each compared alone
			if kv.b[name] != s {
				t.Errorf("%s %s differs:\n%+v\n%+v", kv.name, name, s, kv.b[name])
			}
		}
	}
	if len(sa.TailShares) != len(sb.TailShares) {
		t.Errorf("%d tail shares, %d without the variances", len(sa.TailShares), len(sb.TailShares))
	}
	pa, pb := ca.Plan(), cb.Plan()
	if len(pa) != len(pb) {
		t.Fatalf("%d plan rows, %d without the variances", len(pa), len(pb))
	}
	planned := 0
	for i, row := range pb {
		breakdown := row.Criterion == "contrib-rse" || strings.HasPrefix(row.Driver, "category ") || strings.HasPrefix(row.Driver, "source ")
		if breakdown && pa[i].PeriodsNeeded > 0 {
			planned++
		}
		if breakdown && row.PeriodsNeeded != 0 || !breakdown && row != pa[i] {
			t.Errorf("row %+v, with the variances %+v", row, pa[i])
		}
	}
	if planned == 0 {
		t.Error("no breakdown criterion is planned with the variances")
	}
	ea, err := ca.Analytic()
	if err != nil {
		t.Fatal(err)
	}
	eb, err := cb.Analytic()
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(eb.VarByCategory[0]) || math.IsNaN(ea.VarByCategory[0]) || eb.Var != ea.Var {
		t.Errorf("category variance %g/%g, total variance %g/%g", ea.VarByCategory[0], eb.VarByCategory[0], ea.Var, eb.Var)
	}
}
