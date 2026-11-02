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
